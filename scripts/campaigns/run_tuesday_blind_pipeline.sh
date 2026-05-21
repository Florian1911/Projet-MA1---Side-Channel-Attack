#!/usr/bin/env bash
set -euo pipefail

# Tuesday pipeline:
# - capture known-key profiling sets (same setup/firmware style)
# - capture blind set
# - align all sets with same keyless method
# - train HW and SBOX campaigns
# - fuse HW+SBOX predictions
# - run blind CPA baseline
#
# IMPORTANT:
# - Edit keys and counts below before running.
# - Flash firmware with each KNOWN key before each known capture.
# - For blind capture, flash unknown key and DO NOT use it in attack steps.

PYTHON=${PYTHON:-python}
PLAINTEXTS=${PLAINTEXTS:-plaintexts_no_uart.npy}
N_TRACES_KNOWN=${N_TRACES_KNOWN:-20000}
N_TRACES_BLIND=${N_TRACES_BLIND:-20000}
NUM_SAMPLES=${NUM_SAMPLES:-3968}
TIMEBASE=${TIMEBASE:-1}
TRIG_MV=${TRIG_MV:-1500}
PROBE_ATT=${PROBE_ATT:-10}

# Known profiling keys (edit as needed).
KNOWN_KEYS=(
  "00112233445566778899AABBCCDDEEFF"
  "C6A13B37878F5B826F4F8162A1C8D879"
  "9F4E2D1C7A6B5D3E8C0F12A4B6D9E3F1"
)

# Set this ONLY if you want the blind raw capture to keep key metadata for post-hoc validation.
# Leave empty for strictest process during capture.
BLIND_KEY_HEX="${BLIND_KEY_HEX:-}"

mkdir -p tuesday_runs
cd tuesday_runs

echo "==[1/8] Capture KNOWN datasets =="
idx=0
for K in "${KNOWN_KEYS[@]}"; do
  idx=$((idx + 1))
  out="known_k${idx}_20k_raw.npz"
  echo
  echo "[KNOWN ${idx}] FLASH firmware with key: ${K}"
  echo "Then press Enter to start capture..."
  read -r
  "$PYTHON" ../acquire_no_uart_2000.py \
    --n-traces "$N_TRACES_KNOWN" \
    --num-samples "$NUM_SAMPLES" \
    --timebase "$TIMEBASE" \
    --trig-channel A \
    --trig-direction rising \
    --trig-mv-probe "$TRIG_MV" \
    --probe-att-trig "$PROBE_ATT" \
    --plaintexts "$PLAINTEXTS" \
    --key-hex "$K" \
    --output "$out"
done

echo
echo "==[2/8] Capture BLIND dataset =="
echo "FLASH firmware with UNKNOWN blind key now."
echo "Then press Enter to start blind capture..."
read -r
if [[ -n "${BLIND_KEY_HEX}" ]]; then
  "$PYTHON" ../acquire_no_uart_2000.py \
    --n-traces "$N_TRACES_BLIND" \
    --num-samples "$NUM_SAMPLES" \
    --timebase "$TIMEBASE" \
    --trig-channel A \
    --trig-direction rising \
    --trig-mv-probe "$TRIG_MV" \
    --probe-att-trig "$PROBE_ATT" \
    --plaintexts "$PLAINTEXTS" \
    --key-hex "$BLIND_KEY_HEX" \
    --output unknown_blind_20k_raw.npz
else
  "$PYTHON" ../acquire_no_uart_2000.py \
    --n-traces "$N_TRACES_BLIND" \
    --num-samples "$NUM_SAMPLES" \
    --timebase "$TIMEBASE" \
    --trig-channel A \
    --trig-direction rising \
    --trig-mv-probe "$TRIG_MV" \
    --probe-att-trig "$PROBE_ATT" \
    --plaintexts "$PLAINTEXTS" \
    --output unknown_blind_20k_raw.npz
fi

echo
echo "==[3/8] Keyless alignment for all sets (same method) =="
for f in known_k*_20k_raw.npz unknown_blind_20k_raw.npz; do
  o="${f%.npz}_refal.npz"
  "$PYTHON" ../align_local_per_trace.py \
    --in "$f" \
    --out "$o" \
    --center 520 \
    --window 260 \
    --max-shift 80 \
    --iters 2 \
    --mode ref \
    --ref median
done

echo
echo "==[4/8] Build blind no-key file =="
"$PYTHON" - <<'PY'
import numpy as np
d = np.load("unknown_blind_20k_raw_refal.npz")
np.savez(
    "unknown_blind_20k_raw_refal_nokey.npz",
    traces=d["traces"],
    plaintexts=d["plaintexts"],
)
print("saved unknown_blind_20k_raw_refal_nokey.npz")
PY

echo
echo "==[5/8] Train HW campaign =="
PROF=$(ls known_k*_20k_raw_refal.npz | paste -sd "," -)
"$PYTHON" ../multisession_mlp_16bytes.py \
  --profile-datasets "$PROF" \
  --attack-dataset "unknown_blind_20k_raw_refal.npz" \
  --outdir exp_blind_hw_tuesday \
  --bytes all \
  --label-mode hw \
  --epochs 120 --patience 20 \
  --poi 800 \
  --h1 768 --h2 384 --dropout 0.1 \
  --batch-size 256 --lr 1e-3 --l2 1e-5 \
  --repeats 9 --seed 1337

echo
echo "==[6/8] Train SBOX campaign =="
"$PYTHON" ../multisession_mlp_16bytes.py \
  --profile-datasets "$PROF" \
  --attack-dataset "unknown_blind_20k_raw_refal.npz" \
  --outdir exp_blind_sbox_tuesday \
  --bytes all \
  --label-mode sbox \
  --epochs 80 --patience 15 \
  --poi 500 \
  --h1 512 --h2 256 --dropout 0.1 \
  --batch-size 256 --lr 1e-3 --l2 1e-5 \
  --repeats 7 --seed 1337

echo
echo "==[7/8] Blind fusion HW+SBOX =="
"$PYTHON" ../attack_blind_fusion.py \
  --campaign-a exp_blind_hw_tuesday \
  --campaign-b exp_blind_sbox_tuesday \
  --label-a hw \
  --label-b sbox \
  --confidence-mode margin_norm \
  --out blind_tuesday_fusion.json

echo
echo "==[8/8] Blind CPA baseline (POI from prior reference summary) =="
echo "Edit the poi-json path below if needed."
"$PYTHON" ../cpa_unknown_16bytes_with_poi.py \
  --npz unknown_blind_20k_raw_refal_nokey.npz \
  --poi-json ../unknown_B_20k_aligned_fullscan_summary.json \
  --out-prefix blind_tuesday_cpa \
  --poi-half-window 0

echo
echo "Done. Key outputs:"
echo " - blind_tuesday_fusion.json"
echo " - blind_tuesday_cpa_summary.json"
echo " - exp_blind_hw_tuesday/summary.json"
echo " - exp_blind_sbox_tuesday/summary.json"
