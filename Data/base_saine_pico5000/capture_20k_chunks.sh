#!/usr/bin/env bash
set -euo pipefail

SCRIPT="base_saine_pico5000/acquire_pico5000a_no_uart.py"
MERGE="base_saine_pico5000/merge_npz_traces.py"
PTS="plaintexts_no_uart.npy"

TRACES_PER_CHUNK=1000
CHUNKS=20
NUM_SAMPLES=4000
PRE_TRIGGER=200
TIMEBASE=8
TRIG_MV_PROBE=500
PROBE_ATT_TRIG=10
TIMEOUT_S=180

parts=()

echo "==============================================="
echo "Capture 20k en ${CHUNKS} blocs de ${TRACES_PER_CHUNK}"
echo "IMPORTANT: appuie sur RESET a CHAQUE bloc"
echo "==============================================="

for ((i=0; i<CHUNKS; i++)); do
  start=$((i * TRACES_PER_CHUNK))
  end=$((start + TRACES_PER_CHUNK - 1))
  out=$(printf "part_%05d_%05d.npz" "$start" "$end")
  parts+=("$out")

  echo
  echo "[BLOC $((i+1))/$CHUNKS] sortie: $out"
  echo "Lancement... (quand SCOPE ARME s'affiche, appuie sur RESET)"

  python "$SCRIPT" \
    --n-traces "$TRACES_PER_CHUNK" \
    --pt-offset 0 \
    --num-samples "$NUM_SAMPLES" \
    --pre-trigger "$PRE_TRIGGER" \
    --timebase "$TIMEBASE" \
    --trig-mv-probe "$TRIG_MV_PROBE" \
    --probe-att-trig "$PROBE_ATT_TRIG" \
    --plaintexts "$PTS" \
    --output "$out" \
    --timeout-s "$TIMEOUT_S" \
    --min-duration-factor 0.0

done

echo

echo "Fusion finale..."
python "$MERGE" "${parts[@]}" --output dataset_pico5000a_20k_chunks.npz

echo "[OK] Fini: dataset_pico5000a_20k_chunks.npz"
