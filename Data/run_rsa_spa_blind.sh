#!/usr/bin/env bash
set -euo pipefail

RAW=${1:-rsa_spa_20k_raw.npz}
AL=${2:-rsa_spa_20k_al.npz}
OUT=${3:-rsa_spa_blind_result.json}

python base_saine_pico5000/acquire_pico5000a_no_uart.py \
  --trigger-source ext \
  --ext-threshold-adc 1200 \
  --meas-mode diff_ab \
  --meas-range-a PS5000A_5V \
  --meas-range PS5000A_5V \
  --n-traces 20000 \
  --num-samples 4000 \
  --pre-trigger 200 \
  --timebase 8 \
  --timeout-s 600 \
  --plaintexts plaintexts_no_uart.npy \
  --pt-offset 0 \
  --output "$RAW"

python align_local_per_trace.py \
  --in "$RAW" \
  --out "$AL" \
  --center 1700 --window 220 --max-shift 20 --iters 3 \
  --mode ref --ref median

python rsa_spa_recover_d.py \
  --npz "$AL" \
  --trace-key traces \
  --n-traces 20000 \
  --bitlen 23 \
  --out "$OUT"

echo "[DONE] $OUT"
