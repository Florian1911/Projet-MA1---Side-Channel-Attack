#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COMMON_ARGS=(
  --byte 0
  --key 0x2B
  --center -1
  --epochs 40
  --batch-size 200
  --patience 8
  --preproc center_detrend
)

echo "[1/3] Train on S1, test on S1"
"$PYTHON_BIN" train_ascad_cnn.py \
  --train-npz profile_s1_al.npz \
  --test-npz profile_s1_al.npz \
  --out ascad_s1_to_s1.keras \
  "${COMMON_ARGS[@]}"

echo "[2/3] Train on S1, test on S2"
"$PYTHON_BIN" train_ascad_cnn.py \
  --train-npz profile_s1_al.npz \
  --test-npz profile_s2_al.npz \
  --out ascad_s1_to_s2.keras \
  "${COMMON_ARGS[@]}"

echo "[3/3] Train on S1, test on S3"
"$PYTHON_BIN" train_ascad_cnn.py \
  --train-npz profile_s1_al.npz \
  --test-npz profile_s3_al.npz \
  --out ascad_s1_to_s3.keras \
  "${COMMON_ARGS[@]}"

echo "Done. Metrics files:"
echo "  - ascad_s1_to_s1.json"
echo "  - ascad_s1_to_s2.json"
echo "  - ascad_s1_to_s3.json"
