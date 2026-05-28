#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

INPUT_DIR="demo/input_images"
RGBA_DIR="demo/processed/rgba"
PROCESSED_DIR="demo/processed/rgb"
POINT_OUT_DIR="demo/output_results"

python run_preprocess.py \
  "$INPUT_DIR" \
  "$RGBA_DIR" \
  "$PROCESSED_DIR"

(
  cd "$SCRIPT_DIR"
  python hmr/hmr2.py \
    --img_folder "$PROCESSED_DIR" \
    --batch_size 48 \
    --cur_num 0 \
    --split_num 1
)

python run.py \
  "$PROCESSED_DIR" \
  --output-dir "$POINT_OUT_DIR" \
  --render \
  --render-num-views 36 \
  --no-remove-bg
