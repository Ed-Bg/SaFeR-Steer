#!/usr/bin/env bash
set -euo pipefail

# Configuration - modify these paths as needed
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/datasets/MT_RL/data/DyS2k1.jsonl}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/data/safedy_multiturn}"

python "$PROJECT_DIR/examples/data_preprocess/safedy_multiturn.py" \
  --local_dataset_path "$DATA_PATH" \
  --local_save_dir "$OUT_DIR"
