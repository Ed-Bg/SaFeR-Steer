#!/bin/bash
# Script to merge FSDP checkpoints into HuggingFace format
# Usage: ./merge.sh
# Modify CHECKPOINT_DIR and OUTPUT_DIR according to your experiment

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/safedy_rl/safedy_multiturn_training}"
OUTPUT_DIR="${OUTPUT_DIR:-output/merged_model}"

# Example: merge checkpoint at global_step 10 and 20
for step in 10 20; do
    python -m verl.model_merger merge \
        --backend fsdp \
        --trust-remote-code \
        --local_dir ${CHECKPOINT_DIR}/global_step_${step}/actor \
        --target_dir ${OUTPUT_DIR}-step${step}
done
