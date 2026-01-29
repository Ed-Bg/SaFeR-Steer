#!/bin/bash
# GRPO Training Script for SaFeR-Steer (Stage III: Feedback Dynamics)
# Usage: bash scripts/run_grpo.sh

set -e

# Configuration
SFT_CHECKPOINT="saves/safer-steer-7b/sft/checkpoint-epoch2"
OUTPUT_DIR="saves/safer-steer-7b/grpo"
RL_DATA="data/steer_rl.jsonl"

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_ENABLE=1

echo "=============================================="
echo "SaFeR-Steer GRPO Training (Stage III)"
echo "=============================================="
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "RL Data: ${RL_DATA}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Check verl installation
if ! python -c "import verl" 2>/dev/null; then
    echo "⚠️  verl not found. Please install from the verl directory:"
    echo "    cd verl && pip install -e ."
    exit 1
fi

# GRPO Hyperparameters (from paper)
LEARNING_RATE=1e-6
BATCH_SIZE=64
MINI_BATCH_SIZE=2
ROLLOUTS_PER_PROMPT=5
KL_PENALTY=0.1
TCSR_ALPHA=0.3

# Run GRPO training
python -m verl.trainer.main \
    --model_path ${SFT_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --train_file ${RL_DATA} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --mini_batch_size ${MINI_BATCH_SIZE} \
    --num_rollouts ${ROLLOUTS_PER_PROMPT} \
    --kl_coef ${KL_PENALTY} \
    --tcsr_alpha ${TCSR_ALPHA} \
    --num_train_epochs 1 \
    --bf16 true \
    --logging_steps 10 \
    --save_strategy epoch

echo ""
echo "=============================================="
echo "GRPO Training Complete!"
echo "Final model saved to: ${OUTPUT_DIR}"
echo "=============================================="
