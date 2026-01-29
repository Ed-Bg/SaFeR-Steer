#!/bin/bash
# SFT Training Script for SaFeR-Steer (Stage II: Synthetic Bootstrapping)
# Usage: bash scripts/run_sft.sh

set -e

# Configuration
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="saves/safer-steer-7b/sft"
CONFIG_FILE="training/sft/llamafactory_config.yaml"

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export TRANSFORMERS_OFFLINE=1

echo "=============================================="
echo "SaFeR-Steer SFT Training (Stage II)"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Check LLaMA-Factory installation
if ! python -c "import llamafactory" 2>/dev/null; then
    echo "⚠️  LLaMA-Factory not found. Please install:"
    echo "    pip install llamafactory"
    exit 1
fi

# Run training
llamafactory-cli train ${CONFIG_FILE}

echo ""
echo "=============================================="
echo "SFT Training Complete!"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo "=============================================="
