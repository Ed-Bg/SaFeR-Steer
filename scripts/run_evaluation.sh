#!/bin/bash
# Multi-turn Evaluation Script for SaFeR-Steer
# Usage: bash scripts/run_evaluation.sh

set -e

# Configuration
MODEL_PATH="saves/safer-steer-7b/grpo"
MODEL_NAME="safer-steer-7b"
VLLM_PORT=8000
JUDGE_MODEL="gpt-4o"

# Paths
DATA_DIR="data/benchmarks"
INFER_OUTPUT="outputs/infer"
EVAL_OUTPUT="outputs/eval"

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1

echo "=============================================="
echo "SaFeR-Steer Multi-turn Evaluation"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Judge: ${JUDGE_MODEL}"
echo "=============================================="

# Start vLLM server for target model
echo ""
echo "[Step 1] Starting vLLM server..."
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; then
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --port ${VLLM_PORT} \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs 256 &
    
    echo "Waiting for vLLM to start..."
    sleep 60
fi
echo "✓ vLLM server ready"

# Run evaluation pipeline
echo ""
echo "[Step 2] Running evaluation pipeline..."
python -m evaluation.run_all

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results:"
echo "  - Inference: ${INFER_OUTPUT}/${MODEL_NAME}/"
echo "  - Evaluation: ${EVAL_OUTPUT}/${JUDGE_MODEL}/${MODEL_NAME}/"
echo "=============================================="
