#!/usr/bin/env bash
set -e

# Configuration - modify these according to your environment
MODEL_PATH="${MODEL_PATH:-<YOUR_REWARD_MODEL_PATH>}"
MODEL_NAME="${MODEL_NAME:-reward-model}"
HOST=0.0.0.0
PORT=9010
TP_SIZE=4          # Tensor parallel size (number of GPUs)
MEM_FRACTION=0.9   # Static memory fraction

export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL configuration
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=eth0  # Specify network interface if needed

echo "Starting sglang server with ${TP_SIZE} GPUs..."
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name ${MODEL_NAME} \
  --host ${HOST} \
  --port ${PORT} \
  --tensor-parallel-size ${TP_SIZE} \
  --mem-fraction-static ${MEM_FRACTION} \
  --enable-multimodal \
  --chunked-prefill-size 8192 \
  --max-running-requests 128 \
  --max-prefill-tokens 32768 \
  --schedule-policy fcfs \
  --attention-backend flashinfer \
  --log-level info
