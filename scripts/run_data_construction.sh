#!/bin/bash
# Data Construction Script for SaFeR-Steer
# Usage: bash scripts/run_data_construction.sh

set -e

# Configuration
GENERATOR_MODEL="Qwen/Qwen3-VL-32B-Instruct"
VLLM_PORT=8000
NUM_WORKERS=32

# Paths
INPUT_DIR="data/raw_seeds"
OUTPUT_DIR="data/generated"

echo "=============================================="
echo "SaFeR-Steer Data Construction"
echo "=============================================="

# Check if vLLM is running
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; then
    echo "⚠️  vLLM not running. Starting server..."
    
    # Start vLLM server (adjust parameters as needed)
    python -m vllm.entrypoints.openai.api_server \
        --model ${GENERATOR_MODEL} \
        --port ${VLLM_PORT} \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 64 &
    
    echo "Waiting for vLLM to start..."
    sleep 60
fi

echo "✓ vLLM server ready"

# Generate Red-Team data (Strong adversarial)
echo ""
echo "[1/3] Generating Red-Team data..."
python -c "
from data_construction import DataConstructionPipeline
import json

# Load seeds
seeds = []
with open('${INPUT_DIR}/red_team_seeds.jsonl', 'r') as f:
    for line in f:
        seeds.append(json.loads(line.strip()))

pipeline = DataConstructionPipeline(
    base_url='http://localhost:${VLLM_PORT}/v1',
    model_name='${GENERATOR_MODEL}',
    num_workers=${NUM_WORKERS}
)

pipeline.run(
    input_data=seeds,
    data_type='red_team',
    num_turns=(4, 8),
    output_path='${OUTPUT_DIR}/red_team.jsonl'
)
"

# Generate Obfuscated-Risk data (Progressive disclosure)
echo ""
echo "[2/3] Generating Obfuscated-Risk data..."
python -c "
from data_construction import DataConstructionPipeline
import json

seeds = []
with open('${INPUT_DIR}/obfuscated_seeds.jsonl', 'r') as f:
    for line in f:
        seeds.append(json.loads(line.strip()))

pipeline = DataConstructionPipeline(
    base_url='http://localhost:${VLLM_PORT}/v1',
    model_name='${GENERATOR_MODEL}',
    num_workers=${NUM_WORKERS}
)

pipeline.run(
    input_data=seeds,
    data_type='obfuscated',
    num_turns=(3, 6),
    output_path='${OUTPUT_DIR}/obfuscated.jsonl'
)
"

# Generate Benign data (Capability preservation)
echo ""
echo "[3/3] Generating Benign data..."
python -c "
from data_construction import DataConstructionPipeline
import json

seeds = []
with open('${INPUT_DIR}/benign_seeds.jsonl', 'r') as f:
    for line in f:
        seeds.append(json.loads(line.strip()))

pipeline = DataConstructionPipeline(
    base_url='http://localhost:${VLLM_PORT}/v1',
    model_name='${GENERATOR_MODEL}',
    num_workers=${NUM_WORKERS}
)

pipeline.run(
    input_data=seeds,
    data_type='benign',
    num_turns=(3, 5),
    output_path='${OUTPUT_DIR}/benign.jsonl'
)
"

echo ""
echo "=============================================="
echo "Data construction complete!"
echo "Output files:"
echo "  - ${OUTPUT_DIR}/red_team.jsonl"
echo "  - ${OUTPUT_DIR}/obfuscated.jsonl"
echo "  - ${OUTPUT_DIR}/benign.jsonl"
echo "=============================================="
