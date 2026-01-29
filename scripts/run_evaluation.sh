#!/bin/bash
# Evaluation Script for SaFeR-Steer
# Multi-turn safety benchmark evaluation

echo "=== SaFeR-Steer Evaluation ==="
echo ""
echo "Benchmarks:"
echo "  - STEER-Bench (ours)"
echo "  - MM-SafetyBench"
echo "  - VLSafe"
echo "  - FigStep"
echo ""
echo "Usage:"
echo "  python -m evaluation.run_all \\"
echo "    --model /path/to/checkpoint \\"
echo "    --benchmarks steer_bench,mm_safety,vlsafe,figstep \\"
echo "    --output_dir outputs/"
echo ""
echo "Requires JUDGE_API_KEY environment variable for GPT-5-nano judge."
echo "See paper Section 5 for evaluation setup."
