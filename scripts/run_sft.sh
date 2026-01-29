#!/bin/bash
# SFT Training Script for SaFeR-Steer Stage II
# Uses LLaMA-Factory framework

echo "=== SaFeR-Steer SFT Training ==="
echo "Stage II: Synthetic Bootstrapping"
echo ""
echo "Requirements:"
echo "  - LLaMA-Factory installation"
echo "  - STEER-SFT dataset (6,000 samples)"
echo ""
echo "Usage:"
echo "  llamafactory-cli train training/sft/llamafactory_config.yaml"
echo ""
echo "See paper Table 5 for hyperparameters."
