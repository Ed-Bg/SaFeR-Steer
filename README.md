# SaFeR-Steer

**SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics**

This repository contains the official implementation for ICML 2026 submission.

## Overview

SaFeR-Steer is a progressive multi-turn alignment framework that combines staged synthetic bootstrapping with tutor-in-the-loop GRPO to train vision-language models under adaptive, on-policy attacks.

### Key Features

- **Multi-turn Safety Alignment**: Addresses the static-to-dynamic generalization gap where single-turn aligned models fail under multi-turn context accumulation
- **Synthetic Bootstrapping**: Red-blue agent rollouts for SFT data generation
- **Feedback Dynamics**: Adaptive GRPO with Safety Tutor that generates follow-up attacks based on model behavior
- **TCSR (Trajectory-Consistent Safety Reward)**: Penalizes late-turn safety regressions

## Repository Structure

```
SaFeR-Steer/
├── data_construction/          # Stage I: Intent Decomposition & Reconstruction
│   ├── prompts/               # Prompt templates for 3 data types
│   │   ├── multi_strategy_attack.py   # Strong red-team (high risk)
│   │   ├── progressive_disclosure.py  # Obfuscated risk (medium)
│   │   └── general_query.py           # Benign (capability preservation)
│   └── pipeline.py            # Data generation pipeline
│
├── training/                  # Stage II & III: Training
│   ├── sft/                   # SFT with LLaMA-Factory
│   │   └── llamafactory_config.yaml
│   └── grpo/                  # GRPO with verl
│       ├── reward_prompt.py   # Safety Tutor prompt
│       └── README.md
│
├── evaluation/                # Multi-turn Safety Evaluation
│   ├── multi_turn/            # Core evaluation modules
│   │   ├── infer.py          # Multi-turn dialogue inference
│   │   ├── evaluate.py       # Judge-based evaluation
│   │   ├── aggregate.py      # Statistics aggregation
│   │   └── prompts.py        # Judge prompts
│   ├── utils/                 # Shared utilities
│   └── run_all.py            # Main evaluation entry
│
├── scripts/                   # Running scripts
│   ├── run_data_construction.sh
│   ├── run_sft.sh
│   ├── run_grpo.sh
│   └── run_evaluation.sh
│
├── verl/                      # GRPO training framework (modified verl)
└── configs/                   # Configuration files
```

## Installation

```bash
# Clone repository
git clone https://anonymous.4open.science/r/SaFeR-Steer
cd SaFeR-Steer

# Install dependencies
pip install -r requirements.txt

# Install LLaMA-Factory for SFT
pip install llamafactory

# Install verl for GRPO
cd verl && pip install -e . && cd ..
```

## Quick Start

### 1. Data Construction (Stage I)

Generate multi-turn training data from single-turn seeds:

```bash
bash scripts/run_data_construction.sh
```

This creates three types of data:
- `red_team.jsonl`: Strong adversarial dialogues (4-8 turns)
- `obfuscated.jsonl`: Progressive disclosure dialogues (3-6 turns)
- `benign.jsonl`: Capability preservation dialogues (3-5 turns)

### 2. SFT Training (Stage II)

Train with Synthetic Bootstrapping:

```bash
bash scripts/run_sft.sh
```

### 3. GRPO Training (Stage III)

Train with Feedback Dynamics and TCSR:

```bash
bash scripts/run_grpo.sh
```

### 4. Evaluation

Run multi-turn safety evaluation:

```bash
bash scripts/run_evaluation.sh
```

## Datasets

### STEER Dataset

| Split | Size | Turn Range | Avg. Turns |
|-------|------|------------|------------|
| STEER-SFT | 12,934 | 2-10 | 6.35 |
| STEER-RL | 2,000 | 2-10 | 8.33 |
| STEER-Bench | 3,227 | 2-10 | 8.55 |

### STEER-Bench Subsets

- `steer-beaver`: Multi-turn BeaverTails-V
- `steer-mmsafe`: Multi-turn MM-SafetyBench
- `steer-vls`: Multi-turn VLSBench
- `steer-spa`: Multi-turn SPA-VL
- `steer-dys`: Original dynamic scenarios

## Main Results

### Multi-turn Safety (Table 2 in paper)

| Method | 3B Safety↑ | 3B Help↑ | 7B Safety↑ | 7B Help↑ |
|--------|-----------|----------|-----------|----------|
| Base | 12.55 | 27.13 | 24.66 | 46.48 |
| + SFT | 41.61 | 57.57 | 50.31 | 66.86 |
| + SaFeR-Steer | **55.58** | **70.27** | **64.89** | **72.35** |

## Citation

```bibtex
@inproceedings{safer-steer-2026,
  title={SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics},
  author={Anonymous},
  booktitle={ICML},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
