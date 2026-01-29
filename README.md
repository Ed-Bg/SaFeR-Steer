# SaFeR-Steer

**S**afety **a**lignment with **Fe**edback **R**einforcement via **Steer**ing

> 🔒 **Anonymous Submission to ICML 2026**
> 
> ⚠️ **Note**: This is a **skeleton codebase** for anonymous review. Core implementations are withheld to protect against misuse during the review period. Full code will be released upon paper acceptance.

## Overview

SaFeR-Steer is a three-stage framework for multi-turn multimodal LLM safety alignment:

- **Stage I**: Intent Decomposition and Reconstruction (data construction)
- **Stage II**: Synthetic Bootstrapping (SFT)
- **Stage III**: Tutor-in-the-loop Agentic RL with Feedback Dynamics (GRPO + TCSR)

## Repository Structure

```
SaFeR-Steer/
├── data_construction/       # Stage I: Data synthesis [SKELETON]
│   ├── pipeline.py         # Main pipeline interface
│   └── prompts/            # Prompt templates [WITHHELD]
├── training/
│   ├── sft/                # Stage II: SFT config
│   └── grpo/               # Stage III: GRPO prompts [WITHHELD]
├── evaluation/              # Multi-turn evaluation [SKELETON]
│   └── multi_turn/         # Inference, evaluation, aggregation
├── verl/                    # GRPO training framework [SKELETON]
│   ├── interactions/       # Safety Tutor interaction
│   └── utils/              # TCSR reward aggregation
├── scripts/                 # Run scripts (usage examples)
└── configs/                 # Benchmark & model configs
```

## What's Included

| Component | Status | Description |
|-----------|--------|-------------|
| Pipeline interfaces | ✅ Skeleton | Class/function signatures with docstrings |
| Hyperparameters | ✅ Full | Training configs from paper Table 5 |
| Method descriptions | ✅ Full | Detailed docstrings explaining methodology |
| Core implementations | ❌ Withheld | Will be released upon acceptance |
| Prompts | ❌ Withheld | See paper Appendix E for examples |
| Datasets | ❌ Withheld | STEER-SFT/RL/Bench released upon acceptance |

## Installation

```bash
git clone https://anonymous.4open.science/r/SaFeR-Steer
cd SaFeR-Steer
pip install -r requirements.txt
```

## Usage (After Acceptance)

### Stage I: Data Construction
```bash
bash scripts/run_data_construction.sh
```

### Stage II: SFT Training
```bash
bash scripts/run_sft.sh
```

### Stage III: GRPO Training
```bash
bash scripts/run_grpo.sh
```

### Evaluation
```bash
export JUDGE_API_KEY="your-api-key"
bash scripts/run_evaluation.sh
```


## License

Apache License 2.0

---

*Full implementation will be released upon paper acceptance.*
