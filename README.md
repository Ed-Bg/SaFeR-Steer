# SaFeR-Steer



## Overview

SaFeR-Steer is a three-stage framework for multi-turn multimodal LLM safety alignment:

- **Stage I**: Intent Decomposition and Reconstruction (data construction)
- **Stage II**: Synthetic Bootstrapping (SFT)
- **Stage III**: Tutor-in-the-loop Agentic RL with Feedback Dynamics (GRPO + TCSR)

## Repository Structure

```
SaFeR-Steer/
├── data_construction/       # Stage I: Data synthesis 
│   ├── pipeline.py         # Main pipeline interface
│   └── prompts/            # Prompt templates
├── training/
│   ├── sft/                # Stage II: SFT config
│   └── grpo/               # Stage III: GRPO prompts 
├── evaluation/              # Multi-turn evaluation 
│   └── multi_turn/         # Inference, evaluation, aggregation
├── verl/                    # GRPO training framework 
│   ├── interactions/       # Safety Tutor interaction
│   └── utils/              # TCSR reward aggregation
├── scripts/                 # Run scripts (usage examples)
└── configs/                 # Benchmark & model configs
```

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
