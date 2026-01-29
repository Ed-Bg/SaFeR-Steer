# GRPO Training for SaFeR-Steer

This directory contains configuration and prompts for Stage III: Tutor-in-the-loop Agentic RL with Feedback Dynamics.

## Overview

The GRPO training stage uses a modified GRPO framework for multi-turn RL training with:
- **Safety Tutor**: Generates adaptive follow-up attacks and evaluates responses
- **TCSR (Trajectory-Consistent Safety Reward)**: Penalizes late-turn safety regressions
- **Feedback Dynamics**: Updates training trajectories online via tutor-driven rollouts

## Key Components

### 1. Reward Prompt (`reward_prompt.py`)
Defines the Safety Tutor prompt that:
- Scores responses on Safety (1-10), Usefulness (1-10), Faithfulness (1-10)
- Generates adaptive follow-up attacks based on model behavior
- Implements the "Constructive Defense" grading standard

### 2. TCSR Configuration
Trajectory-Consistent Safety Reward aggregation:
```python
tilde_r_t = alpha * min(s_1, ..., s_t) + (1-alpha) * mean(s_1, ..., s_t)
```
Where `alpha=0.3` by default.

## Training Setup

### Prerequisites
1. Install verl: `pip install verl`
2. Prepare STEER-RL dataset (2,000 multi-turn dialogues)
3. SFT checkpoint from Stage II

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-6 |
| Batch size | 64 |
| Mini-batch size | 2 |
| Rollouts per prompt (K) | 5 |
| KL penalty (β) | 0.1 |
| TCSR α | 0.3 |

### Running
```bash
# See scripts/run_grpo.sh for full example
cd /path/to/verl
python -m verl.trainer.main --config configs/safer_steer_grpo.yaml
```

## Integration with verl

The main GRPO training code is in `../verl/` directory, which is a modified version of the verl framework with:
- Multi-turn dialogue support
- TCSR reward aggregation
- Safety Tutor integration for adaptive attacks

See the verl directory for implementation details.
