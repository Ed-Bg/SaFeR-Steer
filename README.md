# SaFeR-Steer: Safe Multi-turn Reinforcement Learning for LLMs

This repository contains the code for training large language models with multi-turn reinforcement learning for safety alignment, as described in our paper.

## Overview

SaFeR-Steer is a multi-turn RL training framework for safety-aware LLMs. The key components include:

- **SaferdyInteraction**: A multi-turn interaction module that uses a judge model to dynamically generate follow-up questions and compute turn-level rewards
- **Multi-turn Reward Aggregation**: Flexible reward aggregation strategies (mean, min_mean, soft_linear, harmonic) to balance performance across all turns
- **Tool Agent Loop**: An agent loop system that supports multi-turn conversations with tool/interaction integration

## Repository Structure

```
SaFeR-Steer/
├── verl/                           # Core framework
│   ├── interactions/               # Interaction modules
│   │   ├── saferdy_interaction.py  # SafeDy multi-turn interaction (core)
│   │   └── base.py                 # Base interaction class
│   ├── utils/
│   │   └── reward_score/
│   │       └── safedy.py           # Multi-turn reward aggregation
│   ├── experimental/
│   │   └── agent_loop/             # Agent loop for multi-turn RL
│   ├── trainer/                    # PPO trainer
│   └── workers/                    # Distributed workers
│
├── examples/
│   ├── data_preprocess/
│   │   └── safedy_multiturn.py     # Data preprocessing script
│   └── sglang_multiturn/
│       ├── config/                 # Training configurations
│       └── run_*.sh                # Training scripts
│
└── scripts/                        # Utility scripts
```

## Requirements

### Hardware Requirements
- NVIDIA GPUs with CUDA support (tested on A100/H100)
- Minimum 4 GPUs recommended for training
- At least 80GB GPU memory per GPU for 7B models

### Software Requirements
- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 12.0

## Installation

### 1. Create a virtual environment

```bash
conda create -n safedy python=3.10
conda activate safedy
```

### 2. Install PyTorch

```bash
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Flash Attention (optional but recommended)

```bash
pip install flash-attn --no-build-isolation
```

### 5. Install the package

```bash
pip install -e .
```

## Data Preparation

### 1. Prepare your dataset

Your dataset should be in JSONL format with the following structure:

```json
{
  "question": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "metadata": {
    "rest_messages": [{"role": "user", "content": "..."}],
    "question_id": "unique_id"
  },
  "images": ["path/to/image.jpg"],
  "label": "safe/unsafe"
}
```

### 2. Preprocess the dataset

```bash
python examples/data_preprocess/safedy_multiturn.py \
    --local_dataset_path /path/to/your/data.jsonl \
    --local_save_dir data/safedy_multiturn \
    --test_size 0.1
```

This will create `train.parquet` and `test.parquet` in the specified directory.

## Training

### 1. Start the Reward Model Server

The training requires a reward model server for computing turn-level rewards. Start the server using SGLang:

```bash
# Set your model path
export MODEL_PATH=/path/to/your/reward_model

# Start the server
bash sglang_dev1_copy.sh
```

### 2. Configure Training

Edit the configuration file `examples/sglang_multiturn/config/interaction_config/saferdy_interaction_config.yaml`:

```yaml
interaction:
  - name: "saferdy"
    class_name: "verl.interactions.saferdy_interaction.SaferdyInteraction"
    config:
      reward_model_base_url: "http://127.0.0.1:9010"  # Your reward model server URL
      reward_model_name: "reward-model"               # Your model name
      max_turns: 10                                   # Maximum turns per episode
```

### 3. Run Training

```bash
# Set environment variables
export MODEL_PATH=/path/to/your/base_model
export WANDB_API_KEY=your_wandb_key  # Optional, for logging

# Run training
bash examples/sglang_multiturn/run_qwen2.5-3b_safer_multiturn_w_interaction.sh
```

### Training Parameters

Key parameters you can customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 64 | Training batch size |
| `MICRO_BATCH_SIZE` | 2 | Micro batch size per GPU |
| `MODEL_PATH` | - | Path to your base model |
| `OFFLOAD` | False | Enable CPU offloading |

## Model Merging

After training, merge the FSDP checkpoints into a HuggingFace model:

```bash
# Set checkpoint directory
export CHECKPOINT_DIR=checkpoints/
export OUTPUT_DIR=output/merged_model

# Run merging
bash merge.sh
```

## Key Components

### SaferdyInteraction (`verl/interactions/saferdy_interaction.py`)

The core interaction module that:
- Manages multi-turn conversations
- Calls the reward model for turn-level scoring
- Generates dynamic follow-up questions based on judge feedback
- Computes format compliance rewards

### Reward Aggregation (`verl/utils/reward_score/safedy.py`)

Supports multiple aggregation strategies:
- `mean`: Simple average across turns
- `min_mean`: Weighted combination of minimum and mean scores
- `soft_linear`: Gentle recency bias
- `harmonic`: Penalizes low outliers

### Tool Agent Loop (`verl/experimental/agent_loop/tool_agent_loop.py`)

Manages the multi-turn generation process with:
- Tool/interaction integration
- Token-level reward assignment
- Configurable maximum turns

## Docker Support

You can also run training in Docker:

```bash
# Configure paths
export WORKSPACE_PATH=$(pwd)
export MODEL_PATH=/path/to/your/model

# Start container
bash run_docker.sh
```
