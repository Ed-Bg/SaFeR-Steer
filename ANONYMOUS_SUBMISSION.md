# Anonymous Submission Checklist for ICML 2026

This repository is prepared for **anonymous submission** to ICML 2026.

## ✅ Completed Anonymization

| Item | Status | Notes |
|------|--------|-------|
| Author names | ✅ Removed | `setup.py` uses "Anonymous" |
| Email addresses | ✅ Removed | Uses `anonymous@example.com` |
| Institution names | ✅ Removed | No affiliations present |
| Personal GitHub usernames | ✅ Removed | All `TODO(username)` cleaned |
| Volcengine/Bytedance GitHub links | ✅ Removed | Replaced with anonymous URLs |
| API keys | ✅ Secured | Environment variables only |
| Local file paths | ✅ Removed | No hardcoded `/home/`, `/usb/` paths |
| IP addresses | ✅ None | No hardcoded IPs |

## ⚠️ Important Notes

### Third-Party Copyrights (Retained)
The `verl/` directory contains code from open-source projects with their original Apache 2.0 copyright headers. These are **retained as required by the license** and do not reveal author identity:
- SGLang Team (Apache 2.0)
- Bytedance Ltd. (Apache 2.0) - This is the verl framework, not the paper authors
- PRIME team (Apache 2.0)
- PyTorch/HuggingFace references (for algorithm implementations)

### Optional Tracking Backend
`verl/utils/tracking.py` includes optional `volcengine_ml_platform` support. This is an enterprise feature from the verl framework and:
- Only activates if user explicitly sets `vemlp_wandb` as tracking backend
- Does not reveal paper authorship
- Can be safely ignored for reproducibility

## Before Uploading

### 1. Final Verification
```bash
cd SaFeR-Steer

# Check for personal identifiers
grep -ri "your-name\|your-institution" . --include="*.py" --include="*.yaml"

# Check for personal GitHub usernames (should return empty)
grep -ri "zhangchi\|gmsheng\|wangkun" . --include="*.py" --include="*.yaml"

# Check for volcengine GitHub (should return empty)
grep -r "github.com/volcengine" . --include="*.py" --include="*.yaml" --include="*.md"

# Check for hardcoded paths (should return empty)
grep -rE "/home/[a-z]+|/usb/" . --include="*.py" --include="*.yaml" --include="*.sh"
```

### 2. Remove Empty Directories (Optional)
```bash
find . -type d -empty -delete
```

### 3. Verify No Large Data Files
```bash
find . -size +5M -type f
```

## Upload Instructions

### Option 1: Anonymous GitHub (Recommended)
1. Go to https://anonymous.4open.science/
2. Create new anonymous repository
3. Upload the `SaFeR-Steer` directory
4. Copy the anonymous URL for your paper

### Option 2: OpenReview Supplementary
1. Zip the repository: `zip -r SaFeR-Steer.zip SaFeR-Steer -x "*.git*"`
2. Upload to OpenReview as supplementary material

## Environment Setup for Reviewers

```bash
# Clone repository
git clone https://anonymous.4open.science/r/SaFeR-Steer

# Install dependencies
cd SaFeR-Steer
pip install -r requirements.txt

# Set required environment variables
export JUDGE_API_BASE_URL="https://api.openai.com/v1"
export JUDGE_API_KEY="your-api-key"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

## Directory Structure

```
SaFeR-Steer/
├── data_construction/     # Stage I: Data synthesis
├── training/
│   ├── sft/              # Stage II: SFT config
│   └── grpo/             # Stage III: GRPO prompts
├── evaluation/           # Multi-turn evaluation
├── verl/                 # GRPO training framework
├── configs/              # Benchmark & model configs
├── scripts/              # Run scripts
└── examples/             # Usage examples
```

## Reproducibility

- **STEER-SFT/RL/Bench datasets**: Will be released upon acceptance
- **Model checkpoints**: Download links provided upon acceptance
- **Hardware**: 8× A100 80GB GPUs recommended
