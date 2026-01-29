# Anonymous Submission - ICML 2026

## ⚠️ Important Notice

This is a **skeleton codebase** for anonymous review. To protect against potential misuse:

1. **Core implementations are withheld** - replaced with `NotImplementedError`
2. **Detailed prompts are redacted** - placeholders reference paper appendix
3. **Datasets are not included** - will be released upon acceptance

## What Reviewers Can Verify

✅ **Architecture design** - Class structures and interfaces
✅ **Hyperparameters** - Training configs match paper Table 5
✅ **Methodology** - Docstrings explain the approach
✅ **Reproducibility claims** - All settings documented

## What's Withheld

❌ Full implementation of TCSR aggregation
❌ Safety Tutor prompt templates  
❌ Data construction prompts
❌ Complete evaluation pipeline
❌ STEER-SFT/RL/Bench datasets

## Post-Acceptance Release Plan

Upon paper acceptance, we will release:

1. **Complete source code** with all implementations
2. **STEER-SFT dataset** (6,000 samples)
3. **STEER-RL dataset** (2,000 samples)
4. **STEER-Bench** evaluation benchmark
5. **Model checkpoints** for reproducibility
6. **Detailed documentation** and tutorials

## File Statistics

| Category | Count | Notes |
|----------|-------|-------|
| Python files | ~20 | Skeleton interfaces |
| YAML configs | 3 | Full hyperparameters |
| Shell scripts | 4 | Usage examples |

## Verification

```bash
# Verify no sensitive information
grep -r "your-name\|your-email" . --include="*.py"  # Should be empty

# Verify implementation is withheld
grep -r "NotImplementedError" . --include="*.py"    # Should show placeholders
```

## Contact

Questions during review can be directed through OpenReview.
