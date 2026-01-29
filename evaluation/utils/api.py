"""
API Configuration for Judge Model

Configure your API endpoint and key here for the evaluation judge.
Supports OpenAI-compatible APIs.

IMPORTANT: Set environment variables before running:
    export JUDGE_API_BASE_URL="https://api.openai.com/v1"
    export JUDGE_API_KEY="your-api-key"
"""

import os

# Configuration from environment variables (required)
base_url = os.environ.get("JUDGE_API_BASE_URL", "https://api.openai.com/v1")
api_key = os.environ.get("JUDGE_API_KEY", "")

if not api_key:
    import warnings
    warnings.warn(
        "JUDGE_API_KEY environment variable not set. "
        "Please set it before running evaluation: export JUDGE_API_KEY='your-key'"
    )
