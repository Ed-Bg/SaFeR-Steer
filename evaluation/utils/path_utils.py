"""
Path handling utilities
"""

import os
from typing import Dict


def fix_image_path(original_path: str, mapping: Dict[str, str]) -> str:
    """
    Fix image path using prefix mapping (dataset path → actual path).
    
    Args:
        original_path: Original path from dataset
        mapping: Path prefix mapping dictionary
    
    Returns:
        Fixed actual file path
    """
    for old_prefix, new_prefix in mapping.items():
        if original_path.startswith(old_prefix):
            return original_path.replace(old_prefix, new_prefix, 1)
    return original_path


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_benchmark_name_from_file(filename: str, mapping: Dict[str, str]) -> str:
    """
    Get benchmark identifier from filename.
    
    Args:
        filename: JSONL filename
        mapping: BENCHMARKS mapping dictionary
    
    Returns:
        Benchmark identifier
    """
    return mapping.get(filename, filename.replace(".jsonl", ""))
