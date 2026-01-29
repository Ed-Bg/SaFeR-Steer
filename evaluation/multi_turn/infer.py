"""
Multi-turn Dialogue Inference Module

Performs multi-turn dialogue inference using vLLM-compatible API.
Each sample is processed turn-by-turn, maintaining conversation history.
"""

import json
import base64
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import threading

from ..utils.image_utils import check_and_resize_image
from ..utils.path_utils import fix_image_path
from .prompts import DEFAULT_INFER_SYSTEM_PROMPT


# Thread-local storage for HTTP session reuse
_thread_local = threading.local()


def load_benchmark_data(jsonl_path: str) -> List[Dict]:
    """Load benchmark data from JSONL file."""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_system_prompt(config_prompt: Optional[str], data_messages: List[Dict]) -> str:
    """Get system prompt with priority: config > data > default."""
    if config_prompt:
        return config_prompt
    if data_messages and len(data_messages) > 0:
        if data_messages[0].get("role") == "system":
            return data_messages[0]["content"]
    return DEFAULT_INFER_SYSTEM_PROMPT


def extract_text_from_content(content: Union[str, List[Dict]]) -> str:
    """Extract plain text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    return str(content)


def build_first_turn_message(
    user_content: List[Dict],
    image_path: str,
    max_pixels: int,
    min_pixels: int,
    path_mapping: Dict[str, str],
    benchmark_name: str = ""
) -> Dict:
    """Build first turn user message (may include image)."""
    text = extract_text_from_content(user_content)
    is_mmsafetybench = "mmsafetybench" in benchmark_name.lower() if benchmark_name else False
    
    if not image_path or not image_path.strip():
        if is_mmsafetybench:
            return {"role": "user", "content": [{"type": "text", "text": text}]}
        else:
            raise ValueError(f"Image path is empty for dataset: {benchmark_name}")
    
    try:
        fixed_image_path = fix_image_path(image_path, path_mapping)
        
        if not os.path.exists(fixed_image_path):
            if is_mmsafetybench:
                return {"role": "user", "content": [{"type": "text", "text": text}]}
            else:
                raise FileNotFoundError(f"Image file not found: {fixed_image_path}")
        
        img = check_and_resize_image(fixed_image_path, max_pixels, min_pixels)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": text}
            ]
        }
    except Exception as e:
        print(f"⚠️ Image processing failed ({image_path}): {e}, fallback to text-only")
        return {"role": "user", "content": [{"type": "text", "text": text}]}


def build_subsequent_turn_message(user_content: Union[str, List[Dict]]) -> Dict:
    """Build subsequent turn user message (text only)."""
    text = extract_text_from_content(user_content)
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def get_session() -> requests.Session:
    """Get thread-local HTTP session with connection pooling."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(pool_connections=500, pool_maxsize=500, max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def call_vllm(history: List[Dict], model_id: str, port: int, config: Dict) -> str:
    """Call vLLM API to generate response."""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": model_id,
        "messages": history,
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 8192),
        "top_p": config.get("top_p", 1.0),
        "top_k": config.get("top_k", -1),
        "repetition_penalty": config.get("repetition_penalty", 1.1),
    }
    
    session = get_session()
    response = session.post(url, json=payload, timeout=config.get("timeout", 300))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def run_single_sample_inference(
    sample: Dict,
    model_id: str,
    port: int,
    config: Dict,
    path_mapping: Dict[str, str],
    benchmark_name: str = ""
) -> Dict:
    """Run complete multi-turn dialogue inference for a single sample."""
    try:
        question_id = sample.get("question_id")
        if not question_id:
            raise ValueError("Missing required field: question_id")
        
        messages_fixed = sample.get("messages_fixed")
        if not messages_fixed:
            raise ValueError("Missing required field: messages_fixed")
        
        category = sample.get("category", "")
        source_category = sample.get("source_category", category)
        image_path = sample.get("image", "")
        risk_trajectory_gt = sample.get("risk_trajectory", [])
        
        system_prompt = get_system_prompt(config.get("system_prompt"), messages_fixed)
        history = [{"role": "system", "content": system_prompt}]
        
        user_turns = [msg.get("content", "") for msg in messages_fixed 
                     if isinstance(msg, dict) and msg.get("role") == "user"]
        
        conversation = []
        final_response = ""
        
        for turn_idx, user_content in enumerate(user_turns):
            if turn_idx == 0:
                msg = build_first_turn_message(
                    user_content, image_path,
                    config.get("image_max_pixels", 512 * 512),
                    config.get("image_min_pixels", 338 * 338),
                    path_mapping, benchmark_name
                )
                has_image = any(item.get("type") == "image_url" 
                               for item in msg.get("content", []) if isinstance(item, dict))
            else:
                msg = build_subsequent_turn_message(user_content)
                has_image = False
            
            user_text = extract_text_from_content(user_content)
            history.append(msg)
            response = call_vllm(history, model_id, port, config)
            history.append({"role": "assistant", "content": response})
            
            conversation.append({
                "turn": turn_idx + 1,
                "user": user_text,
                "assistant": response,
                "has_image": has_image
            })
            final_response = response
        
        return {
            "question_id": question_id,
            "category": category,
            "source_category": source_category,
            "total_turns": len(user_turns),
            "image_path": image_path,
            "conversation": conversation,
            "final_response": final_response,
            "risk_trajectory_gt": risk_trajectory_gt
        }
        
    except Exception as e:
        return {
            "question_id": sample.get("question_id", "unknown"),
            "error": f"{type(e).__name__}: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


def check_vllm_health(port: int, model_id: str) -> bool:
    """Check if vLLM service is healthy."""
    try:
        health_url = f"http://localhost:{port}/health"
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def run_benchmark_inference(
    benchmark_file: str,
    benchmark_name: str,
    model_name: str,
    model_id: str,
    port: int,
    config: Dict,
    path_mapping: Dict[str, str],
    output_dir: str,
    max_workers: int
) -> str:
    """
    Run batch inference for a single benchmark.
    
    Args:
        benchmark_file: JSONL filename
        benchmark_name: Benchmark identifier
        model_name: Model display name
        model_id: Model ID for API
        port: vLLM service port
        config: Configuration dict
        path_mapping: Image path mapping
        output_dir: Output directory
        max_workers: Number of concurrent workers
    
    Returns:
        Output file path
    """
    print(f"\n{'='*60}")
    print(f"Starting inference: {benchmark_name}")
    print(f"{'='*60}")
    
    # Health check
    if not check_vllm_health(port, model_id):
        raise RuntimeError(f"vLLM service unavailable on port {port}")
    
    # Load data
    data_dir = config.get("data_dir", "safe_dataset/Benchmark/data")
    jsonl_path = os.path.join(data_dir, benchmark_file)
    samples = load_benchmark_data(jsonl_path)
    print(f"Loaded samples: {len(samples)}")
    
    # Prepare output
    output_model_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_model_dir, exist_ok=True)
    output_file = os.path.join(output_model_dir, f"{benchmark_name}_infer.json")
    
    # Resume support
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_results = data.get("samples", {})
        print(f"Existing results: {len(existing_results)}")
    
    skip_existing = config.get("skip_existing", True)
    samples_to_process = [
        s for s in samples 
        if not (skip_existing and s["question_id"] in existing_results 
                and "error" not in existing_results[s["question_id"]])
    ]
    print(f"Samples to process: {len(samples_to_process)}")
    
    # Concurrent inference
    results = dict(existing_results)
    
    if samples_to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(
                    run_single_sample_inference,
                    sample, model_id, port, config, path_mapping, benchmark_name
                ): sample for sample in samples_to_process
            }
            
            for future in tqdm(as_completed(future_to_sample), total=len(samples_to_process), desc="Inference"):
                result = future.result()
                results[result["question_id"]] = result
    
    # Save results
    output_data = {
        "benchmark": benchmark_name,
        "model_name": model_name,
        "model_id": model_id,
        "config": {
            "image_max_pixels": config.get("image_max_pixels", 512 * 512),
            "temperature": config.get("temperature", 0.0),
            "max_tokens": config.get("max_tokens", 8192)
        },
        "total_samples": len(samples),
        "timestamp": datetime.now().isoformat(),
        "samples": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Inference complete, saved to: {output_file}")
    return output_file
