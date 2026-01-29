"""
Multi-turn Dialogue Evaluation Module

Evaluates model responses turn-by-turn using a Judge model.
Computes Safety (Harmless) and Helpfulness scores.
"""

import json
import os
import time
import random
import re
import base64
import threading
from io import BytesIO
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
import numpy as np

from ..utils import api
from ..utils.image_utils import check_and_resize_image
from ..utils.path_utils import fix_image_path
from .prompts import JUDGE_SYSTEM_PROMPT


# Thread-local storage for OpenAI client reuse
_thread_local = threading.local()


def get_client() -> OpenAI:
    """Get thread-local OpenAI client."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=api.api_key, base_url=api.base_url)
    return _thread_local.client


def _strip_code_fence(s: str) -> str:
    """Remove code fence markers."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text."""
    if not isinstance(text, str):
        return None
    s = _strip_code_fence(text)
    
    i1 = s.find("{")
    i2 = s.find("[")
    idxs = [i for i in (i1, i2) if i != -1]
    if not idxs:
        return None
    start = min(idxs)
    s2 = s[start:]

    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(s2)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass
    
    if i1 != -1:
        s3 = s[i1:]
        try:
            obj, _ = dec.raw_decode(s3)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    
    return None


def safe_int(x, default=None) -> Optional[int]:
    """Safe conversion to int."""
    try:
        return int(x)
    except Exception:
        return default


def encode_image_for_judge(
    image_path: str, 
    path_mapping: Dict, 
    max_pixels: int = 512*512, 
    min_pixels: int = 338*338
) -> str:
    """Encode image to base64 for judge."""
    fixed_path = fix_image_path(image_path, path_mapping)
    img = check_and_resize_image(fixed_path, max_pixels, min_pixels)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_turn_eval_messages(
    conversation_history: List[Dict],
    current_turn: int,
    image_path: str,
    path_mapping: Dict
) -> List[Dict]:
    """Build messages for single turn evaluation."""
    context_parts = []
    
    if current_turn > 1:
        context_parts.append("### Previous Conversation History")
        for turn_data in conversation_history[:current_turn - 1]:
            turn_num = turn_data["turn"]
            context_parts.append(f"[Turn {turn_num}]")
            context_parts.append(f"User: {turn_data['user']}")
            context_parts.append(f"Assistant: {turn_data['assistant']}")
            context_parts.append("")
    
    current_data = conversation_history[current_turn - 1]
    context_parts.append(f"### Current Turn {current_turn} (To Evaluate)")
    context_parts.append(f"User Query: {current_data['user']}")
    context_parts.append(f"Model Response: {current_data['assistant']}")
    
    eval_text = "\n".join(context_parts)
    user_content = []
    
    has_image = conversation_history[0].get("has_image", False) if conversation_history else False
    if has_image and image_path and image_path.strip():
        try:
            image_base64 = encode_image_for_judge(image_path, path_mapping)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        except Exception as e:
            print(f"⚠️ Image encoding failed ({image_path}): {e}")
    
    user_content.append({"type": "text", "text": eval_text})
    
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def call_judge_api(messages: List[Dict], judge_model: str, timeout: int = 180) -> str:
    """Call Judge API."""
    client = get_client()
    response = client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=0.0,
        timeout=timeout
    )
    return response.choices[0].message.content or ""


def call_with_retries_json(fn, *, max_tries: int = 3, base_delay: float = 1.0, **kwargs) -> Dict[str, Any]:
    """Call function with retries and JSON parsing."""
    attempt = 0
    last_raw = None
    
    while True:
        try:
            raw = fn(**kwargs)
            last_raw = raw
            if isinstance(raw, dict):
                return raw
            obj = _extract_first_json_obj(raw)
            if obj is None:
                raise ValueError("JSON parsing failed")
            return obj
        except Exception as e:
            attempt += 1
            if attempt >= max_tries:
                raise ValueError(f"Max retries exceeded: {e}, raw: {last_raw}")
            sleep_s = min(8.0, base_delay * (2 ** (attempt - 1))) * (0.5 + random.random())
            time.sleep(sleep_s)


def parse_turn_judge_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Parse judge response for a single turn."""
    return {
        "Answer_Helpful_Rate": safe_int(parsed.get("Answer_Helpful_Rate")),
        "Answer_Harmless_Rate": safe_int(parsed.get("Answer_Harmless_Rate")),
        "Helpful_Rationale": parsed.get("Helpful_Rationale", ""),
        "Harmless_Rationale": parsed.get("Harmless_Rationale", ""),
    }


def compute_sample_scores(turn_scores: List[Dict]) -> Dict[str, Any]:
    """Compute sample-level scores from turn scores."""
    answer_helpful_list = [t["Answer_Helpful_Rate"] for t in turn_scores if t.get("Answer_Helpful_Rate") is not None]
    answer_harmless_list = [t["Answer_Harmless_Rate"] for t in turn_scores if t.get("Answer_Harmless_Rate") is not None]
    
    last_turn = turn_scores[-1] if turn_scores else {}
    
    return {
        "Answer_Helpful": last_turn.get("Answer_Helpful_Rate"),
        "Answer_Harmless": last_turn.get("Answer_Harmless_Rate"),
        "Turn_Helpful": round(float(np.mean(answer_helpful_list)), 4) if answer_helpful_list else None,
        "Turn_Harmless": round(float(np.mean(answer_harmless_list)), 4) if answer_harmless_list else None
    }


def evaluate_single_sample(sample_result: Dict, judge_model: str, path_mapping: Dict) -> Dict:
    """Evaluate a single sample turn by turn."""
    try:
        question_id = sample_result.get("question_id", "unknown")
        category = sample_result.get("category", "")
        image_path = sample_result.get("image_path", "")
        conversation = sample_result.get("conversation", [])
        total_turns = len(conversation)
        
        if not conversation:
            raise ValueError("No conversation data")
        
        turn_scores = []
        
        for turn_idx in range(1, total_turns + 1):
            messages = build_turn_eval_messages(conversation, turn_idx, image_path, path_mapping)
            parsed = call_with_retries_json(call_judge_api, messages=messages, judge_model=judge_model)
            scores = parse_turn_judge_response(parsed)
            
            turn_data = conversation[turn_idx - 1]
            turn_record = {
                "turn": turn_idx,
                "user": turn_data.get("user", ""),
                "assistant": turn_data.get("assistant", ""),
                **scores
            }
            turn_scores.append(turn_record)
        
        sample_scores = compute_sample_scores(turn_scores)
        
        return {
            "question_id": question_id,
            "category": category,
            "total_turns": total_turns,
            "image_path": image_path,
            "Answer_Helpful": sample_scores["Answer_Helpful"],
            "Answer_Harmless": sample_scores["Answer_Harmless"],
            "Turn_Helpful": sample_scores["Turn_Helpful"],
            "Turn_Harmless": sample_scores["Turn_Harmless"],
            "turn_scores": turn_scores
        }
        
    except Exception as e:
        return {
            "question_id": sample_result.get("question_id", "unknown"),
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def check_judge_api_health(judge_model: str) -> bool:
    """Check if Judge API is available."""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
            timeout=30
        )
        return len(response.choices) > 0
    except Exception as e:
        print(f"⚠️ Judge API health check failed: {e}")
        return False


def run_benchmark_evaluation(
    infer_result_path: str,
    benchmark_name: str,
    model_name: str,
    judge_model: str,
    output_dir: str,
    max_workers: int,
    path_mapping: Dict = None
) -> str:
    """
    Run batch evaluation for a single benchmark.
    
    Args:
        infer_result_path: Path to inference result file
        benchmark_name: Benchmark identifier
        model_name: Model name
        judge_model: Judge model name
        output_dir: Output directory
        max_workers: Number of concurrent workers
        path_mapping: Image path mapping
    
    Returns:
        Evaluation result file path
    """
    if path_mapping is None:
        path_mapping = {}
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation: {benchmark_name} (Judge: {judge_model})")
    print(f"{'='*60}")
    
    # Load inference results
    with open(infer_result_path, 'r', encoding='utf-8') as f:
        infer_data = json.load(f)
    
    samples = infer_data.get("samples", {})
    valid_samples = {k: v for k, v in samples.items() if "error" not in v}
    
    print(f"Total samples: {len(samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    
    # Prepare output
    output_judge_dir = os.path.join(output_dir, judge_model, model_name, benchmark_name)
    os.makedirs(output_judge_dir, exist_ok=True)
    output_file = os.path.join(output_judge_dir, "raw_eval.json")
    
    # Resume support
    existing_eval = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_eval = data.get("samples", {})
        print(f"Existing evaluations: {len(existing_eval)}")
    
    samples_to_eval = {
        k: v for k, v in valid_samples.items()
        if k not in existing_eval or "error" in existing_eval.get(k, {})
    }
    print(f"Samples to evaluate: {len(samples_to_eval)}")
    
    # Health check
    if not check_judge_api_health(judge_model):
        raise RuntimeError(f"Judge API unavailable (model: {judge_model})")
    print("✓ Judge API healthy")
    
    # Concurrent evaluation
    eval_results = dict(existing_eval)
    error_count = 0
    
    if samples_to_eval:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qid = {
                executor.submit(evaluate_single_sample, sample_result, judge_model, path_mapping): qid 
                for qid, sample_result in samples_to_eval.items()
            }
            
            for future in tqdm(as_completed(future_to_qid), total=len(samples_to_eval), desc="Evaluation"):
                result = future.result()
                if "error" in result:
                    error_count += 1
                else:
                    eval_results[result["question_id"]] = result
    
    # Save results
    output_data = {
        "benchmark": benchmark_name,
        "model_name": model_name,
        "judge_model": judge_model,
        "total_samples": len(samples),
        "evaluated_samples": len(eval_results),
        "timestamp": datetime.now().isoformat(),
        "samples": eval_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation complete, saved to: {output_file}")
    print(f"Success: {len(eval_results)}, Failed: {error_count}")
    
    return output_file
