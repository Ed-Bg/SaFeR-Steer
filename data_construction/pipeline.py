"""
Data Construction Pipeline

Main pipeline for generating multi-turn safety training data.

Three data types:
1. Strong Red-Team (Multi-Strategy Attack) - High risk, for robustness
2. Obfuscated Risk (Progressive Disclosure) - Medium risk, gradual reveal
3. Benign (General Query) - Safe, capability preservation

Usage:
    from data_construction import DataConstructionPipeline
    
    pipeline = DataConstructionPipeline(
        api_key="your-key",
        base_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen3-VL-32B-Instruct"
    )
    
    results = pipeline.run(
        input_data=samples,
        data_type="red_team",
        num_turns=(4, 8),
        output_path="output.jsonl"
    )
"""

import os
import json
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

from .prompts import (
    PROMPT_FORENSIC_CAPTIONER,
    PROMPT_ADVERSARIAL_REWRITER,
    PROMPT_MULTITURN_PLANNER_TEMPLATE,
    PROMPT_ATTACK_EVALUATOR,
    JAILBREAK_STRATEGIES,
    PROMPT_PROGRESSIVE_DISCLOSURE,
    PROMPT_GENERAL_QUERY,
)


class DataConstructionPipeline:
    """
    Multi-turn safety data construction pipeline.
    
    Supports three generation modes:
    - red_team: Strong adversarial multi-turn dialogues
    - obfuscated: Progressive disclosure medium-risk dialogues
    - benign: General capability-preserving dialogues
    """
    
    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        num_workers: int = 32,
        enable_dedup: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            api_key: API key (use "EMPTY" for local vLLM)
            base_url: API base URL
            model_name: Model name for generation
            num_workers: Number of concurrent workers
            enable_dedup: Enable content deduplication
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.num_workers = num_workers
        self.enable_dedup = enable_dedup
        self._seen_hashes = set()
    
    def _call_api(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Call API with messages."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content or ""
    
    def _extract_json(self, text: str) -> Optional[Any]:
        """Extract JSON from text."""
        import re
        
        # Remove code fences
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        
        # Try to parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Find JSON object or array
        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, content: str) -> bool:
        """Check if content is duplicate."""
        if not self.enable_dedup:
            return False
        h = self._content_hash(content)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False
    
    def generate_red_team(
        self,
        sample: Dict,
        num_turns: Tuple[int, int] = (4, 8)
    ) -> Optional[Dict]:
        """
        Generate strong red-team multi-turn dialogue.
        
        Four-stage pipeline:
        1. Forensic visual analysis
        2. Adversarial intent rewriting
        3. Multi-turn dialogue planning
        4. (Optional) Attack evaluation
        
        Args:
            sample: Input sample with 'image', 'question', etc.
            num_turns: (min_turns, max_turns) range
        
        Returns:
            Generated dialogue dict or None if failed
        """
        try:
            image_path = sample.get("image", "")
            original_question = sample.get("question", "")
            
            # Select random strategy and turn count
            strategy_key = random.choice(list(JAILBREAK_STRATEGIES.keys()))
            strategy = JAILBREAK_STRATEGIES[strategy_key]
            n_turns = random.randint(num_turns[0], num_turns[1])
            
            # Stage 1: Visual analysis (if image exists)
            visual_analysis = ""
            if image_path:
                messages = [
                    {"role": "system", "content": PROMPT_FORENSIC_CAPTIONER},
                    {"role": "user", "content": f"Analyze this image: {image_path}"}
                ]
                visual_analysis = self._call_api(messages, temperature=0.3)
            
            # Stage 2: Adversarial rewriting
            messages = [
                {"role": "system", "content": PROMPT_ADVERSARIAL_REWRITER},
                {"role": "user", "content": f"Visual Context: {visual_analysis}\n\n[Raw Intent]: {original_question}"}
            ]
            rewrite_response = self._call_api(messages, temperature=0.7)
            rewrite_data = self._extract_json(rewrite_response)
            
            if not rewrite_data:
                return None
            
            # Stage 3: Multi-turn planning
            planner_prompt = PROMPT_MULTITURN_PLANNER_TEMPLATE.format(
                num_turns=n_turns,
                strategy_name=strategy["name"]
            )
            
            messages = [
                {"role": "system", "content": planner_prompt},
                {"role": "user", "content": f"Visual: {visual_analysis}\nRewritten Intent: {rewrite_data.get('complex_prompt', original_question)}"}
            ]
            plan_response = self._call_api(messages, temperature=0.7)
            plan_data = self._extract_json(plan_response)
            
            if not plan_data or "dialogue_script" not in plan_data:
                return None
            
            # Build result
            dialogue_turns = []
            for turn in plan_data["dialogue_script"]:
                dialogue_turns.append({
                    "turn": turn.get("turn_id", len(dialogue_turns) + 1),
                    "user": turn.get("user_prompt", ""),
                    "goal": turn.get("goal", "")
                })
            
            # Dedup check
            content_key = json.dumps([t["user"] for t in dialogue_turns])
            if self._is_duplicate(content_key):
                return None
            
            return {
                "question_id": sample.get("question_id", ""),
                "image": image_path,
                "original_question": original_question,
                "data_type": "red_team",
                "strategy": strategy["name"],
                "attack_vector": rewrite_data.get("attack_vector", ""),
                "total_turns": len(dialogue_turns),
                "dialogue": dialogue_turns,
                "risk_curve": plan_data.get("risk_curve", []),
            }
            
        except Exception as e:
            print(f"Red team generation failed: {e}")
            return None
    
    def generate_obfuscated(
        self,
        sample: Dict,
        num_turns: Tuple[int, int] = (3, 6)
    ) -> Optional[Dict]:
        """
        Generate progressive disclosure (obfuscated risk) dialogue.
        
        Args:
            sample: Input sample
            num_turns: (min_turns, max_turns) range
        
        Returns:
            Generated dialogue dict or None
        """
        try:
            image_path = sample.get("image", "")
            original_question = sample.get("question", "")
            
            prompt = PROMPT_PROGRESSIVE_DISCLOSURE.format(
                min_shards=num_turns[0],
                max_shards=num_turns[1]
            )
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Image: {image_path}\nQuestion: {original_question}"}
            ]
            
            response = self._call_api(messages, temperature=0.7)
            turns = self._extract_json(response)
            
            if not turns or not isinstance(turns, list):
                return None
            
            # Dedup check
            content_key = json.dumps(turns)
            if self._is_duplicate(content_key):
                return None
            
            dialogue = [{"turn": i+1, "user": t} for i, t in enumerate(turns)]
            
            return {
                "question_id": sample.get("question_id", ""),
                "image": image_path,
                "original_question": original_question,
                "data_type": "obfuscated",
                "total_turns": len(dialogue),
                "dialogue": dialogue,
            }
            
        except Exception as e:
            print(f"Obfuscated generation failed: {e}")
            return None
    
    def generate_benign(
        self,
        sample: Dict,
        num_turns: Tuple[int, int] = (3, 5)
    ) -> Optional[Dict]:
        """
        Generate benign capability-preserving dialogue.
        
        Args:
            sample: Input sample
            num_turns: (min_turns, max_turns) range
        
        Returns:
            Generated dialogue dict or None
        """
        try:
            image_path = sample.get("image", "")
            original_question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            prompt = PROMPT_GENERAL_QUERY.format(
                min_shards=num_turns[0],
                max_shards=num_turns[1]
            )
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Image: {image_path}\nQuestion: {original_question}\nReference Answer: {answer}"}
            ]
            
            response = self._call_api(messages, temperature=0.5)
            
            # Extract from <answer> tags
            import re
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if match:
                turns = self._extract_json(match.group(1))
            else:
                turns = self._extract_json(response)
            
            if not turns or not isinstance(turns, list):
                return None
            
            # Dedup check
            content_key = json.dumps(turns)
            if self._is_duplicate(content_key):
                return None
            
            dialogue = [{"turn": i+1, "user": t} for i, t in enumerate(turns)]
            
            return {
                "question_id": sample.get("question_id", ""),
                "image": image_path,
                "original_question": original_question,
                "data_type": "benign",
                "total_turns": len(dialogue),
                "dialogue": dialogue,
            }
            
        except Exception as e:
            print(f"Benign generation failed: {e}")
            return None
    
    def run(
        self,
        input_data: List[Dict],
        data_type: str = "red_team",
        num_turns: Tuple[int, int] = (4, 8),
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Run data construction pipeline.
        
        Args:
            input_data: List of input samples
            data_type: "red_team", "obfuscated", or "benign"
            num_turns: (min_turns, max_turns) range
            output_path: Optional path to save results
            max_samples: Optional limit on samples to process
        
        Returns:
            List of generated dialogues
        """
        # Select generator
        generator_map = {
            "red_team": self.generate_red_team,
            "obfuscated": self.generate_obfuscated,
            "benign": self.generate_benign,
        }
        
        if data_type not in generator_map:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        generator = generator_map[data_type]
        
        # Limit samples
        samples = input_data[:max_samples] if max_samples else input_data
        
        print(f"Processing {len(samples)} samples with {data_type} generator...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(generator, sample, num_turns): sample
                for sample in samples
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        print(f"Generated {len(results)} dialogues (success rate: {len(results)/len(samples)*100:.1f}%)")
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved to: {output_path}")
        
        return results
