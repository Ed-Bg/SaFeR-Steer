"""
General Purpose Query Generator Prompt

Used for generating benign multi-turn dialogues to preserve
model capabilities (math, reasoning, coding, comprehension).
"""

PROMPT_GENERAL_QUERY = """
You are simulating multi-turn user queries for alignment research.

TASK:
Rewrite the given single-turn problem into {min_shards}-{max_shards} sequential user queries (a coherent multi-turn conversation).
This is for preserving general capabilities (math reasoning, logic, coding, reading comprehension), NOT for adversarial attacks.

INPUT:
You will receive either:
(A) Text-only: a question/instruction (and optional reference answer),
or
(B) Image + question: use both if an image exists.

CORE GOAL:
Keep the original task objective unchanged (solve the same problem), but split it into a natural multi-turn user conversation.

MULTI-TURN DESIGN RULES:
1) Output ONLY user messages (first-person perspective).
2) Each turn must be ONE short, natural question or statement.
3) Every turn must be answerable and helpful in context.
4) The conversation must flow coherently: later turns refer to earlier turns.
5) The final turn must ask for the final solution / final answer.

OUTPUT FORMAT (STRICT):
1) Think step-by-step in <think>...</think>.
2) Return ONLY a JSON array of strings in <answer>...</answer>.
3) Do NOT put any comments outside the JSON list.

EXAMPLE:
<think>...</think>
<answer>["I'm working on a math problem.", "Can you help me interpret what it's asking?", "What's a good first step?", "Can you verify my intermediate result?", "What is the final answer?"]</answer>
"""
