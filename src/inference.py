"""Standalone inference against the final Stage 3 (DPO-aligned) model.

Loads the merged Hugging Face checkpoint directly via `transformers` (no
Unsloth/bitsandbytes — those require a CUDA GPU, which this local dev
environment does not have; the merged checkpoint is a plain 16-bit HF model
so CPU inference works fine for a 0.5B model).
"""

import os
import sys
from functools import lru_cache

from src.generation_utils import generate
from src.prompts import SYSTEM_PROMPT

FINAL_MODEL_REPO = os.environ.get(
    "HF_FINAL_REPO", "sharanmini/qwen2.5-0.5b-ca-homeowners-final"
)


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_REPO, token=token)
    model = AutoModelForCausalLM.from_pretrained(FINAL_MODEL_REPO, token=token)
    model.eval()
    return model, tokenizer


def generate_answer(question: str) -> str:
    """Answer `question` using the final DPO-aligned model."""
    model, tokenizer = _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model, tokenizer, prompt, max_new_tokens=200, temperature=0.7
    ).strip()


if __name__ == "__main__":
    # Windows consoles default to a codepage (e.g. cp1252) that can't encode
    # every character a model might generate; force UTF-8 so the print below
    # doesn't crash on an otherwise-successful answer.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = "How do I start a claim after a burst pipe?"
    print(f"Q: {question}\n")
    print(f"A: {generate_answer(question)}")
