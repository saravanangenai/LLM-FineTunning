"""Standalone inference against the final Stage 3 (DPO-aligned) model.

Calls the model through a Hugging Face Dedicated Inference Endpoint (see
specs/002-unsloth-ca-homeowners-finetune.md §9) rather than loading weights
in-process, so this only needs `huggingface_hub` locally — no
torch/transformers/GPU required.

Requires the endpoint to already be running (create/pause it via the HF UI
or `huggingface_hub.create_inference_endpoint` — that's a billed action, so
it's not automated here) and these env vars:
    HF_ENDPOINT_URL - the endpoint's URL
    HF_TOKEN        - an HF token with access to the endpoint
"""

import os
import sys
from functools import lru_cache

from huggingface_hub import InferenceClient

from src.prompts import SYSTEM_PROMPT


@lru_cache(maxsize=1)
def _client() -> InferenceClient:
    endpoint_url = os.environ["HF_ENDPOINT_URL"]
    token = os.environ["HF_TOKEN"]
    return InferenceClient(base_url=endpoint_url, token=token)


def generate_answer(question: str) -> str:
    """Answer `question` using the final DPO-aligned model via the HF endpoint."""
    completion = _client().chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    # Windows consoles default to a codepage (e.g. cp1252) that can't encode
    # every character a model might generate; force UTF-8 so the print below
    # doesn't crash on an otherwise-successful answer.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = "How do I start a claim after a burst pipe?"
    print(f"Q: {question}\n")
    print(f"A: {generate_answer(question)}")
