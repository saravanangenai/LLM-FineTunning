"""Helpers for loading and preparing the datasets for all three fine-tuning stages."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.prompts import SYSTEM_PROMPT


def load_raw_paragraphs(path: str | Path) -> list[str]:
    """Read the raw domain text file and return one string per paragraph.

    Lines beginning with '#' are header/metadata and are skipped; paragraphs
    are separated by blank lines.
    """
    text = Path(path).read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return [p.strip() for p in body.split("\n\n") if p.strip()]


def build_text_dataset(paragraphs: list[str], eos_token: str = "") -> Dataset:
    """Wrap paragraphs into a Hugging Face Dataset with a single 'text' column.

    `eos_token` is appended to each paragraph so the model learns paragraph
    boundaries during packed/raw-text training.
    """
    texts = [p + eos_token for p in paragraphs]
    return Dataset.from_dict({"text": texts})


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file (Stage 2 instruction pairs or Stage 3 preference pairs)."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def build_chat_dataset(
    pairs: list[dict[str, str]],
    tokenizer,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dataset:
    """Format Stage 2 {"instruction", "response"} pairs with the chat template.

    Produces a single 'text' column (system + user instruction + assistant
    response, including the tokenizer's EOS), matching the same
    `dataset_text_field="text"` shape SFTTrainer uses in Stage 1.
    """
    texts = []
    for pair in pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return Dataset.from_dict({"text": texts})


def build_preference_dataset(
    pairs: list[dict[str, str]],
    tokenizer,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dataset:
    """Format Stage 3 {"prompt", "chosen", "rejected"} pairs for `DPOTrainer`.

    `prompt` is chat-templated up to the assistant turn (`add_generation_prompt=True`);
    `chosen`/`rejected` are the raw assistant response text plus EOS.
    """
    prompts, chosen, rejected = [], [], []
    for pair in pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["prompt"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        chosen.append(pair["chosen"] + tokenizer.eos_token)
        rejected.append(pair["rejected"] + tokenizer.eos_token)
    return Dataset.from_dict(
        {"prompt": prompts, "chosen": chosen, "rejected": rejected}
    )
