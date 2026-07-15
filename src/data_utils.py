"""Helpers for loading and preparing the Stage 1 non-instruction dataset."""

from pathlib import Path

from datasets import Dataset


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
