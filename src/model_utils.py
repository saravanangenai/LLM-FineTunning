"""Shared Unsloth model-loading and LoRA helpers used across all three fine-tuning stages."""

from typing import Any

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_SEQ_LENGTH = 1024

# Spec 002 §6.3 starting hyperparameters (Stage 1 & 2; Stage 3 DPO overrides lr separately).
DEFAULT_LORA_CONFIG: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 42,
}


def load_base_model(
    model_name: str = BASE_MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
):
    """Load a base (or adapter) model and tokenizer via Unsloth's FastLanguageModel.

    `model_name` may be a HF hub id (base model) or a local/adapter path saved
    by a previous stage.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    return model, tokenizer


def add_lora_adapters(model, **overrides: Any):
    """Attach LoRA adapters to `model` using DEFAULT_LORA_CONFIG, allowing overrides."""
    from unsloth import FastLanguageModel

    config = {**DEFAULT_LORA_CONFIG, **overrides}
    return FastLanguageModel.get_peft_model(model, **config)
