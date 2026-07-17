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


def save_merged_model(
    model, tokenizer, out_dir: str, save_method: str = "merged_16bit"
) -> None:
    """Save a full merged (base weights + LoRA) checkpoint, per spec §6.4.

    The next stage loads this directory as a fresh base model via
    `load_base_model` and attaches its own new LoRA adapters, rather than
    resuming training on top of the previous stage's PEFT wrapper.
    """
    model.save_pretrained_merged(out_dir, tokenizer, save_method=save_method)


def push_folder_to_hub(
    folder: str, repo_id: str, token: str, private: bool = True
) -> None:
    """Upload an already-saved local checkpoint folder (adapter or merged) to the
    Hugging Face Hub, creating the repo if it doesn't exist yet.

    Used to move a checkpoint between Colab sessions without a slow browser
    upload/download round trip (see spec §6.4) — the next stage's notebook
    loads `repo_id` directly via `load_base_model`.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model")
