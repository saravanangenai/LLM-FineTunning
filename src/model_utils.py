"""Shared Unsloth model-loading and LoRA helpers used across all three fine-tuning stages."""

from pathlib import Path
from typing import Any

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_SEQ_LENGTH = 1024

# Qwen/Qwen2.5-0.5B is the base (non-instruct) checkpoint, so its tokenizer
# ships with no chat_template — apply_chat_template() raises without one.
# This is the same ChatML format Qwen2.5-Instruct's tokenizer uses.
QWEN_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

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


def resolve_stage_source(label: str, local_dir: Path, hf_repo: str) -> str:
    """Return `local_dir` if it exists (e.g. the previous stage ran in this
    same Colab session), else `hf_repo` — the shared "load the previous
    stage's checkpoint" fallback used at the start of every stage after the
    first (see spec §6.4).
    """
    if local_dir.exists():
        return str(local_dir)
    print(f"{label} not found locally — pulling from the Hugging Face Hub: {hf_repo}")
    return hf_repo


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
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
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


def push_stage_outputs(adapter_dir: str, merged_dir: str, repo_id: str, token: str) -> None:
    """Push a stage's adapter and merged-model checkpoints to the Hub as
    `{repo_id}-adapter` and `{repo_id}` respectively, then print the merged
    model's URL — the shared shape of every stage's persistence cell.
    """
    push_folder_to_hub(adapter_dir, repo_id + "-adapter", token=token)
    push_folder_to_hub(merged_dir, repo_id, token=token)
    print("Pushed merged model to:", f"https://huggingface.co/{repo_id}")
