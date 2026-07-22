# Fine-Tuning Explanation — LoRA, QLoRA, and the 3-Stage Pipeline

Spec: `specs/002-unsloth-ca-homeowners-finetune.md` §8.2 (Step 11 deliverable).

## Why not full fine-tuning?

Full fine-tuning updates every parameter of the base model. Even at
0.5B parameters, that means storing full-precision gradients and optimizer
state (Adam keeps two extra moments per parameter) for all 500M weights —
roughly 4-8x the model's own memory footprint once you add gradients and
optimizer state on top of the weights themselves. On a free-tier Colab T4
(~15GB VRAM), that's tight even for a model this small, and it produces a
brand-new multi-GB checkpoint per stage with no reuse between stages. It
also risks catastrophic forgetting — overwriting the base model's general
language ability while it learns a narrow domain.

## What LoRA does

LoRA (Low-Rank Adaptation) freezes all of the base model's original
weights and injects a small pair of trainable low-rank matrices
(`A`, `B`) alongside selected weight matrices (here: the attention
projections `q,k,v,o` and the MLP projections `gate,up,down` — spec
§6.2/`src/model_utils.DEFAULT_LORA_CONFIG`). Instead of learning a full
`d x d` weight update, it learns a `d x r` and `r x d` pair where
`r` (the "rank") is much smaller than `d` — 16 in this project. The
weight update at inference is `W + BA`, computed on the fly (or merged in).
This cuts trainable parameters by orders of magnitude, meaning far less
optimizer state, far smaller checkpoints (megabytes, not gigabytes), and a
much lower chance of destroying the base model's general knowledge since
the original weights never change.

## What QLoRA adds

QLoRA is LoRA applied on top of a 4-bit-quantized base model
(`load_in_4bit=True` in `src/model_utils.load_base_model`). The frozen base
weights are stored in 4-bit (NF4) precision instead of 16-bit, roughly
quartering the memory needed just to hold the base model in VRAM, while the
small trainable LoRA matrices stay in higher precision so training quality
isn't compromised. Gradients only ever flow through the LoRA matrices, not
the quantized base weights.

## Why QLoRA suits a T4

The T4's ~15GB VRAM is enough to comfortably hold a 4-bit `Qwen2.5-0.5B`
base plus LoRA adapters, activations, and optimizer state for the (tiny)
trainable parameter count — with headroom to spare, even though at 0.5B
params this model would likely fit in 16-bit too. QLoRA is used here for
consistency with the assignment's stated method and to leave margin for
longer sequences/larger batches without hitting Colab's free-tier memory
ceiling.

## Non-instruction fine-tuning (Stage 1)

Plain causal-language-model continuation training over raw, unstructured
domain paragraphs (`data/1-non_instruction_data.txt`) — no chat template,
no Q&A structure, just `trl`'s `SFTTrainer` with `packing=True` concatenating
paragraphs into fixed-length blocks and training the standard
next-token-prediction loss. The goal isn't to teach the model to *follow
instructions* yet; it's to shift its background knowledge, vocabulary, and
tone toward California-homeowners-insurance terminology (dwelling coverage,
ACV vs. RCV, the CA FAIR Plan, perils, endorsements) before it ever sees a
question-and-answer format.

## Instruction fine-tuning / SFT (Stage 2)

Supervised fine-tuning on `{"instruction", "response"}` pairs
(`data/2-instruction_dataset.jsonl`), chat-templated with a system prompt
via `src/data_utils.build_chat_dataset` and trained with the same
`SFTTrainer`, but now on the *loss over full chat-formatted turns* rather
than raw text. This is what teaches the model to behave like an assistant:
given a question, produce a direct, on-topic answer, continuing from the
Stage 1 domain-adapted checkpoint rather than starting over from the
un-adapted base model.

## DPO (Stage 3)

Direct Preference Optimization trains directly on relative preferences
instead of a single "correct" target. Each example in
`data/3-preference_dataset.jsonl` is a `(prompt, chosen, rejected)` triple;
`trl`'s `DPOTrainer` adjusts the model so it assigns higher likelihood to
`chosen` over `rejected` for the same prompt, using an implicit reward
derived from the log-probability ratio against a reference model (here, the
Stage 2 model itself, `ref_model=None` letting `DPOTrainer` use a frozen
copy). It doesn't teach new facts the way Stage 1/2 do — it sharpens
*which* of several plausible outputs the model prefers: more complete,
safer, more professional answers over generic/wrong/rude ones.

## SFT vs. DPO

SFT teaches the model to imitate a single fixed target response per
example — it has no signal about *why* one phrasing beats another, only
"produce this." DPO instead teaches from contrastive pairs, giving the
model a relative quality signal (chosen > rejected) without needing a
separate reward model (contrast with full RLHF/PPO). In this pipeline, SFT
does the heavy lifting of turning a domain-adapted base model into a
functioning instruction-follower; DPO then polishes response quality
(completeness, safety framing, tone) on top of that already-competent
model — which is why DPO uses a far smaller learning rate and runs for
fewer effective updates than SFT (see table below).

## Actual hyperparameters used (as run)

| Param | Stage 1 (non-instruction) | Stage 2 (instruction) | Stage 3 (DPO) |
|---|---|---|---|
| LoRA rank (r) | 16 | 16 | 16 |
| LoRA alpha | 16 | 16 | 16 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| Target modules | q,k,v,o,gate,up,down proj | same | same |
| Learning rate | 2e-4 | 2e-4 | 5e-6 |
| Per-device batch size | 2 | 2 | 2 |
| Gradient accumulation | 4 | 4 | 2 |
| Effective batch size | 8 | 8 | 4 |
| Max seq length | 1024 | 1024 | 1024 (max_prompt_length=512) |
| Epochs | 3 | 3 | 2 |
| Optimizer | `adamw_8bit` | `adamw_8bit` | `adamw_8bit` |
| LR scheduler | cosine, warmup_ratio=0.03 | cosine, warmup_ratio=0.03 | cosine, warmup_ratio=0.03 |
| DPO beta | — | — | 0.1 |
| Seed | 42 | 42 | 42 |

These match the spec §6.3 starting values exactly — no instability required
adjusting them mid-run. Source: `src/model_utils.DEFAULT_LORA_CONFIG` and
the `TrainingArguments`/`DPOConfig` cells in
`notebooks/1-non_instruction_finetuning.ipynb`,
`notebooks/2-instruction_finetuning.ipynb`, and
`notebooks/3-dpo_alignment.ipynb`.

## Persistence

Each stage saves both a LoRA adapter and a full merged (base + LoRA,
16-bit) checkpoint, then pushes both to the Hugging Face Hub
(`src/model_utils.push_stage_outputs`) rather than relying on Colab's
non-persistent local disk. The next stage pulls the previous stage's
*merged* checkpoint as its new starting "base" and attaches a fresh set of
LoRA adapters (`src/model_utils.resolve_stage_source`), rather than
resuming training on top of the previous stage's PEFT wrapper directly.
