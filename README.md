# CA Homeowners Insurance AI Assistant — Unsloth Fine-Tuning

Spec-driven fine-tuning assignment: adapt `Qwen/Qwen2.5-0.5B` into a
homeowners-insurance domain assistant (California, USA) using
[Unsloth](https://github.com/unslothai/unsloth), through three stages —
non-instruction fine-tuning, instruction fine-tuning, and DPO preference
alignment — with a before/after evaluation at each stage.

> **Data disclosure:** all training data in `data/` is AI-generated
> synthetic content based on general/public knowledge of CA homeowners
> insurance concepts. It is not sourced from any specific carrier's policy
> document and is not real insurance advice.

- Start here: [`CLAUDE.md`](./CLAUDE.md) — workflow & conventions for the coding agent.
- Spec: [`specs/002-unsloth-ca-homeowners-finetune.md`](./specs/002-unsloth-ca-homeowners-finetune.md)

## Problem statement

`Qwen/Qwen2.5-0.5B`'s base weights have no domain knowledge of California
homeowners insurance — see `reports/base_model_evaluation.md` for evidence
(factual hallucinations, backwards claims, generation collapse into
repetition loops). This project fine-tunes it into a usable domain
assistant purely through parametric knowledge injection (no RAG, no
retrieval — see spec §3 for why that's a deliberate choice for this
exercise), and documents the improvement at every stage against a fixed
10-question set.

## Pipeline

```
Base Model (Qwen/Qwen2.5-0.5B)
   -> Stage 1: Non-instruction fine-tuning (raw domain text, LoRA/QLoRA)
   -> Stage 2: Instruction fine-tuning (instruction/response pairs, LoRA/QLoRA)
   -> Stage 3: DPO preference alignment (chosen/rejected pairs)
Final domain-specific AI assistant
```

All three stages use LoRA (QLoRA-style 4-bit base + LoRA adapters), trained
on Google Colab's free-tier T4 GPU via Unsloth, and are chained together —
each stage loads the previous stage's saved *merged* checkpoint as its
starting base model. See `reports/fine_tuning_explanation.md` for what
LoRA/QLoRA/non-instruction-FT/SFT/DPO actually do and the exact
hyperparameters used per stage.

## Dataset

| File | Rows | Format |
|---|---|---|
| `data/1-non_instruction_data.txt` | 50+ paragraphs | Raw prose (no Q&A structure) |
| `data/2-instruction_dataset.jsonl` | 100+ pairs | `{"instruction", "response"}` |
| `data/3-preference_dataset.jsonl` | 50+ pairs | `{"prompt", "chosen", "rejected"}` |

Covers HO-3 policy structure, covered perils/exclusions, the CA FAIR Plan,
wildfire/earthquake nuances, and the claims process. All synthetic — see
disclosure above and spec §5.

## Evaluation

No automated judge/RAGAS pipeline (see spec §3) — a fixed 10-question set
(spec §4) is run through each stage's model and scored qualitatively
against a rubric (correctness, clarity, safety, helpfulness, specificity —
spec §8.1):

| Report | Comparison |
|---|---|
| [`reports/base_model_evaluation.md`](./reports/base_model_evaluation.md) | Base model alone — establishes the "before" baseline |
| [`reports/sft_model_comparison.md`](./reports/sft_model_comparison.md) | Base vs. Stage 2 (instruction-tuned) |
| [`reports/final_evaluation.md`](./reports/final_evaluation.md) | Base vs. Stage 2 (SFT) vs. Stage 3 (DPO) |
| [`reports/fine_tuning_explanation.md`](./reports/fine_tuning_explanation.md) | LoRA/QLoRA/DPO write-up + actual hyperparameters used |

Bottom line: Stage 2 fixes the base model's worst failures (factual
hallucination, generation collapse) on all 10 questions; Stage 3 (DPO)
ties-or-improves on 9/10 questions over Stage 2, with the clearest win
being more complete/correct claims-process guidance. Both fine-tuned stages
still degenerate into repetition after a few sentences — a known,
documented limitation, not glossed over in the reports above.

## Repo layout

```
LLM-FineTunning/
├── CLAUDE.md
├── README.md
├── requirements.txt              # Colab pip install (training deps)
├── pyproject.toml / uv.lock       # local dev tooling (lint/format + inference)
├── specs/
│   └── 002-unsloth-ca-homeowners-finetune.md
├── data/                          # synthetic training data (see disclosure)
├── notebooks/                     # Colab notebooks, one per stage
├── reports/                       # before/after evaluations + write-up
└── src/
    ├── data_utils.py               # dataset loading/formatting
    ├── model_utils.py               # Unsloth model loading, LoRA config, HF push/pull
    ├── generation_utils.py         # shared text-generation helper
    ├── prompts.py                   # system prompt + fixed 10-question eval set
    └── inference.py                # standalone inference against the final model
```

## Models on the Hugging Face Hub

- Stage 1 (merged): [`sharanmini/qwen2.5-0.5b-ca-homeowners-stage1`](https://huggingface.co/sharanmini/qwen2.5-0.5b-ca-homeowners-stage1)
- Stage 2 (LoRA adapter only): [`sharanmini/qwen2.5-0.5b-ca-homeowners-stage2-adapter`](https://huggingface.co/sharanmini/qwen2.5-0.5b-ca-homeowners-stage2-adapter)
- Stage 3 / final (merged): [`sharanmini/qwen2.5-0.5b-ca-homeowners-final`](https://huggingface.co/sharanmini/qwen2.5-0.5b-ca-homeowners-final)

## How to run

**Training** (Colab, T4 GPU): open each notebook in `notebooks/` in order
(1 → 2 → 3) on Google Colab with a T4 runtime. Each notebook clones this
repo, installs Unsloth, and pulls the previous stage's checkpoint from the
Hub automatically — set `HF_USERNAME` and an `HF_TOKEN` Colab secret first.

**Local dev** (linting/formatting only — no local GPU):
```bash
uv sync
ruff check .
black --check .
```

**Inference** (against the final Stage 3 model, via a Hugging Face Dedicated
Inference Endpoint):
```bash
python -m src.inference
```
Create/start the endpoint for `sharanmini/qwen2.5-0.5b-ca-homeowners-final`
via the [HF UI](https://ui.endpoints.huggingface.co) (billed hourly while
running — pause it when done), then set `HF_ENDPOINT_URL` and `HF_TOKEN`
before running the script. Import `generate_answer(question: str) -> str`
from `src.inference` to ask your own.

## Limitations & disclaimers

- Not a licensed-agent replacement or source of binding coverage
  determinations — general educational information only.
- Trained entirely on synthetic data; illustrative figures should not be
  treated as real policy terms (spec §3, §5).
- Both fine-tuned models still degenerate into repetition/off-topic loops
  after a few sentences on the fixed eval set — see
  `reports/final_evaluation.md` for full documentation of this and other
  residual factual imprecisions.
- No retrieval/RAG, no deployment/hosted UI — scope is intentionally
  limited to a fine-tuning demonstration (spec §3).
