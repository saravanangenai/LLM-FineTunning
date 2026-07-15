# CLAUDE.md

Guidance for Claude (or any coding agent) working in this repository.

## Project

A three-stage LLM fine-tuning pipeline, built with **Unsloth**, that turns a
small open-weight base model into a domain-specific assistant for
**homeowners insurance in California, USA**. This is a hands-on fine-tuning
exercise: the goal is to demonstrate the full workflow — raw-text domain
adaptation, instruction tuning, and preference alignment — not to build a
production system.

Pipeline:

```
Base Model (Qwen2.5-0.5B)
   ↓ Stage 1: Non-instruction fine-tuning (raw domain text, LoRA/QLoRA)
   ↓ Stage 2: Instruction fine-tuning (instruction/response pairs, LoRA/QLoRA)
   ↓ Stage 3: DPO preference alignment (chosen/rejected pairs)
Final domain-specific AI assistant
```

Training happens on **Google Colab free-tier GPU (T4, ~15GB VRAM)** via
Unsloth. There is no local GPU in this dev environment — the local `.venv`
is for linting/formatting and running `src/inference.py` against an already
fine-tuned model only.

## Spec-Driven Development Workflow

This repo follows spec-driven development. **Do not start implementation
without a spec.** The workflow is:

1. **Spec** (`specs/NNN-name.md`) — what to build and why, acceptance
   criteria. Read `specs/002-unsloth-ca-homeowners-finetune.md` before
   touching anything.
2. **Plan** — before writing code for a spec section, write a short plan
   (files to touch, approach, risks) and confirm it matches the spec's
   acceptance criteria.
3. **Tasks** — break the plan into small, independently testable tasks.
4. **Implement** — one task/stage at a time.
5. **Verify** — after each fine-tuning stage, run the same fixed question
   set through the model and update the relevant `reports/*.md` comparison
   table before marking the stage done.

If a request conflicts with the spec, flag the conflict instead of silently
reinterpreting the spec.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Training env | Google Colab (free T4 GPU) |
| Fine-tuning | [Unsloth](https://github.com/unslothai/unsloth) + `peft` (LoRA/QLoRA, 4-bit via bitsandbytes) + `trl` (`SFTTrainer`, `DPOTrainer`) |
| Base model | `Qwen/Qwen2.5-0.5B` (see spec §6 for stage-by-stage config) |
| Eval | Manual before/after comparison on a fixed question set, scored against a written rubric (correctness, domain accuracy, clarity, safety, helpfulness) — see `reports/` |
| Inference | Plain Python script (`src/inference.py`), no serving layer |
| Dependency mgmt | `pyproject.toml`/`uv.lock` for local dev tooling; `requirements.txt` at repo root for Colab `pip install` |

## Repo Layout

```
LLM-FineTunning/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── specs/
│   └── 002-unsloth-ca-homeowners-finetune.md
├── data/
│   ├── 1-non_instruction_data.txt      # ≥50 raw domain paragraphs
│   ├── 2-instruction_dataset.jsonl     # ≥100 instruction/response pairs
│   └── 3-preference_dataset.jsonl      # ≥50 chosen/rejected pairs (DPO)
├── notebooks/
│   ├── 1-non_instruction_finetuning.ipynb
│   ├── 2-instruction_finetuning.ipynb
│   └── 3-dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
└── src/
    └── inference.py
```

## Conventions

- **Type hints everywhere.** `ruff` + `black` formatting. No bare `except:`.
- **Notebooks are thin.** Shared logic (data loading, prompt formatting,
  generation helpers) lives in importable `.py` modules under `src/`;
  notebooks orchestrate calls into them so code is diffable and reusable
  across the three stages.
- **This project *is* fact injection, by design.** Unlike a RAG-grounded
  assistant, Stage 1 and Stage 2 deliberately train the model to internalize
  California homeowners-insurance terminology, concepts, and example
  figures. There is no retrieval step and no requirement to hedge on every
  number — that's the point of the exercise. Keep this in mind when writing
  training data: it should read like confident domain knowledge, not
  refusal-heavy RAG output.
- **Dataset provenance matters.** All training data (raw paragraphs,
  instruction pairs, preference pairs) is AI-generated synthetic content
  based on general/public knowledge of CA homeowners insurance (HO-3
  basics, CA FAIR Plan, CA DOI processes, wildfire/earthquake nuances) —
  explicitly disclosed as synthetic in `README.md` and `data/`. Do not
  reproduce copyrighted carrier policy language verbatim, and do not
  present any specific dollar figure/limit as if drawn from a real policy.
- **Every stage needs a report update.** If you touch training data,
  hyperparameters, or prompts for a given stage, re-run the fixed
  10-question set through the affected model(s) and update the
  corresponding `reports/*.md` comparison table — don't leave it stale.
- **Secrets** (HF tokens) go in Colab secrets / environment variables —
  never hardcoded, never committed.

## Commands (reference — see spec for full setup)

```bash
# local dev / linting (not training — training runs in Colab)
uv sync
ruff check .
black --check .

# run inference against the final DPO-aligned model
python src/inference.py
```

## Definition of Done (applies to every task)

- Meets the acceptance criteria in the relevant spec section.
- Passes lint/format checks.
- If model-behavior-affecting: the fixed-question comparison in the
  relevant `reports/*.md` file is updated and reflects the current model.
- No secrets committed. No real customer PII or real carrier policy text
  in `data/`.
