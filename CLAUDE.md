# CLAUDE.md

Guidance for Claude (or any coding agent) working in this repository.

## Project

Fine-tuning an open-weight LLM for the **homeowners insurance domain** (policy Q&A,
coverage explanation, claims process guidance), with hallucination mitigation,
LLM-as-judge evaluation, RAGAS metrics, and deployment as a chat app on Hugging Face Spaces.

Training happens on **Google Colab free-tier GPU (T4, ~15GB VRAM)**, so the model
and training method (QLoRA, 4-bit) must fit that budget.

## Spec-Driven Development Workflow

This repo follows spec-driven development. **Do not start implementation without
a spec.** The workflow is:

1. **Spec** (`specs/NNN-name.md`) — what to build and why, acceptance criteria.
   Read `specs/001-homeowners-insurance-finetune.md` before touching anything.
2. **Plan** — before writing code for a spec section, write a short plan
   (files to touch, approach, risks) and confirm it matches the spec's
   acceptance criteria.
3. **Tasks** — break the plan into small, independently testable tasks.
4. **Implement** — one task at a time. Every task that touches model
   behavior or data must include or update an eval.
5. **Verify** — run the eval suite (RAGAS + LLM-as-judge, see `eval/`) before
   marking a task done. A task is not complete if it regresses faithfulness
   or hallucination scores below the thresholds in the spec.

If a request conflicts with the spec, flag the conflict instead of silently
reinterpreting the spec.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Training env | Google Colab (free T4 GPU) |
| Fine-tuning | PEFT (QLoRA, 4-bit via bitsandbytes), `trl` SFTTrainer / DPOTrainer |
| Base model | Small instruct model that fits T4 in 4-bit (see spec for candidates) |
| Retrieval / grounding | FAISS or Chroma vector store over curated insurance corpus |
| Eval | RAGAS (faithfulness, answer relevancy, context precision/recall) + LLM-as-judge rubric |
| Serving | Hugging Face Spaces, Gradio `ChatInterface` |
| Dependency mgmt | `requirements.txt` per subproject (Colab installs by pip, no lockfile needed) |

## Repo Layout

```
homeowners-insurance-llm/
├── CLAUDE.md
├── specs/
│   └── 001-homeowners-insurance-finetune.md
├── data/              # dataset build scripts, curated QA pairs, source docs (not raw PII)
├── notebooks/         # Colab notebooks (fine-tuning, data prep, eval)
├── eval/              # RAGAS + LLM-as-judge scripts, eval datasets, results
└── app/               # Gradio app for HF Spaces deployment
```

## Conventions

- **Type hints everywhere.** `ruff` + `black` formatting. No bare `except:`.
- **Notebooks are thin.** Business logic (data prep, training loop config, eval
  scoring) lives in importable `.py` modules under `data/`, `eval/`, `app/`;
  notebooks just orchestrate calls into them. This keeps the code testable
  outside Colab and diffable in git.
- **No fact injection during fine-tuning.** Do not fine-tune the model to
  memorize specific numeric coverage limits, dollar amounts, or state-specific
  rules as if they were universal facts — insurance terms vary by policy,
  carrier, and state. Train the model to *explain concepts and reason over
  retrieved policy text*, not to recite memorized figures. See spec §5.
- **Every hallucination-relevant change needs an eval run.** If you touch
  prompts, retrieval, training data, or decoding params, run `eval/run_eval.py`
  and report the before/after RAGAS + judge scores in the PR/commit message.
- **Secrets** (HF tokens, judge-model API keys) go in Colab secrets /
  environment variables — never hardcoded, never committed.
- **Dataset provenance matters.** Every training example must be traceable to
  a source (public ISO/state DOI guidance, licensed corpus, or synthetic-with-
  disclaimer). Do not scrape or fabricate content presented as authoritative
  insurance fact.

## Commands (reference — see spec for full setup)

```bash
# local dev / linting (not training — training runs in Colab)
pip install -r requirements-dev.txt
ruff check .
black --check .

# run eval suite locally against a deployed/local model endpoint
python eval/run_eval.py --config eval/config.yaml

# launch chat app locally before pushing to HF Spaces
python app/app.py
```

## Definition of Done (applies to every task)

- Meets the acceptance criteria in the relevant spec section.
- Passes lint/format checks.
- If model-behavior-affecting: eval suite run, scores meet or exceed the
  thresholds in `specs/001-homeowners-insurance-finetune.md` §8.
- No secrets committed. No raw customer PII in `data/`.
