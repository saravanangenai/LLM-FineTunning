# Homeowners Insurance LLM

Spec-driven project: fine-tune an LLM for the homeowners insurance domain,
with hallucination mitigation, RAGAS + LLM-as-judge evaluation, and
deployment to a Hugging Face Spaces chat app.

- Start here: [`CLAUDE.md`](./CLAUDE.md) — workflow & conventions for the coding agent.
- Spec: [`specs/001-homeowners-insurance-finetune.md`](./specs/001-homeowners-insurance-finetune.md)

## Layout
- `notebooks/` — Colab notebooks (data prep, fine-tuning, eval)
- `data/` — dataset build scripts / curated data (no PII, no raw licensed policy text)
- `eval/` — RAGAS + LLM-as-judge scripts and results
- `app/` — Gradio chat app for HF Spaces
