# CA Homeowners Insurance AI Assistant — Unsloth Fine-Tuning

Spec-driven fine-tuning assignment: adapt `Qwen/Qwen2.5-0.5B` into a
homeowners-insurance domain assistant (California, USA) using
[Unsloth](https://github.com/unslothai/unsloth), through three stages —
non-instruction fine-tuning, instruction fine-tuning, and DPO preference
alignment — with a before/after comparison at each stage.

> **Data disclosure:** all training data in `data/` is AI-generated
> synthetic content based on general/public knowledge of CA homeowners
> insurance concepts. It is not sourced from any specific carrier's policy
> document and is not real insurance advice.

- Start here: [`CLAUDE.md`](./CLAUDE.md) — workflow & conventions for the coding agent.
- Spec: [`specs/002-unsloth-ca-homeowners-finetune.md`](./specs/002-unsloth-ca-homeowners-finetune.md)

## Layout
- `data/` — raw domain text, instruction dataset, preference dataset (all synthetic)
- `notebooks/` — Colab notebooks: non-instruction FT, instruction FT, DPO alignment
- `reports/` — before/after evaluation tables and the LoRA/QLoRA/DPO write-up
- `src/` — shared helper modules + `inference.py` for the final model
