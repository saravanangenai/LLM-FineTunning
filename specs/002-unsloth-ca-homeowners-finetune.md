# Spec 002: California Homeowners Insurance Assistant — Unsloth 3-Stage Fine-Tuning

Status: Draft
Owner: (you)
Last updated: 2026-07-14

## 1. Summary

Fine-tune `Qwen/Qwen2.5-0.5B` into a California-homeowners-insurance domain
assistant using [Unsloth](https://github.com/unslothai/unsloth), through
three stages — non-instruction fine-tuning, instruction fine-tuning, and DPO
preference alignment — with a before/after comparison at each stage. This is
a training/assignment deliverable demonstrating the end-to-end fine-tuning
workflow, not a production RAG system (see §3).

## 2. Goals

- G1: A raw-domain-text corpus (≥50 paragraphs) and a Stage-1
  non-instruction-fine-tuned model that has absorbed CA homeowners-insurance
  vocabulary, tone, and background knowledge.
- G2: An instruction dataset (≥100 pairs) and a Stage-2 instruction-tuned
  model that answers domain questions clearly and specifically, materially
  better than the untuned base model.
- G3: A preference dataset (≥50 chosen/rejected pairs) and a Stage-3
  DPO-aligned model that further improves response quality over Stage 2.
- G4: Written before/after evaluation at each stage (base vs. SFT vs. DPO)
  against a fixed set of 10 domain questions, scored on correctness, domain
  accuracy, clarity, safety, and helpfulness.
- G5: A standalone inference script (`src/inference.py`) that loads the
  final DPO-aligned model/adapter and answers a question end-to-end.
- G6: A written explanation (`reports/fine_tuning_explanation.md`) of LoRA,
  QLoRA, non-instruction FT, SFT, and DPO, plus the actual hyperparameters
  used.
- G7: Entire pipeline trains within Google Colab's free-tier GPU (T4,
  ~15GB VRAM, ~12hr session limit, no guaranteed persistence).

## 3. Non-Goals

- **No retrieval/RAG.** This project deliberately trains the model to
  answer from its own fine-tuned parametric knowledge. There is no vector
  store, no citation UI, no "insufficient context" refusal path tied to
  retrieved passages.
- **No RAGAS or automated judge pipeline.** Evaluation is a human
  (Claude-authored) comparison table against a written rubric, not an
  automated metrics gate.
- **No deployment.** No Hugging Face Spaces app, no hosted chat UI. The
  deliverable stops at a local inference script.
- Not a licensed-agent replacement or source of binding coverage
  determinations — responses should carry an implicit "example only, not
  insurance advice" framing in the system prompt, but heavy hedging is not
  the point of this exercise (contrast with the retired RAG design).
- Not fine-tuning on real, copyrighted carrier policy language, or on any
  real customer/policy PII.
- Not optimizing for languages other than English.

## 4. Domain, Users & Core Use Cases

Domain: **Homeowners insurance, California, USA.**

Target user (in-fiction): a homeowner asking an internal company assistant
general questions about HO-3-style coverage, the CA FAIR Plan, wildfire and
earthquake-related nuances, and the claims process.

Representative questions (also the basis for the fixed 10-question eval
set, §8):
1. What does dwelling coverage mean in a homeowners policy?
2. What's the difference between actual cash value and replacement cost?
3. Does a standard homeowners policy cover wildfire damage in California?
4. What is the CA FAIR Plan and when would I need it?
5. How do I start a claim after a burst pipe?
6. What is typically excluded from a standard homeowners policy?
7. Do I need separate earthquake coverage in California?
8. What is personal liability coverage?
9. How does a home inventory help with a claim?
10. What factors affect homeowners insurance premiums in California?

## 5. Data Requirements

All data in `data/` is **AI-generated synthetic content**, written from
general/public knowledge of CA homeowners insurance concepts (not sourced
from any specific carrier's policy document), and explicitly labeled as
synthetic in `README.md`. No real dollar figures, limits, or terms should
be presented as if drawn from an actual policy — illustrative example
figures are fine and expected (per §3, this project does not hedge away
from concrete numbers the way the retired RAG design did), but they must
read as generic/illustrative, not as a specific real carrier's terms.

### 5.1 Stage 1 — Non-instruction dataset (`data/1-non_instruction_data.txt`)
- ≥50 paragraphs of raw domain text (no Q&A structure) covering: policy
  structure (HO-3 basics), covered perils, exclusions, the CA FAIR Plan,
  earthquake/wildfire considerations, claims process, terminology
  (dwelling coverage, personal property, ACV vs. RCV, deductible,
  endorsement, declarations page, peril, liability coverage, loss of use).
- Plain paragraphs, varied phrasing/sentence structure (this is what the
  model learns *style* and *vocabulary* from) — not a list of facts.

### 5.2 Stage 2 — Instruction dataset (`data/2-instruction_dataset.jsonl`)
- ≥100 examples, each `{"instruction": ..., "response": ...}`.
- Cover the representative questions in §4 plus variations (rephrasing,
  follow-ups, edge cases like "what's not covered").
- Responses should be clear, specific, plain-language, and noticeably
  better than a generic non-domain answer.

### 5.3 Stage 3 — Preference dataset (`data/3-preference_dataset.jsonl`)
- ≥50 examples, each `{"prompt": ..., "chosen": ..., "rejected": ...}`.
- `chosen`: correct, helpful, safe, professional, domain-specific.
- `rejected`: one or more of wrong / incomplete / unsafe / rude / generic /
  not domain-specific — varied failure modes across the set, not the same
  failure repeated 50 times.

### 5.4 Data hygiene
- No real customer PII, no real policy numbers, no verbatim copyrighted
  carrier policy text anywhere in the repo.
- Each dataset file's header/README notes it is synthetic/AI-generated and
  for training-demonstration purposes only.

## 6. Model & Training Approach

### 6.1 Base model
`Qwen/Qwen2.5-0.5B` (fixed by assignment requirement) via Unsloth's
`FastLanguageModel` loader, 4-bit where useful — at 0.5B params this may
comfortably fit in 16-bit on a T4, but load in 4-bit (QLoRA) for
consistency with the stated method and to leave VRAM headroom.

### 6.2 Method, stage by stage

| Stage | Trainer | Data | Objective |
|---|---|---|---|
| 1. Non-instruction FT | Unsloth + `trl` `SFTTrainer` (raw text / packed sequences, no chat template) | §5.1 | Causal LM continuation loss over raw domain paragraphs |
| 2. Instruction FT | Unsloth + `trl` `SFTTrainer` (chat-templated instruction/response) | §5.2, continuing from Stage 1 adapter | Instruction-following on domain Q&A |
| 3. DPO alignment | Unsloth + `trl` `DPOTrainer` | §5.3, continuing from Stage 2 model | Preference optimization, chosen > rejected |

All three stages use LoRA (QLoRA-style 4-bit base + LoRA adapters), not
full fine-tuning.

### 6.3 Starting hyperparameters (sized for a 0.5B model on a T4; document
actual final values used in `reports/fine_tuning_explanation.md`, adjust if
training is unstable or Colab session time runs out)

| Param | Stage 1 (non-instruction) | Stage 2 (instruction) | Stage 3 (DPO) |
|---|---|---|---|
| LoRA rank (r) | 16 | 16 | 16 |
| LoRA alpha | 16 | 16 | 16 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| Target modules | q,k,v,o,gate,up,down proj | same | same |
| Learning rate | 2e-4 | 2e-4 | 5e-6 |
| Effective batch size | 8 (e.g. 2 × grad-accum 4) | 8 | 4 |
| Max seq length | 1024 | 1024 | 1024 |
| Epochs | 3 | 3 | 1–2 |
| Optimizer | `adamw_8bit` | `adamw_8bit` | `adamw_8bit` |
| LR scheduler | cosine | cosine | cosine |

### 6.4 Persistence
- Save the LoRA adapter (and/or merged model) after each stage to Google
  Drive or a private HF model repo — do not rely on Colab local disk.
- Stage 2 continues training from the Stage 1 adapter; Stage 3 continues
  from the Stage 2 model. Each notebook loads the previous stage's saved
  adapter as its starting point.

## 7. Notebooks

| Notebook | Must include |
|---|---|
| `notebooks/1-non_instruction_finetuning.ipynb` | Load raw text → clean/chunk → load base model via Unsloth → apply LoRA/QLoRA → train → save adapter → test generation post-training |
| `notebooks/2-instruction_finetuning.ipynb` | Install/setup → load tokenizer → load Stage 1 model/adapter → format instruction dataset (chat template) → apply LoRA/QLoRA → train → save adapter → run inference on the 10 fixed questions |
| `notebooks/3-dpo_alignment.ipynb` | Load Stage 2 SFT model → load preference dataset → format prompt/chosen/rejected → configure `DPOTrainer` → run DPO → save final adapter/model → test on the 10 fixed questions |

Each notebook keeps business logic (data loading, formatting, generation
helpers) in importable modules under `src/` rather than inline cells, per
CLAUDE.md conventions.

## 8. Evaluation

No RAGAS, no automated judge API — evaluation is a human/Claude-authored
comparison against a written rubric, run on the same fixed 10 questions
(§4) at each stage.

### 8.1 Rubric (score qualitatively per response, not a numeric formula)
- Correctness / domain accuracy
- Clarity (plain language, non-expert-readable)
- Safety (no discouraging a legitimate claim, no definitive legal
  determinations presented as certain)
- Helpfulness
- Specificity (less generic / more clearly domain-grounded than a
  non-fine-tuned answer)

### 8.2 Required reports
- `reports/base_model_evaluation.md` — base model's answers to the 10
  questions + a "Problem" column identifying genericness/gaps. (§ Step 5)
- `reports/sft_model_comparison.md` — base vs. Stage 2 instruction-tuned
  model, table with Question / Base Answer / Fine-Tuned Answer / Which is
  Better / Reason. (§ Step 7)
- `reports/final_evaluation.md` — base vs. Stage 2 (SFT) vs. Stage 3 (DPO),
  table with Question / Base / SFT / DPO / Best Answer / Reason. (§ Step 10)
- `reports/fine_tuning_explanation.md` — plain-language write-up covering:
  why full fine-tuning is expensive, what LoRA does, what QLoRA does, why
  QLoRA suits a T4, what non-instruction FT is, what instruction FT (SFT)
  is, what DPO is, SFT vs. DPO, and the actual rank/alpha/dropout/lr/batch
  values used per stage (§6.3, updated with final-as-run values). (§ Step 11)

### 8.3 Pass bar
Not a numeric gate (contrast with the retired RAG design's RAGAS
thresholds) — the qualitative bar is: Stage 2 must be visibly less generic
and more domain-specific than base on a majority of the 10 questions, and
Stage 3 must be judged equal-or-better than Stage 2 on a majority of the 10
questions (no regression in safety or correctness even if tone/preference
differs).

## 9. Inference Script

`src/inference.py` — loads the final DPO-aligned adapter/model and exposes
a `generate_answer(question: str) -> str` function, plus a `__main__` block
demonstrating one example (per assignment's example: "How can I apply for
reimbursement?" pattern, adapted to a homeowners-insurance question).

## 10. Repo Layout

See CLAUDE.md §Repo Layout — `data/`, `notebooks/`, `reports/`, `src/`,
`requirements.txt`, `README.md` at repo root, numbered files matching the
assignment's required final structure.

## 11. Milestones

1. Spec finalized (this document) — done.
2. Stage 1 data (`data/1-non_instruction_data.txt`, ≥50 paragraphs).
3. Stage 1 notebook + adapter + smoke-test generations.
4. Stage 2 data (`data/2-instruction_dataset.jsonl`, ≥100 pairs).
5. Base model evaluation report (`reports/base_model_evaluation.md`) —
   must happen *before* Stage 2 training so it reflects the true untuned
   base model.
6. Stage 2 notebook + adapter + `reports/sft_model_comparison.md`.
7. Stage 3 preference data (`data/3-preference_dataset.jsonl`, ≥50 pairs).
8. Stage 3 notebook + final model + `reports/final_evaluation.md`.
9. `reports/fine_tuning_explanation.md`.
10. `src/inference.py` + `requirements.txt` + `README.md` finalized.

## 12. Acceptance Criteria (Definition of Done for v1)

- [ ] `data/1-non_instruction_data.txt` has ≥50 paragraphs of raw domain text.
- [ ] `data/2-instruction_dataset.jsonl` has ≥100 instruction/response pairs.
- [ ] `data/3-preference_dataset.jsonl` has ≥50 chosen/rejected pairs.
- [ ] All three notebooks run end-to-end in Colab (T4) and save their
      stage's adapter/model.
- [ ] `reports/base_model_evaluation.md`, `reports/sft_model_comparison.md`,
      `reports/final_evaluation.md`, and `reports/fine_tuning_explanation.md`
      all exist and are complete per §8.2.
- [ ] `src/inference.py` runs locally against the final saved model/adapter
      and returns an answer for an example question.
- [ ] `README.md` covers all 14 required points from the assignment brief.
- [ ] No secrets, real PII, or verbatim copyrighted policy text committed.

## 13. Open Questions

- Merge LoRA adapters into a single merged checkpoint for `src/inference.py`,
  or load base + stacked adapters at inference time? (Decide at Milestone 10
  — merging is simpler for a standalone script; stacked adapters are smaller
  to store.)
- Push the final model/adapter to a public/private HF model repo, or keep
  it Drive-only? (Not required by the assignment; decide if a shareable
  link is wanted.)
