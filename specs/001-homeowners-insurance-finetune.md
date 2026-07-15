# Spec 001: Homeowners Insurance Domain LLM — Fine-Tune, Ground, Evaluate, Deploy

Status: Draft
Owner: (you)
Last updated: 2026-07-14

## 1. Summary

Fine-tune an open-weight instruct LLM to act as a homeowners-insurance domain
assistant (explains coverage concepts, policy terms, claims process, common
endorsements) that is trained and evaluated with hallucination mitigation as a
first-class requirement, using free Google Colab GPU, and deployed as a public
chat app on Hugging Face Spaces.

## 2. Goals

- G1: A fine-tuned model that communicates homeowners insurance concepts
  clearly, in plain language, at a similar or better quality than the base
  instruct model on domain phrasing/tone.
- G2: A retrieval-grounded answer path so factual/numeric claims are tied to
  retrieved source text, not memorized parametric "facts."
- G3: A measurable, repeatable hallucination-mitigation strategy (data
  design + refusal training + RAG + eval gate).
- G4: RAGAS-based automated evaluation integrated into the workflow.
- G5: An LLM-as-judge rubric for qualitative correctness/tone/safety scoring.
- G6: A deployed, working chat interface on Hugging Face Spaces.
- G7: Entire training pipeline runs within Google Colab's free-tier GPU
  (T4, ~15GB VRAM, ~12hr session limit, no guaranteed persistence).

## 3. Non-Goals

- Not a licensed-agent replacement or source of binding coverage
  determinations — output must carry a "not insurance advice" disclaimer.
- Not building a production-grade multi-tenant RAG system — a single shared
  curated knowledge base is sufficient for v1.
- Not fine-tuning on proprietary/licensed carrier policy language without
  verifying redistribution rights.
- Not optimizing for languages other than English in v1.

## 4. Users & Core Use Cases

Target user: a homeowner trying to understand their policy or the claims
process before/after a loss event.

Representative queries:
- "What's the difference between dwelling coverage and personal property coverage?"
- "Does my policy cover a burst pipe in winter?"
- "What is ACV vs replacement cost value?"
- "What's typically excluded from a standard HO-3 policy?"
- "How do I start a claim after a roof leak?"
- Out-of-scope probe: "What will my exact deductible be?" → should not
  fabricate a number; should explain it depends on the policy and direct the
  user to check their declarations page.

## 5. Data Requirements

### 5.1 Sources (must be traceable, licensing-checked)
- Public regulatory/consumer-education material (e.g., state Department of
  Insurance consumer guides, NAIC consumer resources, III.org educational
  content) — verify each source's reuse terms before inclusion.
- ISO/standard policy form *concepts* described generically (do not
  reproduce copyrighted ISO form text verbatim).
- Synthetic Q&A generated from the above sources, reviewed for faithfulness
  to the source before being used as training targets.

### 5.2 Composition
- **Concept-explanation pairs**: term → plain-language explanation.
- **Grounded QA pairs**: (retrieved passage, question, answer-derived-only-
  from-passage).
- **Refusal / hedging examples** (critical — target ~15-20% of the dataset):
  questions where the correct answer is "this varies by policy/state, check
  your declarations page" or "I don't have enough information," including:
  - questions asking for specific dollar amounts/limits,
  - questions requiring the user's actual policy document,
  - questions outside homeowners insurance scope entirely.
- **Adversarial/distractor examples**: retrieved context that does *not*
  answer the question, with target output correctly declining to answer
  from it.

### 5.3 Data hygiene
- No real customer PII or real policy numbers anywhere in the repo.
- Every example tagged with `source_id` for traceability.
- Train/validation/test split stratified so refusal examples appear in all
  three splits.

## 6. Model & Training Approach

### 6.1 Base model candidates (must run 4-bit QLoRA on a T4)
Evaluate and pick one:
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct` (fallback if VRAM/session-time is tight —
  smaller, faster to iterate on free Colab)

Decision criteria: license permits the intended use, tokenizer handles
domain terms reasonably, fits 4-bit + LoRA adapters + gradient checkpointing
in ~15GB VRAM with the target sequence length (recommend 2048–4096 tokens).

### 6.2 Method
- **QLoRA**: 4-bit NF4 base weights via `bitsandbytes`, LoRA adapters via
  `peft`, trained with `trl`'s `SFTTrainer`.
- Stage 1 — SFT on the grounded QA + refusal dataset (§5.2).
- Stage 2 (optional, recommended) — **DPO** using preference pairs of
  (grounded/correctly-hedged answer) vs (fabricated/overconfident answer) to
  further push down hallucination rate. Use `trl`'s `DPOTrainer`.
- Checkpoint to Google Drive or push adapters to a private HF model repo
  after each Colab session — do not rely on Colab local disk persisting.

### 6.3 Inference-time grounding (RAG)
- Build a vector index (FAISS or Chroma) over the curated source corpus
  (§5.1), chunked with metadata (source, section).
- At inference: retrieve top-k passages → inject into the prompt as
  context → model must answer only from context or explicitly say the
  context doesn't cover it.
- Retrieved source snippets are shown to the user (citation UI) so claims
  are checkable, not just asserted.

## 7. Hallucination Mitigation Strategy (cross-cutting requirement)

This is not a single feature — it's enforced across the pipeline:

1. **Data-level**: no fact injection beyond what the base model already
   generally knows about insurance concepts (see CLAUDE.md); heavy inclusion
   of refusal/hedging examples; adversarial distractor-context examples.
2. **Training-level**: DPO stage penalizing fabricated/overconfident
   answers relative to grounded/hedged ones.
3. **Inference-level**: mandatory RAG — no ungrounded factual claims about
   coverage specifics; system prompt instructs the model to decline when
   retrieved context is insufficient.
4. **Output-level**: every response involving a specific figure, limit, or
   coverage determination must either (a) cite retrieved context or (b)
   include an explicit hedge directing the user to their policy/agent.
5. **Eval-level (gate)**: RAGAS faithfulness/context-precision scores and
   LLM-as-judge hallucination rate must meet thresholds (§8) before a model
   version is promoted to deployment.

## 8. Evaluation

### 8.1 RAGAS metrics (automated, run against a held-out eval set with
retrieved contexts)
- **Faithfulness** — target ≥ 0.85
- **Answer relevancy** — target ≥ 0.80
- **Context precision** — target ≥ 0.75
- **Context recall** — target ≥ 0.75

### 8.2 LLM-as-judge rubric
Use a strong external judge model (e.g., via API) scoring each response 1-5 on:
- Factual grounding (does every specific claim trace to retrieved context?)
- Appropriate hedging (does it correctly decline/hedge when it should?)
- Clarity/tone for a non-expert homeowner
- Safety (no discouraging someone from filing a legitimate claim, no
  definitive legal/coverage determinations)

Judge prompts, rubric, and scoring script live in `eval/`. Track scores per
model version in `eval/results/`.

### 8.3 Regression gate
No model version is deployed to the HF Space if it scores below §8.1/§8.2
thresholds on the held-out set, or regresses more than 2% from the
previously deployed version on any metric without an explicit override note.

## 9. Deployment

- Package: LoRA adapter (or merged model) + retrieval index + Gradio app.
- Host: Hugging Face Spaces (Gradio SDK), `ChatInterface`.
- App must show: chat window, source citations for grounded answers, a
  persistent "not insurance advice" disclaimer, and a visible model/version
  tag.
- Secrets (HF token, judge API key if used at runtime) via HF Spaces secrets,
  not committed.
- If free CPU Spaces hardware is insufficient for the chosen base model at
  inference, use 4-bit inference or an inference API (HF Inference Endpoints,
  or the free Inference API for the base model + adapter) rather than
  upgrading to paid Spaces hardware by default — flag if paid tier becomes
  necessary.

## 10. Milestones

1. Environment setup (Colab notebook skeleton, repo scaffold) — this spec.
2. Data collection + curation + refusal/adversarial set (§5).
3. Baseline eval of the un-tuned base model on the eval set (establishes
   before/after comparison).
4. SFT fine-tune (QLoRA) + eval.
5. DPO stage + eval.
6. RAG integration + citation UI + eval with retrieval in the loop.
7. LLM-as-judge + RAGAS gate automated in `eval/run_eval.py`.
8. Deploy to HF Spaces; smoke test; publish model card documenting
   limitations and the "not insurance advice" disclaimer.

## 11. Acceptance Criteria (Definition of Done for v1)

- [ ] Eval set of ≥150 held-out questions (mix of answerable, hedge-required,
      and adversarial-context) exists and is version-controlled.
- [ ] Fine-tuned model beats base model on judge hallucination score by a
      measurable margin on the eval set.
- [ ] RAGAS thresholds in §8.1 met.
- [ ] Deployed HF Space is publicly reachable, shows citations, and shows
      the disclaimer.
- [ ] Model card published describing training data sources, known
      limitations, and explicit non-use as binding insurance advice.

## 12. Open Questions

- Which base model fits best given actual T4 session-time limits once
  dataset size is known? (Decide at Milestone 3.)
- Which judge model/API will be used for LLM-as-judge, and what's the cost
  budget for eval runs?
- Do we need state-specific handling (insurance regulation varies by state),
  or is v1 explicitly general/non-state-specific with a disclaimer?
