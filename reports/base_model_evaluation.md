# Base Model Evaluation — `Qwen/Qwen2.5-0.5B` (pre-fine-tuning)

Spec: `specs/002-unsloth-ca-homeowners-finetune.md` §4, §8.2 (Step 5 deliverable).

## Method

- Model: `Qwen/Qwen2.5-0.5B` (base, not `-Instruct`), loaded directly from
  Hugging Face via `transformers.AutoModelForCausalLM`, unmodified — no
  fine-tuning applied. This is the exact starting checkpoint Stage 1 will
  adapt.
- Decoding: greedy (`do_sample=False`), `max_new_tokens=150`, no repetition
  penalty — deliberately plain settings so the raw base-model behavior is
  visible, rather than tuned to look better than it is.
- Prompting: each question was wrapped with the tokenizer's chat template
  (`apply_chat_template`, single user turn) to give the base model its best
  chance at a conversational answer, even though it has not been
  instruction-tuned by Qwen or by this project yet.
- Run locally on CPU (not Colab) — for a 0.5B-parameter model doing pure
  inference (no training), CPU vs. GPU does not change the output for
  greedy/deterministic decoding, so this is a faithful "before" snapshot of
  the same weights Stage 1 starts from.
- Answers below are the model's actual, unedited output. Several responses
  degenerate into token repetition loops, which is real base-model behavior
  on an out-of-distribution chat prompt, not an artifact of editing —
  long repeated segments are truncated with `[repeats]` for readability,
  full raw output is preserved in `reports/base_model_answers_raw.json`.

## Results

| # | Question | Base Model Answer | Problem |
|---|---|---|---|
| 1 | What does dwelling coverage mean in a homeowners policy? | "dwelling coverage means that the policy covers the dwelling property, which is the house or apartment where the policyholder lives. This coverage is typically provided by the homeowner's insurance company and is used to protect the policyholder from damage or loss to the property due to natural disasters, accidents, or other events." | Vague and circular ("dwelling coverage covers the dwelling"); no mention of Coverage A, rebuild-cost basis, or how it differs from market value. Not wrong, just generic — could describe any coverage type. |
| 2 | What's the difference between actual cash value and replacement cost? | "The actual cash value is the amount of money that you would have to pay to replace the asset if it were to be sold at its current market value. The replacement cost is the amount of money that you would have to pay to replace the asset if it were to be purchased at its current market value. `[repeats: "yük / Comey / What is the difference between actual cash value and replacement cost?" ~5x]`" | Factually blurs the two terms (both definitions describe "cost to replace," with no mention of depreciation, which is the actual distinguishing concept) and then degenerates into a repetition loop instead of stopping. |
| 3 | Does a standard homeowners policy cover wildfire damage in California? | "wildfire damage in California is not covered by a standard homeowners policy. wildfire damage is typically covered by a wildfire insurance policy specifically designed for wildfire damage... `[repeats a near-identical sentence ~3x, then trails into garbled tokens]`" | Factually backwards — wildfire is a named peril generally covered by a standard HO-3 policy (the *actual* CA-specific issue is insurer non-renewal/availability in high-risk zones, not a blanket exclusion). Invents a nonexistent "wildfire insurance policy" product. Degenerates into repetition. |
| 4 | What is the CA FAIR Plan and when would I need it? | "The CA FAIR Plan is a set of guidelines and recommendations for ensuring that the data used in research is accurate, complete, and reliable. It was developed by the Center for Advanced Research in Education (CA RE)..." | Outright hallucination — confuses "FAIR" (the California FAIR Plan insurance pool) with the unrelated "FAIR data principles" from research/data science, inventing a fictional organization. Completely wrong domain, stated with full confidence. |
| 5 | How do I start a claim after a burst pipe? | "You are a helpful assistant.rible \| libertine \nYou are a helpful assistant.rible \| libertine `[repeats this fragment ~15x]`" | Total generation failure — the model echoes back what looks like a leaked system-prompt fragment and loops indefinitely. Zero relevant content; does not answer the question at all. |
| 6 | What is typically excluded from a standard homeowners policy? | "A standard homeowners policy typically excludes certain types of coverage, such as: `[repeats the single word "cougar" ~65x]`" | Starts correctly (correct framing) but immediately collapses into meaningless token repetition before listing a single actual exclusion. No usable content. |
| 7 | Do I need separate earthquake coverage in California? | "Yes, you do. California is a state with a high risk of earthquakes, and earthquake coverage is essential for any traveler. `[repeats "You are a helpful assistant.icode" ~9x]`" | Gets the core "yes" correct but for the wrong reason ("essential for any traveler" — nonsensical, sounds like a travel-insurance answer bleeding in), no mention of the CEA, and degenerates into repetition. |
| 8 | What is personal liability coverage? | "Personal liability coverage is a type of insurance that covers the financial responsibility of a person or entity for any damages or losses that they may suffer as a result of their actions or negligence... Personal liability coverage is typically provided by insurance companies and is often included in auto insurance policies. `[repeats "yük / What is the difference between personal liability and general liability coverage" ~2x]`" | Reasonably close on the general concept, but incorrectly frames it as primarily an auto-insurance feature rather than a homeowners policy coverage (Coverage E), and again drifts into repetition/self-questioning instead of stopping. |
| 9 | How does a home inventory help with a claim? | "A home inventory is a document that lists all the items in a home that are needed for a claim. It helps to ensure that all the necessary items are included in the claim, and that the claim is accurate and complete... `[repeats "navigationOptions / You are a helpful assistant." ~5x]`" | The core idea is roughly right and is the best answer of the ten, but it's generic (no mention of photos, purchase dates, faster settlement) and still trails into a repetition loop instead of ending cleanly. |
| 10 | What factors affect homeowners insurance premiums in California? | "Homeowners insurance premiums in California are affected by a variety of factors, including: 1. Home Value... 2. Property Type... 3. Property Condition..." (cut off mid-list at the 150-token limit, no repetition) | The only fully coherent, non-degenerate answer — but entirely generic (home value, property type, condition) with zero California-specific content: no mention of wildfire hazard zones, roof age, claims history, or the FAIR Plan/CEA context that actually drives CA premiums. |

## Summary

Across the 10 questions, the untuned base model shows three consistent
problems that the fine-tuning stages are meant to fix:

1. **Genericness** — even the "best" answers (Q1, Q9, Q10) describe
   concepts in a domain-agnostic way with no California-specific detail
   (FAIR Plan, CEA, wildfire risk zones, percentage deductibles).
2. **Domain hallucination** — Q4 confidently invents an entirely wrong
   meaning for "CA FAIR Plan," and Q3 states the opposite of reality about
   wildfire coverage.
3. **Generation instability** — 7 of 10 answers degenerate into repetition
   loops or off-topic fragments ("You are a helpful assistant," "cougar,"
   "yük") well before the 150-token limit, indicating the base model has
   no real instruction-following behavior yet — exactly what Stage 2
   (instruction fine-tuning) targets.

This becomes the "before" baseline for `reports/sft_model_comparison.md`
(Stage 2) and `reports/final_evaluation.md` (Stage 3).
