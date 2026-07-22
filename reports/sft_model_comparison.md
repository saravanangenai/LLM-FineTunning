# SFT Model Comparison — Base vs. Stage 2 (Instruction Fine-Tuned)

Spec: `specs/002-unsloth-ca-homeowners-finetune.md` §4, §8.2 (Step 7 deliverable).

## Method

- Base model answers: reused verbatim from `reports/base_model_evaluation.md`
  / `reports/base_model_answers_raw.json` (same weights, same prompting, same
  decoding settings — a true "before" baseline).
- Fine-tuned model: Stage 2 was never pushed to the Hub as a merged
  checkpoint (only `sharanmini/qwen2.5-0.5b-ca-homeowners-stage2-adapter`,
  the LoRA adapter, exists there — Stage 3 consumed the in-session Stage 2
  model directly on Colab rather than pulling a merged repo). Reproduced
  the equivalent model locally by loading the public Stage 1 merged
  checkpoint (`sharanmini/qwen2.5-0.5b-ca-homeowners-stage1`) as the base
  and applying the public Stage 2 LoRA adapter via `peft.PeftModel`, which
  is mathematically identical to the intended merged Stage 2 model.
- Decoding: greedy (`do_sample=False`), `max_new_tokens=150`, no repetition
  penalty — identical settings to the base-model eval for a fair
  comparison.
- Prompting: chat template with `SYSTEM_PROMPT` (`src/prompts.py`), same as
  base eval.
- Run locally on CPU. Full raw output in
  `reports/sft_model_answers_raw.json`; long repeated tails are truncated
  below with `[...]` for readability.

## Results

| # | Question | Base Answer | Fine-Tuned (Stage 2) Answer | Which is Better | Reason |
|---|---|---|---|---|---|
| 1 | What does dwelling coverage mean in a homeowners policy? | Vague/circular ("dwelling coverage covers the dwelling"); no rebuild-cost framing. | "Dwelling coverage is a special type of coverage that pays for the cost of repairing or rebuilding a dwelling... after a fire, flood, or other catastrophic event. It's often included in a standard homeowners policy, but some policies may offer a higher limit..." [degenerates into an off-topic repeated question loop] | Fine-Tuned | Correctly frames it as rebuild/repair-cost coverage rather than a circular restatement — concrete improvement, though still trails into repetition. |
| 2 | What's the difference between actual cash value and replacement cost? | Blurs both terms into near-identical definitions, no mention of depreciation, then degenerates. | "The actual cash value is the replacement cost of the structure minus the cost of rebuilding it... The replacement cost is the cost of rebuilding the structure..." [degenerates into repeated off-topic question loop] | Fine-Tuned (partial) | Cleaner prose and no more mid-sentence repetition loop, but the core concept is still muddled (never mentions depreciation, the actual distinguishing factor) — a real remaining gap. |
| 3 | Does a standard homeowners policy cover wildfire damage in California? | States the opposite of reality — "not covered," invents a fictional "wildfire insurance policy" product. | "Yes, most standard homeowners policies do cover wildfire damage, but the extent of the coverage can vary depending on the specific policy and the location." [degenerates into repeated `<quote>` tokens] | Fine-Tuned | Corrects the single biggest factual error in the base model's answers — this is the most important fix in the set. |
| 4 | What is the CA FAIR Plan and when would I need it? | Total hallucination — confuses it with unrelated "FAIR data principles" research guidelines. | "The CA FAIR Plan is a voluntary... plan that California homeowners can choose to purchase to reduce the likelihood of a wildfire-related loss..." [degenerates into a repeated off-topic question loop] | Fine-Tuned | Fixes the hallucination entirely and stays in the correct domain, though "reduce the likelihood of a loss" is imprecise (FAIR Plan is a coverage-of-last-resort pool, not a risk-reduction program) — improved but not fully accurate. |
| 5 | How do I start a claim after a burst pipe? | Total generation failure — repeats a leaked system-prompt fragment, zero relevant content. | "You should immediately contact your insurance company or a licensed contractor to report the leak and request a repair estimate... you can also report the leak to your local fire department..." [degenerates into repeated off-topic question loop] | Fine-Tuned | Goes from zero usable content to actual actionable claim-start steps — one of the largest before/after gaps in the set. |
| 6 | What is typically excluded from a standard homeowners policy? | Collapses into 65x repetition of the word "cougar," no usable content. | "...typically excludes coverage for flood damage, fire, and theft, as well as certain types of personal property loss..." [degenerates into repeated garbled tokens] | Fine-Tuned | Produces an actual (if imperfect — listing "fire" as excluded is questionable, fire is normally a covered peril) exclusions list instead of pure noise. |
| 7 | Do I need separate earthquake coverage in California? | Gets "yes" right for the wrong reason ("essential for any traveler"), then degenerates. | "Yes, you need separate earthquake coverage in California. It's a standard requirement for homeowners insurance..." followed by a real explanatory continuation contrasting standard vs. earthquake policies. | Fine-Tuned (partial) | Correct "yes" with on-topic reasoning this time, though "standard requirement" overstates it — earthquake coverage (CEA) is optional, not mandatory. Still a clear improvement over the base's nonsensical justification. |
| 8 | What is personal liability coverage? | Frames it primarily as an auto-insurance feature. | "Personal liability coverage is a supplemental coverage that pays for the cost of a personal injury or property damage claim against a homeowner, even if the homeowner is not at fault..." [degenerates into repeated `}` tokens at the token limit] | Fine-Tuned | Correctly grounds the concept in homeowners coverage rather than auto insurance. |
| 9 | How does a home inventory help with a claim? | Generic — no mention of photos, dates, or faster settlement. | "A home inventory is a record of the contents of your home, including the value of each item, the date it was purchased, and the current condition. This information can be used to estimate the cost of a rebuild, which can help you negotiate a fair settlement." [degenerates into repeated off-topic question loop] | Fine-Tuned | Adds the specific detail (purchase date, condition, negotiating a settlement) the base model's answer was missing. |
| 10 | What factors affect homeowners insurance premiums in California? | Fully coherent but entirely generic (home value, property type, condition) — zero CA-specific content. | "...Premiums are typically higher in coastal areas, rural areas, and older homes... You can find more information about California homeowners insurance in the CA FAIR Plan..." [degenerates into repeated off-topic question loop] | Fine-Tuned | Adds CA-relevant drivers (coastal/rural risk, older homes) and references the FAIR Plan — more domain-grounded than the base's generic list. |

## Summary

Stage 2 is visibly less generic and more domain-specific than the base
model on all 10 questions — the pass bar in spec §8.3 ("Stage 2 must be
visibly less generic and more domain-specific than base on a majority of
the 10 questions") is met with room to spare. Two consistent improvements
stand out:

1. **Factual corrections** — the two worst base-model failures (wildfire
   "not covered," the CA FAIR Plan hallucination) are both fixed.
2. **Instruction-following** — every answer now opens with a direct,
   on-topic response instead of degenerating immediately; the base model's
   near-total generation collapse (7/10 answers) is gone from the *opening*
   of each answer.

One limitation carries over from the base model and is **not** fully
resolved by Stage 2: every answer still degenerates into repetition or
off-topic loops after roughly 2-4 sentences, well before the 150-token
limit. Stage 2's SFT data did not include enough signal about *when to
stop* — this is a real remaining gap, not glossed over here, and is a
natural target for Stage 3 (DPO) to improve on. See
`reports/final_evaluation.md` for whether DPO helps.
