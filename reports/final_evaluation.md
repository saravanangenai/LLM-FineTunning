# Final Evaluation — Base vs. SFT (Stage 2) vs. DPO (Stage 3)

Spec: `specs/002-unsloth-ca-homeowners-finetune.md` §4, §8.2 (Step 10 deliverable).

## Method

- Base and Stage 2 (SFT) answers: reused from `reports/base_model_evaluation.md`
  and `reports/sft_model_comparison.md` (same weights, same decoding
  settings).
- Stage 3 (DPO) answers: the final merged model,
  `sharanmini/qwen2.5-0.5b-ca-homeowners-final`, loaded directly via
  `transformers` (already a full 16-bit merged checkpoint).
- Decoding: greedy (`do_sample=False`), `max_new_tokens=150`, no repetition
  penalty, chat-templated with `SYSTEM_PROMPT` — identical settings across
  all three models for a fair three-way comparison.
- Run locally on CPU. Full raw output in
  `reports/dpo_model_answers_raw.json` (Stage 3) and
  `reports/sft_model_answers_raw.json` (Stage 2); long repeated tails are
  truncated below with `[...]` for readability.
- Note on greedy decoding: since DPO nudges token *probabilities* rather
  than teaching new content, several answers below are identical or
  near-identical between SFT and DPO under greedy/argmax decoding — the
  preference shift is real but subtle enough that it doesn't always change
  which token wins at each step. Where DPO's output differs from SFT's,
  that's captured explicitly in the Reason column below.

## Results

| # | Question | Base | SFT (Stage 2) | DPO (Stage 3) | Best Answer | Reason |
|---|---|---|---|---|---|---|
| 1 | What does dwelling coverage mean in a homeowners policy? | Vague/circular restatement. | "Dwelling coverage is a special type of coverage that pays for the cost of repairing or rebuilding a dwelling..." | Identical to SFT. | Tie (SFT = DPO), both ≫ Base | DPO made no change here; both fine-tuned stages are a clear improvement over base's circularity. |
| 2 | What's the difference between actual cash value and replacement cost? | Blurs both terms, no depreciation concept, degenerates mid-sentence. | "The actual cash value is the replacement cost of the structure minus the cost of rebuilding it... " (still conceptually muddled). | Identical to SFT. | Tie (SFT = DPO) | Neither fine-tuned stage fixes the core conceptual gap (depreciation is never mentioned) — a genuine remaining weakness in both. |
| 3 | Does a standard homeowners policy cover wildfire damage in California? | Factually backwards — claims it's not covered. | "Yes, most standard homeowners policies do cover wildfire damage..." | Identical to SFT. | Tie (SFT = DPO), both ≫ Base | The critical factual fix happened at SFT; DPO preserves it without regression. |
| 4 | What is the CA FAIR Plan and when would I need it? | Total hallucination (confuses with "FAIR data principles"). | "...can choose to purchase to reduce the likelihood of a wildfire-related loss... through a California homeowners association." | Same opening, but continuation adds a direct FAIR-Plan-vs-standard-policy comparison and says "through a California FAIR Plan broker" (more plausible than SFT's "homeowners association"). | DPO | Small but real wording correction plus more useful bonus content in the continuation. |
| 5 | How do I start a claim after a burst pipe? | Total generation failure, no content. | "...you can also report the leak to your local fire department, which may be able to help with the repair." | "...Keep a log of the repair, including the date, time, and estimated cost, so you can submit a claim later." | DPO | SFT's fire-department suggestion is off-base for a burst pipe; DPO's documentation advice (date/time/cost log) is standard, correct claims guidance — a genuine quality improvement from preference tuning. |
| 6 | What is typically excluded from a standard homeowners policy? | Degenerates into 65x "cougar," no content. | "...excludes coverage for flood damage, fire, and theft, as well as certain types of personal property loss." | "...excludes coverage for flood damage, fire, and theft, as well as certain natural disasters like hurricanes, tornadoes, and earthquakes. It may also exclude coverage for personal property damage..." | DPO | More complete exclusions list. (Both stages share the same inaccuracy of listing "fire" as excluded — fire is normally a covered peril on a standard HO-3 — so this is an improvement in completeness, not a full correctness fix.) |
| 7 | Do I need separate earthquake coverage in California? | Correct "yes" but nonsensical justification ("essential for any traveler"). | "Yes... It's a standard requirement..." continuation cuts off mid-sentence ("Standard earthquake policies, on the"). | Same opening, continuation completes the thought: "...Standard earthquake policies... are specifically designed to cover earthquakes, including the possibility of a major, widespread, and destructive earthquake." | DPO | More coherent, complete explanatory content. (Both stages still overstate earthquake coverage as a "standard requirement" rather than an optional CEA add-on — a shared correctness gap.) |
| 8 | What is personal liability coverage? | Frames it as primarily an auto-insurance feature. | "Personal liability coverage is a supplemental coverage that pays for the cost of a personal injury or property damage claim against a homeowner..." | Identical to SFT. | Tie (SFT = DPO), both ≫ Base | Correct homeowners-context framing established at SFT, preserved by DPO. |
| 9 | How does a home inventory help with a claim? | Generic, no specifics. | "A home inventory is a record of the contents of your home, including the value of each item, the date it was purchased, and the current condition..." | Identical to SFT. | Tie (SFT = DPO), both ≫ Base | No change from DPO; SFT's specificity gain over base is preserved. |
| 10 | What factors affect homeowners insurance premiums in California? | Generic, zero CA-specific detail. | "...Premiums are typically higher in coastal areas, rural areas, and older homes... You can find more information... in the CA FAIR Plan..." | "...California homeowners insurance rates are generally higher than those in other states, and homeowners should review their local rates and policies..." (drops the coastal/rural/older-home detail and the FAIR Plan mention). | SFT | The one case where DPO is slightly *less* domain-specific than SFT — a minor regression in specificity, though not a safety or correctness issue. |

## Summary

Judged against the spec §8.3 pass bar — "Stage 3 must be judged
equal-or-better than Stage 2 on a majority of the 10 questions, no
regression in safety or correctness even if tone/preference differs" — DPO
passes clearly: strictly better on 4/10 (Q4, Q5, Q6, Q7), tied on 5/10
(Q1, Q2, Q3, Q8, Q9), and only marginally behind on 1/10 (Q10, a
specificity nit, not a safety/correctness issue). No question shows DPO
introducing a new factual error or unsafe claim relative to SFT.

The clearest DPO win is **Q5** (burst-pipe claim): it replaces an
off-topic suggestion (contacting the fire department) with concrete,
correct claims-documentation advice (logging date/time/cost) — exactly the
kind of "more helpful, more complete" preference the Stage 3 dataset
(`data/3-preference_dataset.jsonl`) was designed to reinforce.

**Carried-over limitations, present in both SFT and DPO** (not fixed by
either fine-tuning stage, worth calling out honestly rather than
overselling the result):
1. Every answer still degenerates into repetition/off-topic loops after
   roughly 2-4 sentences — the models never learned reliable stopping
   behavior. The *initial* answer is what should be read; the tail is
   noise.
2. A few specific factual imprecisions persist unchanged from SFT into
   DPO: ACV vs. replacement cost never mentions depreciation (Q2), "fire"
   is incorrectly listed among common exclusions (Q6), and earthquake
   coverage is overstated as a "standard requirement" rather than an
   optional CEA add-on (Q7). DPO's preference data improved *helpfulness
   and completeness* on several questions but did not correct these
   specific residual inaccuracies — a good target for a future preference
   dataset iteration.

**Best answer overall per question**: DPO ties or wins on 9/10 questions
and is the recommended final model, consistent with it being the last
stage in the pipeline.
