"""Shared system prompt and fixed evaluation question set used across all three stages."""

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers general questions about California "
    "homeowners insurance, including HO-3 policies, the CA FAIR Plan, and related "
    "topics. Answers are for general educational purposes only, not licensed "
    "insurance advice or a binding coverage determination."
)

# Spec 002 §4 — the fixed 10-question set used for the base/SFT/DPO comparison
# reports in reports/*.md. Keep this list identical across all three stages so
# the before/after tables stay directly comparable.
EVAL_QUESTIONS: list[str] = [
    "What does dwelling coverage mean in a homeowners policy?",
    "What's the difference between actual cash value and replacement cost?",
    "Does a standard homeowners policy cover wildfire damage in California?",
    "What is the CA FAIR Plan and when would I need it?",
    "How do I start a claim after a burst pipe?",
    "What is typically excluded from a standard homeowners policy?",
    "Do I need separate earthquake coverage in California?",
    "What is personal liability coverage?",
    "How does a home inventory help with a claim?",
    "What factors affect homeowners insurance premiums in California?",
]
