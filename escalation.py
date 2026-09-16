"""
Decide: auto-handle or escalate to a human, with a stated reason.

Escalation is a SAFETY decision, not a quality decision -- it should be
conservative and explainable, so it's rule-based on top of signals we
already have (intent, classifier confidence, retrieval similarity, and a
small set of hard triggers), rather than another LLM call whose own
uncertainty we'd then have to police (decision_log.md #8).
"""
import re

HARD_TRIGGER_PATTERNS = [
    (r"\b(lawyer|legal action|sue|attorney)\b", "legal threat mentioned"),
    (r"\bbbb\b|better business bureau", "regulatory complaint (BBB) mentioned"),
    (r"\b(kill myself|suicide|self harm)\b", "self-harm language detected"),
    (r"\bfraud\b|unauthorized charge|stolen card|identity theft", "possible fraud/security issue"),
]

# Intents that are inherently higher-stakes even without a hard trigger.
HIGH_STAKES_INTENTS = {"complaint_escalation", "account_access"}

CONF_THRESHOLD = 0.55       # below this, classifier itself is unsure
RETRIEVAL_SIM_THRESHOLD = 0.12  # below this, we have weak precedent to ground on


def decide(customer_text: str, intent: str, intent_confidence: float,
           top_retrieval_similarity: float) -> dict:
    text_lower = customer_text.lower()

    for pattern, reason in HARD_TRIGGER_PATTERNS:
        if re.search(pattern, text_lower):
            return {"action": "escalate", "reason": reason, "rule": "hard_trigger"}

    if intent in HIGH_STAKES_INTENTS:
        return {
            "action": "escalate",
            "reason": f"intent '{intent}' is high-stakes by policy (angry/account-security cases go to a human)",
            "rule": "high_stakes_intent",
        }

    if intent_confidence < CONF_THRESHOLD:
        return {
            "action": "escalate",
            "reason": f"intent classifier confidence too low ({intent_confidence:.2f} < {CONF_THRESHOLD})",
            "rule": "low_classifier_confidence",
        }

    if top_retrieval_similarity < RETRIEVAL_SIM_THRESHOLD:
        return {
            "action": "escalate",
            "reason": f"no close historical precedent found (top similarity {top_retrieval_similarity:.2f} < {RETRIEVAL_SIM_THRESHOLD})",
            "rule": "weak_grounding",
        }

    return {
        "action": "auto_handle",
        "reason": "confident intent, non-high-stakes, strong precedent to ground the reply",
        "rule": "default_auto",
    }


if __name__ == "__main__":
    print(decide("I want a refund NOW or I'm calling my lawyer", "refund_request", 0.9, 0.3))
    print(decide("where is my order", "order_status", 0.8, 0.4))
    print(decide("where is my order", "order_status", 0.3, 0.4))
