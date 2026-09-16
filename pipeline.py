"""
End-to-end pipeline: raw customer message -> (intent, reply, escalation decision).
This is what a real inbound webhook would call per message.
"""
from intents import predict as predict_intent
from retrieval import retrieve
from reply_gen import draft_reply
from escalation import decide
from ingest import clean_text


def run(customer_text: str) -> dict:
    clean = clean_text(customer_text)

    (intent, confidence), = predict_intent([clean])

    precedents = retrieve(clean, k=3)
    top_sim = precedents[0]["similarity"] if precedents else 0.0

    reply_out = draft_reply(clean, k=3)

    decision = decide(clean, intent, confidence, top_sim)

    return {
        "input": customer_text,
        "intent": intent,
        "intent_confidence": round(float(confidence), 3),
        "reply": reply_out["reply"],
        "reply_generation_mode": reply_out["generation_mode"],
        "grounded_on": precedents,
        "decision": decision,
    }


if __name__ == "__main__":
    import json
    examples = [
        "where the heck is my package, it's been 2 weeks and no update",
        "third time asking, refund me now or I'm contacting my lawyer",
        "just wanted to say thanks, you guys fixed my issue super fast!",
        "can't log in, tried resetting password 3 times, nothing works",
    ]
    for ex in examples:
        out = run(ex)
        print(json.dumps({k: v for k, v in out.items() if k != "grounded_on"}, indent=2))
        print("-" * 60)
