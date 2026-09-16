"""
Draft a reply grounded in retrieved historical resolutions.

Two modes:
  - LLM mode (real deployment): if ANTHROPIC_API_KEY is set, calls Claude
    with the retrieved precedent as context and asks it to draft a reply
    that follows the brand's actual resolution pattern. This is the
    intended production path.
  - Fallback mode (this sandbox, no network): a deterministic template
    that adapts the single closest retrieved resolution to the new
    order/case id. This is clearly weaker (no real language generation)
    but keeps the pipeline runnable end-to-end without internet, and
    isolates "does retrieval find the right precedent" from "can an LLM
    write a good reply given that precedent" -- see report.md failure
    analysis, which is run against the fallback and says explicitly which
    failures would disappear under the LLM path.
"""
import os
import re
import json

from retrieval import retrieve

SYSTEM_PROMPT = """You are a customer support agent for AmazonHelp on Twitter.
You will be given a new customer message and 1-3 examples of how this brand
has resolved similar issues before. Draft a reply that:
- Matches the brand's tone (warm, apologetic where appropriate, concrete next step)
- Is grounded in the precedent shown -- do not invent policies not reflected in the examples
- Is under 280 characters
- Does not promise anything the precedent doesn't support
Respond with ONLY the reply text, nothing else."""


def _call_claude(customer_text: str, precedents: list) -> str:
    import urllib.request

    precedent_str = "\n".join(
        f"- Similar case: \"{p['customer_text']}\" -> Resolution: \"{p['resolution']}\""
        for p in precedents
    )
    user_msg = f"New customer message: \"{customer_text}\"\n\nPrecedent:\n{precedent_str}"

    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return "".join(b["text"] for b in data["content"] if b["type"] == "text").strip()


def _fallback_template(customer_text: str, precedents: list) -> str:
    """No-network fallback: adapt the closest resolution's case-id to the new message."""
    if not precedents:
        return "Thanks for reaching out -- could you share your order number so we can look into this?"
    best = precedents[0]
    ids_in_query = re.findall(r"\d{5,6}-\d{6,7}", customer_text)
    reply = best["resolution"]
    old_ids = re.findall(r"\d{5,6}-\d{6,7}", reply)
    if ids_in_query and old_ids:
        reply = reply.replace(old_ids[0], ids_in_query[0])
    return reply


def draft_reply(customer_text: str, k: int = 3) -> dict:
    precedents = retrieve(customer_text, k=k)
    used_llm = "ANTHROPIC_API_KEY" in os.environ
    try:
        if used_llm:
            reply = _call_claude(customer_text, precedents)
        else:
            reply = _fallback_template(customer_text, precedents)
    except Exception as e:
        used_llm = False
        reply = _fallback_template(customer_text, precedents)
    return {
        "reply": reply,
        "grounded_on": precedents,
        "generation_mode": "llm" if used_llm else "template_fallback",
    }


if __name__ == "__main__":
    out = draft_reply("my order still hasn't shipped and it's been 6 days, can I get an update")
    print(f"[{out['generation_mode']}] {out['reply']}")
