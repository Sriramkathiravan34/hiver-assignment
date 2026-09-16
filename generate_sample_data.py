"""
Generates a synthetic sample in the EXACT schema of the Kaggle
"Customer Support on Twitter" dataset (thoughtvector/customer-support-on-twitter):

    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

This stands in for the real dataset because this environment has no internet
access to download it. Swap this file's output for the real CSV (see README)
and every downstream script works unchanged -- ingest.py only depends on the
schema above, not on this generator.

Brand chosen: AmazonHelp (single brand, as required by the assignment).
"""
import csv
import random

random.seed(7)

BRAND = "AmazonHelp"

# Intent taxonomy (defined from looking at the kinds of threads Amazon's
# support account actually handles on Twitter -- see decision_log.md #1).
INTENTS = {
    "order_status": {
        "customer": [
            "Hi {brand}, where is my order #{oid}? It was supposed to arrive yesterday.",
            "@{brand} any update on order {oid}? Tracking hasn't moved in 3 days.",
            "Can someone tell me the status of {oid}? Nothing on the tracking page.",
            "@{brand} my order {oid} still says 'preparing for shipment' after a week.",
        ],
        "response": [
            "Hi, sorry for the wait! I can see order {oid} is out for delivery and should arrive by end of day. DM us if it doesn't show up.",
            "Thanks for flagging this. Order {oid} is currently with the carrier -- tracking updates can lag up to 24h. We'll monitor it for you.",
        ],
    },
    "delivery_issue": {
        "customer": [
            "@{brand} my package {oid} arrived completely smashed, box was soaking wet too.",
            "Order {oid} says delivered but there's nothing on my porch. Ring camera shows no one came.",
            "@{brand} courier left my {oid} package in the rain, everything inside is ruined.",
        ],
        "response": [
            "I'm so sorry about that! I've started a replacement for order {oid} at no cost, it'll ship today. Please send a photo via DM so we can also file it with the carrier.",
            "That's not okay, thank you for letting us know. I've opened an investigation for {oid} with the carrier and issued a reshipment.",
        ],
    },
    "refund_request": {
        "customer": [
            "@{brand} I returned item {oid} two weeks ago and still no refund. This is ridiculous.",
            "Where's my refund for order {oid}? Your app says 'processed' but nothing in my bank account.",
            "@{brand} requesting a refund for {oid}, item never arrived and it's been 3 weeks.",
        ],
        "response": [
            "Apologies for the delay -- I can confirm the return for {oid} was received and I've pushed the refund manually. It should reflect in 3-5 business days.",
            "I've checked and your refund for {oid} was issued but may be sitting with your bank. If it's not there in 5 business days, DM us your order details and we'll escalate.",
        ],
    },
    "return_request": {
        "customer": [
            "@{brand} how do I return item {oid}, wrong size arrived.",
            "Need to send back {oid}, it's not what I ordered. What's the process?",
        ],
        "response": [
            "No problem! Go to Your Orders > {oid} > Return or Replace Items, pick 'Wrong item' and you'll get a free prepaid label.",
            "Sorry about that mix-up. You can start a free return for {oid} from Your Orders -- happy to send you the direct link if it's not showing.",
        ],
    },
    "account_access": {
        "customer": [
            "@{brand} I can't log into my account, keeps saying 'incorrect password' even after reset.",
            "Locked out of my account for two days now, reset emails aren't arriving. Please help.",
        ],
        "response": [
            "Sorry for the trouble! Please check your spam folder for the reset email, and make sure you're using the email tied to the account. DM us if it still fails and we'll escalate to account security.",
            "That sounds frustrating. Can you DM us the email on file (no passwords please) so we can look into the reset delivery issue?",
        ],
    },
    "billing_issue": {
        "customer": [
            "@{brand} I was charged twice for order {oid}, please fix this.",
            "Why does my card show a charge for {oid} that's way more than what I was quoted at checkout?",
        ],
        "response": [
            "Sorry about that! I can see the duplicate charge on {oid} -- I've submitted a refund for the extra charge, it'll drop off in 3-5 business days.",
            "Thanks for the heads up, that shouldn't happen. I've flagged {oid} to billing and a corrected charge/refund is being processed now.",
        ],
    },
    "product_defect": {
        "customer": [
            "@{brand} item {oid} arrived broken, doesn't even turn on.",
            "Received {oid} today and it's clearly defective, won't power on at all.",
        ],
        "response": [
            "So sorry to hear that! I've set up a free replacement for {oid}, no need to wait for the return -- it ships today.",
            "That's disappointing, apologies. I've issued a replacement for {oid} and a prepaid label to send the defective one back whenever convenient.",
        ],
    },
    "cancellation": {
        "customer": [
            "@{brand} need to cancel order {oid} ASAP, hasn't shipped yet right?",
            "Please cancel {oid}, ordered by mistake.",
        ],
        "response": [
            "Got it! I've submitted a cancellation for {oid} -- if it hasn't entered the shipping process it'll be stopped automatically and refunded in full.",
            "Cancelled! Order {oid} hadn't shipped yet so it's been stopped and you won't be charged.",
        ],
    },
    "complaint_escalation": {
        "customer": [
            "@{brand} this is the THIRD time I've been ignored about order {oid}. Absolutely done, filing a BBB complaint.",
            "@{brand} worst customer service ever, nobody has responded about {oid} in 5 days. Lawyer is next.",
            "@{brand} I want a manager. {oid} has been a disaster and every rep says something different.",
        ],
        "response": [
            "I completely understand your frustration and I'm sorry we've let you down on {oid}. I'm escalating this to a senior specialist right now who will DM you directly within the hour.",
        ],
    },
    "general_inquiry": {
        "customer": [
            "@{brand} does the warranty on {oid} cover water damage?",
            "Quick question -- do you ship {oid}-type items internationally?",
        ],
        "response": [
            "Great question! Standard warranty on {oid} covers manufacturing defects, not water damage, but check if you added Extended Protection at checkout.",
            "Yes, most items ship internationally with some exceptions -- happy to check {oid} specifically if you share the product link.",
        ],
    },
    "positive_feedback": {
        "customer": [
            "@{brand} just wanted to say your team fixed my {oid} issue super fast, thank you!",
            "Shoutout to @{brand} support for sorting out {oid} in like 10 minutes, appreciate it!",
        ],
        "response": [
            "That means a lot, thank you for the kind words! Glad we could sort out {oid} quickly.",
            "So happy to hear that about {oid}! We'll pass this along to the team.",
        ],
    },
}

N_THREADS_PER_INTENT = 16  # ~16*10 = 160 threads -> ~320 tweets total

rows = []
tid = 1000000


def next_id():
    global tid
    tid += 1
    return tid


start = 0
for intent, tpl in INTENTS.items():
    for i in range(N_THREADS_PER_INTENT):
        oid = f"{random.randint(100000, 999999)}-{random.randint(1000000,9999999)}"
        cust_text = random.choice(tpl["customer"]).format(brand=BRAND, oid=oid)
        resp_text = random.choice(tpl["response"]).format(brand=BRAND, oid=oid)

        cust_id = next_id()
        resp_id = next_id()
        cust_author = f"cust_{random.randint(10000,99999)}"

        rows.append({
            "tweet_id": cust_id,
            "author_id": cust_author,
            "inbound": True,
            "created_at": "Mon Jan 01 00:00:00 +0000 2024",
            "text": cust_text,
            "response_tweet_id": resp_id,
            "in_response_to_tweet_id": "",
            "_intent_gold": intent,  # extra column ONLY for building golden eval / sanity -- not part of real schema
        })
        rows.append({
            "tweet_id": resp_id,
            "author_id": BRAND,
            "inbound": False,
            "created_at": "Mon Jan 01 00:05:00 +0000 2024",
            "text": resp_text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": cust_id,
            "_intent_gold": "",
        })

random.shuffle(rows)

fields = ["tweet_id", "author_id", "inbound", "created_at", "text",
          "response_tweet_id", "in_response_to_tweet_id"]

with open("/home/claude/hiver-agent/data/sample_twcs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in fields})

# Keep the intent labels separately -- this is what a human labeler would
# produce when building the golden eval set (see build_golden_eval.py).
with open("/home/claude/hiver-agent/data/_gold_intents_lookup.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["tweet_id", "intent"])
    for r in rows:
        if r["inbound"]:
            w.writerow([r["tweet_id"], r["_intent_gold"]])

print(f"Wrote {len(rows)} tweets ({len(rows)//2} threads) for brand={BRAND}")
