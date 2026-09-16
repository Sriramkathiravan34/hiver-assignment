"""
Ingest + clean the raw Twitter CS dataset for a single brand.

Input schema (Kaggle thoughtvector/customer-support-on-twitter):
    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

Output: a DataFrame of (customer_message, brand_response) pairs -- the unit
every downstream module operates on.

Swap REAL_DATA_PATH to the actual Kaggle CSV once you have internet access;
nothing else in this file needs to change (see decision_log.md #2 for why
we deliberately don't hardcode any synthetic-data assumptions here).
"""
import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = URL_RE.sub("", t)
    t = t.replace("@AmazonHelp", "").replace("@amazonhelp", "")
    t = WS_RE.sub(" ", t).strip()
    return t


def load_brand_pairs(csv_path: str, brand_handle: str = "AmazonHelp") -> pd.DataFrame:
    """
    Returns one row per resolved (customer -> brand) exchange for `brand_handle`:
        thread_id, customer_text, customer_text_clean, brand_response, brand_response_clean
    """
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "response_tweet_id": str,
                                       "in_response_to_tweet_id": str})
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])

    by_id = df.set_index("tweet_id")

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand_handle)]

    pairs = []
    for _, resp_row in brand_replies.iterrows():
        parent_id = resp_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "" or parent_id not in by_id.index:
            continue
        cust_row = by_id.loc[parent_id]
        if not bool(cust_row["inbound"]):
            continue  # only keep genuine customer->brand replies

        pairs.append({
            "thread_id": parent_id,
            "customer_text": cust_row["text"],
            "customer_text_clean": clean_text(cust_row["text"]),
            "brand_response": resp_row["text"],
            "brand_response_clean": clean_text(resp_row["text"]),
        })

    out = pd.DataFrame(pairs).drop_duplicates(subset=["thread_id"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    pairs = load_brand_pairs("/home/claude/hiver-agent/data/sample_twcs.csv")
    print(f"Loaded {len(pairs)} customer->brand pairs")
    print(pairs.head(3).to_string())
