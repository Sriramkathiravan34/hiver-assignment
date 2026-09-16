"""
Builds the golden evaluation set: a held-out, hand-checked sample of
(customer_text, true_intent, reference_resolution) that NEVER touches
training. See report.md "Problem framing" for sampling rationale, and
decision_log.md #5 for why we stratify by intent instead of sampling
uniformly at random.

On the real 3M-row dataset this script's job is identical: stratified
sample -> a human (you) reads and labels/corrects each one -> golden_eval.csv.
Here, because the source data is itself synthetically generated with known
labels, "labelling" collapses to sampling + a light manual sanity pass,
which is disclosed in report.md as a real limitation of this demo run.
"""
import pandas as pd
from ingest import load_brand_pairs

N_PER_INTENT = 4  # -> ~44 examples for this demo (176 threads total).
# On the real 3M-tweet dataset this constant should be raised so that
# N_PER_INTENT * num_intents lands in the required 150-250 range.

pairs = load_brand_pairs("/home/claude/hiver-agent/data/sample_twcs.csv")
lookup = pd.read_csv("/home/claude/hiver-agent/data/_gold_intents_lookup.csv", dtype=str)
df = pairs.merge(lookup, left_on="thread_id", right_on="tweet_id")

golden = pd.concat(
    [g.sample(min(N_PER_INTENT, len(g)), random_state=42) for _, g in df.groupby("intent")]
).reset_index(drop=True)

golden_out = golden[["thread_id", "customer_text", "customer_text_clean",
                      "intent", "brand_response_clean"]].rename(
    columns={"brand_response_clean": "reference_resolution", "intent": "true_intent"}
)
golden_out.to_csv("/home/claude/hiver-agent/eval/golden_eval.csv", index=False)

# Remaining data becomes the training set for the classifier -- strict
# separation from golden_eval to avoid leakage (decision_log.md #6).
train_df = df[~df["thread_id"].isin(golden_out["thread_id"])].reset_index(drop=True)
train_df.to_csv("/home/claude/hiver-agent/data/train_pairs.csv", index=False)

print(f"golden_eval.csv: {len(golden_out)} rows across {golden_out['true_intent'].nunique()} intents")
print(f"train_pairs.csv: {len(train_df)} rows")
print(golden_out["true_intent"].value_counts())
