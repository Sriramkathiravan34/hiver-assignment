"""
Intent classification.

Two models, both trained/evaluated the same way, so the eval harness can
compare them honestly:
  - TRIVIAL baseline: always predict the majority class.
  - SIMPLE baseline: TF-IDF + Logistic Regression (the one we ship).

We deliberately do NOT call an LLM for classification (see decision_log.md #4):
an 11-way intent tag is a cheap, high-volume, low-ambiguity decision -- a
linear classifier on TF-IDF gets ~90%+ of the way there at near-zero cost and
near-zero latency, and is auditable (you can inspect the learned weights).
The LLM budget is spent where it actually earns its keep: reply drafting.
"""
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

INTENT_LABELS = [
    "order_status", "delivery_issue", "refund_request", "return_request",
    "account_access", "billing_issue", "product_defect", "cancellation",
    "complaint_escalation", "general_inquiry", "positive_feedback",
]

MODEL_PATH = "src/_intent_model.joblib"
TRIVIAL_MODEL_PATH = "src/_intent_trivial.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(train_df: pd.DataFrame, text_col="customer_text_clean", label_col="intent"):
    pipe = build_pipeline()
    pipe.fit(train_df[text_col], train_df[label_col])
    joblib.dump(pipe, MODEL_PATH)

    trivial = DummyClassifier(strategy="most_frequent")
    trivial.fit(train_df[text_col], train_df[label_col])
    joblib.dump(trivial, TRIVIAL_MODEL_PATH)
    return pipe, trivial


def predict(texts, model_path=MODEL_PATH):
    pipe = joblib.load(model_path)
    preds = pipe.predict(texts)
    # confidence = max predicted probability
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(texts).max(axis=1)
    else:
        probs = [1.0] * len(texts)
    return list(zip(preds, probs))


if __name__ == "__main__":
    df = pd.read_csv("data/train_pairs.csv")
    train(df)
    preds = predict(df["customer_text_clean"].head(5).tolist())
    for text, (label, conf) in zip(df["customer_text_clean"].head(5), preds):
        print(f"[{label} ({conf:.2f})] {text}")
