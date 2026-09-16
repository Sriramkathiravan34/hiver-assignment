"""
Retrieval grounding: given a new customer message, find the most similar
*historically resolved* threads for this brand so the reply generator has
real precedent to ground on instead of hallucinating a policy.

TF-IDF cosine similarity, not embeddings (decision_log.md #7): at this
data scale, and given no network access to download an embedding model in
this environment, TF-IDF is deterministic, needs no downloads, and is easy
to audit ("why did it retrieve this example?" has a legible answer -- shared
n-grams). Swap in a real embedding model when you have GPU/internet.
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "src/_retrieval_index.joblib"


def build_index(train_df: pd.DataFrame, text_col="customer_text_clean"):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vec.fit_transform(train_df[text_col])
    joblib.dump((vec, matrix, train_df.reset_index(drop=True)), INDEX_PATH)
    return vec, matrix, train_df


def retrieve(query_text: str, k: int = 3, index_path: str = INDEX_PATH):
    vec, matrix, train_df = joblib.load(index_path)
    q = vec.transform([query_text])
    sims = cosine_similarity(q, matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({
            "similarity": float(sims[i]),
            "customer_text": train_df.iloc[i]["customer_text_clean"],
            "resolution": train_df.iloc[i]["brand_response_clean"],
        })
    return results


if __name__ == "__main__":
    train_df = pd.read_csv("data/train_pairs.csv")
    build_index(train_df)
    hits = retrieve("my package never showed up and tracking is stuck", k=3)
    for h in hits:
        print(f"[{h['similarity']:.2f}] {h['customer_text']} -> {h['resolution']}")
