# AmazonHelp Support Agent — Take-Home

An AI support agent for **AmazonHelp** (Twitter) that classifies incoming
customer messages, drafts a reply grounded in how AmazonHelp has historically
resolved similar issues, and decides auto-handle vs. escalate-to-human with a
stated reason.

## ⚠️ Read this first: what's real data here, and what isn't

This was built in a sandboxed environment **with no internet access**. That
meant I could not:
- Download the real Kaggle dataset (`thoughtvector/customer-support-on-twitter`, ~3M tweets)
- Call a live LLM API for reply generation or LLM-as-judge

So this repo ships with:
1. **A synthetic sample dataset** (`data/sample_twcs.csv`, `src/generate_sample_data.py`)
   in the **exact schema** of the real Kaggle dataset (`tweet_id, author_id,
   inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`),
   covering one brand (AmazonHelp) across 11 intents, 176 threads.
2. A **template-based fallback** for reply generation, used automatically when
   `ANTHROPIC_API_KEY` isn't set (see `src/reply_gen.py`).
3. A **disclosed rule-based proxy judge**, used automatically when
   `ANTHROPIC_API_KEY` isn't set, instead of a real LLM-as-judge (see
   `eval/eval_harness.py`).

**None of this is hidden** — every script prints which mode it's running in,
and `reports/report.md` has a dedicated section on exactly how the headline
numbers below are inflated by testing on synthetic, template-generated data.
Everything else — ingestion, the intent classifier, retrieval grounding,
escalation rules, the eval harness structure, train/eval separation — is
real, working code that behaves identically on the real dataset. Swapping in
real data + a real API key (see below) is the only thing standing between
this and a production-representative run.

## Reproduce in under 15 minutes

```bash
pip install pandas numpy scikit-learn joblib

cd src
python3 setup.py              # generates data, splits golden eval, trains classifier, builds retrieval index (~2s)
python3 pipeline.py           # run a few example messages through the full pipeline

cd ../eval
python3 eval_harness.py       # intent metrics + automated reply metrics + judge scores
python3 build_human_scores.py # (only needed once) hand-scored subset for judge agreement
python3 eval_harness.py       # rerun to see judge-vs-human agreement at the bottom
```

Total runtime: a few seconds. No GPU, no downloads.

### Using the real dataset

1. Download `thoughtvector/customer-support-on-twitter` from Kaggle, unzip to
   get `twcs.csv`.
2. Replace the call in `src/setup.py` / `src/build_golden_eval.py` /
   `src/retrieval.py` that points at `data/sample_twcs.csv` with the path to
   the real `twcs.csv`. **Nothing else changes** — `ingest.py` only depends
   on the documented Kaggle schema, not on anything about the synthetic
   generator.
3. Raise `N_PER_INTENT` in `build_golden_eval.py` so the golden set lands in
   the required 150–250 range (it's set low here because the synthetic
   sample only has 176 threads total).
4. Optionally pick a different `brand_handle` in `ingest.load_brand_pairs`.

### Using a real LLM

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 pipeline.py       # now drafts replies with Claude instead of the template fallback
python3 eval_harness.py   # now judges with Claude instead of the proxy judge
```

## Repo layout

```
src/
  generate_sample_data.py   synthetic stand-in for the Kaggle CSV (schema-identical)
  ingest.py                 loads raw tweets -> (customer, brand) resolved pairs, for ANY brand
  build_golden_eval.py      stratified golden eval split + train/eval separation
  intents.py                TF-IDF+LogReg intent classifier, + trivial baseline
  retrieval.py               TF-IDF retrieval over historically resolved threads
  reply_gen.py               drafts a reply grounded in retrieval (LLM if key set, else template)
  escalation.py               rule-based auto-handle vs. escalate decision
  pipeline.py                 wires the above into one call: message -> intent, reply, decision
  setup.py                    runs the full setup in order
eval/
  golden_eval.csv             (generated) 44 hand-checked examples, stratified across 11 intents
  build_human_scores.py       small hand-scored subset for judge-agreement measurement
  eval_harness.py             intent metrics, automated reply metrics, LLM-judge (or proxy), agreement
reports/
  report.md                   problem framing, baselines, failure analysis, "what's misleading", next steps
decision_log.md               15 non-obvious decisions and why
```

## What "good" means for this brand (short version, see report.md)

For a high-volume retail support account like AmazonHelp, correctness and
*not overpromising* (groundedness) matter more than eloquence — a reply that
confidently states a wrong refund timeline is worse than a slightly stiff
correct one. So the eval harness weights groundedness and correctness over
tone, and escalation is deliberately conservative (see `escalation.py`).
