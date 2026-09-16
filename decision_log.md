# Decision Log

Non-obvious decisions made while building this, and why. (See inline comments in the
referenced files for the full reasoning — this is the condensed version.)

1. **Defined 11 intents from scratch by reading what an Amazon-style support account
   actually handles**, rather than reusing Banking77's 77 labels. Banking77 is
   banking-specific (card issues, exchange rates) and doesn't map cleanly onto retail
   support; a smaller, brand-appropriate taxonomy is more useful for the escalation
   logic downstream, which needs a handful of clearly high-stakes buckets, not 77
   fine-grained ones. (`src/generate_sample_data.py`)

2. **`ingest.py` only depends on the documented Kaggle schema**, never on anything
   specific to how the synthetic sample was generated. This was deliberate so the exact
   same ingestion code works unchanged the moment someone points it at the real
   `twcs.csv` — the only thing that should ever need to change is the file path.

3. **Picked one brand (AmazonHelp) and hardcoded nothing brand-specific into `ingest.py`**
   — `load_brand_pairs(csv_path, brand_handle)` takes the handle as a parameter, so
   testing on a second brand from the real dataset is a one-line change, even though the
   assignment only requires one.

4. **Used TF-IDF + Logistic Regression for intent classification instead of an LLM call.**
   An 11-way tag is cheap, high-volume, and low-ambiguity — a linear model gets there at
   near-zero latency/cost and is auditable (you can literally inspect which n-grams drive
   each class). Reserving the LLM budget for reply generation, where it actually earns
   its cost, seemed like the right call for a high-volume support account.

5. **Stratified the golden eval sample by intent (`N_PER_INTENT` per class) instead of
   uniform random sampling.** Random sampling from an unbalanced real dataset would
   under-represent rare-but-important intents like `complaint_escalation` in the eval
   set, which is exactly the intent the escalation logic most needs to be right about.

6. **Strictly separated golden eval from training data** (`build_golden_eval.py` writes
   `train_pairs.csv` explicitly excluding every golden thread_id) rather than doing a
   generic train/test split after the fact — this makes leakage structurally impossible
   rather than something to remember to check.

7. **Used TF-IDF cosine similarity for retrieval instead of embeddings.** No
   internet access in this environment to download an embedding model, but even setting
   that aside: TF-IDF retrieval decisions are legible ("retrieved because these n-grams
   overlap"), which matters for debugging failure mode #2/#3 in the report (wrong
   precedent retrieved). Would revisit with real data volume, where embeddings' better
   semantic matching would likely pay for the loss of interpretability.

8. **Made escalation rule-based rather than another LLM call.** Escalation is a safety
   decision, not a quality one — if the escalation decision itself has an LLM's own
   uncertainty baked in, you've just moved the problem, not solved it. Rules on top of
   signals we already computed (classifier confidence, retrieval similarity, keyword
   triggers) are slower to write well but their failure modes are enumerable and testable.

9. **Made the reply-generation fallback intentionally "dumb" (near-verbatim precedent
   reuse) rather than trying to fake sophistication with more string manipulation.** A
   fancier fallback would have made the automated "grounding overlap" metric look more
   meaningful than it is, obscuring the fact that it's not a real generation model. Better
   to have an honestly weak fallback than a dishonestly convincing one — see report.md
   Section 4.

10. **Built the LLM-judge and the proxy judge with the exact same interface/rubric**
    (`llm_judge()` / `proxy_judge()` in `eval_harness.py`) so switching from proxy to real
    judge is a zero-code-change, one-env-var operation, and so the judge-vs-human
    agreement numbers measure the same rubric either way.

11. **The "human" scores in `human_scores.csv` are explicitly disclosed as not being from
    an actual independent human** in this environment (see that file's docstring) —
    reporting an agreement number without that disclosure would be actively misleading,
    since a same-model "human" stand-in is the single easiest way to fake a good
    agreement score.

12. **Chose word-overlap as the automated grounding metric over something like ROUGE.**
    At this scale (44 short examples, no long-form text), ROUGE's n-gram precision/recall
    machinery adds complexity without adding real signal over a simpler ratio; it's also
    easier for a reader of the report to sanity-check by eye.

13. **Escalation thresholds (0.55 classifier confidence, 0.12 retrieval similarity) are
    explicit named constants, not tuned to this eval set.** They're placeholders pending
    real traffic (report.md Section 5, item 6) — tuning them to make this specific eval
    set look better would be fitting the demo, not building a real threshold.

14. **`positive_feedback` and `general_inquiry` are included as intents even though they
    don't need "resolution" in the refund/replacement sense**, because a real support
    queue receives them and misrouting them (e.g., treating a thank-you as needing
    escalation) would itself be a failure worth measuring.

15. **Kept the whole thing dependency-light (pandas/numpy/sklearn/joblib only)**, no
    heavier ML stack, both because this sandbox has no internet to install anything
    exotic, and because the eval harness should be something a reviewer can run in
    seconds, not something that needs a GPU box to reproduce.
