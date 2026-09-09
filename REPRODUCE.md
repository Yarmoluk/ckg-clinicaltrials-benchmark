# Reproducing the Evaluation

## Verify the Frozen Retrieval Comparison Without API Calls

The current comparison uses the exact integrity-v3 240-query set and requires
the BGE embedding model and MiniLM cross-encoder revisions recorded in
`results/frozen-retrieval-v2/manifest.json`.

```bash
python evaluation/test_retrieval_comparison.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/verify_retrieval_comparison.py
```

The verifier reconstructs the frozen edge-text documents, both hybrid retrieval
paths, every saved context and ranking, all CKG reuse and router decisions, all
structural scores, and all aggregates. It makes no API calls. On CPU, the full
rebuild can take 10–25 minutes.

Rebuild the deterministic edge-text corpus independently:

```bash
python evaluation/build_edge_text_corpus.py
```

Run retrieval and scoring with placeholder answers and no API key:

```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/retrieval_comparison_eval.py
```

Make the 480 new answer-model calls only when intentionally creating a new
replication. Use a new directory so the frozen run remains unchanged:

```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/retrieval_comparison_eval.py \
  --generate \
  --output-dir results/frozen-retrieval-v2-replication
```

The CKG branch reuses integrity-v3 raw outputs after hash, query, context, model,
and score validation. The router reuses selected branch outputs, so neither adds
new model calls. API output can vary between replications.

## Verify Integrity-v3 Without API Calls

Install the dependencies below, then run:

```bash
python evaluation/test_integrity_v3.py
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/verify_integrity_v3.py
```

The integrity-v3 verifier recomputes every structural score and aggregate,
regenerates the seeded sample, validates gold structures against the graph,
rebuilds the local RAG indexes, reconstructs every CKG and RAG context, checks
query/model/prompt parity, validates both controls, and checks hashes of all
frozen input trees. The thread limits avoid an Apple Accelerate/OpenMP crash
observed on one macOS environment.

## Re-run Integrity-v3

```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/integrity_v3_eval.py \
  --systems ckg rag no_context question_echo \
  --workers 8 \
  --output-dir results/integrity-v3-replication
```

The sample uses four questions of each T1-T5 type per domain, for 240 matched
questions and 720 paid model calls. API output can vary, so write replications
to a new output directory and preserve the generated manifest.

To validate retrieval, sampling, scoring, and local RAG indexes without paid
model calls:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/integrity_v3_eval.py --dry-run
```

## Verify Superseded Runs

Integrity-v2 remains frozen as an audit trail:

```bash
python evaluation/test_integrity_eval.py
python evaluation/verify_integrity_v2.py
```

The original verifier also makes no API calls. Its pass establishes arithmetic
consistency only; see `INTEGRITY_V3_REPORT.md` for the current methodology.

```bash
python3 evaluation/verify_results.py
```

It recomputes query counts, mean F1, mean tokens, API cost, pairwise wins, and
query-ID alignment from the frozen JSONL files. A successful run ends with:

```text
PASS: all published aggregate results match the frozen raw outputs
```

## Install the Live Harnesses

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
```

Provide an Anthropic key through the environment or an untracked `.env` file:

```bash
export ANTHROPIC_API_KEY="your-key"
```

Never commit an API key. `.env` is excluded by `.gitignore`.

## Domain IDs

Define the evaluation set once in the current shell:

```bash
clinical_domains=(
  ct-oncology-solid-tumors
  ct-hematologic-malignancies
  ct-cardiovascular
  ct-neurology-alzheimers
  ct-immunology-autoimmune
  ct-infectious-disease-vaccines
  ct-metabolic-endocrine
  ct-rare-genetic
  ct-respiratory-pulmonary
  ct-gastro-hepatology
  ct-nephrology-renal
  ct-psychiatry-behavioral
)
```

## Re-run One Domain Without API Calls

```bash
python evaluation/ckg_harness.py \
  --domain ct-cardiovascular \
  --dry-run

python evaluation/rag_harness.py \
  --domain ct-cardiovascular \
  --dry-run \
  --reindex
```

## Re-run the Superseded Matched Sample

```bash
python evaluation/ckg_harness.py \
  --domains "${clinical_domains[@]}" \
  --stratified-limit 20 \
  --seed 42

python evaluation/rag_harness.py \
  --domains "${clinical_domains[@]}" \
  --stratified-limit 20 \
  --seed 42 \
  --overwrite

python evaluation/small_model_harness.py \
  --model claude-haiku-4-5-20251001 \
  --backend anthropic \
  --mode baseline \
  --domains "${clinical_domains[@]}" \
  --n 20 \
  --seed 42
```

API output can vary between runs. Preserve raw outputs and report the exact model
identifier, pricing assumptions, query IDs, and run date with any replication.

## Rebuild the Dataset From the Live API

This contacts ClinicalTrials.gov and may produce a different snapshot as trial
records change:

```bash
python evaluation/clinicaltrials_probe_builder.py \
  --all \
  --max-trials 500
```

Use the frozen `benchmark/` and `corpus/` directories when reproducing the
published September 9, 2026 evaluation. Rebuilding from the live API is a
freshness test, not an exact replication.
