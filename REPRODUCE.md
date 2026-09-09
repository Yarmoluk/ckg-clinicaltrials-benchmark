# Reproducing the Evaluation

## Verify the Published Raw Outputs

The verifier requires only Python 3 and makes no API calls:

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

## Re-run the Matched Sample

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
