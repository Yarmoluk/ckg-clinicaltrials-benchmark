# CKG ClinicalTrials.gov Benchmark

**Author:** Daniel Yarmoluk, Graphify.md

**Run date:** September 9, 2026

**Status:** Author-run, openly auditable evaluation; not independently replicated

This repository evaluates a Compressed Knowledge Graph (CKG) as an AI context
layer over structural questions derived from ClinicalTrials.gov records. It is a
standalone project by Daniel Yarmoluk. It does not modify, extend, or share
authorship with the separate Yarmoluk-McCreary CKG benchmark paper.

## Headline Result

On an identical, stratified sample of 240 questions across 12 therapeutic
domains, using the same Claude Haiku 4.5 model for answer generation:

| System | Macro-F1 | Mean tokens/query | API cost |
| --- | ---: | ---: | ---: |
| **CKG** | **0.4691** | **231.8** | **$0.1214** |
| Vanilla RAG | 0.1300 | 3,231.1 | $0.9130 |
| No context | 0.0784 | 336.4 | $0.3480 |

Against the configured vanilla RAG baseline, CKG produced:

- **3.61x higher macro-F1**
- **13.94x fewer tokens per query**
- **7.52x lower API cost**
- **223 wins in 240 paired comparisons (92.9%)**

The separate full CKG run covered all **2,064 questions** and reached **0.5139
macro-F1** at **277.2 mean tokens per query**. It is reported separately because
RAG and no-context systems have not yet been run over all 2,064 questions.

## What This Supports

The result is evidence that a declared, traversable semantic structure is a
better context architecture than this vanilla vector-RAG configuration for the
tested relationship-dependent workload. The strongest results appear on direct
dependencies, multi-hop paths, and category aggregates.

It is not evidence that CKG universally replaces RAG. These are structural,
graph-grounded questions, not open-ended medical document QA. RAG remains useful
for fuzzy discovery and the unstructured long tail.

## Evaluation Surface

- 12 therapeutic domains
- 5,587 ClinicalTrials.gov studies processed
- 2,160 graph nodes and 2,716 declared edges
- 2,064 deterministic benchmark questions
- Five query classes: entity, dependency, path, aggregate, and cross-concept
- Same answer model and token-F1 scorer across compared systems
- Vanilla RAG: all-MiniLM-L6-v2 embeddings, FAISS, 512-token chunks, 50-token
  overlap, top-5 retrieval

Read the [full report](REPORT.md), inspect the
[machine-readable results](results/aggregate/), or follow the
[reproduction instructions](REPRODUCE.md).

## Repository Contents

| Path | Contents |
| --- | --- |
| `benchmark/` | Frozen graph CSVs, source provenance, query sets, and domain manifest |
| `corpus/` | Frozen prose representation used by the RAG baseline |
| `evaluation/` | Graph builder, CKG/RAG harnesses, and result verifier |
| `results/aggregate/` | Machine-readable full-run and paired-sample summaries |
| `results/raw/full-ckg/` | All 2,064 full CKG outputs |
| `results/raw/paired/` | Exact 240-query CKG, RAG, and no-context outputs |
| `REVIEW.md` | Protocol for independent technical review |

## Independence and Review

Daniel Yarmoluk is the sole author of this repository and evaluation. External
reviewers may be acknowledged after they reproduce, critique, or validate the
work. Review does not imply coauthorship or endorsement.

## Appropriate Citation

> Yarmoluk, Daniel. "CKG ClinicalTrials.gov Benchmark: Structural Context
> Evaluation Across 12 Therapeutic Domains." Graphify.md, 2026.

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Important Scope Notice

ClinicalTrials.gov is the source of the public trial records. This project is not
affiliated with or endorsed by ClinicalTrials.gov, the National Library of
Medicine, or the National Institutes of Health. The benchmark does not evaluate
clinical safety, treatment recommendations, or medical decision support.

## Licenses

- Evaluation code: [MIT](LICENSE)
- Derived benchmark graphs, query sets, and results:
  [CC BY 4.0](LICENSES/DATA.md)
- Underlying trial records: source terms and notices remain with
  ClinicalTrials.gov
