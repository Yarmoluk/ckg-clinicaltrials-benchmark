# Standalone ClinicalTrials.gov Benchmark Plan

## Objective

Publish the September 2026 ClinicalTrials.gov CKG evaluation as an independent
Daniel Yarmoluk / Graphify.md repository without modifying or extending the
Yarmoluk-McCreary v0.6.2 paper.

## Deliverables

1. A concise repository README with the scoped finding and separate full-run and
   paired-sample tables.
2. A detailed report documenting data construction, systems, metrics,
   limitations, and defensible interpretation.
3. Machine-readable aggregate results and the raw outputs needed to audit them.
4. Frozen benchmark graphs, query sets, source provenance, and scripts required
   to rebuild or rerun the evaluation.
5. A reviewer protocol that allows independent evaluation without assigning
   coauthorship.

## Guardrails

- Do not edit `Yarmoluk/ckg-benchmark` paper files, version, DOI, authorship, or
  headline result tables.
- Keep the 2,064-query full CKG run distinct from the 240-query paired comparison.
- Describe the task as ClinicalTrials.gov-derived structural graph-grounded QA,
  not open-ended medical QA.
- Do not claim universal superiority over RAG or independent verification.
- Attribute ClinicalTrials.gov as the public data source and identify all models,
  retrieval settings, seeds, and known limitations.

## Publication

Create and validate the package locally, commit it to this repository, and push
to the private GitHub remote. Public visibility remains a separate owner decision
after review.
