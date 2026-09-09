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

## Integrity-v2 Correction Plan

The first independent model audit found that the frozen arithmetic is internally
consistent but that token-set F1 and annotation-assisted CKG retrieval do not
support a relationship-reasoning claim. Preserve those artifacts as the audit
trail and run a separate corrected evaluation.

1. Select exactly four questions of each T1-T5 type per domain, for 240 total
   questions and 48 questions per type. Rotate T4 taxonomy categories across
   domains so the sample is not limited to singleton AREA queries.
2. Resolve CKG retrieval exclusively from the natural-language question. The
   retrieval function must receive no concept IDs, path IDs, taxonomy IDs, or
   ground-truth labels.
3. Require both CKG and RAG to return the same compact JSON answer schema. Score
   taxonomy classification, dependency sets, ordered path edges, aggregate sets,
   and directed dependency relations instead of unordered word overlap.
4. Include a deterministic question-echo control, retain token-set F1 only as a
   diagnostic, and record evidence recall separately from answer correctness.
5. Stamp every row and run manifest with model, prompt, retrieval settings, seed,
   timestamp, git commit, pricing, selected query IDs, and hashes of frozen inputs.
6. Add unit and integrity tests before any API call. Then run the minimum matched
   CKG/RAG evaluation with Claude Haiku 4.5 and publish a separate corrected
   report without modifying the Yarmoluk-McCreary paper.

## Integrity-v3 Hardening Plan

An adversarial model audit of integrity-v2 found that the evidence metric was
only substring coverage, T1 ignored extra labels, RAG contexts were not rebuilt
by the verifier, and CKG/RAG retrieval used differently worded questions.

1. Use identical original query text for CKG and RAG retrieval.
2. Add a same-model, same-prompt no-context condition.
3. Score every T1 label and penalize extras.
4. Rename the context diagnostic to exact target coverage and use term
   boundaries; do not represent it as factual or relational evidence.
5. Validate every sampled gold structure against the frozen graph.
6. Rebuild all RAG indexes during verification and reconstruct both retrieval
   paths from frozen inputs.
7. Publish integrity-v3 as the headline result and retain all prior versions as
   a visible audit trail.
