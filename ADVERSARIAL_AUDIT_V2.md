# Adversarial Model Audit: Frozen Retrieval v2

**Audit date:** September 9, 2026  
**Audit model:** `gpt-6-astra` through Codex  
**Access:** Fresh read-only session  
**Verdict:** **VALID AS SCOPED**  
**Status:** Model-based adversarial review; not an independent replication

## Scope

The audit inspected the frozen-v2 report, manifest, aggregate, representative
raw rows, retrieval and edge-text implementations, verifier, tests,
integrity-v3 parent manifest and scorer, and the invalid v1 audit trail. It
checked annotation leakage, graph/gold leakage, query and model/prompt parity,
preprocessing, retrieval fairness, CKG reuse, router semantics, scorer scope,
arithmetic, reproducibility, and claim boundaries.

## Finding And Correction

The first pass returned `NEEDS CORRECTION` for claim scope, not numerical or
artifact integrity:

- Hard NCT filtering excluded path-root evidence in all eight NCT-containing
  T3 edge-text queries, creating a configuration-specific disadvantage.
- Edge-text changes information availability and document granularity as well
  as representation, so it does not isolate a pure representation effect.

The public report and README were revised to disclose both issues and remove
causal language. The re-audit then returned `VALID AS SCOPED` with no remaining
critical findings.

## Confirmed

- All 960 saved rows reproduce the reported summaries and pairwise counts.
- Every frozen-v2 headline and per-type table value matches the aggregate.
- Input, implementation, edge-text, and parent hashes reconcile.
- The 4,876 edge-text documents reconstruct from the unchanged graph CSVs.
- No unintended annotation or gold leakage was found in retrieval.
- The 240 query IDs, model, prompt, output limit, and scorer match integrity-v3.
- CKG source-row reuse and router branch reuse hashes reconcile.
- All chapter NCT identifiers survive v2 prose preprocessing.

The audit confirmed F1 values of `0.174264`, `0.925485`, `1.000000`, and
`0.750000` for modern RAG, edge-text RAG, CKG, and router respectively; CKG vs.
edge-text had 35 wins, zero losses, and 205 ties. It also confirmed the `2.79x`
answer-model token ratio, `2.06x` answer-model inference-cost ratio, and
`$1.134875` new run spend.

## Claim Boundary

The result supports this specific graph-derived structural benchmark
comparison. It does not establish isolated causal effects, clinical
correctness, universal superiority, independent replication, or total
operating-cost savings. See [the full report](FROZEN_RETRIEVAL_REPORT.md).
