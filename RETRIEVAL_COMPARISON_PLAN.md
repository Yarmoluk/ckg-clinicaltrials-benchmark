# Frozen Retrieval Comparison Plan

## Objective

Add a separate, reproducible comparison over the exact 240 integrity-v3 query
IDs without changing integrity-v3 gold, source graphs, or prose corpora.

## Systems

- `modern_rag`: domain-filtered BM25 plus dense retrieval, reciprocal-rank
  fusion, and local cross-encoder reranking over the frozen prose corpus.
- `edge_text_rag`: the same retrieval stack over deterministic node and typed
  edge documents generated only from the frozen graph CSVs.
- `ckg`: copied from integrity-v3 raw CKG rows after query, prompt, model,
  context, input-tree, and row-score validation.
- `router`: query-type-only routing. Default mapping is T2/T3/T4 to `ckg` and
  T1/T5 to `modern_rag`; the mapping is recorded in the manifest.

## Controls

1. Pin the v3 manifest hash, ordered query IDs, seed, domains, model, prompt,
   source-tree hashes, and structural scorer.
2. Generate edge-text documents deterministically and record both per-file and
   whole-tree hashes.
3. Keep retrieval annotations blind to gold labels, concept IDs, path IDs, and
   answer arrays. Domain and query-text NCT metadata are allowed.
4. Make paid answer generation opt-in with `--generate`; dry-run builds and
   verifies contexts without an API key.
5. Reconstruct both RAG retrieval paths in the verifier and independently
   re-score every saved answer.
6. Report representation, retrieval, generation, token, and cost limitations;
   make no clinical-correctness claim.

## Verification

- Unit tests for BM25, RRF, metadata handling, deterministic edge documents,
  frozen-query loading, CKG reuse, routing, and score compatibility.
- Full offline verifier over all 240 IDs and all four systems.
- Secret scan and clean-worktree review before publication.

