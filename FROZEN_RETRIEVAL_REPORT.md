# Frozen 240-Question Retrieval Comparison

**Run date:** September 9, 2026  
**Parent:** Frozen integrity-v3 query set and CKG outputs  
**Answer model:** `claude-haiku-4-5-20251001`  
**Status:** Author-run and reproducible; not an independent replication

## Result

This comparison tests whether the integrity-v3 result survives a modern hybrid
RAG baseline and whether its advantage comes from graph representation or from
deterministic graph traversal. It uses the same 240 query IDs, original
retrieval-question text, clarified generation questions, answer model, system
prompt, and structural scorer as integrity-v3.

| System | Structural F1 | Exact match | Target coverage | Tokens/query | Inference cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modern raw-prose RAG | 0.174264 | 0.150000 | 0.387288 | 3,023.829 | $0.775967 |
| Edge-text RAG | 0.925485 | 0.854167 | 0.940281 | 1,249.350 | $0.358908 |
| **Annotation-blind CKG** | **1.000000** | **1.000000** | **1.000000** | **447.863** | **$0.174087** |
| Type-rule router | 0.750000 | 0.750000 | 0.883333 | 1,450.800 | $0.421540 |

The main finding is not that CKG is 5.74 times better than modern RAG. That
ratio compares CKG with raw prose and still combines representation and
retrieval effects. The edge-text control is the important comparison:

- Making the graph's declared facts available as searchable text raises F1
  from 0.174264 to 0.925485.
- CKG traversal raises F1 from 0.925485 to 1.000000: 35 pairwise wins, zero
  losses, and 205 ties.
- Edge-text RAG uses 2.79 times as many answer-model tokens and costs 2.06
  times as much as CKG in this run.

For these specific pipelines, the result is consistent with explicit semantic
representation accounting for most of the raw-prose gap. It does not cleanly
isolate representation from information availability, document granularity,
or retrieval policy. Deterministic, task-scoped traversal records the remaining
measured accuracy and answer-model token efficiency under this configuration.

The two new RAG systems cost $1.134875 to run. CKG reused the exact 240
integrity-v3 raw outputs after reconstruction and re-scoring; the router reused
already generated branch outputs. Their displayed costs are deployment-equivalent
inference costs, not new benchmark spend.

## Per-Type Results

| Query type | Modern RAG | Edge-text RAG | CKG | Router |
| --- | ---: | ---: | ---: | ---: |
| T1 entity taxonomy | 0.666667 | 1.000000 | 1.000000 | 0.666667 |
| T2 direct dependency | 0.000000 | 0.979167 | 1.000000 | 1.000000 |
| T3 directed path | 0.000000 | 0.845833 | 1.000000 | 1.000000 |
| T4 category aggregate | 0.121322 | 0.802423 | 1.000000 | 1.000000 |
| T5 directed relation | 0.083333 | 1.000000 | 1.000000 | 0.083333 |

The router uses only the frozen query-type field: T2/T3/T4 go to CKG and
T1/T5 go to modern raw-prose RAG. It is a fixed policy control, not a learned
router and not an optimized production recommendation. Its lower score follows
directly from routing T1 and T5 away from the better-performing system.

## Retrieval Systems

### Modern raw-prose RAG

- Frozen integrity-v3 prose corpus
- Conservative presentation-markup removal that preserves literal clinical
  comparison text such as `<26`; all chapter NCT identifiers remain indexable
- Existing 512-token chunks with 50-token overlap
- One index per domain
- Exact NCT metadata filter parsed only from query text, when matching chunks
  exist
- Local Okapi BM25 (`k1=1.5`, `b=0.75`)
- `BAAI/bge-base-en-v1.5` dense embeddings, revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- Reciprocal-rank fusion with `k=60`, retaining 40 candidates
- `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`
- Top five documents supplied to the answer model

### Edge-text RAG

The same hybrid and reranking stack indexes 4,876 deterministic documents made
only from the unchanged graph CSVs:

- 2,160 node documents: label, taxonomy type, direct prerequisites, direct
  dependents, and NCT identifiers present in those graph labels
- 2,716 typed-edge documents: source label and type, `depends_on`, target label
  and type, and NCT identifiers present on the endpoints

No document contains a query, gold answer, path annotation, or query-specific
concept identifier. This system intentionally receives the graph's declared
facts as text. It narrows the comparison between searchable serialization and
deterministic traversal, but also changes information availability and document
granularity relative to raw prose; it is not an independent evidence source.

### CKG

CKG uses the integrity-v3 annotation-blind question resolver and graph
traversal. Retrieval sees only query type and original question text, not gold
labels, concept IDs, path IDs, or taxonomy IDs. Each saved row records the hash
of the exact integrity-v3 source row it reused.

## Frozen Controls

- Parent manifest SHA-256:
  `1543e41d1ab7f831725bc3babfca919a94145c09fa309a6aef76691d066ad2e4`
- Seed: `20260909`
- Questions: 240 unique IDs, 48 each of T1-T5, across the same 12 domains
- Edge-text tree SHA-256:
  `b7fbd362961523d5bfd1c2e9fcb510694bf8e8fb6816b3e8b90166628077b8c5`
- Same retrieval-question text for all systems
- Same answer model, system prompt, output limit, generation question, and
  structural scorer
- Saved top-five document IDs plus BM25, dense, RRF, and reranker trace fields

## Verification

```bash
python evaluation/test_retrieval_comparison.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/verify_retrieval_comparison.py
```

The verifier makes no paid calls. It reconstructs the edge-text corpus from the
CSV files, checks all input and parent hashes, reloads the pinned local models,
rebuilds both RAG index families, reproduces every top-five context and ranking,
re-scores all 960 rows, validates CKG reuse and router selection, and recomputes
all aggregates. Cross-encoder logits allow `1e-4` CPU floating-point drift;
document IDs, order, ranks, RRF values, and contexts must match.

## Limitations

1. **Graph-derived questions and gold.** The benchmark measures recovery of
   relationships declared by the graph. It does not establish clinical
   correctness, completeness, safety, or medical usefulness.
2. **Edge-text is graph-derived.** Its high score demonstrates the value of
   explicit representation. It is not evidence that independently extracted
   prose facts agree with the graph.
3. **One modern RAG configuration.** This is a credible BM25+dense+reranker
   baseline, not an exhaustive search over query expansion, multi-query RAG,
   contextual retrieval, GraphRAG, or task-specific retrieval training.
4. **Top-five aggregate constraint.** T4 asks for complete category membership
   while both RAG systems provide five documents. Node documents include
   neighbor labels, but top-five retrieval can still cap aggregate recall.
5. **Hard NCT filtering affects paths.** The query-text NCT filter narrows the
   eligible document set before ranking. In all eight NCT-containing T3
   questions, eligible edge-text documents omitted the path root; all eight
   missed exact match. Part of the measured traversal gap therefore comes from
   this retrieval policy, not traversal alone.
6. **Question-only signal remains.** T1 entity names reveal taxonomy signal, as
   the integrity-v3 no-context control already showed.
7. **T5 remains positive-only.** Both endpoints appear in the question and all
   examples use `depends_on`; a question-only formatter can score 48/48 by
   copying the endpoints and emitting that relation. T5 does not establish
   retrieval use. T3 also scores one graph-defined reference path.
8. **Single answer model and run.** Results can vary with generation. Retrieval
   contexts, raw outputs, and scores are frozen for this run.
9. **Reused CKG generation.** CKG calls were not simultaneous with the two new
   systems, but model, prompt, questions, and source rows are hash-checked.
10. **Cost scope.** Reported cost covers answer-model tokens. It excludes local
    embedding, reranking, graph construction, indexing, storage, and latency;
    it is not an end-to-end operating-cost comparison.
11. **Environment scope.** Model revisions and implementation hashes are
    pinned, but package versions are not locked to a fully reproducible image.
12. **Superseded parser run.** `frozen-retrieval-v1` is retained only as an
   invalid audit trail. Its inherited broad HTML-tag regex removed clinical
   text between literal angle brackets. The corrected v2 loader preserves all
   NCT identifiers found in the frozen chapter Markdown.

## Defensible Claim

> On the same frozen 240 graph-derived structural questions, a BM25+BGE+RRF+
> reranker raw-prose baseline scored 0.174 structural F1. Giving that same
> retrieval stack text representations of the graph's declared nodes and edges
> raised it to 0.925. Annotation-blind CKG traversal scored 1.000 while using
> fewer answer-model tokens; edge-text RAG used 2.79 times as many. For these
> pipelines, the result is consistent with substantial benefit from explicit
> semantic representation plus task-scoped traversal. It does not isolate those
> effects from information availability, document granularity, retrieval policy,
> or local retrieval compute; test clinical correctness; or establish universal
> superiority over RAG.
