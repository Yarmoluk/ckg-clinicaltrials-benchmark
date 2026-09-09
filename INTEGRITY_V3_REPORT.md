# ClinicalTrials.gov CKG Integrity-v3 Evaluation

**Run date:** September 9, 2026  
**Scope:** Structural graph recovery across 12 ClinicalTrials.gov-derived domains  
**Model:** `claude-haiku-4-5-20251001`  
**Status:** Author-run, reproducible evaluation; not an independent third-party replication

## Result

Integrity-v3 responds to an adversarial model audit of integrity-v2. It makes
the CKG and RAG retrieval queries identical, adds a same-model no-context
baseline, applies strict T1 scoring, replaces the overstated evidence metric,
and reconstructs both retrieval paths during verification.

| System | Structural F1 | Exact match | Target coverage | Tokens/query | API cost |
|---|---:|---:|---:|---:|---:|
| Annotation-blind CKG | **1.000000** | **1.000000** | **1.000000** | **447.863** | **$0.174087** |
| Configured raw-prose RAG | 0.143044 | 0.133333 | 0.304143 | 3,207.550 | $0.831588 |
| Same-model no context | 0.116667 | 0.116667 | 0.000000 | 246.400 | $0.093252 |
| Plain question echo | 0.000000 | 0.000000 | 0.000000 | 0 | $0 |

Against this RAG configuration, CKG produced **6.99x the structural F1**.
RAG used **7.16x as many model tokens** and cost **4.78x as much**. CKG had
208 pairwise wins, zero losses, and 32 ties. The paid v3 run cost $1.098927.

The no-context result matters. It scored 0.562500 on T1 entity taxonomy and
0.116667 overall, compared with RAG's 0.604167 on T1 and 0.143044 overall.
Entity names expose substantial T1 signal without retrieval. The difference
between RAG and no context is only 0.026377 absolute structural F1, so RAG's
full score must not be described as retrieval lift.

## Changes From Integrity-v2

1. **Strict T1 scoring.** Every returned taxonomy label is scored. A correct
   label plus an extra label no longer receives perfect precision or exact
   match.
2. **Same retrieval query.** CKG and RAG both retrieve with the original frozen
   natural-language question. Both then answer the same clarified evaluation
   question.
3. **Same-model no-context control.** The identical model, system prompt,
   output limit, and evaluation question are run with an empty context.
4. **Accurate context and format diagnostics.** `context_target_coverage` measures exact
   answer-target presence. It is not described as evidence, factual support, or
   relationship support. Exact term boundaries prevent `INTR` from matching
   words such as `intraoperative`. JSON reporting distinguishes an extractable
   object from strict whole-response JSON compliance.
5. **Graph-validated gold.** Verification checks T1 taxonomy, T2 dependency
   sets, every T3 path edge, T4 membership sets, and every directed T5 edge
   against the frozen graph.
6. **Both retrieval paths reconstructed.** Verification rebuilds RAG indexes
   from frozen prose and embeddings in memory, reconstructs all 240 RAG
   contexts, and independently reconstructs all 240 annotation-blind CKG
   contexts.
7. **Expanded parity checks.** The verifier regenerates the seeded sample,
   compares saved annotations to source queries, and checks model, backend,
   prompt hash, retrieval questions, generation questions, tokens, cost, and
   aggregates.

## Per-Type Results

| Type | CKG F1 | RAG F1 | No-context F1 | CKG coverage | RAG coverage |
|---|---:|---:|---:|---:|---:|
| T1 entity taxonomy | 1.000000 | 0.604167 | 0.562500 | 1.000000 | 0.895833 |
| T2 direct dependency | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.083333 |
| T3 directed path | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.173611 |
| T4 category aggregate | 1.000000 | 0.048553 | 0.000000 | 1.000000 | 0.065853 |
| T5 directed relation | 1.000000 | 0.062500 | 0.020833 | 1.000000 | 0.302083 |

All 720 model responses contained an extractable JSON object. Strict
whole-response JSON compliance was 1.000000 for CKG, 0.516667 for RAG, and
0.545833 for no context. The structural scorer uses the extractable object.
CKG scored 1.0 in all 12 domains.

## Evaluation Design

- 240 matched questions: four T1-T5 questions per domain
- 48 questions in each query class
- All nine taxonomy categories represented in T4
- CKG retrieval receives only query type and original question text
- RAG retrieves over frozen source-derived prose using the same question text
- Same model, system prompt, output limit, and clarified generation question
- Structural scoring for taxonomy sets, dependency sets, path edges, category
  membership, and directed relations
- NCT registry identifier used as canonical trial identity

The deterministic plain-text echo scores 0.000000 structurally but 0.461484
under the superseded token-overlap metric. This control explains why token F1
is retained only as a diagnostic.

## Systems

### CKG

- Frozen 180-node graph per domain
- Deterministic question-text resolution and graph traversal
- No embeddings and no answer annotations available to retrieval
- Task-scoped serialized subgraph supplied to the answer model

### RAG

- Frozen prose generated from the same normalized ClinicalTrials.gov records
- `all-MiniLM-L6-v2` embeddings and FAISS cosine similarity
- 512-token chunks, 50-token overlap, and top-five retrieval

## Verification

Run from the repository root with the documented environment:

```bash
python evaluation/test_integrity_v3.py
python evaluation/verify_integrity_v3.py
```

The verifier performs no paid model calls. It rebuilds embeddings locally and
checks the raw artifacts rather than trusting the published aggregate.

## Limitations

1. **Graph-generated evaluation.** Questions and gold answers come from the
   graph. The result measures recovery of declared structure, not independent
   clinical correctness or completeness.
2. **Representation advantage.** CKG explicitly encodes taxonomy and directed
   edges. The raw prose baseline does not contain every generated graph fact.
   This comparison includes representation, retrieval, and context efficiency;
   it does not isolate traversal alone.
3. **Weak RAG baseline.** Only one vanilla dense configuration is tested. There
   is no BM25 hybrid retrieval, reranker, contextual retrieval, edge-text RAG,
   or GraphRAG baseline.
4. **Question-only signal.** Entity names permit substantial T1 classification
   without context, as measured by the no-context baseline.
5. **T5 is not adversarially balanced.** T5 contains only positive
   `depends_on` examples, exposes both endpoints in the question, and allows
   only one positive relation label. The same-model no-context control usually
   abstained, but a purpose-built structured endpoint copier could exploit this
   task. Add negative relation pairs before treating T5 as a robust relation
   classification benchmark.
6. **No expert holdout.** No blinded clinician or trial-intelligence expert has
   authored or adjudicated an independent question set.
7. **Single model and run.** Results use one Claude Haiku snapshot and one run.
8. **Historical raw-payload gap.** Normalized fields consumed by the graph
   builder are frozen, but complete original ClinicalTrials.gov API response
   objects were not retained. Historical per-trial source hashes cannot be
   independently recomputed.

## Defensible Public Claim

> On 240 balanced, graph-derived structural questions across 12
> ClinicalTrials.gov-derived domains, annotation-blind CKG traversal achieved
> 1.000 structural F1. A configured raw-prose vector-RAG baseline using the
> same retrieval questions and answer model scored 0.143 and used 7.16 times
> as many model tokens. A same-model no-context control scored 0.117. This
> supports explicit semantic structure for recovering the graph's declared
> relationships; it does not establish universal superiority over RAG or
> independent clinical correctness.

## Next Tests

1. Build an edge-text vector baseline containing the same declared facts as
   CKG to isolate traversal from representation.
2. Add hybrid BM25/vector retrieval and reranking.
3. Commission a blinded expert-authored holdout that is not generated from the
   evaluated graph.
4. Repeat with a second model family and multiple seeds.
