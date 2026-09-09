# ClinicalTrials.gov CKG Integrity-v2 Evaluation

**Run date:** September 9, 2026  
**Scope:** Structural graph-grounded questions across 12 ClinicalTrials.gov-derived domains  
**Model:** `claude-haiku-4-5-20251001`  
**Status:** Author-run, reproducible evaluation; not an independent third-party replication

## Result

Integrity-v2 corrects two material defects found in the first evaluation: CKG
retrieval could read answer annotations, and unordered token-set F1 rewarded
labels copied directly from the question.

| System | Structural F1 | Exact match | Evidence recall | Tokens/query | API cost |
|---|---:|---:|---:|---:|---:|
| Annotation-blind CKG | **1.000000** | **1.000000** | **1.000000** | **448.038** | **$0.174297** |
| Configured raw-prose RAG | 0.138716 | 0.125000 | 0.290945 | 3,113.800 | $0.805720 |
| Question-echo control | 0.000000 | 0.000000 | 0.000000 | 0 | $0 |

On this 240-question structural evaluation, CKG produced **7.21x the
structural F1**, used **6.95x fewer tokens**, and cost **4.62x less** than the
configured raw-prose RAG baseline. CKG had 210 wins, zero losses, and 30 ties.

These ratios describe this configuration and task. They are not evidence that
CKG is universally superior to RAG.

## What Changed

1. **Annotation-blind retrieval.** The CKG retriever receives only the natural-
   language question and query type. It cannot inspect `concept_id`,
   `path_ids`, `taxonomy_id`, or `ground_truth`.
2. **Balanced query sampling.** The paired set contains 48 questions from each
   T1-T5 class: four of each type per domain. T4 covers all nine taxonomy
   categories instead of only singleton `AREA` questions.
3. **Structural scoring.** Both systems return the same JSON schema. The scorer
   evaluates taxonomy labels, dependency sets, directed path edges, category
   member sets, and directed `depends_on` relations.
4. **Question-echo control.** Repeating the clarified question scores zero on
   the structural metric. The same control still scores 0.461484 under the old
   token-overlap metric, demonstrating why token F1 is diagnostic only.
5. **Stable trial identity.** Clinical trials are matched by their NCT registry
   identifiers. Long title text is descriptive and is not required when the
   correct NCT ID is returned.
6. **Execution provenance.** Every row records its model, backend, run ID,
   retrieval mode, token counts, cost, and harness version. The manifest records
   prompt, sampling, model configuration, selected IDs, Git revisions, and
   hashes of all frozen input trees.

## Query Tasks

| Type | Structural requirement | Questions |
|---|---|---:|
| T1 | Return the entity's taxonomy code | 48 |
| T2 | Return the exact direct prerequisite set | 48 |
| T3 | Return the correct directed path edges | 48 |
| T4 | Return the taxonomy category's member set | 48 |
| T5 | Return the directed `source depends_on target` relation | 48 |

Question text is clarified for generation so that the structural task is
explicit. Retrieval still resolves from the original frozen natural-language
question, without answer annotations.

## Per-Type Results

| Type | CKG F1 | RAG F1 | CKG evidence | RAG evidence |
|---|---:|---:|---:|---:|
| T1 entity taxonomy | 1.000000 | 0.604167 | 1.000000 | 0.833333 |
| T2 direct dependency | 1.000000 | 0.000000 | 1.000000 | 0.062500 |
| T3 directed path | 1.000000 | 0.000000 | 1.000000 | 0.175347 |
| T4 category aggregate | 1.000000 | 0.068581 | 1.000000 | 0.081463 |
| T5 directed relation | 1.000000 | 0.020833 | 1.000000 | 0.302083 |

All model responses were valid JSON. CKG scored 1.0 in each of the 12 domains.
RAG domain-level structural F1 ranged from 0.059091 to 0.209091.

## Systems

### CKG

- Frozen 180-node graph per domain
- Deterministic query-text resolution and graph traversal
- No embeddings
- No answer annotations available to retrieval
- Only the task-scoped subgraph enters the model context

### RAG

- Frozen prose generated from the same normalized ClinicalTrials.gov records
- `all-MiniLM-L6-v2` embeddings
- FAISS cosine similarity
- 512-token chunks with 50-token overlap
- Top five chunks

Both systems use the same model, output limit, provider-default temperature,
structured-answer prompt, questions, and pricing assumptions.

## Verification

Run:

```bash
python evaluation/verify_integrity_v2.py
```

The verifier independently checks 240 unique matched IDs, balanced query types,
T4 category coverage, every row's score/tokens/cost, exact reproduction of CKG
contexts from annotation-blind retrieval, the question-echo control, aggregates,
pairwise wins, and hashes of all frozen input trees.

## Limitations

1. **Graph-generated evaluation.** Questions and gold structural answers are
   generated from the evaluated graph. This establishes whether each system can
   recover declared structure; it does not independently prove that structure
   is clinically correct or complete.
2. **Representation is part of the comparison.** RAG searches source-derived
   prose that does not explicitly contain all generated taxonomy labels and
   dependency edges. This does not isolate traversal from representation.
3. **One RAG configuration.** No BM25 hybrid search, reranker, contextual
   retrieval, edge-text RAG, or Microsoft GraphRAG baseline is included.
4. **No expert-authored holdout.** A blinded clinician or trial-intelligence
   expert has not authored or adjudicated the questions and answers.
5. **Single model and run.** Generation used one Claude Haiku model snapshot and
   one execution. Provider outputs may vary on repetition.
6. **Historical raw-payload gap.** Every normalized source field consumed by the
   builder is frozen, but the complete original API response objects were not
   retained. Historical per-trial hashes cannot be independently recomputed.

## Defensible Interpretation

> On 240 balanced structural questions generated from 12 ClinicalTrials.gov-
> derived graphs, annotation-blind CKG traversal recovered the declared
> structure with 1.000 structural F1. The configured raw-prose RAG baseline
> scored 0.139, while using 6.95 times more model tokens. This demonstrates the
> value of an explicit semantic context layer for recovering its declared
> relationships; it does not establish universal superiority over RAG or
> independent clinical correctness.

## Next Integrity Tests

1. Commission expert-authored, blinded trial-intelligence questions that are
   not generated from the graph.
2. Add an edge-text vector baseline containing the same declared facts as CKG,
   isolating deterministic traversal from representation advantage.
3. Add hybrid BM25/vector retrieval plus reranking and repeat across at least
   one additional model family.
