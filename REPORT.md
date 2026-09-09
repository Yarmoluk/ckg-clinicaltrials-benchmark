# ClinicalTrials.gov CKG Structural Context Evaluation

> **Superseded as the primary result.** An independent model audit found that
> this first run's arithmetic is reproducible but its token-overlap scorer and
> annotation-assisted CKG retrieval do not support a relationship-reasoning
> claim. See [INTEGRITY_V2_REPORT.md](INTEGRITY_V2_REPORT.md) for the corrected
> annotation-blind evaluation. These original results remain frozen as an audit
> trail.

## Abstract

This evaluation tests whether a Compressed Knowledge Graph (CKG) provides more
effective model context than vanilla retrieval-augmented generation (RAG) for
relationship-dependent questions derived from ClinicalTrials.gov records.

A deterministic pipeline processed 5,587 studies across 12 therapeutic domains
into 2,160 concept nodes, 2,716 declared edges, 60 prose chapters, and 2,064
structural questions. A matched 240-question sample compared CKG, vanilla RAG,
and no context using the same answer model. CKG achieved macro-F1 0.4691 versus
0.1300 for RAG while consuming 231.8 versus 3,231.1 mean tokens per query. This
corresponds to 3.61x higher F1, 13.94x fewer tokens, and 7.52x lower measured API
cost for the tested configuration.

These results support an architectural advantage for CKG on the evaluated
structural workload. They do not establish universal superiority over RAG or
clinical-answering validity.

## 1. Research Question

When an LLM must answer questions about declared entities, dependencies,
categories, and multi-hop relationships, does a compact graph substructure
provide better context than semantically similar prose chunks?

The variable under evaluation is the representation delivered to the answer
model:

- **CKG:** deterministic traversal of declared nodes and edges.
- **Vanilla RAG:** semantic retrieval of prose chunks from the same domain data.
- **No context:** the same model answering from parametric knowledge alone.

## 2. Data Construction

The source was the ClinicalTrials.gov API v2. The domain manifest contains five
condition searches for each of 12 commercially relevant therapeutic areas. The
selection is a pragmatic evaluation set, not a claim about live trial-volume
rankings.

| Domain | Studies | Nodes | Edges | Questions |
| --- | ---: | ---: | ---: | ---: |
| Oncology - solid tumors | 446 | 180 | 227 | 172 |
| Hematologic malignancies | 362 | 180 | 249 | 172 |
| Cardiovascular disease | 477 | 180 | 213 | 172 |
| Neurology and neurodegeneration | 494 | 180 | 217 | 172 |
| Immunology and autoimmune disease | 473 | 180 | 222 | 172 |
| Infectious disease and vaccines | 493 | 180 | 255 | 172 |
| Metabolic and endocrine disease | 471 | 180 | 224 | 172 |
| Rare and genetic disease | 487 | 180 | 220 | 172 |
| Respiratory and pulmonary disease | 480 | 180 | 218 | 172 |
| Gastroenterology and hepatology | 456 | 180 | 225 | 172 |
| Nephrology and renal disease | 474 | 180 | 226 | 172 |
| Psychiatry and behavioral health | 474 | 180 | 220 | 172 |
| **Total** | **5,587** | **2,160** | **2,716** | **2,064** |

The builder extracts conditions, interventions, outcomes, sponsors, phases,
statuses, and selected trial records. It then creates graph nodes, typed
taxonomies, declared dependencies, source records, a prose corpus, and
deterministic structural questions. No LLM is used to construct the graphs.

The frozen graph and prose forms are both present in this repository. This avoids
comparing a graph built from one source against RAG built from another source.

## 3. Query Classes

| Type | Task |
| --- | --- |
| T1 entity | Identify a graph entity and its class |
| T2 dependency | Return a concept's declared prerequisite |
| T3 path | Return a prerequisite chain between concepts |
| T4 aggregate | Enumerate concepts belonging to a taxonomy class |
| T5 cross-concept | Explain a relationship between two graph concepts |

The query set is intentionally structural. It measures whether a context system
preserves and delivers relationships; it is not a general document-QA suite.

## 4. Systems

### 4.1 CKG

The CKG harness reads `learning-graph.csv`, selects the target entity or traverses
the relevant declared edges, serializes only that subgraph, and passes it to the
answer model. Retrieval requires no embeddings or graph database.

### 4.2 Vanilla RAG

The RAG harness indexes the frozen prose corpus with:

- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS inner-product search over normalized vectors
- 512-token chunks
- 50-token overlap
- top-5 retrieval

### 4.3 No-context baseline

The same answer model receives the question without graph or retrieved prose.

### 4.4 Shared generation and scoring

- Answer model: `claude-haiku-4-5-20251001`
- Scoring: normalized token-level F1 against deterministic ground-truth labels
- Cost assumption recorded at run time: $1.00 per million input tokens and $5.00
  per million output tokens
- Paired sample: 20 stratified questions per domain, seed 42

## 5. Matched 240-Question Result

This is the primary comparison because every system answered the same query IDs.

| System | Questions | Macro-F1 | Mean tokens/query | API cost |
| --- | ---: | ---: | ---: | ---: |
| **CKG** | 240 | **0.4691** | **231.8** | **$0.1214** |
| Vanilla RAG | 240 | 0.1300 | 3,231.1 | $0.9130 |
| No context | 240 | 0.0784 | 336.4 | $0.3480 |

CKG versus vanilla RAG:

- F1 ratio: **3.61x**
- Absolute F1 lift: **0.3391**
- Token ratio: **13.94x fewer**
- Cost ratio: **7.52x lower**
- Pairwise record: **223 wins, 17 losses, 0 ties**

### 5.1 Matched result by query type

| Type | n | CKG F1 | RAG F1 | No-context F1 | CKG/RAG |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 entity | 72 | 0.2021 | 0.1158 | 0.0597 | 1.75x |
| T2 dependency | 72 | 0.5030 | 0.0053 | 0.0018 | 95.32x |
| T3 path | 36 | 0.8429 | 0.2436 | 0.1967 | 3.46x |
| T4 aggregate | 12 | 0.8709 | 0.1523 | 0.0000 | 5.72x |
| T5 cross-concept | 48 | 0.4378 | 0.2475 | 0.1526 | 1.77x |

### 5.2 Matched result by domain

| Domain | CKG F1 | RAG F1 | CKG/RAG |
| --- | ---: | ---: | ---: |
| Cardiovascular | 0.4877 | 0.1290 | 3.78x |
| Gastroenterology/hepatology | 0.4559 | 0.0952 | 4.79x |
| Hematologic malignancies | 0.5103 | 0.1380 | 3.70x |
| Immunology/autoimmune | 0.4305 | 0.1285 | 3.35x |
| Infectious disease/vaccines | 0.4534 | 0.1495 | 3.03x |
| Metabolic/endocrine | 0.4212 | 0.1209 | 3.48x |
| Nephrology/renal | 0.4860 | 0.1280 | 3.80x |
| Neurology/neurodegeneration | 0.4398 | 0.1256 | 3.50x |
| Oncology/solid tumors | 0.5264 | 0.1204 | 4.37x |
| Psychiatry/behavioral | 0.4852 | 0.1103 | 4.40x |
| Rare/genetic | 0.4668 | 0.1709 | 2.73x |
| Respiratory/pulmonary | 0.4655 | 0.1434 | 3.25x |

CKG outperformed the configured RAG baseline in every domain in the paired
sample. This cross-domain consistency is stronger evidence than a single-domain
result, while remaining specific to this task and baseline.

## 6. Full CKG Run

The full CKG run evaluates all 2,064 questions. It is a scale and stability result,
not a full-system comparison.

| Metric | Result |
| --- | ---: |
| Questions | 2,064 |
| Domains | 12 |
| Macro-F1 | 0.5139 |
| Mean tokens/query | 277.2 |
| API cost | $1.2024 |

### 6.1 Full CKG F1 by query type

| Type | F1 |
| --- | ---: |
| T1 entity | 0.2866 |
| T2 dependency | 0.5455 |
| T3 path | 0.8657 |
| T4 aggregate | 0.8970 |
| T5 cross-concept | 0.4493 |

### 6.2 Full CKG F1 by hop depth

| Hop depth | F1 |
| ---: | ---: |
| 0 | 0.3797 |
| 1 | 0.5040 |
| 2 | 0.8506 |
| 3 | 0.9287 |

The strongest behavior occurs where explicit structure should matter most:
dependency, path, aggregate, and deeper-hop questions. Entity-definition lookup
remains the weakest CKG class.

## 7. Interpretation

The evidence supports this scoped statement:

> Across 12 ClinicalTrials.gov-derived domains, CKG was demonstrably superior to
> the configured vanilla RAG baseline for structural, relationship-dependent
> questions: 3.61x higher macro-F1, 13.94x fewer tokens, and 7.52x lower API cost
> on an identical 240-question sample.

The mechanism is consistent with the result. RAG retrieves passages that are
semantically similar to a question and asks the model to infer the needed
relationship. CKG traverses a relationship already declared in the context
structure and sends the model a much smaller task-specific subgraph.

The appropriate production pattern is hybrid: use CKG for the governed,
high-value structural core and use RAG for fuzzy discovery or details outside
the graph.

## 8. Limitations and Threats to Validity

1. **Structural task alignment.** Questions are generated from the graph schema.
   This is a valid test of structural context delivery, but it favors systems
   capable of preserving that structure.
2. **No expert clinical grading.** Token-F1 measures overlap with deterministic
   graph labels. It does not establish clinical correctness, usefulness, or
   safety.
3. **Partial comparison at scale.** Only CKG completed all 2,064 questions. The
   headline comparison therefore uses the matched 240-question subset.
4. **Single RAG configuration.** The comparison covers one common vanilla RAG
   design. It does not include reranking, hybrid lexical retrieval, query
   rewriting, long-context prompting, GraphRAG, LightRAG, HippoRAG, or RAPTOR.
5. **Author-run evaluation.** The artifacts are public and auditable, but the run
   has not yet been independently reproduced.
6. **Model stochasticity.** Generation can vary between runs. The frozen raw
   outputs establish what was scored in this run.
7. **Registry scope.** ClinicalTrials.gov records describe registered studies;
   they are not a complete clinical knowledge base.
8. **Graph quality risk.** A missing, stale, or incorrect graph relationship can
   produce a systematic miss or a faithfully grounded wrong answer.

## 9. Strongest Next Tests

1. Run vanilla RAG across all 2,064 questions.
2. Add independently written, held-out questions created from raw trial records
   rather than from graph structure.
3. Add expert grading for answer correctness, citation entailment, and refusal.
4. Compare against hybrid RAG, reranked RAG, long-context prompting, and modern
   graph-retrieval baselines.
5. Publish confidence intervals and repeated-run variance.
6. Ask an external reviewer to reproduce the frozen 240-question comparison.

## 10. Authorship and Relationship to Other Work

This repository and evaluation are authored by Daniel Yarmoluk / Graphify.md.
It is intentionally separate from the Yarmoluk-McCreary benchmark paper and does
not change that paper's version, DOI, claims, files, or authorship.

An external review may be acknowledged by name after completion. Independent
review does not make the reviewer an author and does not imply institutional or
ClinicalTrials.gov endorsement.
