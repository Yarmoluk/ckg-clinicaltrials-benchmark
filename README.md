# CKG ClinicalTrials.gov Benchmark

<p align="center">
  <a href="INTEGRITY_V3_REPORT.md"><img alt="Status: integrity v3" src="https://img.shields.io/badge/status-integrity--v3%20hardened-0f766e"></a>
  <a href="REPRODUCE.md"><img alt="Reproducible benchmark" src="https://img.shields.io/badge/benchmark-reproducible-2563eb"></a>
  <a href="REVIEW.md"><img alt="Review protocol included" src="https://img.shields.io/badge/review-protocol%20included-7c3aed"></a>
  <img alt="Updated: 2026-09-09" src="https://img.shields.io/badge/updated-2026--09--09-16a34a">
  <img alt="Domains: 12" src="https://img.shields.io/badge/domains-12-0891b2">
  <img alt="Studies processed: 5,587" src="https://img.shields.io/badge/studies-5%2C587-0284c7">
  <img alt="Graph: 2,160 nodes and 2,716 edges" src="https://img.shields.io/badge/graph-2%2C160%20nodes%20%7C%202%2C716%20edges-0d9488">
  <img alt="Questions: 240 matched, 2,064 generated" src="https://img.shields.io/badge/questions-240%20matched%20%7C%202%2C064%20generated-4f46e5">
  <img alt="MCP companion recommended" src="https://img.shields.io/badge/MCP-companion%20recommended-111827">
  <a href="LICENSE"><img alt="Code license: MIT" src="https://img.shields.io/badge/code%20license-MIT-111827"></a>
  <a href="LICENSES/DATA.md"><img alt="Data license: CC BY 4.0" src="https://img.shields.io/badge/data%20license-CC%20BY%204.0-f59e0b"></a>
</p>

This repository evaluates a Compressed Knowledge Graph (CKG) as a
relationship-aware retrieval and context layer over structural questions derived
from ClinicalTrials.gov records. It is a standalone project by Daniel Yarmoluk.
It does not modify, extend, or share authorship with the separate
Yarmoluk-McCreary CKG benchmark paper.

## At a Glance

| Item | Current value |
| --- | --- |
| Primary result | Integrity-v3 hardened evaluation |
| Corpus source | Public ClinicalTrials.gov records, normalized and frozen locally |
| Workload | Relationship-dependent retrieval and structural questions |
| Compared systems | Annotation-blind CKG, configured raw-prose RAG, same-model no context, question echo |
| Best-supported claim | Declared graph structure improved relationship-aware retrieval and recovered the benchmark's generated relationships with fewer model tokens than this RAG configuration |
| Not claimed | Clinical correctness, medical safety, independent replication, or universal superiority over every RAG/GraphRAG design |

## What You Can Do With This Repo

- Reproduce the integrity-v3 verification without paid model calls.
- Inspect the frozen graphs, normalized source records, query set, raw outputs,
  aggregates, and run manifest.
- Compare relationship-aware CKG retrieval against the included raw-prose
  vector-RAG baseline.
- Replace the public ClinicalTrials.gov corpus with an approved internal corpus
  and rerun the same evaluation pattern.
- Use the [MCP companion design](MCP.md) as the next step for exposing the
  benchmark to agent clients.

## How to Use This

Use this repository in three layers: inspect the published result, verify the
frozen artifacts, then adapt the evaluation pattern to a domain where
relationships matter.

| User | Start here | What to do |
| --- | --- | --- |
| Technical reviewer | [`INTEGRITY_V3_REPORT.md`](INTEGRITY_V3_REPORT.md) | Read the limitations first, then run the verifier against the frozen artifacts. |
| Life sciences team | [`benchmark/manifest.json`](benchmark/manifest.json) and [`sources/README.md`](sources/README.md) | Inspect the domain structure, relationship types, and provenance model before mapping an internal corpus. |
| RAG or platform team | [`evaluation/`](evaluation/) | Compare the included CKG, raw-prose RAG, no-context, and question-echo controls, then substitute your own retrieval stack. |
| Semantic-layer team | [`benchmark/domains/`](benchmark/domains/) | Treat the graphs as an example of metric-like relationships: entity, dependency, path, aggregate, and cross-concept queries. |
| Agent/MCP builder | [`MCP.md`](MCP.md) | Expose the frozen benchmark through read-only tools so agents can inspect domains, relationships, outputs, and limitations. |

To verify the current result locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
python evaluation/test_integrity_v3.py
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python evaluation/verify_integrity_v3.py
```

To adapt the benchmark for an internal life sciences or semantic-layer use
case, keep the evaluation shape but replace the corpus and schema:

1. Choose a bounded corpus: protocols, trial records, medical-affairs claims,
   regulatory references, metric definitions, or lineage metadata.
2. Define the relationships the organization needs agents to recover.
3. Build a frozen CKG and a matching prose corpus from the same approved
   sources.
4. Freeze a representative question set across entity, dependency, path,
   aggregate, and cross-concept tasks.
5. Run CKG, your RAG stack, and no-context controls with the same model and
   scoring rules.
6. Have subject-matter reviewers inspect misses before using the workflow in
   production.

## Hardened Headline Result

Integrity-v3 uses annotation-blind CKG retrieval, identical retrieval queries
for CKG and RAG, equal sampling across all five query types, structural scoring,
all nine taxonomy categories, and a same-model no-context control. On 240
matched questions across 12 therapeutic domains:

| System | Structural F1 | Target coverage | Mean tokens/query | API cost |
| --- | ---: | ---: | ---: | ---: |
| **Annotation-blind CKG** | **1.000000** | **1.000000** | **447.863** | **$0.174087** |
| Configured raw-prose RAG | 0.143044 | 0.304143 | 3,207.550 | $0.831588 |
| Same-model no context | 0.116667 | 0.000000 | 246.400 | $0.093252 |
| Question echo | 0.000000 | 0.000000 | 0 | $0 |

Target coverage is exact answer-label presence in retrieved context. It is a
diagnostic, not a measure of factual entailment or relationship support.
All model responses contained an extractable JSON object; strict whole-response
JSON compliance was 100.00% for CKG, 51.67% for RAG, and 54.58% for no context.

Against the configured vanilla RAG baseline, CKG produced:

- **6.99x the structural F1**
- **RAG used 7.16x as many model tokens per query**
- **RAG cost 4.78x as much**
- **208 wins, zero losses, and 32 ties**

The no-context model scored 0.116667, including 0.562500 on T1 entity taxonomy.
This shows that entity names alone reveal part of T1 and that RAG's 0.143044
score should not be interpreted entirely as retrieval lift.

The first run and integrity-v2 remain frozen for auditability but are
superseded. Read the [hardened report](INTEGRITY_V3_REPORT.md) before using any
benchmark claim.

## What This Supports

The hardened result shows that annotation-blind, relationship-aware retrieval
recovers the structure declared in these graphs more accurately and efficiently
than this raw-prose vector-RAG configuration.

It is not evidence that CKG universally replaces RAG or that the generated graph
is clinically correct. Questions and gold answers are graph-generated, and the
RAG corpus does not explicitly encode all graph edges or taxonomy labels.

## What a Life Sciences Team Can Do With This

For a life sciences company, this repository is a reproducible scaffold for
testing whether declared semantic structure should sit in front of an agent,
analyst, or knowledge workflow. The same pattern can be adapted to compare a CKG
against an internal RAG stack on questions the organization already needs to
answer:

- Trial intelligence: condition -> intervention -> endpoint -> eligibility ->
  phase -> sponsor
- Portfolio and competitive maps: asset -> indication -> mechanism -> trial ->
  comparator -> status
- Evidence traceability: claim -> source record -> endpoint -> population ->
  limitation
- Protocol and operations context: study design -> inclusion criteria ->
  exclusion criteria -> site constraint -> measurable outcome
- Regulatory and medical affairs review: claim -> evidence -> citation ->
  approval boundary -> review owner
- Semantic layers: metric -> dimension -> cohort -> join path -> source table ->
  lineage -> access policy

The practical extension is not to replace every document with a graph. It is to
encode the relationship-dependent core, expose it through agent-readable tools,
and keep RAG for fuzzy discovery and long-tail source lookup.

A production evaluation would swap the public ClinicalTrials.gov corpus for an
approved internal corpus, define the relationships the business cares about,
freeze a representative question set, run CKG/RAG/no-context controls, and have
subject-matter reviewers inspect the misses before any workflow is used
operationally.

## MCP Companion

Yes: this benchmark should have a read-only MCP companion. The repository should
remain the system of record, while the MCP server gives agents a clean interface
for traversing the frozen graphs, inspecting results, and explaining scope.

Recommended first tools:

- `list_domains`
- `search_concepts`
- `get_relationship_context`
- `compare_system_outputs`
- `summarize_benchmark_result`
- `explain_scope_and_limitations`

The MCP should not provide clinical advice, live ClinicalTrials.gov lookup,
write actions, payment actions, or private data access. Start with local stdio
for reviewers, then add Streamable HTTP only when authentication and hosting are
ready. See [MCP.md](MCP.md) for the companion design.

## Evaluation Surface

- 12 therapeutic domains
- 5,587 ClinicalTrials.gov studies processed
- 2,160 graph nodes and 2,716 declared edges
- 2,064 deterministic benchmark questions
- Five query classes: entity, dependency, path, aggregate, and cross-concept
- Same answer model, structured output prompt, and relation-aware scorer
- 48 questions from each T1-T5 class; all nine taxonomy categories represented
- Question-only CKG retrieval with no IDs, paths, taxonomy tags, or answer keys
- Identical original query text used for CKG and RAG retrieval
- Same-model empty-context control
- Deterministic question-echo control
- Vanilla RAG: all-MiniLM-L6-v2 embeddings, FAISS, 512-token chunks, 50-token
  overlap, top-5 retrieval

Read the [hardened report](INTEGRITY_V3_REPORT.md), inspect the
[machine-readable integrity-v3 results](results/integrity-v3/), or follow the
[reproduction instructions](REPRODUCE.md).

## Repository Contents

| Path | Contents |
| --- | --- |
| `benchmark/` | Frozen graph CSVs, source provenance, query sets, and domain manifest |
| `corpus/` | Frozen prose representation used by the RAG baseline |
| `sources/` | Frozen normalized source records and provenance disclosure |
| `evaluation/` | Original harnesses plus integrity-v2/v3 runners, scorers, tests, and verifiers |
| `results/integrity-v3/` | Current raw outputs, aggregate, and run manifest |
| `results/integrity-v2/` | Superseded corrected run retained as an audit trail |
| `results/aggregate/` | Superseded first-run summaries retained as an audit trail |
| `results/raw/full-ckg/` | All 2,064 full CKG outputs |
| `results/raw/paired/` | Exact 240-query CKG, RAG, and no-context outputs |
| `MCP.md` | Proposed read-only MCP companion for agent access |
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
