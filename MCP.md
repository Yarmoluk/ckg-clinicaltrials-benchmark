# MCP Companion Design

**Status:** Recommended next artifact; not deployed in this repository yet.

This benchmark is already structured for agent access. The frozen graph files,
query sets, raw outputs, aggregate results, and scope notices can be exposed
through a read-only Model Context Protocol server so reviewers and life sciences
teams can inspect the benchmark without manually browsing the repository.

The MCP companion should be a benchmark interface, not a medical assistant.

## Intended Users

- Technical reviewers validating the integrity-v3 result
- Life sciences teams evaluating CKG as an agent context layer
- Data and semantic-layer teams comparing graph traversal with RAG
- Internal Graphify.md demos where an agent needs to inspect the benchmark

## Recommended Tools

| Tool | Purpose |
| --- | --- |
| `list_domains` | Return the 12 therapeutic domains and graph sizes. |
| `search_concepts` | Find concept nodes by label, synonym, taxonomy, or source identifier. |
| `get_relationship_context` | Return the local typed edges around a concept or trial identifier. |
| `get_query` | Return one frozen benchmark question and its gold structural target. |
| `compare_system_outputs` | Show CKG, RAG, no-context, and question-echo outputs for one query. |
| `summarize_benchmark_result` | Return aggregate or per-domain integrity-v3 metrics. |
| `explain_scope_and_limitations` | Return the public scope notice and benchmark limitations. |

## Recommended Resources

| Resource | Backing path |
| --- | --- |
| `benchmark://manifest` | `benchmark/manifest.json` |
| `benchmark://domain/{domain}` | `benchmark/domains/{domain}/learning-graph.csv` |
| `benchmark://sources/{domain}` | `benchmark/domains/{domain}/sources.json` |
| `benchmark://queries/{domain}` | `benchmark/queries/queries_{domain}.jsonl` |
| `results://integrity-v3/aggregate` | `results/integrity-v3/aggregate.json` |
| `results://integrity-v3/manifest` | `results/integrity-v3/manifest.json` |

## Guardrails

- Read-only tools only.
- No clinical safety, treatment, diagnosis, enrollment, or care
  recommendations.
- No live ClinicalTrials.gov API calls in the first version.
- No write actions, user tracking, payment actions, or private data access.
- No use of `clientInfo` as a security control.
- Every answer should cite the local file or result artifact it used.

## Transport Plan

Start with a local stdio server for reviewers. It is simplest to run, keeps the
benchmark self-contained, and can use environment/config credentials if any are
ever needed.

Add Streamable HTTP only after the local interface is stable and authentication
is designed. Do not implement the deprecated SSE transport.

## Suggested Implementation

1. Add a small `mcp/` package that loads the frozen benchmark files.
2. Implement the seven read-only tools above.
3. Add `python -m mcp_server` or equivalent local run instructions.
4. Add tests that compare MCP tool output to the frozen CSV/JSON artifacts.
5. Add PostHog instrumentation only for hosted/demo usage, with query text
   hashing and no raw private data capture.

## Non-Goals

- This is not a clinical decision support system.
- This is not an independent ClinicalTrials.gov mirror.
- This is not a replacement for a sponsor's validated clinical, regulatory, or
  medical-affairs systems.
- This is not evidence that CKG universally replaces RAG.

The first MCP version should make the benchmark easier to inspect. A production
life sciences deployment would require an approved internal corpus, explicit
schema choices, reviewer signoff, access control, and domain-specific validation.
