#!/usr/bin/env python3
"""Verify integrity-v3 inputs, retrieval contexts, rows, and aggregates."""

from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
from pathlib import Path

import ckg_harness
import integrity_v3_eval
import rag_harness


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "integrity-v3"
TOLERANCE = 1e-9
SYSTEMS = ("ckg", "rag", "no_context", "question_echo")


def assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_nested(label: str, actual: dict, expected: dict) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: key mismatch {set(actual) ^ set(expected)}")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            assert_nested(f"{label}.{key}", actual_value, expected_value)
        elif isinstance(expected_value, float):
            assert_close(f"{label}.{key}", float(actual_value), expected_value)
        elif actual_value != expected_value:
            raise AssertionError(f"{label}.{key}: expected {expected_value}, got {actual_value}")


def validate_gold(query: dict, concepts: dict[int, ckg_harness.Concept]) -> None:
    """Verify that every saved gold structure is actually declared by the graph."""
    query_type = query["type"]
    if query_type == "T1_entity":
        concept = concepts[int(query["concept_id"])]
        if concept.taxonomy_id != query["ground_truth"][-1]:
            raise AssertionError(f"{query['id']} T1 gold disagrees with graph")
    elif query_type == "T2_dependency":
        concept = concepts[int(query["concept_id"])]
        actual = [concepts[dependency].label for dependency in concept.dependencies]
        if integrity_v3_eval.set_f1(actual, query["ground_truth"])["exact"] is not True:
            raise AssertionError(f"{query['id']} T2 gold disagrees with graph")
    elif query_type == "T3_path":
        path_ids = [int(value) for value in query["path_ids"]]
        actual = [concepts[value].label for value in path_ids]
        if [integrity_v3_eval.canonical_label(value) for value in actual] != [
            integrity_v3_eval.canonical_label(value) for value in query["ground_truth"]
        ]:
            raise AssertionError(f"{query['id']} T3 labels disagree with graph")
        for child, parent in zip(path_ids, path_ids[1:]):
            if parent not in concepts[child].dependencies:
                raise AssertionError(f"{query['id']} T3 edge is absent from graph")
    elif query_type == "T4_aggregate":
        actual = [
            concept.label for concept in concepts.values()
            if concept.taxonomy_id == query["taxonomy_id"]
        ]
        if integrity_v3_eval.set_f1(actual, query["ground_truth"])["exact"] is not True:
            raise AssertionError(f"{query['id']} T4 gold disagrees with graph")
    elif query_type == "T5_cross_concept":
        source_id = int(query["concept_id_a"])
        target_id = int(query["concept_id_b"])
        if target_id not in concepts[source_id].dependencies:
            raise AssertionError(f"{query['id']} T5 edge is absent from graph")
        actual = [concepts[source_id].label, concepts[target_id].label]
        if [integrity_v3_eval.canonical_label(value) for value in actual] != [
            integrity_v3_eval.canonical_label(value) for value in query["ground_truth"]
        ]:
            raise AssertionError(f"{query['id']} T5 labels disagree with graph")


def build_fresh_rag(domain: str, embed_model):
    docs = rag_harness.load_corpus_docs(domain)
    chunks = rag_harness.chunk_documents(docs)
    vectors = rag_harness.embed_texts([chunk["text"] for chunk in chunks], embed_model)
    index = rag_harness.faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def format_rag_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks
    )


def main() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    aggregate = json.loads((RESULTS / "aggregate.json").read_text())
    if manifest["harness_version"] != integrity_v3_eval.HARNESS_VERSION:
        raise AssertionError("Harness version mismatch")
    if manifest["model"] != integrity_v3_eval.MODEL:
        raise AssertionError("Manifest model mismatch")
    if manifest["systems"] != list(SYSTEMS):
        raise AssertionError("Unexpected system list or execution order")
    prompt_hash = hashlib.sha256(integrity_v3_eval.SYSTEM_PROMPT.encode()).hexdigest()
    if manifest["system_prompt_sha256"] != prompt_hash:
        raise AssertionError("System prompt hash mismatch")

    domains = manifest["sample"]["domain_ids"]
    selected = {
        domain: integrity_v3_eval.balanced_sample(
            integrity_v3_eval.load_queries(domain),
            manifest["sample"]["per_type_per_domain"],
            manifest["seed"],
            index,
        )
        for index, domain in enumerate(domains)
    }
    selected_queries = [query for domain in domains for query in selected[domain]]
    selected_ids = [query["id"] for query in selected_queries]
    if selected_ids != manifest["sample"]["selected_query_ids"]:
        raise AssertionError("Manifest sample cannot be regenerated from seed")
    source_queries = {query["id"]: query for query in selected_queries}

    systems = SYSTEMS
    rows = {
        system: integrity_v3_eval.load_result_rows(RESULTS / "raw" / system)
        for system in systems
    }

    expected_ids = set(selected_ids)
    if len(expected_ids) != 240 or manifest["sample"]["total"] != 240:
        raise AssertionError("Manifest must contain 240 unique selected query IDs")

    maps = {}
    for system, system_rows in rows.items():
        maps[system] = {row["id"]: row for row in system_rows}
        if len(maps[system]) != len(system_rows):
            raise AssertionError(f"Duplicate query IDs in {system}")
        if set(maps[system]) != expected_ids:
            raise AssertionError(f"Selected query-ID mismatch in {system}")
        if len(system_rows) != 240:
            raise AssertionError(f"Expected 240 {system} rows")
        if Counter(row["type"] for row in system_rows) != Counter({name: 48 for name in integrity_v3_eval.QUERY_TYPES}):
            raise AssertionError(f"Unbalanced query types in {system}")

    t4_categories = {row["taxonomy_id"] for row in rows["ckg"] if row["type"] == "T4_aggregate"}
    if t4_categories != set(integrity_v3_eval.TAXONOMY_ALIASES):
        raise AssertionError(f"Incomplete T4 taxonomy diversity: {sorted(t4_categories)}")

    graph_cache = {}
    for system, system_rows in rows.items():
        for row in system_rows:
            original = source_queries[row["id"]]
            for key, value in original.items():
                if row.get(key) != value:
                    raise AssertionError(f"{system}:{row['id']} changed source annotation {key}")
            domain = row["domain"]
            if domain not in graph_cache:
                graph_cache[domain] = ckg_harness.load_graph(
                    ROOT / "benchmark" / "domains" / domain / "learning-graph.csv"
                )
            concepts = graph_cache[domain]
            validate_gold(original, concepts)
            expected_question = integrity_v3_eval.evaluation_question(original, concepts)
            if row["evaluation_query"] != expected_question:
                raise AssertionError(f"{system}:{row['id']} generation question mismatch")
            if row["retrieval_query"] != original["query"]:
                raise AssertionError(f"{system}:{row['id']} retrieval question mismatch")
            rescored = integrity_v3_eval.relation_score(row, row["predicted_answer"], concepts)
            for key in (
                "structural_precision", "structural_recall", "structural_f1",
                "exact_match", "valid_json", "expected_structural_answer",
                "parsed_structural_answer",
            ):
                if row[key] != rescored[key]:
                    raise AssertionError(f"{system}:{row['id']} stale {key}")
            token_f1 = ckg_harness.token_f1(row["predicted_answer"], row["ground_truth"])["f1"]
            assert_close(f"{system}:{row['id']}.token_f1", row["token_f1_diagnostic"], token_f1)
            coverage = (
                integrity_v3_eval.context_target_coverage(row, row["retrieved_context"], concepts)
                if row["retrieved_context"] else 0.0
            )
            assert_close(
                f"{system}:{row['id']}.context_target_coverage",
                row["context_target_coverage"],
                coverage,
            )
            expected_cost = (
                row["prompt_tokens"] * integrity_v3_eval.INPUT_PRICE
                + row["completion_tokens"] * integrity_v3_eval.OUTPUT_PRICE
            )
            assert_close(f"{system}:{row['id']}.cost", row["cost_usd"], round(expected_cost, 8))
            if row["total_tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
                raise AssertionError(f"{system}:{row['id']} token total mismatch")
            if row["run_id"] != manifest["run_id"] or row["harness_version"] != integrity_v3_eval.HARNESS_VERSION:
                raise AssertionError(f"{system}:{row['id']} run metadata mismatch")

            if system in {"ckg", "rag", "no_context"}:
                if row["model"] != manifest["model"] or row["backend"] != "anthropic":
                    raise AssertionError(f"{system}:{row['id']} model/backend mismatch")
            elif row["model"] != "deterministic-question-echo" or row["backend"] != "deterministic":
                raise AssertionError(f"{system}:{row['id']} control metadata mismatch")

            if system == "ckg":
                context, _ = ckg_harness.retrieve_honest(concepts, integrity_v3_eval.public_query(row))
                if row["retrieved_context"] != context:
                    raise AssertionError(f"{row['id']} does not match annotation-blind retrieval")
            elif system == "question_echo":
                if row["predicted_answer"] != row["evaluation_query"] or row["structural_f1"] != 0.0:
                    raise AssertionError(f"{row['id']} invalid question-echo control")
            elif system == "no_context":
                if row["retrieved_context"] or row["retrieval_mode"] != "none":
                    raise AssertionError(f"{row['id']} invalid no-context control")

    for query_id in selected_ids:
        generation_questions = {maps[system][query_id]["evaluation_query"] for system in systems}
        if len(generation_questions) != 1:
            raise AssertionError(f"{query_id} generation-query parity failure")
        retrieval_questions = {
            maps[system][query_id]["retrieval_query"] for system in ("ckg", "rag")
        }
        if len(retrieval_questions) != 1:
            raise AssertionError(f"{query_id} retrieval-query parity failure")

    embed_model = rag_harness.get_embed_model()
    rag_cache = {domain: build_fresh_rag(domain, embed_model) for domain in domains}
    for row in rows["rag"]:
        index, chunks = rag_cache[row["domain"]]
        retrieved = rag_harness.retrieve_chunks(
            row["retrieval_query"], index, chunks, embed_model
        )
        if row["retrieved_context"] != format_rag_context(retrieved):
            raise AssertionError(f"{row['id']} RAG context reconstruction failed")

    input_paths = {
        "graphs": list((ROOT / "benchmark" / "domains").glob("ct-*/*.csv")),
        "queries": list((ROOT / "benchmark" / "queries").glob("queries_ct-*.jsonl")),
        "corpus_markdown": list((ROOT / "corpus").glob("ct-*/docs/**/*.md")),
        "normalized_source_snapshots": list((ROOT / "sources" / "frozen-normalized").glob("ct-*/*.json")),
    }
    for name, paths in input_paths.items():
        assert_nested(f"manifest.input_trees.{name}", integrity_v3_eval.tree_hash(paths), manifest["input_trees"][name])

    actual_systems = {name: integrity_v3_eval.summarize(system_rows) for name, system_rows in rows.items()}
    assert_nested("aggregate.systems", actual_systems, aggregate["systems"])
    assert_nested("aggregate.ckg_vs_rag", integrity_v3_eval.comparison(rows["ckg"], rows["rag"]), aggregate["ckg_vs_rag"])
    assert_nested(
        "aggregate.ckg_vs_question_echo",
        integrity_v3_eval.comparison(rows["ckg"], rows["question_echo"]),
        aggregate["ckg_vs_question_echo"],
    )
    assert_nested(
        "aggregate.ckg_vs_no_context",
        integrity_v3_eval.comparison(rows["ckg"], rows["no_context"]),
        aggregate["ckg_vs_no_context"],
    )

    ckg = actual_systems["ckg"]
    rag = actual_systems["rag"]
    no_context = actual_systems["no_context"]
    print("PASS: integrity-v3 artifacts, gold edges, and both retrieval paths verified")
    print(f"paired IDs: {len(expected_ids)}; type counts: 48 each; T4 categories: {len(t4_categories)}")
    print(
        f"structural-F1: CKG={ckg['structural_f1']:.6f}; "
        f"RAG={rag['structural_f1']:.6f}; no-context={no_context['structural_f1']:.6f}"
    )
    print(
        f"context-target coverage: CKG={ckg['context_target_coverage']:.6f}; "
        f"RAG={rag['context_target_coverage']:.6f}"
    )
    print(f"CKG tokens/query={ckg['mean_tokens']:.3f}; RAG={rag['mean_tokens']:.3f}")
    print(
        f"cost: CKG=${ckg['cost_usd']:.6f}; RAG=${rag['cost_usd']:.6f}; "
        f"no-context=${no_context['cost_usd']:.6f}"
    )


if __name__ == "__main__":
    main()
