#!/usr/bin/env python3
"""Verify integrity-v2 raw outputs, retrieval, manifests, and aggregates."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import ckg_harness
import integrity_eval


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "integrity-v2"
TOLERANCE = 1e-9


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


def main() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    aggregate = json.loads((RESULTS / "aggregate.json").read_text())
    systems = ("ckg", "rag", "question_echo")
    rows = {
        system: integrity_eval.load_result_rows(RESULTS / "raw" / system)
        for system in systems
    }

    expected_ids = set(manifest["sample"]["selected_query_ids"])
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
        if Counter(row["type"] for row in system_rows) != Counter({name: 48 for name in integrity_eval.QUERY_TYPES}):
            raise AssertionError(f"Unbalanced query types in {system}")

    t4_categories = {row["taxonomy_id"] for row in rows["ckg"] if row["type"] == "T4_aggregate"}
    if len(t4_categories) < 8:
        raise AssertionError(f"Insufficient T4 taxonomy diversity: {sorted(t4_categories)}")

    graph_cache = {}
    for system, system_rows in rows.items():
        for row in system_rows:
            domain = row["domain"]
            if domain not in graph_cache:
                graph_cache[domain] = ckg_harness.load_graph(
                    ROOT / "benchmark" / "domains" / domain / "learning-graph.csv"
                )
            concepts = graph_cache[domain]
            rescored = integrity_eval.relation_score(row, row["predicted_answer"], concepts)
            for key in (
                "structural_precision", "structural_recall", "structural_f1",
                "exact_match", "valid_json", "expected_structural_answer",
                "parsed_structural_answer",
            ):
                if row[key] != rescored[key]:
                    raise AssertionError(f"{system}:{row['id']} stale {key}")
            token_f1 = ckg_harness.token_f1(row["predicted_answer"], row["ground_truth"])["f1"]
            assert_close(f"{system}:{row['id']}.token_f1", row["token_f1_diagnostic"], token_f1)
            expected_cost = (
                row["prompt_tokens"] * integrity_eval.INPUT_PRICE
                + row["completion_tokens"] * integrity_eval.OUTPUT_PRICE
            )
            assert_close(f"{system}:{row['id']}.cost", row["cost_usd"], round(expected_cost, 8))
            if row["total_tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
                raise AssertionError(f"{system}:{row['id']} token total mismatch")
            if row["run_id"] != manifest["run_id"] or row["harness_version"] != integrity_eval.HARNESS_VERSION:
                raise AssertionError(f"{system}:{row['id']} run metadata mismatch")

            if system == "ckg":
                context, _ = ckg_harness.retrieve_honest(concepts, integrity_eval.public_query(row))
                if row["retrieved_context"] != context:
                    raise AssertionError(f"{row['id']} does not match annotation-blind retrieval")
            elif system == "question_echo":
                if row["predicted_answer"] != row["evaluation_query"] or row["structural_f1"] != 0.0:
                    raise AssertionError(f"{row['id']} invalid question-echo control")

    input_paths = {
        "graphs": list((ROOT / "benchmark" / "domains").glob("ct-*/*.csv")),
        "queries": list((ROOT / "benchmark" / "queries").glob("queries_ct-*.jsonl")),
        "corpus_markdown": list((ROOT / "corpus").glob("ct-*/docs/**/*.md")),
        "normalized_source_snapshots": list((ROOT / "sources" / "frozen-normalized").glob("ct-*/*.json")),
    }
    for name, paths in input_paths.items():
        assert_nested(f"manifest.input_trees.{name}", integrity_eval.tree_hash(paths), manifest["input_trees"][name])

    actual_systems = {name: integrity_eval.summarize(system_rows) for name, system_rows in rows.items()}
    assert_nested("aggregate.systems", actual_systems, aggregate["systems"])
    assert_nested("aggregate.ckg_vs_rag", integrity_eval.comparison(rows["ckg"], rows["rag"]), aggregate["ckg_vs_rag"])
    assert_nested(
        "aggregate.ckg_vs_question_echo",
        integrity_eval.comparison(rows["ckg"], rows["question_echo"]),
        aggregate["ckg_vs_question_echo"],
    )

    ckg = actual_systems["ckg"]
    rag = actual_systems["rag"]
    print("PASS: integrity-v2 raw outputs and aggregates verified")
    print(f"paired IDs: {len(expected_ids)}; type counts: 48 each; T4 categories: {len(t4_categories)}")
    print(f"CKG structural-F1={ckg['structural_f1']:.6f}; RAG={rag['structural_f1']:.6f}")
    print(f"CKG evidence recall={ckg['evidence_recall']:.6f}; RAG={rag['evidence_recall']:.6f}")
    print(f"CKG tokens/query={ckg['mean_tokens']:.3f}; RAG={rag['mean_tokens']:.3f}")
    print(f"CKG cost=${ckg['cost_usd']:.6f}; RAG=${rag['cost_usd']:.6f}")


if __name__ == "__main__":
    main()
