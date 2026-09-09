#!/usr/bin/env python3
"""Rebuild and verify the frozen retrieval comparison without paid model calls."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import build_edge_text_corpus
import ckg_harness
import integrity_v3_eval
import retrieval_comparison_eval as comparison


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / comparison.HARNESS_VERSION
SYSTEMS = comparison.SYSTEMS
TOLERANCE = 1e-8


def assert_nested(label: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{label}: dictionary keys differ")
        for key, value in expected.items():
            assert_nested(f"{label}.{key}", actual[key], value)
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise AssertionError(f"{label}: expected {expected}, got {actual}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_retrieval_trace(
    label: str, actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    """Require identical rankings while allowing harmless CPU float drift."""
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: trace length mismatch")
    exact_fields = ("document_id", "metadata_filter", "bm25_rank", "dense_rank")
    for rank, (actual_row, expected_row) in enumerate(zip(actual, expected), start=1):
        for field in exact_fields:
            if actual_row[field] != expected_row[field]:
                raise AssertionError(f"{label}: rank {rank} changed {field}")
        if not math.isclose(
            actual_row["rrf_score"], expected_row["rrf_score"], abs_tol=1e-12
        ):
            raise AssertionError(f"{label}: rank {rank} changed RRF score")
        if not math.isclose(
            actual_row["reranker_score"],
            expected_row["reranker_score"],
            rel_tol=1e-5,
            abs_tol=1e-4,
        ):
            raise AssertionError(f"{label}: rank {rank} changed reranker score")


def verify_edge_text(edge_manifest: dict[str, Any]) -> None:
    expected_files = {
        metadata["document_file"] for metadata in edge_manifest["domains"].values()
    }
    actual_files = {path.name for path in comparison.EDGE_TEXT_ROOT.glob("*.jsonl")}
    if actual_files != expected_files:
        raise AssertionError("Edge-text directory contains missing or unmanifested JSONL files")
    total_nodes = total_edges = total_documents = 0
    tree_digest = hashlib.sha256()
    for domain, metadata in sorted(edge_manifest["domains"].items()):
        graph_path = ROOT / metadata["graph_path"]
        if build_edge_text_corpus.sha256_file(graph_path) != metadata["graph_sha256"]:
            raise AssertionError(f"{domain}: graph hash differs from edge manifest")
        concepts = ckg_harness.load_graph(graph_path)
        rebuilt = build_edge_text_corpus.build_documents(domain, concepts)
        path = comparison.EDGE_TEXT_ROOT / metadata["document_file"]
        expected_bytes = build_edge_text_corpus.serialize_documents(rebuilt).encode()
        if path.read_bytes() != expected_bytes:
            raise AssertionError(f"{domain}: frozen edge text cannot be reconstructed")
        file_hash = hashlib.sha256(expected_bytes).hexdigest()
        if file_hash != metadata["document_sha256"]:
            raise AssertionError(f"{domain}: edge-text file hash mismatch")
        nodes = sum(row["kind"] == "node" for row in rebuilt)
        edges = sum(row["kind"] == "edge" for row in rebuilt)
        if (nodes, edges, len(rebuilt)) != (
            metadata["nodes"], metadata["edges"], metadata["documents"]
        ):
            raise AssertionError(f"{domain}: edge-text document counts mismatch")
        for row in rebuilt:
            forbidden = {"ground_truth", "path_ids", "concept_id_a", "concept_id_b"}
            if forbidden & set(row):
                raise AssertionError(f"{row['document_id']}: answer annotation leaked")
        total_nodes += nodes
        total_edges += edges
        total_documents += len(rebuilt)
        tree_digest.update(domain.encode())
        tree_digest.update(b"\0")
        tree_digest.update(file_hash.encode())
        tree_digest.update(b"\n")
    expected_totals = {
        "domains": len(edge_manifest["domains"]),
        "nodes": total_nodes,
        "edges": total_edges,
        "documents": total_documents,
    }
    if edge_manifest["totals"] != expected_totals:
        raise AssertionError("Edge-text aggregate counts mismatch")
    if edge_manifest["tree_sha256"] != tree_digest.hexdigest():
        raise AssertionError("Edge-text tree hash mismatch")


def verify_row(
    system: str,
    row: dict[str, Any],
    query: dict[str, Any],
    concepts: dict[int, ckg_harness.Concept],
    manifest: dict[str, Any],
) -> None:
    if row.get("system") != system:
        raise AssertionError(f"{system}:{row['id']} system label mismatch")
    for key, value in query.items():
        if row.get(key) != value:
            raise AssertionError(f"{system}:{row['id']} changed source annotation {key}")
    if row["retrieval_query"] != query["query"]:
        raise AssertionError(f"{system}:{row['id']} retrieval question mismatch")
    expected_question = integrity_v3_eval.evaluation_question(query, concepts)
    if row["evaluation_query"] != expected_question:
        raise AssertionError(f"{system}:{row['id']} generation question mismatch")
    if row["model"] != manifest["model"] or row["backend"] != "anthropic":
        raise AssertionError(f"{system}:{row['id']} model/backend mismatch")
    if row["harness_version"] != comparison.HARNESS_VERSION:
        raise AssertionError(f"{system}:{row['id']} harness mismatch")
    if row["run_id"] != manifest["run_id"]:
        raise AssertionError(f"{system}:{row['id']} run mismatch")
    rescored = integrity_v3_eval.relation_score(query, row["predicted_answer"], concepts)
    for key in (
        "structural_precision",
        "structural_recall",
        "structural_f1",
        "exact_match",
        "extractable_json",
        "strict_json",
        "expected_structural_answer",
        "parsed_structural_answer",
    ):
        if row[key] != rescored[key]:
            raise AssertionError(f"{system}:{row['id']} stale {key}")
    coverage = integrity_v3_eval.context_target_coverage(
        query, row["retrieved_context"], concepts
    )
    if not math.isclose(row["context_target_coverage"], coverage, abs_tol=TOLERANCE):
        raise AssertionError(f"{system}:{row['id']} context coverage mismatch")
    token_f1 = ckg_harness.token_f1(
        row["predicted_answer"], query.get("ground_truth", [])
    )["f1"]
    if not math.isclose(row["token_f1_diagnostic"], token_f1, abs_tol=TOLERANCE):
        raise AssertionError(f"{system}:{row['id']} token diagnostic mismatch")
    if row["total_tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
        raise AssertionError(f"{system}:{row['id']} token total mismatch")
    expected_cost = round(
        row["prompt_tokens"] * integrity_v3_eval.INPUT_PRICE
        + row["completion_tokens"] * integrity_v3_eval.OUTPUT_PRICE,
        8,
    )
    if not math.isclose(row["cost_usd"], expected_cost, abs_tol=TOLERANCE):
        raise AssertionError(f"{system}:{row['id']} cost mismatch")


def main() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    aggregate = json.loads((RESULTS / "aggregate.json").read_text())
    if manifest["systems"] != list(SYSTEMS):
        raise AssertionError("Unexpected system list or order")
    if manifest["model"] != comparison.MODEL:
        raise AssertionError("Answer model mismatch")
    prompt_hash = hashlib.sha256(comparison.SYSTEM_PROMPT.encode()).hexdigest()
    if manifest["system_prompt_sha256"] != prompt_hash:
        raise AssertionError("System prompt mismatch")
    if manifest["input_trees"] != comparison.current_input_trees():
        raise AssertionError("Frozen input trees have drifted")
    if manifest["implementation_files"] != comparison.implementation_hashes():
        raise AssertionError("Frozen implementation files have drifted")

    v3_manifest_bytes = comparison.V3_MANIFEST.read_bytes()
    if hashlib.sha256(v3_manifest_bytes).hexdigest() != manifest["frozen_parent"]["manifest_sha256"]:
        raise AssertionError("Parent v3 manifest hash mismatch")
    v3_manifest, selected = comparison.load_frozen_selection()
    if manifest["sample"] != v3_manifest["sample"] or manifest["seed"] != v3_manifest["seed"]:
        raise AssertionError("Frozen sample or seed mismatch")

    edge_manifest_path = comparison.EDGE_TEXT_ROOT / "manifest.json"
    if integrity_v3_eval.sha256_file(edge_manifest_path) != manifest["edge_text"]["manifest_sha256"]:
        raise AssertionError("Edge-text manifest hash mismatch")
    edge_manifest = json.loads(edge_manifest_path.read_text())
    verify_edge_text(edge_manifest)
    if edge_manifest["tree_sha256"] != manifest["edge_text"]["tree_sha256"]:
        raise AssertionError("Edge-text tree differs from run manifest")

    rows = {
        system: integrity_v3_eval.load_result_rows(RESULTS / "raw" / system)
        for system in SYSTEMS
    }
    ordered_queries = [query for values in selected.values() for query in values]
    expected_ids = {query["id"] for query in ordered_queries}
    query_map = {query["id"]: query for query in ordered_queries}
    graph_cache = {
        domain: ckg_harness.load_graph(
            ROOT / "benchmark" / "domains" / domain / "learning-graph.csv"
        )
        for domain in selected
    }
    row_maps = {}
    for system, system_rows in rows.items():
        row_maps[system] = {row["id"]: row for row in system_rows}
        if len(system_rows) != 240 or set(row_maps[system]) != expected_ids:
            raise AssertionError(f"{system}: frozen query-ID mismatch")
        if Counter(row["type"] for row in system_rows) != Counter(
            {query_type: 48 for query_type in integrity_v3_eval.QUERY_TYPES}
        ):
            raise AssertionError(f"{system}: query types are not balanced")
        for row in system_rows:
            verify_row(
                system,
                row,
                query_map[row["id"]],
                graph_cache[row["domain"]],
                manifest,
            )

    v3_source_rows = comparison.load_v3_ckg_rows(v3_manifest, selected)
    for query_id, row in row_maps["ckg"].items():
        source = v3_source_rows[query_id]
        if row["source_row_sha256"] != comparison.stable_row_hash(source):
            raise AssertionError(f"ckg:{query_id} source row hash mismatch")
        for key in (
            "predicted_answer", "retrieved_context", "prompt_tokens",
            "completion_tokens", "total_tokens", "cost_usd", "structural_f1"
        ):
            if row[key] != source[key]:
                raise AssertionError(f"ckg:{query_id} does not reuse v3 {key}")
        if row["new_api_spend_usd"] != 0.0:
            raise AssertionError(f"ckg:{query_id} claims new API spend")
        if row["generation_status"] != "reused_integrity_v3_raw_output":
            raise AssertionError(f"ckg:{query_id} generation status mismatch")
        if row["retrieval_mode"] != "reused_integrity_v3_annotation_blind_ckg":
            raise AssertionError(f"ckg:{query_id} retrieval mode mismatch")

    retrieval = manifest["retrieval"]
    if not retrieval["dense"]["revision"] or not retrieval["reranker"]["revision"]:
        raise AssertionError("Retrieval model revisions must be pinned in the manifest")
    models = comparison.load_models(
        retrieval["dense"]["model"],
        retrieval["reranker"]["model"],
        retrieval["dense"]["revision"],
        retrieval["reranker"]["revision"],
    )
    for system, loader in (
        ("modern_rag", comparison.prose_documents),
        ("edge_text_rag", comparison.edge_text_documents),
    ):
        print(f"Rebuilding {system} indexes and traces")
        indexes = comparison.build_indexes(selected, loader, models)
        for count, query in enumerate(ordered_queries, start=1):
            row = row_maps[system][query["id"]]
            documents, trace = indexes[query["domain"]].retrieve(
                query["query"],
                candidate_k=retrieval["candidate_k"],
                top_k=retrieval["top_k"],
                rrf_k=retrieval["fusion"]["rrf_k"],
            )
            if row["retrieved_context"] != comparison.format_context(documents):
                raise AssertionError(f"{system}:{query['id']} context reconstruction failed")
            assert_retrieval_trace(
                f"{system}:{query['id']}", row["retrieval_trace"], trace
            )
            if row["new_api_spend_usd"] != row["cost_usd"]:
                raise AssertionError(f"{system}:{query['id']} new spend mismatch")
            if row["generation_status"] != "generated":
                raise AssertionError(f"{system}:{query['id']} generation status mismatch")
            expected_mode = (
                f"bm25_dense_rrf{retrieval['candidate_k']}_cross_encoder_"
                f"top{retrieval['top_k']}"
            )
            if row["retrieval_mode"] != expected_mode:
                raise AssertionError(f"{system}:{query['id']} retrieval mode mismatch")
            if count % 60 == 0:
                print(f"  {system}: {count}/{len(ordered_queries)}")

    for query_id, row in row_maps["router"].items():
        selected_system = comparison.ROUTER_MAP[row["type"]]
        source = row_maps[selected_system][query_id]
        if row["router_selected_system"] != selected_system:
            raise AssertionError(f"router:{query_id} selected wrong system")
        if row["router_rule_input"] != row["type"]:
            raise AssertionError(f"router:{query_id} used more than query type")
        if row["source_row_sha256"] != comparison.stable_row_hash(source):
            raise AssertionError(f"router:{query_id} source hash mismatch")
        for key in (
            "predicted_answer", "retrieved_context", "prompt_tokens",
            "completion_tokens", "total_tokens", "cost_usd", "structural_f1"
        ):
            if row[key] != source[key]:
                raise AssertionError(f"router:{query_id} changed selected {key}")
        if row["new_api_spend_usd"] != 0.0:
            raise AssertionError(f"router:{query_id} claims duplicate API spend")
        if row["generation_status"] != f"reused_{selected_system}_output":
            raise AssertionError(f"router:{query_id} generation status mismatch")
        if row["retrieval_mode"] != f"query_type_router_to_{selected_system}":
            raise AssertionError(f"router:{query_id} retrieval mode mismatch")

    actual_systems = {
        system: comparison.summarize(system_rows) for system, system_rows in rows.items()
    }
    assert_nested("aggregate.systems", actual_systems, aggregate["systems"])
    for right in ("modern_rag", "edge_text_rag", "router"):
        expected = integrity_v3_eval.comparison(rows["ckg"], rows[right])
        assert_nested(f"aggregate.ckg_vs_{right}", expected, aggregate[f"ckg_vs_{right}"])
    spend = round(
        sum(row["new_api_spend_usd"] for system in ("modern_rag", "edge_text_rag") for row in rows[system]),
        6,
    )
    if aggregate["actual_new_api_spend_usd"] != spend:
        raise AssertionError("Actual new API spend mismatch")

    print("PASS: frozen retrieval comparison reconstructed and rescored")
    print(f"paired IDs: {len(expected_ids)}; edge-text documents: {edge_manifest['totals']['documents']}")
    for system in SYSTEMS:
        summary = actual_systems[system]
        print(
            f"{system}: F1={summary['structural_f1']:.6f}; "
            f"coverage={summary['context_target_coverage']:.6f}; "
            f"tokens/query={summary['mean_tokens']:.3f}; cost=${summary['cost_usd']:.6f}"
        )
    print(f"new API spend: ${spend:.6f}")


if __name__ == "__main__":
    main()
