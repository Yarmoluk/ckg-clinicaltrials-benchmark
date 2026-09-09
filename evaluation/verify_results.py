#!/usr/bin/env python3
"""Verify published aggregate metrics against frozen raw JSONL outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICE = 1.0 / 1_000_000
OUTPUT_PRICE = 5.0 / 1_000_000
TOLERANCE = 1e-9


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summarize(rows: list[dict], recalculate_cost: bool = False) -> dict:
    count = len(rows)
    if not count:
        raise AssertionError("No result rows found")
    cost = sum(float(row.get("cost_usd", 0.0)) for row in rows)
    if recalculate_cost:
        cost = sum(
            int(row.get("prompt_tokens", 0)) * INPUT_PRICE
            + int(row.get("completion_tokens", 0)) * OUTPUT_PRICE
            for row in rows
        )
    return {
        "n": count,
        "macro_f1": sum(float(row["f1"]) for row in rows) / count,
        "mean_tokens": sum(int(row["total_tokens"]) for row in rows) / count,
        "cost_usd": cost,
    }


def assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_summary(label: str, actual: dict, expected: dict) -> None:
    if actual["n"] != expected["n"]:
        raise AssertionError(f"{label}.n: expected {expected['n']}, got {actual['n']}")
    for key in ("macro_f1", "mean_tokens", "cost_usd"):
        assert_close(f"{label}.{key}", actual[key], float(expected[key]))


def map_by_id(rows: list[dict]) -> dict[str, dict]:
    mapped = {row["id"]: row for row in rows}
    if len(mapped) != len(rows):
        raise AssertionError("Duplicate query IDs found")
    return mapped


def main() -> None:
    full_expected = load_json(ROOT / "results/aggregate/full-ckg.json")
    paired_expected = load_json(ROOT / "results/aggregate/paired-sample.json")

    full_rows = load_jsonl(ROOT / "results/raw/full-ckg")
    full_actual = summarize(full_rows)
    assert_summary(
        "full_ckg",
        full_actual,
        {
            "n": full_expected["n_queries"],
            "macro_f1": full_expected["macro_f1"],
            "mean_tokens": full_expected["mean_tokens"],
            "cost_usd": full_expected["total_cost_usd"],
        },
    )

    paired_rows = {
        "ckg": load_jsonl(ROOT / "results/raw/paired/ckg"),
        "rag": load_jsonl(ROOT / "results/raw/paired/rag"),
        "baseline": load_jsonl(ROOT / "results/raw/paired/baseline"),
    }
    paired_actual = {
        name: summarize(rows, recalculate_cost=(name == "baseline"))
        for name, rows in paired_rows.items()
    }
    for name, actual in paired_actual.items():
        expected = paired_expected["systems"][name]
        assert_summary(
            f"paired_{name}",
            actual,
            {
                "n": expected["n"],
                "macro_f1": expected["macro_f1"],
                "mean_tokens": expected["mean_tokens"],
                "cost_usd": expected["cost_usd"],
            },
        )

    maps = {name: map_by_id(rows) for name, rows in paired_rows.items()}
    query_ids = set(maps["ckg"])
    for name in ("rag", "baseline"):
        if set(maps[name]) != query_ids:
            raise AssertionError(f"Paired query-ID mismatch for {name}")

    wins = losses = ties = 0
    for query_id in query_ids:
        ckg_f1 = float(maps["ckg"][query_id]["f1"])
        rag_f1 = float(maps["rag"][query_id]["f1"])
        if ckg_f1 > rag_f1:
            wins += 1
        elif ckg_f1 < rag_f1:
            losses += 1
        else:
            ties += 1

    expected_comparison = paired_expected["ckg_vs_rag"]
    observed = {"wins": wins, "losses": losses, "ties": ties}
    for key, actual in observed.items():
        expected = int(expected_comparison[key])
        if actual != expected:
            raise AssertionError(f"paired_ckg_vs_rag.{key}: expected {expected}, got {actual}")

    graph_files = sorted((ROOT / "benchmark/domains").glob("ct-*/learning-graph.csv"))
    query_files = sorted((ROOT / "benchmark/queries").glob("queries_ct-*.jsonl"))
    if len(graph_files) != 12 or len(query_files) != 12:
        raise AssertionError("Expected 12 graph files and 12 query files")

    node_count = 0
    edge_count = 0
    for path in graph_files:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                node_count += 1
                edge_count += len(
                    [value for value in row.get("Dependencies", "").split("|") if value.strip()]
                )
    query_count = sum(
        1
        for path in query_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (node_count, edge_count, query_count) != (2160, 2716, 2064):
        raise AssertionError(
            "Benchmark shape mismatch: expected nodes=2160, edges=2716, queries=2064; "
            f"got nodes={node_count}, edges={edge_count}, queries={query_count}"
        )

    print(
        f"full CKG: n={full_actual['n']}, "
        f"F1={full_actual['macro_f1']:.4f}, "
        f"tokens/query={full_actual['mean_tokens']:.1f}"
    )
    print(
        "paired: "
        + ", ".join(
            f"{name} F1={values['macro_f1']:.4f}"
            for name, values in paired_actual.items()
        )
    )
    print(f"CKG vs RAG: {wins} wins, {losses} losses, {ties} ties")
    print(f"benchmark: {node_count} nodes, {edge_count} edges, {query_count} queries")
    print("PASS: all published aggregate results match the frozen raw outputs")


if __name__ == "__main__":
    main()
