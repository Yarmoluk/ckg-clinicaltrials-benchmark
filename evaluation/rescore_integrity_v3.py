#!/usr/bin/env python3
"""Recompute integrity-v3 scores from frozen model outputs without API calls."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import ckg_harness
import integrity_v3_eval


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "integrity-v3")
    args = parser.parse_args()

    systems = sorted(path.name for path in (args.results / "raw").iterdir() if path.is_dir())
    rescored: dict[str, list[dict]] = {}
    graph_cache = {}
    for system in systems:
        rows = integrity_v3_eval.load_result_rows(args.results / "raw" / system)
        for row in rows:
            domain = row["domain"]
            if domain not in graph_cache:
                graph_cache[domain] = ckg_harness.load_graph(
                    ROOT / "benchmark" / "domains" / domain / "learning-graph.csv"
                )
            concepts = graph_cache[domain]
            row.pop("valid_json", None)
            row.update(integrity_v3_eval.relation_score(row, row["predicted_answer"], concepts))
            row["token_f1_diagnostic"] = ckg_harness.token_f1(
                row["predicted_answer"], row.get("ground_truth", [])
            )["f1"]
            row["context_target_coverage"] = (
                integrity_v3_eval.context_target_coverage(row, row["retrieved_context"], concepts)
                if row.get("retrieved_context") else 0.0
            )
        rescored[system] = rows
        integrity_v3_eval.write_rows(args.results, system, rows)

    aggregate = {
        "run_id": next(iter(rescored.values()))[0]["run_id"],
        "systems": {name: integrity_v3_eval.summarize(rows) for name, rows in rescored.items()},
    }
    if "ckg" in rescored and "rag" in rescored:
        aggregate["ckg_vs_rag"] = integrity_v3_eval.comparison(rescored["ckg"], rescored["rag"])
    if "ckg" in rescored and "question_echo" in rescored:
        aggregate["ckg_vs_question_echo"] = integrity_v3_eval.comparison(
            rescored["ckg"], rescored["question_echo"]
        )
    if "ckg" in rescored and "no_context" in rescored:
        aggregate["ckg_vs_no_context"] = integrity_v3_eval.comparison(
            rescored["ckg"], rescored["no_context"]
        )
    (args.results / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    manifest_path = args.results / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scoring_git_commit"] = integrity_v3_eval.git_commit()
    manifest["rescored_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["trial_identity_rule"] = "NCT registry identifier; title text is descriptive"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
