#!/usr/bin/env python3
"""Build deterministic node and typed-edge documents from frozen graph CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import ckg_harness


ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = ROOT / "benchmark" / "domains"
DEFAULT_OUTPUT = ROOT / "benchmark" / "edge-text"
GENERATOR_VERSION = "edge-text-v1"
NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nct_ids(*values: str) -> list[str]:
    return sorted({match.upper() for value in values for match in NCT_PATTERN.findall(value)})


def build_documents(
    domain: str, concepts: dict[int, ckg_harness.Concept]
) -> list[dict[str, Any]]:
    """Create one document per node and one per declared dependency edge."""
    dependents: defaultdict[int, list[int]] = defaultdict(list)
    for source in concepts.values():
        for target_id in source.dependencies:
            if target_id not in concepts:
                raise ValueError(f"{domain}: node {source.id} references missing node {target_id}")
            dependents[target_id].append(source.id)

    documents: list[dict[str, Any]] = []
    for node_id in sorted(concepts):
        node = concepts[node_id]
        prerequisites = [concepts[value] for value in node.dependencies]
        children = [concepts[value] for value in sorted(dependents[node_id])]
        ids = nct_ids(
            node.label,
            *(item.label for item in prerequisites),
            *(item.label for item in children),
        )
        prerequisite_text = "; ".join(
            f"{item.label} [{item.taxonomy_id}]" for item in prerequisites
        ) or "none"
        dependent_text = "; ".join(
            f"{item.label} [{item.taxonomy_id}]" for item in children
        ) or "none"
        lines = [
            "Document kind: graph node",
            f"Domain: {domain}",
            f"Node label: {node.label}",
            f"Node type: {node.taxonomy_id}",
            f"Direct prerequisites (outgoing depends_on targets): {prerequisite_text}",
            f"Direct dependents (incoming depends_on sources): {dependent_text}",
        ]
        if ids:
            lines.append(f"NCT/source identifiers present in this graph neighborhood: {', '.join(ids)}")
        documents.append(
            {
                "document_id": f"{domain}:node:{node.id}",
                "domain": domain,
                "kind": "node",
                "node_id": node.id,
                "label": node.label,
                "taxonomy_ids": [node.taxonomy_id],
                "nct_ids": ids,
                "text": "\n".join(lines),
            }
        )

    for source_id in sorted(concepts):
        source = concepts[source_id]
        for target_id in sorted(source.dependencies):
            target = concepts[target_id]
            ids = nct_ids(source.label, target.label)
            lines = [
                "Document kind: typed graph edge",
                f"Domain: {domain}",
                f"Typed edge: {source.label} depends_on {target.label}",
                f"Source node: {source.label} [{source.taxonomy_id}]",
                f"Target node: {target.label} [{target.taxonomy_id}]",
            ]
            if ids:
                lines.append(f"NCT/source identifiers present on this edge: {', '.join(ids)}")
            documents.append(
                {
                    "document_id": f"{domain}:edge:{source.id}:{target.id}",
                    "domain": domain,
                    "kind": "edge",
                    "source_node_id": source.id,
                    "target_node_id": target.id,
                    "label": f"{source.label} depends_on {target.label}",
                    "taxonomy_ids": sorted({source.taxonomy_id, target.taxonomy_id}),
                    "nct_ids": ids,
                    "text": "\n".join(lines),
                }
            )
    return documents


def serialize_documents(documents: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(item, sort_keys=True) + "\n" for item in documents)


def build_all(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    domains: dict[str, Any] = {}
    for graph_path in sorted(GRAPH_ROOT.glob("ct-*/learning-graph.csv")):
        domain = graph_path.parent.name
        concepts = ckg_harness.load_graph(graph_path)
        documents = build_documents(domain, concepts)
        output_path = output_dir / f"{domain}.jsonl"
        output_path.write_text(serialize_documents(documents), encoding="utf-8")
        domains[domain] = {
            "graph_path": str(graph_path.relative_to(ROOT)),
            "graph_sha256": sha256_file(graph_path),
            "document_file": output_path.name,
            "document_sha256": sha256_file(output_path),
            "nodes": sum(item["kind"] == "node" for item in documents),
            "edges": sum(item["kind"] == "edge" for item in documents),
            "documents": len(documents),
        }

    tree_digest = hashlib.sha256()
    for domain, metadata in sorted(domains.items()):
        tree_digest.update(domain.encode())
        tree_digest.update(b"\0")
        tree_digest.update(metadata["document_sha256"].encode())
        tree_digest.update(b"\n")
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "source": "existing benchmark/domains/*/learning-graph.csv files only",
        "document_schema": "one document per node plus one document per typed dependency edge",
        "domains": domains,
        "totals": {
            "domains": len(domains),
            "nodes": sum(item["nodes"] for item in domains.values()),
            "edges": sum(item["edges"] for item in domains.values()),
            "documents": sum(item["documents"] for item in domains.values()),
        },
        "tree_sha256": tree_digest.hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_all(args.output_dir)
    print(json.dumps(manifest["totals"], sort_keys=True))
    print(f"tree_sha256={manifest['tree_sha256']}")


if __name__ == "__main__":
    main()
