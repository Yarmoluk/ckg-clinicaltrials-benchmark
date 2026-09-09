#!/usr/bin/env python3
"""Integrity-v2 evaluation with annotation-blind retrieval and structural scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

import ckg_harness
import rag_harness


ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-haiku-4-5-20251001"
HARNESS_VERSION = "integrity-v2"
INPUT_PRICE = 1.0 / 1_000_000
OUTPUT_PRICE = 5.0 / 1_000_000
DEFAULT_OUTPUT = ROOT / "results" / HARNESS_VERSION
QUERY_TYPES = (
    "T1_entity",
    "T2_dependency",
    "T3_path",
    "T4_aggregate",
    "T5_cross_concept",
)

TAXONOMY_ALIASES = {
    "AREA": {"area", "domain", "landscape"},
    "FRAME": {"frame", "framework"},
    "COND": {"cond", "condition", "disease"},
    "INTR": {"intr", "intervention", "treatment"},
    "OUTC": {"outc", "outcome", "endpoint", "measure"},
    "SPON": {"spon", "sponsor"},
    "PHASE": {"phase", "trial phase"},
    "STATUS": {"status", "recruitment status"},
    "TRIAL": {"trial", "study"},
}

SYSTEM_PROMPT = """You answer structural questions using only the provided context.
Return exactly one JSON object and no prose.

For an entity-type question: {"labels":["TAXONOMY_CODE"]}
For a direct-prerequisite question: {"labels":["exact prerequisite label"]}
For a path question: {"labels":["start label","intermediate label","end label"]}
For an aggregate question: {"labels":["every exact member label"]}
For a relationship question: {"source":"exact source label","relation":"depends_on","target":"exact target label"}

Valid taxonomy codes are AREA, FRAME, COND, INTR, OUTC, SPON, PHASE, STATUS, and TRIAL.
Use an empty labels list, or relation "unknown", when the context does not support an answer."""


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.lower())).strip()


def public_query(query: dict[str, Any]) -> dict[str, str]:
    """Return the only fields retrieval is allowed to inspect."""
    return {
        "type": str(query.get("query_type") or query.get("type", "")),
        "query": str(query["query"]),
    }


def load_queries(domain: str) -> list[dict[str, Any]]:
    path = ROOT / "benchmark" / "queries" / f"queries_{domain}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def balanced_sample(
    queries: list[dict[str, Any]],
    per_type: int,
    seed: int,
    domain_index: int,
) -> list[dict[str, Any]]:
    """Choose an equal count per type and rotate T4 categories across domains."""
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        grouped[str(query.get("query_type") or query.get("type", ""))].append(query)

    selected: list[dict[str, Any]] = []
    for query_type in QUERY_TYPES:
        candidates = sorted(grouped[query_type], key=lambda item: item["id"])
        if len(candidates) < per_type:
            raise ValueError(f"{query_type} has {len(candidates)} rows; need {per_type}")
        if query_type == "T4_aggregate":
            offset = (domain_index * per_type) % len(candidates)
            chosen = [candidates[(offset + index) % len(candidates)] for index in range(per_type)]
        else:
            rng = random.Random(f"{seed}:{queries[0]['domain']}:{query_type}")
            chosen = rng.sample(candidates, per_type)
        selected.extend(chosen)

    random.Random(f"{seed}:{queries[0]['domain']}:shuffle").shuffle(selected)
    return selected


def extract_json_object(answer: str) -> dict[str, Any] | None:
    stripped = answer.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def set_f1(predicted: list[str], expected: list[str]) -> dict[str, float | bool]:
    predicted_set = {canonical(item) for item in predicted if canonical(item)}
    expected_set = {canonical(item) for item in expected if canonical(item)}
    if not expected_set:
        exact = not predicted_set
        return {"precision": float(exact), "recall": float(exact), "f1": float(exact), "exact": exact}
    overlap = predicted_set & expected_set
    precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
    recall = len(overlap) / len(expected_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact": predicted_set == expected_set,
    }


def edge_f1(predicted: list[str], expected: list[str]) -> dict[str, float | bool]:
    predicted_edges = {
        (canonical(left), canonical(right)) for left, right in zip(predicted, predicted[1:])
    }
    expected_edges = {
        (canonical(left), canonical(right)) for left, right in zip(expected, expected[1:])
    }
    overlap = predicted_edges & expected_edges
    precision = len(overlap) / len(predicted_edges) if predicted_edges else 0.0
    recall = len(overlap) / len(expected_edges) if expected_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact": predicted_edges == expected_edges and bool(expected_edges),
    }


def taxonomy_value(value: str) -> str:
    normalized = canonical(value)
    for code, aliases in TAXONOMY_ALIASES.items():
        if normalized == canonical(code) or normalized in aliases:
            return code
    return normalized.upper()


def relation_score(
    query: dict[str, Any],
    answer: str,
    concepts: dict[int, ckg_harness.Concept],
) -> dict[str, Any]:
    query_type = str(query.get("query_type") or query.get("type", ""))
    parsed = extract_json_object(answer)
    labels = parsed.get("labels", []) if parsed else []
    if not isinstance(labels, list):
        labels = []
    labels = [str(value) for value in labels]

    if query_type == "T1_entity":
        expected = str(query["ground_truth"][-1])
        predicted = [taxonomy_value(labels[0])] if labels else []
        score = set_f1(predicted, [expected])
        expected_value: Any = [expected]
        predicted_value: Any = predicted
    elif query_type in {"T2_dependency", "T4_aggregate"}:
        expected = [str(value) for value in query["ground_truth"]]
        predicted = labels
        score = set_f1(predicted, expected)
        expected_value = expected
        predicted_value = predicted
    elif query_type == "T3_path":
        expected = list(reversed([str(value) for value in query["ground_truth"]]))
        predicted = labels
        score = edge_f1(predicted, expected)
        expected_value = expected
        predicted_value = predicted
    elif query_type == "T5_cross_concept":
        source = concepts[int(query["concept_id_a"])].label
        target = concepts[int(query["concept_id_b"])].label
        predicted_source = str(parsed.get("source", "")) if parsed else ""
        predicted_target = str(parsed.get("target", "")) if parsed else ""
        predicted_relation = canonical(str(parsed.get("relation", ""))) if parsed else ""
        correct = (
            canonical(predicted_source) == canonical(source)
            and canonical(predicted_target) == canonical(target)
            and predicted_relation in {"depends on", "depends_on", "dependency"}
        )
        score = {
            "precision": float(correct),
            "recall": float(correct),
            "f1": float(correct),
            "exact": correct,
        }
        expected_value = {"source": source, "relation": "depends_on", "target": target}
        predicted_value = {
            "source": predicted_source,
            "relation": predicted_relation,
            "target": predicted_target,
        }
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    return {
        "structural_precision": score["precision"],
        "structural_recall": score["recall"],
        "structural_f1": score["f1"],
        "exact_match": bool(score["exact"]),
        "valid_json": parsed is not None,
        "expected_structural_answer": expected_value,
        "parsed_structural_answer": predicted_value,
    }


def evidence_recall(query: dict[str, Any], context: str, concepts: dict[int, ckg_harness.Concept]) -> float:
    query_type = str(query.get("query_type") or query.get("type", ""))
    if query_type == "T1_entity":
        expected = [str(query["ground_truth"][-1])]
    elif query_type == "T3_path":
        expected = list(reversed([str(value) for value in query["ground_truth"]]))
    elif query_type == "T5_cross_concept":
        expected = [
            concepts[int(query["concept_id_a"])].label,
            concepts[int(query["concept_id_b"])].label,
        ]
    else:
        expected = [str(value) for value in query["ground_truth"]]
    if not expected:
        return 0.0
    normalized = canonical(context)
    found = sum(canonical(label) in normalized for label in expected)
    return round(found / len(expected), 6)


def evaluation_question(query: dict[str, Any], concepts: dict[int, ckg_harness.Concept]) -> str:
    query_type = str(query.get("query_type") or query.get("type", ""))
    if query_type == "T1_entity":
        label = concepts[int(query["concept_id"])].label
        return f"What taxonomy type is {label}?"
    if query_type == "T2_dependency":
        label = concepts[int(query["concept_id"])].label
        return f"In the benchmark knowledge structure, what are the direct prerequisites for {label}?"
    if query_type == "T3_path":
        return "In the benchmark knowledge structure, " + query["query"][0].lower() + query["query"][1:]
    if query_type == "T4_aggregate":
        return "In the benchmark knowledge structure, " + query["query"][0].lower() + query["query"][1:]
    if query_type == "T5_cross_concept":
        return str(query["query"])
    return str(query["query"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(paths: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(paths, key=lambda item: str(item.relative_to(ROOT))):
        relative = str(path.relative_to(ROOT))
        file_hash = sha256_file(path)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
        count += 1
        size += path.stat().st_size
    return {"sha256": digest.hexdigest(), "file_count": count, "bytes": size}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def call_model(client: anthropic.Anthropic, context: str, question: str) -> tuple[str, int, int, int]:
    user_message = f"Context:\n{context}\n\nQuestion:\n{question}"
    started = time.time()
    response = None
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    assert response is not None
    answer = response.content[0].text
    return (
        answer,
        int(response.usage.input_tokens),
        int(response.usage.output_tokens),
        int((time.time() - started) * 1000),
    )


def prepare_contexts(
    system: str,
    selected: dict[str, list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[int, ckg_harness.Concept], str, str]]:
    prepared = []
    embed_model = rag_harness.get_embed_model() if system == "rag" else None
    for domain, queries in selected.items():
        concepts = ckg_harness.load_graph(ROOT / "benchmark" / "domains" / domain / "learning-graph.csv")
        if system == "rag":
            index, chunks = rag_harness.build_or_load_index(domain, embed_model)
        for query in queries:
            question = evaluation_question(query, concepts)
            retrieval_input = public_query(query)
            if system == "ckg":
                context, _ = ckg_harness.retrieve_honest(concepts, retrieval_input)
            elif system == "rag":
                top_chunks = rag_harness.retrieve_chunks(question, index, chunks, embed_model)
                context = "\n\n---\n\n".join(
                    f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in top_chunks
                )
            elif system == "question_echo":
                context = ""
            else:
                raise ValueError(system)
            prepared.append((query, concepts, question, context))
    return prepared


def evaluate_system(
    system: str,
    selected: dict[str, list[dict[str, Any]]],
    run_id: str,
    workers: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    prepared = prepare_contexts(system, selected)
    client = None if system == "question_echo" or dry_run else anthropic.Anthropic()

    def evaluate_one(item: tuple[dict[str, Any], dict[int, ckg_harness.Concept], str, str]) -> dict[str, Any]:
        query, concepts, question, context = item
        if system == "question_echo":
            answer, prompt_tokens, completion_tokens, latency_ms = question, 0, 0, 0
        elif dry_run:
            answer, prompt_tokens, completion_tokens, latency_ms = '{"labels":[]}', 0, 0, 0
        else:
            answer, prompt_tokens, completion_tokens, latency_ms = call_model(client, context, question)

        structural = relation_score(query, answer, concepts)
        diagnostic = ckg_harness.token_f1(answer, query.get("ground_truth", []))
        total_tokens = prompt_tokens + completion_tokens
        row = {
            **query,
            "evaluation_query": question,
            "system": system,
            "model": "deterministic-question-echo" if system == "question_echo" else MODEL,
            "backend": "deterministic" if system == "question_echo" else "anthropic",
            "harness_version": HARNESS_VERSION,
            "run_id": run_id,
            "retrieval_mode": (
                "question_text_only_graph_traversal" if system == "ckg"
                else "minilm_faiss_top5_raw_prose" if system == "rag"
                else "none"
            ),
            "retrieved_context": context,
            "predicted_answer": answer,
            **structural,
            "token_f1_diagnostic": diagnostic["f1"],
            "evidence_recall": evidence_recall(query, context, concepts) if context else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(prompt_tokens * INPUT_PRICE + completion_tokens * OUTPUT_PRICE, 8),
            "latency_ms": latency_ms,
        }
        return row

    if system == "question_echo" or dry_run or workers == 1:
        return [evaluate_one(item) for item in prepared]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, item) for item in prepared]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0:
                print(f"  {system}: {index}/{len(futures)}")
    return sorted(rows, key=lambda row: row["id"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    def block(values: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(values)
        return {
            "n": count,
            "structural_f1": round(sum(row["structural_f1"] for row in values) / count, 6),
            "exact_match_rate": round(sum(row["exact_match"] for row in values) / count, 6),
            "evidence_recall": round(sum(row["evidence_recall"] for row in values) / count, 6),
            "token_f1_diagnostic": round(sum(row["token_f1_diagnostic"] for row in values) / count, 6),
            "valid_json_rate": round(sum(row["valid_json"] for row in values) / count, 6),
            "mean_tokens": round(sum(row["total_tokens"] for row in values) / count, 3),
            "cost_usd": round(sum(row["cost_usd"] for row in values), 6),
        }

    output = block(rows)
    output["by_type"] = {key: block(value) for key, value in sorted(by_type.items())}
    return output


def comparison(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_map = {row["id"]: row for row in left}
    right_map = {row["id"]: row for row in right}
    if set(left_map) != set(right_map):
        raise ValueError("Comparison query IDs do not match")
    wins = losses = ties = 0
    for query_id in left_map:
        left_score = left_map[query_id]["structural_f1"]
        right_score = right_map[query_id]["structural_f1"]
        if left_score > right_score:
            wins += 1
        elif left_score < right_score:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "n": len(left_map)}


def write_rows(output_dir: Path, system: str, rows: list[dict[str, Any]]) -> None:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    target = output_dir / "raw" / system
    target.mkdir(parents=True, exist_ok=True)
    for domain, domain_rows in sorted(grouped.items()):
        path = target / f"{system}_{domain}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in domain_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", choices=("ckg", "rag", "question_echo"),
                        default=("ckg", "rag", "question_echo"))
    parser.add_argument("--per-type", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--domains", nargs="+", help="Optional subset of domain IDs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and any(system != "question_echo" for system in args.systems):
        ckg_harness.load_secret_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY or use --dry-run")

    available_domains = sorted(
        path.parent.name for path in (ROOT / "benchmark" / "domains").glob("ct-*/learning-graph.csv")
    )
    domains = args.domains or available_domains
    unknown_domains = sorted(set(domains) - set(available_domains))
    if unknown_domains:
        raise SystemExit(f"Unknown domains: {', '.join(unknown_domains)}")
    selected = {
        domain: balanced_sample(load_queries(domain), args.per_type, args.seed, index)
        for index, domain in enumerate(domains)
    }
    selected_ids = [query["id"] for domain in domains for query in selected[domain]]
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{HARNESS_VERSION}-{run_timestamp.replace(':', '').replace('+00:00', 'Z')}"

    graph_inputs = list((ROOT / "benchmark" / "domains").glob("ct-*/*.csv"))
    query_inputs = list((ROOT / "benchmark" / "queries").glob("queries_ct-*.jsonl"))
    corpus_inputs = list((ROOT / "corpus").glob("ct-*/docs/**/*.md"))
    normalized_source_inputs = list((ROOT / "sources" / "frozen-normalized").glob("ct-*/*.json"))
    manifest = {
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp,
        "git_commit_before_run": git_commit(),
        "model": MODEL,
        "temperature": "provider_default",
        "max_output_tokens": 1024,
        "request_attempts": 5,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "systems": list(args.systems),
        "seed": args.seed,
        "sample": {"domains": len(domains), "per_type_per_domain": args.per_type,
                   "total": len(selected_ids), "selected_query_ids": selected_ids},
        "retrieval": {
            "ckg": "natural-language question only; no answer annotations",
            "rag": {"embedding_model": rag_harness.EMBED_MODEL, "chunk_tokens": rag_harness.CHUNK_TOKENS,
                    "overlap_tokens": rag_harness.OVERLAP_TOKENS, "top_k": rag_harness.TOP_K},
        },
        "pricing_usd_per_million": {"input": 1.0, "output": 5.0},
        "input_trees": {
            "graphs": tree_hash(graph_inputs),
            "queries": tree_hash(query_inputs),
            "corpus_markdown": tree_hash(corpus_inputs),
            "normalized_source_snapshots": tree_hash(normalized_source_inputs),
        },
    }

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for system in args.systems:
        print(f"Running {system} ({len(selected_ids)} queries)")
        rows = evaluate_system(system, selected, run_id, args.workers, args.dry_run)
        all_rows[system] = rows
        print(json.dumps(summarize(rows), indent=2))

    if args.dry_run:
        print("DRY RUN: no result artifacts written")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for system, rows in all_rows.items():
        write_rows(args.output_dir, system, rows)
    aggregate = {"run_id": run_id, "systems": {name: summarize(rows) for name, rows in all_rows.items()}}
    if "ckg" in all_rows and "rag" in all_rows:
        aggregate["ckg_vs_rag"] = comparison(all_rows["ckg"], all_rows["rag"])
    if "ckg" in all_rows and "question_echo" in all_rows:
        aggregate["ckg_vs_question_echo"] = comparison(all_rows["ckg"], all_rows["question_echo"])
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(f"Wrote corrected artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
