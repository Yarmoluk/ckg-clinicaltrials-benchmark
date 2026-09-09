#!/usr/bin/env python3
"""Frozen 240-query hybrid-RAG, edge-text, CKG, and router comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import ckg_harness
import integrity_v3_eval
import rag_harness


ROOT = Path(__file__).resolve().parents[1]
HARNESS_VERSION = "frozen-retrieval-v2"
DEFAULT_OUTPUT = ROOT / "results" / HARNESS_VERSION
V3_RESULTS = ROOT / "results" / "integrity-v3"
V3_MANIFEST = V3_RESULTS / "manifest.json"
EDGE_TEXT_ROOT = ROOT / "benchmark" / "edge-text"
IMPLEMENTATION_FILES = (
    "evaluation/build_edge_text_corpus.py",
    "evaluation/ckg_harness.py",
    "evaluation/integrity_v3_eval.py",
    "evaluation/rag_harness.py",
    "evaluation/retrieval_comparison_eval.py",
    "evaluation/verify_retrieval_comparison.py",
)
MODEL = integrity_v3_eval.MODEL
SYSTEM_PROMPT = integrity_v3_eval.SYSTEM_PROMPT
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SYSTEMS = ("modern_rag", "edge_text_rag", "ckg", "router")
ROUTER_MAP = {
    "T1_entity": "modern_rag",
    "T2_dependency": "ckg",
    "T3_path": "ckg",
    "T4_aggregate": "ckg",
    "T5_cross_concept": "modern_rag",
}
NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.I)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*")


def stable_row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(value)]


def extract_nct_ids(value: str) -> list[str]:
    return sorted({match.upper() for match in NCT_PATTERN.findall(value)})


def strip_retrieval_markdown(text: str) -> str:
    """Remove known presentation markup without treating clinical '<...>' as HTML."""
    text = re.sub(r"<details\b[^>]*>.*?</details>", "", text, flags=re.DOTALL | re.I)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"^!!!\s+\w+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_prose_corpus(domain: str) -> list[dict[str, str]]:
    """Load frozen Markdown while preserving comparison operators and trial text."""
    docs: list[dict[str, str]] = []
    chapters_dir = ROOT / "corpus" / domain / "docs" / "chapters"
    if chapters_dir.exists():
        for chapter_path in sorted(chapters_dir.iterdir()):
            index_file = chapter_path / "index.md"
            if not chapter_path.is_dir() or not index_file.exists():
                continue
            text = strip_retrieval_markdown(
                index_file.read_text(encoding="utf-8", errors="ignore")
            )
            if len(text) > 200:
                docs.append({"source": f"{domain}/{chapter_path.name}", "text": text})
    for name in ("glossary.md", "course-description.md"):
        path = ROOT / "corpus" / domain / "docs" / name
        if path.exists():
            text = strip_retrieval_markdown(
                path.read_text(encoding="utf-8", errors="ignore")
            )
            if len(text) > 200:
                docs.append({"source": f"{domain}/{name}", "text": text})
    return docs


class BM25Index:
    """Small deterministic Okapi BM25 implementation for per-domain corpora."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [Counter(tokenize(text)) for text in texts]
        self.lengths = np.asarray(
            [sum(document.values()) for document in self.documents], dtype=np.float32
        )
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        frequencies: Counter[str] = Counter()
        for document in self.documents:
            frequencies.update(document.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }

    def scores(self, query: str) -> np.ndarray:
        values = np.zeros(len(self.documents), dtype=np.float32)
        if not self.documents or not self.average_length:
            return values
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, document in enumerate(self.documents):
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / self.average_length
                )
                values[index] += idf * frequency * (self.k1 + 1.0) / denominator
        return values


def reciprocal_rank_fusion(
    rankings: list[list[int]], limit: int, rrf_k: int = 60
) -> list[tuple[int, float]]:
    scores: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, document_index in enumerate(ranking, start=1):
            scores[document_index] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def model_revision(model: Any) -> str | None:
    candidates = [getattr(model, "model", None)]
    try:
        candidates.extend(list(model))
    except TypeError:
        pass
    for candidate in candidates:
        auto_model = getattr(candidate, "auto_model", candidate)
        config = getattr(auto_model, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def dense_query_text(query: str, embedding_model_name: str) -> str:
    if "bge-" in embedding_model_name.lower():
        return BGE_QUERY_INSTRUCTION + query
    return query


@dataclass
class RetrievalModels:
    embedding: Any
    reranker: Any
    embedding_name: str
    reranker_name: str
    embedding_revision: str | None
    reranker_revision: str | None


def load_models(
    embedding_name: str,
    reranker_name: str,
    embedding_revision: str | None = None,
    reranker_revision: str | None = None,
) -> RetrievalModels:
    print(f"Loading dense model: {embedding_name}")
    embedding = SentenceTransformer(
        embedding_name, revision=embedding_revision, trust_remote_code=False
    )
    print(f"Loading reranker: {reranker_name}")
    reranker = CrossEncoder(
        reranker_name,
        revision=reranker_revision,
        trust_remote_code=False,
        max_length=512,
    )
    return RetrievalModels(
        embedding=embedding,
        reranker=reranker,
        embedding_name=embedding_name,
        reranker_name=reranker_name,
        embedding_revision=model_revision(embedding),
        reranker_revision=model_revision(reranker),
    )


def prose_documents(domain: str) -> list[dict[str, Any]]:
    documents = []
    chunks = rag_harness.chunk_documents(load_prose_corpus(domain))
    for chunk in chunks:
        documents.append(
            {
                "document_id": f"{domain}:prose:{chunk['source']}:{chunk['start_token']}",
                "domain": domain,
                "kind": "prose_chunk",
                "source": chunk["source"],
                "nct_ids": extract_nct_ids(chunk["text"]),
                "taxonomy_ids": [],
                "text": chunk["text"],
            }
        )
    return documents


def edge_text_documents(domain: str) -> list[dict[str, Any]]:
    path = EDGE_TEXT_ROOT / f"{domain}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Build frozen edge text first: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class HybridRerankIndex:
    def __init__(self, documents: list[dict[str, Any]], models: RetrievalModels) -> None:
        if not documents:
            raise ValueError("Cannot index an empty document set")
        self.documents = documents
        self.models = models
        self.bm25 = BM25Index([document["text"] for document in documents])
        self.vectors = models.embedding.encode(
            [document["text"] for document in documents],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def retrieve(
        self,
        query: str,
        candidate_k: int = 40,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        query_nct_ids = set(extract_nct_ids(query))
        eligible = list(range(len(self.documents)))
        metadata_filter = "domain"
        if query_nct_ids:
            matching = [
                index
                for index, document in enumerate(self.documents)
                if query_nct_ids & set(document.get("nct_ids", []))
            ]
            if matching:
                eligible = matching
                metadata_filter = "domain+nct_from_query_text"

        bm25_scores = self.bm25.scores(query)
        query_vector = self.models.embedding.encode(
            [dense_query_text(query, self.models.embedding_name)],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        dense_scores = self.vectors @ query_vector
        bm25_rank = sorted(
            eligible, key=lambda index: (-float(bm25_scores[index]), index)
        )[:candidate_k]
        dense_rank = sorted(
            eligible, key=lambda index: (-float(dense_scores[index]), index)
        )[:candidate_k]
        fused = reciprocal_rank_fusion([bm25_rank, dense_rank], candidate_k, rrf_k)
        fused_indices = [index for index, _ in fused]
        rerank_scores = self.models.reranker.predict(
            [(query, self.documents[index]["text"]) for index in fused_indices],
            batch_size=16,
            show_progress_bar=False,
        )
        reranked = sorted(
            zip(fused, [float(value) for value in np.asarray(rerank_scores).reshape(-1)]),
            key=lambda item: (-item[1], -item[0][1], item[0][0]),
        )
        selected = reranked[:top_k]
        documents = [self.documents[item[0][0]] for item in selected]
        bm25_positions = {value: rank for rank, value in enumerate(bm25_rank, start=1)}
        dense_positions = {value: rank for rank, value in enumerate(dense_rank, start=1)}
        trace = []
        for (index, fused_score), reranker_score in selected:
            trace.append(
                {
                    "document_id": self.documents[index]["document_id"],
                    "metadata_filter": metadata_filter,
                    "bm25_rank": bm25_positions.get(index),
                    "dense_rank": dense_positions.get(index),
                    "rrf_score": round(fused_score, 12),
                    "reranker_score": round(reranker_score, 8),
                }
            )
        return documents, trace


def format_context(documents: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        f"[Document: {document['document_id']}]\n{document['text']}"
        for document in documents
    )


def current_input_trees() -> dict[str, dict[str, Any]]:
    return {
        "graphs": integrity_v3_eval.tree_hash(
            list((ROOT / "benchmark" / "domains").glob("ct-*/*.csv"))
        ),
        "queries": integrity_v3_eval.tree_hash(
            list((ROOT / "benchmark" / "queries").glob("queries_ct-*.jsonl"))
        ),
        "corpus_markdown": integrity_v3_eval.tree_hash(
            list((ROOT / "corpus").glob("ct-*/docs/**/*.md"))
        ),
        "normalized_source_snapshots": integrity_v3_eval.tree_hash(
            list((ROOT / "sources" / "frozen-normalized").glob("ct-*/*.json"))
        ),
    }


def implementation_hashes() -> dict[str, str]:
    return {
        relative: integrity_v3_eval.sha256_file(ROOT / relative)
        for relative in IMPLEMENTATION_FILES
    }


def load_frozen_selection() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads(V3_MANIFEST.read_text())
    if current_input_trees() != manifest["input_trees"]:
        raise ValueError("Frozen v3 graph/query/corpus/source input trees have drifted")
    ordered_ids = manifest["sample"]["selected_query_ids"]
    query_map: dict[str, dict[str, Any]] = {}
    for domain in manifest["sample"]["domain_ids"]:
        for query in integrity_v3_eval.load_queries(domain):
            query_map[query["id"]] = query
    if len(ordered_ids) != 240 or len(set(ordered_ids)) != 240:
        raise ValueError("The v3 manifest does not contain 240 unique frozen query IDs")
    if any(query_id not in query_map for query_id in ordered_ids):
        raise ValueError("A frozen v3 query ID is absent from the query source files")
    selected: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id in ordered_ids:
        query = query_map[query_id]
        selected[query["domain"]].append(query)
    if list(selected) != manifest["sample"]["domain_ids"]:
        raise ValueError("Frozen v3 domain order has drifted")
    return manifest, dict(selected)


def load_v3_ckg_rows(
    v3_manifest: dict[str, Any], selected: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    rows = integrity_v3_eval.load_result_rows(V3_RESULTS / "raw" / "ckg")
    row_map = {row["id"]: row for row in rows}
    expected_ids = [query["id"] for queries in selected.values() for query in queries]
    if set(row_map) != set(expected_ids) or len(rows) != 240:
        raise ValueError("Integrity-v3 CKG raw rows do not match the frozen query IDs")
    for domain, queries in selected.items():
        concepts = ckg_harness.load_graph(
            ROOT / "benchmark" / "domains" / domain / "learning-graph.csv"
        )
        for query in queries:
            row = row_map[query["id"]]
            for key, value in query.items():
                if row.get(key) != value:
                    raise ValueError(
                        f"{query['id']}: v3 CKG row changed source annotation {key}"
                    )
            expected_context, _ = ckg_harness.retrieve_honest(
                concepts, integrity_v3_eval.public_query(query)
            )
            if row["retrieved_context"] != expected_context:
                raise ValueError(f"{query['id']}: v3 CKG context cannot be reconstructed")
            if row["retrieval_query"] != query["query"]:
                raise ValueError(f"{query['id']}: v3 CKG retrieval question drift")
            rescored = integrity_v3_eval.relation_score(
                query, row["predicted_answer"], concepts
            )
            if row["structural_f1"] != rescored["structural_f1"]:
                raise ValueError(f"{query['id']}: stale v3 CKG score")
            if row["model"] != v3_manifest["model"]:
                raise ValueError(f"{query['id']}: v3 CKG model mismatch")
    return row_map


def reused_ckg_rows(
    source_rows: dict[str, dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    run_id: str,
) -> list[dict[str, Any]]:
    output = []
    for query in [item for values in selected.values() for item in values]:
        source = source_rows[query["id"]]
        output.append(
            {
                **source,
                "system": "ckg",
                "harness_version": HARNESS_VERSION,
                "run_id": run_id,
                "retrieval_mode": "reused_integrity_v3_annotation_blind_ckg",
                "generation_status": "reused_integrity_v3_raw_output",
                "source_harness_version": source["harness_version"],
                "source_run_id": source["run_id"],
                "source_row_sha256": stable_row_hash(source),
                "new_api_spend_usd": 0.0,
            }
        )
    return output


def make_rag_row(
    system: str,
    query: dict[str, Any],
    concepts: dict[int, ckg_harness.Concept],
    context: str,
    trace: list[dict[str, Any]],
    answer: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    run_id: str,
    generated: bool,
    candidate_k: int,
    top_k: int,
) -> dict[str, Any]:
    structural = integrity_v3_eval.relation_score(query, answer, concepts)
    total_tokens = prompt_tokens + completion_tokens
    cost = round(
        prompt_tokens * integrity_v3_eval.INPUT_PRICE
        + completion_tokens * integrity_v3_eval.OUTPUT_PRICE,
        8,
    )
    return {
        **query,
        "evaluation_query": integrity_v3_eval.evaluation_question(query, concepts),
        "retrieval_query": query["query"],
        "system": system,
        "model": MODEL,
        "backend": "anthropic" if generated else "dry-run",
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "retrieval_mode": (
            f"bm25_dense_rrf{candidate_k}_cross_encoder_top{top_k}"
        ),
        "retrieval_trace": trace,
        "retrieved_context": context,
        "predicted_answer": answer,
        **structural,
        "token_f1_diagnostic": ckg_harness.token_f1(
            answer, query.get("ground_truth", [])
        )["f1"],
        "context_target_coverage": integrity_v3_eval.context_target_coverage(
            query, context, concepts
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "new_api_spend_usd": cost if generated else 0.0,
        "latency_ms": latency_ms,
        "generation_status": "generated" if generated else "dry_run_placeholder",
    }


def evaluate_rag_system(
    system: str,
    selected: dict[str, list[dict[str, Any]]],
    indexes: dict[str, HybridRerankIndex],
    run_id: str,
    workers: int,
    generate: bool,
    candidate_k: int,
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    prepared = []
    for domain, queries in selected.items():
        concepts = ckg_harness.load_graph(
            ROOT / "benchmark" / "domains" / domain / "learning-graph.csv"
        )
        for query in queries:
            documents, trace = indexes[domain].retrieve(
                query["query"], candidate_k=candidate_k, top_k=top_k, rrf_k=rrf_k
            )
            prepared.append((query, concepts, format_context(documents), trace))

    client = None
    if generate:
        import anthropic

        client = anthropic.Anthropic()

    def evaluate_one(item: tuple[Any, ...]) -> dict[str, Any]:
        query, concepts, context, trace = item
        question = integrity_v3_eval.evaluation_question(query, concepts)
        if generate:
            answer, prompt_tokens, completion_tokens, latency_ms = (
                integrity_v3_eval.call_model(client, context, question)
            )
        else:
            answer, prompt_tokens, completion_tokens, latency_ms = (
                '{"labels":[]}', 0, 0, 0
            )
        return make_rag_row(
            system,
            query,
            concepts,
            context,
            trace,
            answer,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            run_id,
            generate,
            candidate_k,
            top_k,
        )

    if not generate or workers == 1:
        return [evaluate_one(item) for item in prepared]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, item) for item in prepared]
        for count, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if count % 25 == 0:
                print(f"  {system}: {count}/{len(futures)}")
    return sorted(rows, key=lambda row: row["id"])


def router_rows(
    ckg_rows: list[dict[str, Any]],
    modern_rows: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    sources = {
        "ckg": {row["id"]: row for row in ckg_rows},
        "modern_rag": {row["id"]: row for row in modern_rows},
    }
    expected = set(sources["ckg"])
    if set(sources["modern_rag"]) != expected:
        raise ValueError("Router source systems do not share query IDs")
    output = []
    for query_id in sorted(expected):
        query_type = sources["ckg"][query_id]["type"]
        selected_system = ROUTER_MAP[query_type]
        source = sources[selected_system][query_id]
        output.append(
            {
                **source,
                "system": "router",
                "harness_version": HARNESS_VERSION,
                "run_id": run_id,
                "retrieval_mode": f"query_type_router_to_{selected_system}",
                "router_selected_system": selected_system,
                "router_rule_input": query_type,
                "source_row_sha256": stable_row_hash(source),
                "new_api_spend_usd": 0.0,
                "generation_status": f"reused_{selected_system}_output",
            }
        )
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = integrity_v3_eval.summarize(rows)
    summary["new_api_spend_usd"] = round(
        sum(float(row.get("new_api_spend_usd", 0.0)) for row in rows), 6
    )
    return summary


def build_indexes(
    selected: dict[str, list[dict[str, Any]]],
    loader: Callable[[str], list[dict[str, Any]]],
    models: RetrievalModels,
) -> dict[str, HybridRerankIndex]:
    return {
        domain: HybridRerankIndex(loader(domain), models) for domain in selected
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument(
        "--generate", action="store_true", help="Make paid answer-model calls"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    requested = list(args.systems)
    if "router" in requested and not {"ckg", "modern_rag"}.issubset(requested):
        raise SystemExit("router requires ckg and modern_rag in --systems")
    if args.top_k > args.candidate_k or args.top_k < 1:
        raise SystemExit("Require 1 <= top-k <= candidate-k")
    if args.generate:
        ckg_harness.load_secret_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY or omit --generate")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
            raise SystemExit(
                f"Refusing to overwrite non-empty {args.output_dir}; pass --force"
            )

    v3_manifest, selected = load_frozen_selection()
    edge_manifest_path = EDGE_TEXT_ROOT / "manifest.json"
    if not edge_manifest_path.exists():
        raise SystemExit("Run evaluation/build_edge_text_corpus.py first")
    edge_manifest = json.loads(edge_manifest_path.read_text())
    source_ckg = load_v3_ckg_rows(v3_manifest, selected)
    models = load_models(
        args.embedding_model,
        args.reranker_model,
        args.embedding_revision,
        args.reranker_revision,
    )
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{HARNESS_VERSION}-{run_timestamp.replace(':', '').replace('+00:00', 'Z')}"

    all_rows: dict[str, list[dict[str, Any]]] = {}
    ckg_rows = reused_ckg_rows(source_ckg, selected, run_id)
    if "ckg" in requested:
        all_rows["ckg"] = ckg_rows

    for system, loader in (
        ("modern_rag", prose_documents),
        ("edge_text_rag", edge_text_documents),
    ):
        if system not in requested:
            continue
        print(f"Building {system} indexes")
        indexes = build_indexes(selected, loader, models)
        action = "generating" if args.generate else "dry-running"
        print(f"Retrieving and {action} {system}")
        all_rows[system] = evaluate_rag_system(
            system,
            selected,
            indexes,
            run_id,
            args.workers,
            args.generate,
            args.candidate_k,
            args.top_k,
            args.rrf_k,
        )
        print(json.dumps(summarize(all_rows[system]), indent=2))

    if "router" in requested:
        all_rows["router"] = router_rows(
            ckg_rows, all_rows["modern_rag"], run_id
        )

    if not args.generate:
        for system, rows in all_rows.items():
            print(system, json.dumps(summarize(rows), sort_keys=True))
        print("DRY RUN: no result artifacts written and no answer-model calls made")
        return

    manifest = {
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp,
        "git_commit_before_run": integrity_v3_eval.git_commit(),
        "systems": requested,
        "model": MODEL,
        "temperature": v3_manifest["temperature"],
        "max_output_tokens": v3_manifest["max_output_tokens"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "seed": v3_manifest["seed"],
        "sample": v3_manifest["sample"],
        "frozen_parent": {
            "path": str(V3_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": integrity_v3_eval.sha256_file(V3_MANIFEST),
            "run_id": v3_manifest["run_id"],
            "harness_version": v3_manifest["harness_version"],
        },
        "input_trees": current_input_trees(),
        "implementation_files": implementation_hashes(),
        "edge_text": {
            "path": str(edge_manifest_path.relative_to(ROOT)),
            "manifest_sha256": integrity_v3_eval.sha256_file(edge_manifest_path),
            "tree_sha256": edge_manifest["tree_sha256"],
            "document_count": edge_manifest["totals"]["documents"],
        },
        "retrieval": {
            "same_question_text": True,
            "domain_filter": "one index per frozen domain",
            "nct_filter": (
                "exact NCT IDs parsed from query text; applied only when matching "
                "documents exist"
            ),
            "prose_preprocessing": (
                "remove only details blocks, HTML comments, MkDocs directives, "
                "images, and URLs; preserve clinical comparison text containing "
                "literal angle brackets"
            ),
            "bm25": {"implementation": "local_okapi", "k1": 1.5, "b": 0.75},
            "dense": {
                "model": models.embedding_name,
                "revision": models.embedding_revision,
                "normalized_cosine": True,
                "query_instruction": (
                    BGE_QUERY_INSTRUCTION
                    if "bge-" in models.embedding_name.lower()
                    else ""
                ),
            },
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "rrf_k": args.rrf_k,
            },
            "candidate_k": args.candidate_k,
            "reranker": {
                "model": models.reranker_name,
                "revision": models.reranker_revision,
                "max_length": 512,
            },
            "top_k": args.top_k,
        },
        "router": {
            "input": "frozen query type only",
            "mapping": ROUTER_MAP,
            "additional_model_calls": 0,
        },
        "ckg_reuse": {
            "source": "integrity-v3 raw CKG outputs",
            "additional_model_calls": 0,
        },
        "pricing_usd_per_million": v3_manifest["pricing_usd_per_million"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for system, rows in all_rows.items():
        integrity_v3_eval.write_rows(args.output_dir, system, rows)
    aggregate = {
        "run_id": run_id,
        "systems": {name: summarize(rows) for name, rows in all_rows.items()},
    }
    for right in ("modern_rag", "edge_text_rag", "router"):
        if "ckg" in all_rows and right in all_rows:
            aggregate[f"ckg_vs_{right}"] = integrity_v3_eval.comparison(
                all_rows["ckg"], all_rows[right]
            )
    aggregate["actual_new_api_spend_usd"] = round(
        sum(
            row["new_api_spend_usd"]
            for system, rows in all_rows.items()
            if system != "router"
            for row in rows
        ),
        6,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote frozen comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
