#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_edge_text_corpus as edge_builder
import ckg_harness
import integrity_v3_eval
import retrieval_comparison_eval as comparison
import verify_retrieval_comparison as verifier


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "ct-cardiovascular"


class FakeEmbedding:
    def encode(self, texts, **kwargs):
        values = []
        for text in texts:
            normalized = text.lower()
            vector = np.asarray(
                [normalized.count("alpha"), normalized.count("nct01234567") + 0.1],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            values.append(vector / norm if norm else np.asarray([0.0, 1.0], dtype=np.float32))
        return np.asarray(values, dtype=np.float32)


class FakeReranker:
    def predict(self, pairs, **kwargs):
        return np.asarray(
            [sum(token in document.lower() for token in comparison.tokenize(query)) for query, document in pairs],
            dtype=np.float32,
        )


class RetrievalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.concepts = ckg_harness.load_graph(
            ROOT / "benchmark" / "domains" / DOMAIN / "learning-graph.csv"
        )

    def test_bm25_ranks_matching_document_first(self):
        index = comparison.BM25Index(["alpha beta", "gamma delta"])
        self.assertGreater(index.scores("alpha")[0], index.scores("alpha")[1])

    def test_rrf_rewards_document_present_in_both_rankings(self):
        fused = comparison.reciprocal_rank_fusion([[0, 1], [1, 2]], 3, rrf_k=1)
        self.assertEqual(fused[0][0], 1)

    def test_edge_text_is_one_doc_per_node_and_edge(self):
        documents = edge_builder.build_documents(DOMAIN, self.concepts)
        edge_count = sum(len(concept.dependencies) for concept in self.concepts.values())
        self.assertEqual(len(documents), len(self.concepts) + edge_count)
        self.assertEqual(sum(row["kind"] == "node" for row in documents), 180)
        self.assertEqual(sum(row["kind"] == "edge" for row in documents), edge_count)

    def test_edge_text_is_deterministic_and_annotation_free(self):
        first = edge_builder.build_documents(DOMAIN, self.concepts)
        second = edge_builder.build_documents(DOMAIN, self.concepts)
        self.assertEqual(first, second)
        serialized = json.dumps(first)
        for forbidden in ("ground_truth", "path_ids", "query_type", "gold_answer"):
            self.assertNotIn(forbidden, serialized)
        self.assertIn("depends_on", serialized)

    def test_edge_builder_writes_reproducible_manifest(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = edge_builder.build_all(Path(first_dir))
            second = edge_builder.build_all(Path(second_dir))
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            self.assertEqual(first["totals"], second["totals"])

    def test_frozen_selection_uses_v3_order_and_ids(self):
        manifest, selected = comparison.load_frozen_selection()
        ids = [query["id"] for values in selected.values() for query in values]
        self.assertEqual(ids, manifest["sample"]["selected_query_ids"])
        self.assertEqual(len(ids), 240)

    def test_query_text_nct_filter_uses_no_annotations(self):
        documents = [
            {
                "document_id": "plain",
                "text": "alpha NCT99999999",
                "nct_ids": ["NCT99999999"],
            },
            {
                "document_id": "match",
                "text": "record NCT01234567",
                "nct_ids": ["NCT01234567"],
            },
        ]
        models = comparison.RetrievalModels(
            FakeEmbedding(), FakeReranker(), "fake", "fake", "1", "1"
        )
        index = comparison.HybridRerankIndex(documents, models)
        retrieved, trace = index.retrieve("What is NCT01234567?", candidate_k=2, top_k=1)
        self.assertEqual(retrieved[0]["document_id"], "match")
        self.assertEqual(trace[0]["metadata_filter"], "domain+nct_from_query_text")

    def test_prose_preprocessing_preserves_clinical_angle_bracket_text(self):
        raw = (
            "Eligibility: age <26 and value > 4. NCT03047369 remains.\n"
            "<!-- generated separator -->\n"
            "[source](https://example.com)"
        )
        cleaned = comparison.strip_retrieval_markdown(raw)
        self.assertIn("<26", cleaned)
        self.assertIn("NCT03047369", cleaned)
        self.assertNotIn("generated separator", cleaned)

    def test_prose_loader_preserves_all_chapter_nct_ids(self):
        chapter_paths = (ROOT / "corpus" / DOMAIN / "docs" / "chapters").glob(
            "*/index.md"
        )
        raw_ids = {
            nct
            for path in chapter_paths
            for nct in comparison.extract_nct_ids(path.read_text())
        }
        loaded_ids = {
            nct
            for document in comparison.load_prose_corpus(DOMAIN)
            for nct in comparison.extract_nct_ids(document["text"])
        }
        self.assertEqual(raw_ids, loaded_ids)

    def test_v3_ckg_rows_are_reusable_without_model_calls(self):
        manifest, selected = comparison.load_frozen_selection()
        source = comparison.load_v3_ckg_rows(manifest, selected)
        rows = comparison.reused_ckg_rows(source, selected, "test-run")
        self.assertEqual(len(rows), 240)
        self.assertTrue(all(row["new_api_spend_usd"] == 0.0 for row in rows))
        self.assertTrue(all(row["structural_f1"] == 1.0 for row in rows))

    def test_router_uses_only_frozen_query_type(self):
        ckg = [
            {"id": "a", "type": "T2_dependency", "structural_f1": 1.0},
            {"id": "b", "type": "T1_entity", "structural_f1": 0.0},
        ]
        modern = [
            {"id": "a", "type": "T2_dependency", "structural_f1": 0.0},
            {"id": "b", "type": "T1_entity", "structural_f1": 0.5},
        ]
        rows = comparison.router_rows(ckg, modern, "test-run")
        mapped = {row["id"]: row for row in rows}
        self.assertEqual(mapped["a"]["router_selected_system"], "ckg")
        self.assertEqual(mapped["b"]["router_selected_system"], "modern_rag")
        self.assertEqual(mapped["a"]["router_rule_input"], "T2_dependency")

    def test_comparison_reuses_v3_structural_scorer(self):
        query = next(
            row
            for row in integrity_v3_eval.load_queries(DOMAIN)
            if row["type"] == "T1_entity"
        )
        expected = query["ground_truth"][-1]
        score = integrity_v3_eval.relation_score(
            query, json.dumps({"labels": [expected]}), self.concepts
        )
        self.assertEqual(score["structural_f1"], 1.0)

    def test_trace_verification_allows_float_drift_not_rank_drift(self):
        saved = [{
            "document_id": "a", "metadata_filter": "domain", "bm25_rank": 1,
            "dense_rank": 2, "rrf_score": 0.02, "reranker_score": -5.123456,
        }]
        rebuilt = [{**saved[0], "reranker_score": -5.123459}]
        verifier.assert_retrieval_trace("test", saved, rebuilt)
        rebuilt[0]["document_id"] = "b"
        with self.assertRaises(AssertionError):
            verifier.assert_retrieval_trace("test", saved, rebuilt)

    def test_verifier_rejects_wrong_system_label(self):
        manifest, selected = comparison.load_frozen_selection()
        source = comparison.load_v3_ckg_rows(manifest, selected)
        query = selected[DOMAIN][0]
        row = comparison.reused_ckg_rows(source, selected, "test-run")[0]
        row["system"] = "not_ckg"
        with self.assertRaises(AssertionError):
            verifier.verify_row("ckg", row, query, self.concepts, {
                "model": comparison.MODEL,
                "run_id": "test-run",
            })


if __name__ == "__main__":
    unittest.main()
