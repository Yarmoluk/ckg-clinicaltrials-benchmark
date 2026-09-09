#!/usr/bin/env python3

import json
import unittest
from collections import Counter
from pathlib import Path

import ckg_harness
import integrity_eval


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "ct-cardiovascular"


class IntegrityEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = integrity_eval.load_queries(DOMAIN)
        cls.concepts = ckg_harness.load_graph(
            ROOT / "benchmark" / "domains" / DOMAIN / "learning-graph.csv"
        )

    def test_public_query_excludes_answer_annotations(self):
        public = integrity_eval.public_query(self.queries[0])
        self.assertEqual(set(public), {"type", "query"})
        self.assertNotIn("ground_truth", public)
        self.assertNotIn("path_ids", public)
        self.assertNotIn("concept_id", public)

    def test_balanced_sample_has_equal_type_counts_and_varied_t4(self):
        selected = []
        t4_categories = set()
        domains = sorted(path.parent.name for path in (ROOT / "benchmark" / "domains").glob("ct-*/learning-graph.csv"))
        for index, domain in enumerate(domains):
            sample = integrity_eval.balanced_sample(
                integrity_eval.load_queries(domain), 4, 20260909, index
            )
            self.assertEqual(Counter(item["type"] for item in sample), Counter({name: 4 for name in integrity_eval.QUERY_TYPES}))
            selected.extend(sample)
            t4_categories.update(item["taxonomy_id"] for item in sample if item["type"] == "T4_aggregate")
        self.assertEqual(len(selected), 240)
        self.assertGreaterEqual(len(t4_categories), 8)

    def test_question_echo_scores_zero_for_balanced_sample(self):
        domains = sorted(path.parent.name for path in (ROOT / "benchmark" / "domains").glob("ct-*/learning-graph.csv"))
        scores = []
        for index, domain in enumerate(domains):
            concepts = ckg_harness.load_graph(ROOT / "benchmark" / "domains" / domain / "learning-graph.csv")
            sample = integrity_eval.balanced_sample(integrity_eval.load_queries(domain), 4, 20260909, index)
            for query in sample:
                question = integrity_eval.evaluation_question(query, concepts)
                scores.append(integrity_eval.relation_score(query, question, concepts)["structural_f1"])
        self.assertEqual(sum(scores), 0.0)

    def test_annotation_blind_ckg_retrieval_has_full_sample_evidence(self):
        domains = sorted(path.parent.name for path in (ROOT / "benchmark" / "domains").glob("ct-*/learning-graph.csv"))
        recalls = []
        for index, domain in enumerate(domains):
            concepts = ckg_harness.load_graph(ROOT / "benchmark" / "domains" / domain / "learning-graph.csv")
            sample = integrity_eval.balanced_sample(integrity_eval.load_queries(domain), 4, 20260909, index)
            for query in sample:
                context, _ = ckg_harness.retrieve_honest(concepts, integrity_eval.public_query(query))
                recalls.append(integrity_eval.evidence_recall(query, context, concepts))
        self.assertEqual(sum(recalls), 240.0)

    def test_path_requires_correct_directed_edges(self):
        query = next(item for item in self.queries if item["type"] == "T3_path")
        expected = list(reversed(query["ground_truth"]))
        correct = json.dumps({"labels": expected})
        endpoints_only = json.dumps({"labels": [expected[0], expected[-1]]})
        reversed_path = json.dumps({"labels": list(reversed(expected))})
        self.assertEqual(integrity_eval.relation_score(query, correct, self.concepts)["structural_f1"], 1.0)
        self.assertEqual(integrity_eval.relation_score(query, endpoints_only, self.concepts)["structural_f1"], 0.0)
        self.assertEqual(integrity_eval.relation_score(query, reversed_path, self.concepts)["structural_f1"], 0.0)

    def test_relationship_requires_direction(self):
        query = next(item for item in self.queries if item["type"] == "T5_cross_concept")
        source = self.concepts[query["concept_id_a"]].label
        target = self.concepts[query["concept_id_b"]].label
        correct = json.dumps({"source": source, "relation": "depends_on", "target": target})
        reversed_relation = json.dumps({"source": target, "relation": "depends_on", "target": source})
        self.assertEqual(integrity_eval.relation_score(query, correct, self.concepts)["structural_f1"], 1.0)
        self.assertEqual(integrity_eval.relation_score(query, reversed_relation, self.concepts)["structural_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
