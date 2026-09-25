from __future__ import annotations

import unittest

from src.pain_point_analysis import extract_response_text, validate_analysis


class PainPointAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.articles = [
            {"article_id": "A1", "title": "Original one"},
            {"article_id": "A2", "title": "Original two"},
        ]
        self.analysis = {
            "top_problems": [{"supporting_article_ids": ["A1"]}],
            "evidence_ledger": [{"supporting_article_ids": ["A1", "A2"]}],
            "article_analysis": [
                {
                    "article_id": "A2",
                    "title": "Changed title",
                    "no_supported_inference": True,
                    "problems": [],
                },
                {
                    "article_id": "A1",
                    "title": "Changed title",
                    "no_supported_inference": False,
                    "problems": [{"problem": "Example"}],
                },
            ],
        }

    def test_extract_response_text(self) -> None:
        response = {
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]},
            ]
        }
        self.assertEqual(extract_response_text(response), '{"ok":true}')

    def test_validation_restores_titles_and_input_order(self) -> None:
        result = validate_analysis(self.analysis, self.articles)
        rows = result["article_analysis"]
        self.assertEqual([row["article_id"] for row in rows], ["A1", "A2"])
        self.assertEqual([row["title"] for row in rows], ["Original one", "Original two"])

    def test_validation_rejects_unknown_citation(self) -> None:
        self.analysis["top_problems"][0]["supporting_article_ids"] = ["A3"]
        with self.assertRaisesRegex(ValueError, "unknown article"):
            validate_analysis(self.analysis, self.articles)

    def test_validation_rejects_missing_article(self) -> None:
        self.analysis["article_analysis"].pop()
        with self.assertRaisesRegex(ValueError, "every supplied article"):
            validate_analysis(self.analysis, self.articles)


if __name__ == "__main__":
    unittest.main()
