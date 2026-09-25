from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.event_deduplication import (
    DedupeConfig,
    EmbeddingBatch,
    deduplicate_articles,
    evaluate_fixture,
)


FIXTURE = Path(__file__).parent / "fixtures" / "event_deduplication_v1.json"


def _semantic_provider(texts: list[str]) -> EmbeddingBatch:
    vectors = []
    for text in texts:
        lowered = text.casefold()
        if "40 million" in lowered or "40m" in lowered:
            vectors.append([1.0, 0.0, 0.0])
        elif "outage" in lowered or "disruption" in lowered:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return EmbeddingBatch(
        np.asarray(vectors), "deterministic-test-semantic", True,
        {"model": "test-event-vectors-v1"},
    )


class EventDeduplicationTests(unittest.TestCase):
    def test_uses_title_and_description_and_retains_newest_article(self) -> None:
        articles = [
            {
                "title": "Fintech growth round announced",
                "description": "Acme raises $40 million Series B for its fraud platform.",
                "publishedAt": "2026-09-20T12:00:00Z",
            },
            {
                "title": "Investors back fraud prevention company",
                "description": "Acme's $40 million Series B will expand its payments product.",
                "publishedAt": "2026-09-21T12:00:00Z",
            },
        ]

        result = deduplicate_articles(articles, embedding_provider=_semantic_provider)

        self.assertEqual(len(result.articles), 1)
        self.assertEqual(result.articles[0]["publishedAt"], "2026-09-21T12:00:00Z")
        self.assertEqual(result.diagnostics.removed_duplicates, 1)
        self.assertTrue(result.diagnostics.semantic_active)
        self.assertEqual(result.diagnostics.merges[0].representative_index, 1)
        self.assertIn("acme", result.diagnostics.merges[0].evidence.shared_entities)

    def test_does_not_merge_different_events_for_same_company(self) -> None:
        articles = [
            {
                "title": "Acme raises $40 million Series B",
                "description": "Acme funds expansion of its fraud product.",
                "publishedAt": "2026-09-20T12:00:00Z",
            },
            {
                "title": "Acme launches payments monitoring product",
                "description": "Acme introduced a new dashboard for banks.",
                "publishedAt": "2026-09-21T12:00:00Z",
            },
        ]

        result = deduplicate_articles(articles, embedding_provider=_semantic_provider)

        self.assertEqual(len(result.articles), 2)
        self.assertEqual(result.diagnostics.removed_duplicates, 0)

    def test_conflicting_funding_amounts_block_a_merge(self) -> None:
        articles = [
            {
                "title": "Acme raises $40 million Series B",
                "description": "Acme expands its fraud platform.",
                "publishedAt": "2026-09-20T12:00:00Z",
            },
            {
                "title": "Acme raises $12 million Series A",
                "description": "Acme expands its fraud platform.",
                "publishedAt": "2026-09-19T12:00:00Z",
            },
        ]

        result = deduplicate_articles(articles, embedding_provider=_semantic_provider)

        self.assertEqual(len(result.articles), 2)

    def test_same_amount_with_omitted_currency_can_merge(self) -> None:
        articles = [
            {
                "title": "Acme raises $40 million Series B",
                "description": "Acme expands its fraud platform.",
                "publishedAt": "2026-09-20T12:00:00Z",
            },
            {
                "title": "Acme lands 40m Series B round",
                "description": "Acme expands its fraud platform.",
                "publishedAt": "2026-09-20T11:00:00Z",
            },
        ]

        result = deduplicate_articles(articles, embedding_provider=_semantic_provider)

        self.assertEqual(len(result.articles), 1)

    def test_fallback_is_visible_and_conservative(self) -> None:
        articles = [
            {
                "title": "CloudCo outage disrupts US customers",
                "description": "CloudCo services failed for three hours.",
                "publishedAt": "2026-09-20T12:00:00Z",
            },
            {
                "title": "CloudCo outage disrupts US customers",
                "description": "",
                "publishedAt": "2026-09-20T11:00:00Z",
            },
        ]

        def unavailable(_texts: list[str]) -> EmbeddingBatch:
            raise RuntimeError("embedding service unavailable")

        result = deduplicate_articles(articles, embedding_provider=unavailable)

        self.assertTrue(result.diagnostics.fallback_used)
        self.assertFalse(result.diagnostics.semantic_active)
        self.assertIn("embedding service unavailable", result.diagnostics.fallback_reason or "")
        self.assertEqual(result.diagnostics.merges[0].evidence.decision_reason, "lexical_fallback_match")

    def test_limit_preserves_newest_first_behavior(self) -> None:
        articles = [
            {
                "title": f"Company{i} launches product {i}",
                "description": f"Distinct event {i}",
                "publishedAt": f"2026-09-{i + 1:02d}T12:00:00Z",
            }
            for i in range(30)
        ]
        vectors = np.eye(30)

        result = deduplicate_articles(
            articles,
            limit=25,
            embedding_provider=lambda _texts: EmbeddingBatch(vectors, "test", True),
        )

        self.assertEqual(len(result.articles), 25)
        self.assertEqual(result.diagnostics.scanned_articles, 25)
        dates = [str(article["publishedAt"]) for article in result.articles]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_versioned_fixture_reports_reproducible_metrics(self) -> None:
        evaluation = evaluate_fixture(FIXTURE)

        self.assertEqual(evaluation.fixture_version, "1.0.0")
        self.assertEqual(evaluation.pair_count, 10)
        self.assertGreaterEqual(evaluation.precision, 0.9)
        self.assertGreaterEqual(evaluation.recall, 0.8)
        self.assertGreaterEqual(evaluation.f1, 0.85)
        self.assertEqual(
            evaluation.pair_count,
            evaluation.true_positives + evaluation.false_positives
            + evaluation.true_negatives + evaluation.false_negatives,
        )

    def test_fixture_is_human_inspectable_and_covers_required_cases(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cases = " ".join(str(pair["case"]) for pair in payload["pairs"])
        for expected in ("syndicated", "different date", "different organization", "amount", "acquisition", "sparse", "follow-up"):
            self.assertIn(expected, cases)
        self.assertEqual(payload["threshold_selection"]["selected"], payload["settings"]["threshold"])
        self.assertIn("precision", payload["threshold_selection"]["selection_rule"])


if __name__ == "__main__":
    unittest.main()
