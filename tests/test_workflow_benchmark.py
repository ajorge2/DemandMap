from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.workflow_benchmark import (
    TASK_DEFINITION,
    WorkflowDependencies,
    aggregate_records,
    append_jsonl,
    load_jsonl,
    run_automated_benchmark,
)


def _analysis(articles: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model": "test-model",
        "top_problems": [
            {"problem_name": "Reporting pressure", "supporting_article_ids": ["A1"]}
        ],
        "article_analysis": [
            {"article_id": f"A{index + 1}"} for index in range(len(articles))
        ],
    }


def _dependencies(*, query_error: BaseException | None = None) -> WorkflowDependencies:
    def query(_: str, __: list[str], model: str) -> tuple[dict[str, object], str]:
        if query_error:
            raise query_error
        return {"q": "fintech AND compliance"}, model

    def fetch(_: str, __: str, ___: str, page: int) -> dict[str, object]:
        return {
            "status": "ok",
            "totalResults": 1,
            "articles": [{"title": "A regulation changed"}] if page == 1 else [],
        }

    def dedupe(
        articles: list[dict[str, object]], _: int
    ) -> tuple[list[dict[str, object]], int, int]:
        return articles, 0, len(articles)

    def analyze(
        _: str, __: list[str], articles: list[dict[str, object]], ___: str
    ) -> dict[str, object]:
        return _analysis(articles)

    return WorkflowDependencies(query, fetch, dedupe, analyze)


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _automated_record(
    duration: float,
    selection: str,
    *,
    status: str = "success",
    quality: bool = True,
) -> dict[str, object]:
    return {
        "record_type": "automated",
        "status": status,
        "duration_seconds": duration,
        "task_definition": dict(TASK_DEFINITION),
        "input": {"persona_selection_id": selection},
        "stage_durations_seconds": {
            "query_generation": 10.0,
            "news_retrieval_and_deduplication": 20.0,
            "pain_synthesis": 30.0,
        },
        "output": {
            "quality_validation_passed": quality,
            "article_count": 5,
            "pain_hypothesis_count": 3,
        },
    }


def _manual_record(active: float, selection: str) -> dict[str, object]:
    return {
        "record_type": "manual",
        "status": "success",
        "task_definition": dict(TASK_DEFINITION),
        "input": {"persona_selection_id": selection},
        "observation": {
            "observed": True,
            "quality_validation_passed": True,
            "elapsed_seconds": active + 600,
            "active_seconds": active,
        },
    }


class WorkflowBenchmarkTests(unittest.TestCase):
    def test_records_real_boundaries_and_separate_monotonic_durations(self) -> None:
        now_values = iter(
            [
                datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
                datetime(2026, 9, 24, 12, 2, tzinfo=timezone.utc),
            ]
        )
        record = run_automated_benchmark(
            persona_descriptions=["A fintech data leader"],
            openai_api_key="not-real",
            news_api_key="not-real",
            dataset_label="fixture",
            persona_source="fixture.json",
            persona_selection_label="fintech",
            model="test-model",
            dependencies=_dependencies(),
            monotonic=_Clock([0, 1, 3, 4, 9, 10, 16, 20]),
            now=lambda: next(now_values),
        )

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["duration_seconds"], 20)
        self.assertEqual(
            record["stage_durations_seconds"],
            {
                "query_generation": 2,
                "news_retrieval_and_deduplication": 5,
                "pain_synthesis": 6,
            },
        )
        self.assertEqual(record["output"]["article_count"], 1)
        self.assertTrue(record["output"]["quality_validation_passed"])
        self.assertEqual(record["started_at_utc"], "2026-09-24T12:00:00Z")

    def test_retains_failure_stage_and_timeout(self) -> None:
        now = lambda: datetime(2026, 9, 24, tzinfo=timezone.utc)
        record = run_automated_benchmark(
            persona_descriptions=["A persona"],
            openai_api_key="not-real",
            news_api_key="not-real",
            dataset_label="fixture",
            persona_source="fixture.json",
            persona_selection_label="failure",
            dependencies=_dependencies(query_error=TimeoutError("model timed out")),
            monotonic=_Clock([0, 1, 4, 5]),
            now=now,
        )

        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["failure"]["stage"], "query_generation")
        self.assertEqual(record["stage_durations_seconds"]["query_generation"], 3)
        self.assertEqual(record["duration_seconds"], 5)

    def test_aggregation_reports_distribution_and_failures(self) -> None:
        records = [
            _automated_record(100, "a"),
            _automated_record(200, "b"),
            _automated_record(300, "c"),
            _automated_record(0, "d", status="failure"),
            _automated_record(0, "e", status="timeout"),
        ]
        summary = aggregate_records(records)

        self.assertEqual(summary["automated"]["successes"], 3)
        self.assertEqual(summary["automated"]["failures"], 1)
        self.assertEqual(summary["automated"]["timeouts"], 1)
        self.assertEqual(summary["automated"]["duration_seconds"]["median"], 200)
        self.assertEqual(summary["automated"]["duration_seconds"]["p95"], 300)

    def test_claim_is_gated_on_samples_personas_quality_and_thresholds(self) -> None:
        automated = [
            _automated_record(duration, selection)
            for duration, selection in [
                (300, "a"), (400, "b"), (500, "c"), (550, "a"), (590, "b")
            ]
        ]
        manual = [
            _manual_record(active, selection)
            for active, selection in [(7200, "a"), (7500, "b"), (7800, "c")]
        ]
        summary = aggregate_records(automated + manual)

        self.assertTrue(summary["claim_gate"]["eligible"])
        self.assertGreater(summary["claim_gate"]["median_time_reduction_percent"], 90)

        automated[0]["output"]["quality_validation_passed"] = False
        blocked = aggregate_records(automated + manual)
        self.assertFalse(blocked["claim_gate"]["eligible"])
        self.assertIn(
            "at_least_five_quality_automated_runs",
            blocked["claim_gate"]["unmet_requirements"],
        )

    def test_jsonl_round_trip_preserves_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            append_jsonl(path, _automated_record(0, "a", status="failure"))
            append_jsonl(path, _automated_record(120, "a"))

            loaded = load_jsonl(path)

        self.assertEqual([row["status"] for row in loaded], ["failure", "success"])


if __name__ == "__main__":
    unittest.main()
