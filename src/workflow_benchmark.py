from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence
from uuid import uuid4

from . import config as project_config
from .event_deduplication import (
    DedupeConfig,
    deduplicate_articles as deduplicate_news_events,
)
from .news_query import generate_news_query
from .pain_point_analysis import DEFAULT_OPENAI_MODEL, analyze_pain_points
from .serve_dashboard import (
    LOOKBACK_DAYS,
    MAX_PAGES,
    NEWSAPI_PAGE_SIZE,
    RESULT_LIMIT,
    fetch_news_page,
    load_local_env,
)


SCHEMA_VERSION = "1.0"
BENCHMARK_VERSION = "1.0"
TASK_DEFINITION = {
    "lookback_days": 30,
    "maximum_distinct_events": 25,
    "deduplication_required": True,
    "deliverable": "ranked operational-pain hypotheses with source citations",
    "timing_start": "persona descriptions are ready",
    "timing_end": "cited deliverable is ready for review",
}


QueryGenerator = Callable[[str, list[str], str], tuple[dict[str, object], str]]
NewsFetcher = Callable[[str, str, str, int], dict[str, object]]
ArticleDeduplicator = Callable[
    [list[dict[str, object]], int],
    tuple[list[dict[str, object]], int, int],
]
PainAnalyzer = Callable[
    [str, list[str], list[dict[str, object]], str],
    dict[str, object],
]


@dataclass(frozen=True)
class WorkflowDependencies:
    generate_query: QueryGenerator
    fetch_news_page: NewsFetcher
    deduplicate_articles: ArticleDeduplicator
    analyze_pain_points: PainAnalyzer


def _production_deduplicate_articles(
    articles: list[dict[str, object]], limit: int
) -> tuple[list[dict[str, object]], int, int]:
    result = deduplicate_news_events(
        articles,
        limit,
        config=DedupeConfig(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            embedding_backend="openai",
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", project_config.OPENAI_EMBEDDING_MODEL
            ),
            cache_dir=project_config.EMBEDDING_CACHE_DIR,
        ),
    )
    return (
        list(result.articles),
        result.diagnostics.removed_duplicates,
        result.diagnostics.scanned_articles,
    )


def _offline_deduplicate_articles(
    articles: list[dict[str, object]], limit: int
) -> tuple[list[dict[str, object]], int, int]:
    result = deduplicate_news_events(
        articles,
        limit,
        config=DedupeConfig(embedding_backend="tfidf-lsa"),
    )
    return (
        list(result.articles),
        result.diagnostics.removed_duplicates,
        result.diagnostics.scanned_articles,
    )


PRODUCTION_DEPENDENCIES = WorkflowDependencies(
    generate_query=generate_news_query,
    fetch_news_page=fetch_news_page,
    deduplicate_articles=_production_deduplicate_articles,
    analyze_pain_points=analyze_pain_points,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _persona_selection_id(persona_descriptions: Sequence[str]) -> str:
    normalized = json.dumps(
        [" ".join(value.split()) for value in persona_descriptions],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _git_revision(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _base_context(
    persona_descriptions: Sequence[str],
    *,
    dataset_label: str,
    persona_source: str,
    persona_selection_label: str,
    model: str,
) -> dict[str, object]:
    project_dir = Path(__file__).resolve().parents[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "task_definition": dict(TASK_DEFINITION),
        "input": {
            "dataset_label": dataset_label,
            "persona_source": persona_source,
            "persona_selection_label": persona_selection_label,
            "persona_selection_id": _persona_selection_id(persona_descriptions),
            "persona_count": len(persona_descriptions),
            "persona_character_count": sum(len(value) for value in persona_descriptions),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model": model,
            "code_revision": _git_revision(project_dir),
        },
    }


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(
        getattr(exc, "reason", None), (TimeoutError, socket.timeout)
    )


def _safe_error_message(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def _quality_validation(
    analysis: dict[str, object], articles: Sequence[dict[str, object]]
) -> tuple[bool, dict[str, object]]:
    article_rows = analysis.get("article_analysis")
    top_problems = analysis.get("top_problems")
    rows = article_rows if isinstance(article_rows, list) else []
    problems = top_problems if isinstance(top_problems, list) else []
    expected_ids = {f"A{index + 1}" for index in range(len(articles))}
    returned_ids = {
        str(row.get("article_id")) for row in rows if isinstance(row, dict)
    }
    citations_valid = True
    citation_count = 0
    for problem in problems:
        if not isinstance(problem, dict):
            citations_valid = False
            continue
        support = problem.get("supporting_article_ids")
        if not isinstance(support, list) or not support:
            citations_valid = False
            continue
        citation_count += len(support)
        if any(str(article_id) not in expected_ids for article_id in support):
            citations_valid = False
    coverage_valid = returned_ids == expected_ids and len(rows) == len(articles)
    has_deliverable = bool(articles) and bool(problems)
    passed = coverage_valid and citations_valid and has_deliverable
    return passed, {
        "article_coverage_valid": coverage_valid,
        "citations_valid": citations_valid,
        "has_cited_pain_deliverable": has_deliverable,
        "citation_count": citation_count,
    }


def _retrieve_and_deduplicate(
    news_api_key: str,
    query: str,
    from_date: str,
    dependencies: WorkflowDependencies,
) -> dict[str, object]:
    all_articles: list[dict[str, object]] = []
    unique_articles: list[dict[str, object]] = []
    total_results = 0
    duplicates_removed = 0
    scanned_results = 0
    pages_fetched = 0
    for page in range(1, MAX_PAGES + 1):
        upstream = dependencies.fetch_news_page(news_api_key, query, from_date, page)
        if upstream.get("status") == "error":
            raise RuntimeError(str(upstream.get("message") or "NewsAPI returned an error"))
        page_articles = upstream.get("articles")
        if not isinstance(page_articles, list):
            page_articles = []
        all_articles.extend(
            article for article in page_articles if isinstance(article, dict)
        )
        total_results = int(upstream.get("totalResults") or 0)
        pages_fetched = page
        unique_articles, duplicates_removed, scanned_results = (
            dependencies.deduplicate_articles(all_articles, RESULT_LIMIT)
        )
        if len(unique_articles) >= RESULT_LIMIT:
            break
        if not page_articles or len(all_articles) >= total_results:
            break
    return {
        "articles": unique_articles,
        "article_count": len(unique_articles),
        "duplicates_removed": duplicates_removed,
        "scanned_results": scanned_results,
        "total_results": total_results,
        "pages_fetched": pages_fetched,
    }


def run_automated_benchmark(
    *,
    persona_descriptions: Sequence[str],
    openai_api_key: str,
    news_api_key: str,
    dataset_label: str,
    persona_source: str,
    persona_selection_label: str,
    model: str = DEFAULT_OPENAI_MODEL,
    dependencies: WorkflowDependencies = PRODUCTION_DEPENDENCIES,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = utc_now,
) -> dict[str, object]:
    """Measure one real persona-to-cited-pain workflow run.

    Dependency injection exists solely so offline tests can replace network boundaries.
    Live execution uses ``PRODUCTION_DEPENDENCIES`` by default.
    """
    personas = [value.strip() for value in persona_descriptions if value.strip()]
    if not personas:
        raise ValueError("At least one non-empty persona description is required")
    record = _base_context(
        personas,
        dataset_label=dataset_label,
        persona_source=persona_source,
        persona_selection_label=persona_selection_label,
        model=model,
    )
    # Environment fingerprinting is harness setup, not part of the measured
    # persona-ready to deliverable-ready workflow.
    started_at = now()
    started = monotonic()
    record.update(
        {
            "record_type": "automated",
            "run_id": uuid4().hex,
            "started_at_utc": _iso_utc(started_at),
            "status": "running",
            "stage_durations_seconds": {},
            "output": {
                "article_count": 0,
                "duplicates_removed": 0,
                "scanned_results": 0,
                "total_results": 0,
                "pages_fetched": 0,
                "pain_hypothesis_count": 0,
                "quality_validation_passed": False,
            },
        }
    )
    stage_durations = record["stage_durations_seconds"]
    output = record["output"]
    current_stage = "query_generation"
    try:
        stage_started = monotonic()
        try:
            query_spec, query_model = dependencies.generate_query(
                openai_api_key, personas, model
            )
        finally:
            stage_durations[current_stage] = monotonic() - stage_started
        record["query"] = {
            "q": str(query_spec.get("q") or ""),
            "model": query_model,
        }

        current_stage = "news_retrieval_and_deduplication"
        stage_started = monotonic()
        try:
            from_date = (started_at - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
            retrieval = _retrieve_and_deduplicate(
                news_api_key,
                str(query_spec["q"]),
                from_date,
                dependencies,
            )
        finally:
            stage_durations[current_stage] = monotonic() - stage_started
        articles = retrieval.pop("articles")
        output.update(retrieval)

        current_stage = "pain_synthesis"
        stage_started = monotonic()
        try:
            analysis = dependencies.analyze_pain_points(
                openai_api_key, personas, articles, model
            )
        finally:
            stage_durations[current_stage] = monotonic() - stage_started
        problems = analysis.get("top_problems")
        output["pain_hypothesis_count"] = len(problems) if isinstance(problems, list) else 0
        quality_passed, quality_details = _quality_validation(analysis, articles)
        output["quality_validation_passed"] = quality_passed
        output["quality_validation"] = quality_details
        output["analysis_model"] = str(analysis.get("model") or model)
        record["status"] = "success"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        record["status"] = "timeout" if _is_timeout_error(exc) else "failure"
        record["failure"] = {
            "stage": current_stage,
            "error_type": type(exc).__name__,
            "message": _safe_error_message(exc),
        }
    finally:
        record["ended_at_utc"] = _iso_utc(now())
        record["duration_seconds"] = monotonic() - started
    return record


def record_manual_observation(
    *,
    persona_descriptions: Sequence[str],
    dataset_label: str,
    persona_source: str,
    persona_selection_label: str,
    elapsed_minutes: float,
    active_minutes: float,
    observer: str,
    quality_validation_passed: bool,
    notes: str = "",
    now: Callable[[], datetime] = utc_now,
) -> dict[str, object]:
    if elapsed_minutes <= 0 or active_minutes <= 0:
        raise ValueError("Observed manual durations must be positive")
    if active_minutes > elapsed_minutes:
        raise ValueError("Active manual time cannot exceed elapsed manual time")
    personas = [value.strip() for value in persona_descriptions if value.strip()]
    if not personas:
        raise ValueError("At least one non-empty persona description is required")
    record = _base_context(
        personas,
        dataset_label=dataset_label,
        persona_source=persona_source,
        persona_selection_label=persona_selection_label,
        model="manual",
    )
    record.update(
        {
            "record_type": "manual",
            "run_id": uuid4().hex,
            "recorded_at_utc": _iso_utc(now()),
            "status": "success",
            "observation": {
                "observed": True,
                "observer": observer,
                "elapsed_seconds": elapsed_minutes * 60.0,
                "active_seconds": active_minutes * 60.0,
                "quality_validation_passed": quality_validation_passed,
                "notes": notes,
            },
        }
    )
    return record


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(value)
    return records


def _percentile_95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _metric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": _percentile_95(values),
        "maximum": max(values) if values else None,
    }


def _task_is_comparable(record: dict[str, object]) -> bool:
    return record.get("task_definition") == TASK_DEFINITION


def aggregate_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(records)
    automated = [row for row in rows if row.get("record_type") == "automated"]
    manual = [row for row in rows if row.get("record_type") == "manual"]
    successful_auto = [row for row in automated if row.get("status") == "success"]
    quality_auto = [
        row
        for row in successful_auto
        if isinstance(row.get("output"), dict)
        and row["output"].get("quality_validation_passed") is True
        and int(row["output"].get("article_count") or 0) > 0
        and int(row["output"].get("pain_hypothesis_count") or 0) > 0
        and _task_is_comparable(row)
    ]
    successful_manual = [
        row
        for row in manual
        if row.get("status") == "success"
        and isinstance(row.get("observation"), dict)
        and row["observation"].get("observed") is True
        and row["observation"].get("quality_validation_passed") is True
        and _task_is_comparable(row)
    ]
    auto_durations = [float(row["duration_seconds"]) for row in successful_auto]
    quality_auto_durations = [float(row["duration_seconds"]) for row in quality_auto]
    manual_elapsed = [
        float(row["observation"]["elapsed_seconds"]) for row in successful_manual
    ]
    manual_active = [
        float(row["observation"]["active_seconds"]) for row in successful_manual
    ]
    stage_names = (
        "query_generation",
        "news_retrieval_and_deduplication",
        "pain_synthesis",
    )
    stage_summary: dict[str, object] = {}
    for stage in stage_names:
        values = [
            float(row["stage_durations_seconds"][stage])
            for row in successful_auto
            if isinstance(row.get("stage_durations_seconds"), dict)
            and stage in row["stage_durations_seconds"]
        ]
        stage_summary[stage] = _metric_summary(values)

    auto_ids = {
        str(row.get("input", {}).get("persona_selection_id"))
        for row in quality_auto
        if isinstance(row.get("input"), dict)
    }
    manual_ids = {
        str(row.get("input", {}).get("persona_selection_id"))
        for row in successful_manual
        if isinstance(row.get("input"), dict)
    }
    auto_median = statistics.median(quality_auto_durations) if quality_auto_durations else None
    manual_active_median = statistics.median(manual_active) if manual_active else None
    requirements = {
        "at_least_five_quality_automated_runs": len(quality_auto) >= 5,
        "at_least_three_persona_selections": len(auto_ids) >= 3,
        "at_least_three_quality_manual_observations": len(successful_manual) >= 3,
        "manual_personas_have_automated_matches": bool(manual_ids) and manual_ids <= auto_ids,
        "automated_median_below_10_minutes": auto_median is not None and auto_median < 600,
        "manual_active_median_at_least_2_hours": (
            manual_active_median is not None and manual_active_median >= 7200
        ),
    }
    eligible = all(requirements.values())
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _iso_utc(utc_now()),
        "task_definition": dict(TASK_DEFINITION),
        "automated": {
            "total_runs": len(automated),
            "successes": len(successful_auto),
            "quality_validated_successes": len(quality_auto),
            "failures": sum(row.get("status") == "failure" for row in automated),
            "timeouts": sum(row.get("status") == "timeout" for row in automated),
            "duration_seconds": _metric_summary(auto_durations),
            "quality_validated_duration_seconds": _metric_summary(quality_auto_durations),
            "stage_duration_seconds": stage_summary,
            "distinct_persona_selections": len(auto_ids),
        },
        "manual": {
            "total_observations": len(manual),
            "quality_validated_observations": len(successful_manual),
            "elapsed_seconds": _metric_summary(manual_elapsed),
            "active_seconds": _metric_summary(manual_active),
        },
        "claim_gate": {
            "claim": "compressed GTM research from hours to under 10 minutes",
            "eligible": eligible,
            "requirements": requirements,
            "unmet_requirements": [name for name, passed in requirements.items() if not passed],
        },
    }
    if eligible and auto_median and manual_active_median:
        summary["claim_gate"]["median_time_reduction_percent"] = (
            1.0 - auto_median / manual_active_median
        ) * 100.0
    return summary


def _load_persona_file(path: Path) -> tuple[list[str], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        personas = value
        label = path.stem
    elif isinstance(value, dict):
        personas = value.get("persona_descriptions")
        label = str(value.get("label") or path.stem)
    else:
        raise ValueError("Persona file must contain a JSON list or object")
    if not isinstance(personas, list) or any(not isinstance(item, str) for item in personas):
        raise ValueError("persona_descriptions must be a JSON array of strings")
    cleaned = [item.strip() for item in personas if item.strip()]
    if not cleaned:
        raise ValueError("Persona file contains no non-empty descriptions")
    return cleaned, label


def _offline_dependencies() -> WorkflowDependencies:
    articles = [
        {
            "title": "Payments provider launches compliance platform",
            "description": "The release adds reporting controls.",
            "url": "https://example.test/one",
        },
        {
            "title": "Payments provider unveils compliance platform",
            "description": "The same release adds reporting controls.",
            "url": "https://example.test/two",
        },
    ]

    def query(_: str, __: list[str], model: str) -> tuple[dict[str, object], str]:
        return ({"q": "payments AND compliance"}, model)

    def fetch(_: str, __: str, ___: str, page: int) -> dict[str, object]:
        return {"status": "ok", "totalResults": 2, "articles": articles if page == 1 else []}

    def analyze(
        _: str, __: list[str], retained: list[dict[str, object]], model: str
    ) -> dict[str, object]:
        return {
            "model": model,
            "top_problems": [
                {
                    "problem_name": "Compliance reporting changes",
                    "supporting_article_ids": ["A1"],
                }
            ],
            "article_analysis": [
                {"article_id": f"A{index + 1}"} for index in range(len(retained))
            ],
        }

    return WorkflowDependencies(query, fetch, _offline_deduplicate_articles, analyze)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the persona-to-cited-market-pain workflow."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    offline = commands.add_parser("offline", help="Run deterministic fixtures without network access")
    offline.add_argument("--output", type=Path)

    live = commands.add_parser("live", help="Run the real paid/network workflow")
    live.add_argument("--persona-file", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--dataset-label", required=True)
    live.add_argument("--selection-label")
    live.add_argument("--runs", type=int, default=1)
    live.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that this command uses API credits and network access",
    )

    manual = commands.add_parser("record-manual", help="Append an observed manual baseline")
    manual.add_argument("--persona-file", type=Path, required=True)
    manual.add_argument("--output", type=Path, required=True)
    manual.add_argument("--dataset-label", required=True)
    manual.add_argument("--selection-label")
    manual.add_argument("--elapsed-minutes", type=float, required=True)
    manual.add_argument("--active-minutes", type=float, required=True)
    manual.add_argument("--observer", required=True)
    manual.add_argument("--notes", default="")
    manual.add_argument("--quality-validated", action="store_true")

    summarize = commands.add_parser("summarize", help="Aggregate raw JSONL records")
    summarize.add_argument("results", type=Path)
    summarize.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "offline":
        personas = ["Data leaders at payments companies responsible for compliance reporting."]
        record = run_automated_benchmark(
            persona_descriptions=personas,
            openai_api_key="offline-not-a-key",
            news_api_key="offline-not-a-key",
            dataset_label="offline-fixture",
            persona_source="built-in deterministic fixture",
            persona_selection_label="payments compliance fixture",
            dependencies=_offline_dependencies(),
        )
        if args.output:
            append_jsonl(args.output, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0 if record["status"] == "success" else 1

    if args.command == "live":
        if not args.confirm_live:
            raise SystemExit("Live runs require --confirm-live because they use network APIs and credits.")
        if args.runs < 1:
            raise SystemExit("--runs must be at least 1")
        personas, default_label = _load_persona_file(args.persona_file)
        load_local_env(Path(__file__).resolve().parents[1] / ".env")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        news_key = os.getenv("NEWSAPI_KEY", "").strip()
        if not openai_key or not news_key:
            raise SystemExit("OPENAI_API_KEY and NEWSAPI_KEY must be configured in the environment or project .env.")
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        exit_code = 0
        for _ in range(args.runs):
            record = run_automated_benchmark(
                persona_descriptions=personas,
                openai_api_key=openai_key,
                news_api_key=news_key,
                dataset_label=args.dataset_label,
                persona_source=str(args.persona_file),
                persona_selection_label=args.selection_label or default_label,
                model=model,
            )
            append_jsonl(args.output, record)
            print(json.dumps(record, indent=2, sort_keys=True))
            if record["status"] != "success":
                exit_code = 1
        return exit_code

    if args.command == "record-manual":
        personas, default_label = _load_persona_file(args.persona_file)
        record = record_manual_observation(
            persona_descriptions=personas,
            dataset_label=args.dataset_label,
            persona_source=str(args.persona_file),
            persona_selection_label=args.selection_label or default_label,
            elapsed_minutes=args.elapsed_minutes,
            active_minutes=args.active_minutes,
            observer=args.observer,
            quality_validation_passed=args.quality_validated,
            notes=args.notes,
        )
        append_jsonl(args.output, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    summary = aggregate_records(load_jsonl(args.results))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
