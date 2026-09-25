# Persona-to-market-pain benchmark protocol

This protocol measures the project before making any time-compression claim. It preserves raw observations in JSONL and mechanically separates measured evidence from an aspirational resume statement.

## Comparable task

Manual and automated runs must start with the same finalized persona descriptions and end with the same review-ready deliverable:

- search English-language news from the preceding 30 days;
- retain no more than 25 distinct events, treating syndicated or semantically equivalent coverage as duplicates;
- produce ranked operational-pain hypotheses;
- attach source citations to every retained hypothesis; and
- account for every retained article in the analysis.

Timing starts when the persona descriptions are ready. It ends when the cited pain-point deliverable is ready for review. Credential setup, persona creation, and downstream feature ideation are outside both timings.

## Deterministic offline verification

This checks the timing harness and production deduplicator without making network calls or spending API credits:

```bash
cd /Users/ajorge/life/fuzzy-semantic-clustering
/usr/local/bin/python3.13 -m src.workflow_benchmark offline
/usr/local/bin/python3.13 -m unittest tests.test_workflow_benchmark
```

The offline result is a fixture validation, not performance evidence, and must not be used in a resume claim.

## Persona manifests

Store each benchmark selection in a small JSON file outside source control. Either supported shape is valid:

```json
{
  "label": "fintech data leaders",
  "persona_descriptions": [
    "VP or Head of Data at a mid-market fintech company responsible for analytics infrastructure, data quality, and governance."
  ]
}
```

The raw result records the manifest path, label, count, character count, and a stable content hash, but not the persona text. Keep the manifest alongside the raw results so the exact selection remains reproducible without copying customer-derived descriptions into every log line.

## Automated live observations

Live runs are deliberately opt-in. They reuse the same `.env` variables as the dashboard (`OPENAI_API_KEY`, `NEWSAPI_KEY`, and optional `OPENAI_MODEL`) and call the production query generator, NewsAPI page fetcher, semantic deduplicator, and pain analyzer directly.

```bash
cd /Users/ajorge/life/fuzzy-semantic-clustering
/usr/local/bin/python3.13 -m src.workflow_benchmark live \
  --confirm-live \
  --persona-file /absolute/path/to/persona-selection.json \
  --dataset-label customer-export-v1 \
  --output /absolute/path/to/benchmark-results.jsonl \
  --runs 2
```

Run at least five successful, quality-validated observations spanning at least three materially different persona selections. Do not delete failures or retries. The record retains UTC timestamps, monotonic end-to-end and per-stage durations, environment and revision metadata, status, timeouts, article counts, duplicates, pages, pain counts, and citation/coverage validation. API keys and original customer rows are never logged.

## Observed manual baseline

For each manual run, give the researcher the identical persona manifest and task definition. Record both:

- **elapsed time:** wall-clock time from ready persona to review-ready deliverable;
- **active time:** time the researcher actually spent searching, deduplicating, reasoning, citing, and formatting.

Enter measured values only. Do not estimate how long the task “normally” takes. Mark `--quality-validated` only after confirming that citations exist and every retained article was considered.

```bash
/usr/local/bin/python3.13 -m src.workflow_benchmark record-manual \
  --persona-file /absolute/path/to/persona-selection.json \
  --dataset-label customer-export-v1 \
  --output /absolute/path/to/benchmark-results.jsonl \
  --elapsed-minutes 145 \
  --active-minutes 130 \
  --observer "researcher-1" \
  --quality-validated
```

Collect at least three completed manual observations. A manual observation used by the claim gate must have a matching automated persona-selection hash.

## Aggregate and gate the claim

```bash
/usr/local/bin/python3.13 -m src.workflow_benchmark summarize \
  /absolute/path/to/benchmark-results.jsonl \
  --output /absolute/path/to/benchmark-summary.json
```

The summary reports samples, successes, failures, timeouts, median, p95, maximum, and stage-level latency. The statement **“compressed GTM research from hours to under 10 minutes”** is emitted as eligible only if all of these are true:

1. at least five quality-validated automated runs exist;
2. those runs span at least three persona selections;
3. at least three observed, quality-validated manual runs exist;
4. every manual persona selection has a matching automated run;
5. the automated median is below 600 seconds; and
6. the manual active-time median is at least 7,200 seconds.

Until the gate passes, report the measured automated runtime or say that the system consolidates query generation, research, deduplication, and cited synthesis into one workflow. Do not claim the unmeasured gap.
