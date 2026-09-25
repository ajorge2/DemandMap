from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import pandas as pd

from . import config
from .customer_datasets import CustomerDataset, make_customer_dataset, parse_uploaded_csv
from .embeddings import generate_combined_field_embeddings, generate_embeddings
from .event_deduplication import (
    DedupeConfig,
    deduplicate_articles as deduplicate_news_events,
    diagnostics_to_dict,
)
from .fuzzy_cluster import fit_fuzzy_cmeans
from .news_query import generate_news_query
from .pain_point_analysis import DEFAULT_OPENAI_MODEL, analyze_pain_points
from .persona_generation import (
    build_fallback_grounding,
    build_persona_model_input,
    generate_persona_descriptions,
)
from .persona_labels import generate_persona_labels
from .representative_accounts import build_representative_accounts
from .visualize import pca_2d


NEWS_API_URL = "https://newsapi.org/v2/everything"
LOOKBACK_DAYS = 30
RESULT_LIMIT = 25
NEWSAPI_PAGE_SIZE = 100
MAX_PAGES = 5
MAX_ACTIVE_UPLOADS = 8
_DATASETS: OrderedDict[str, CustomerDataset] = OrderedDict()
_DATASET_LOCK = Lock()


def _demo_dataset() -> CustomerDataset:
    with _DATASET_LOCK:
        existing = _DATASETS.get("demo")
        if existing is not None:
            return existing
        dataset = make_customer_dataset(
            "demo", "B2B AI Startups Marketing Tech Export.csv",
            pd.read_csv(config.INPUT_CSV), is_demo=True,
        )
        dataset.clusterable_fields = tuple(
            field for field in config.CLUSTERABLE_FIELDS if field in dataset.dataframe.columns
        )
        dataset.default_fields = (config.SEMANTIC_FIELD,)
        dataset.display_fields = ("Full Name", "Job Title", "Company")
        _DATASETS["demo"] = dataset
        return dataset


def get_dataset(dataset_id: str) -> CustomerDataset:
    if dataset_id == "demo":
        return _demo_dataset()
    with _DATASET_LOCK:
        dataset = _DATASETS.get(dataset_id)
        if dataset is None:
            raise ValueError("This uploaded dataset is no longer available. Upload the CSV again.")
        _DATASETS.move_to_end(dataset_id)
        return dataset


def register_uploaded_dataset(filename: str, csv_text: str) -> CustomerDataset:
    dataset_id = uuid4().hex
    dataset = make_customer_dataset(
        dataset_id, filename, parse_uploaded_csv(csv_text), is_demo=False
    )
    with _DATASET_LOCK:
        _DATASETS[dataset_id] = dataset
        uploaded_ids = [key for key in _DATASETS if key != "demo"]
        while len(uploaded_ids) > MAX_ACTIVE_UPLOADS:
            _DATASETS.pop(uploaded_ids.pop(0), None)
    return dataset


def dashboard_embedding_backend() -> str:
    return config.EMBEDDING_BACKEND

def fetch_news_page(api_key: str, query: str, from_date: str, page: int) -> dict[str, object]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": NEWSAPI_PAGE_SIZE,
            "page": page,
        }
    )
    request = urllib.request.Request(
        f"{NEWS_API_URL}?{params}",
        headers={"X-Api-Key": api_key, "User-Agent": "fuzzy-persona-news-prototype/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


@lru_cache(maxsize=64)
def build_clustering_payload(
    dataset_id: str, selected_fields: tuple[str, ...]
) -> dict[str, object]:
    dataset = get_dataset(dataset_id)
    df = dataset.dataframe
    load_local_env(config.PROJECT_DIR / ".env")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL", config.OPENAI_EMBEDDING_MODEL
    ).strip()
    texts: list[str] = []
    for _, row in df.iterrows():
        values = []
        for field in selected_fields:
            value = row.get(field)
            if pd.notna(value) and str(value).strip():
                values.append(f"{field}: {str(value).strip()}")
        texts.append(" | ".join(values) or "No populated values in selected fields")

    embedding_backend = dashboard_embedding_backend()
    if embedding_backend == "openai":
        embedding = generate_combined_field_embeddings(
            {
                field: df[field].tolist()
                for field in selected_fields
            },
            api_key=openai_key,
            model_name=embedding_model,
            cache_dir=config.EMBEDDING_CACHE_DIR,
        )
    else:
        embedding = generate_embeddings(
            texts,
            backend=embedding_backend,
            model_name=config.SENTENCE_TRANSFORMER_MODEL,
            lsa_dimensions=config.LSA_DIMENSIONS,
        )
    embedding_trace: dict[str, object] = {
        "id": "embedding-generation",
        "kind": "model" if embedding.backend == "openai" else "local",
        "label": "Embed selected CSV fields",
        "status": "completed",
        "model": str(embedding.metadata.get("model") or embedding.backend),
        "input": "Each nonblank selected CSV cell, cached by text hash",
        "output": "One combined normalized vector per customer row",
        "detail": (
            f"{embedding.metadata.get('model')} · "
            f"{embedding.metadata.get('cache_hits', 0)} cached values · "
            f"{embedding.metadata.get('api_inputs', 0)} newly embedded values"
            if embedding.backend == "openai"
            else embedding.backend
        ),
    }
    points = pca_2d(embedding.vectors)
    models: dict[str, object] = {}
    memberships_by_k: dict[str, np.ndarray] = {}
    minimum_k = 3 if len(df) >= 4 else 2
    maximum_k = min(max(config.K_RANGE), len(df) - 1)
    k_values = range(minimum_k, maximum_k + 1)
    for k in k_values:
        if k >= len(df):
            continue
        result = fit_fuzzy_cmeans(
            embedding.vectors,
            k,
            fuzziness=config.FUZZINESS,
            n_initializations=config.N_INITIALIZATIONS,
            max_iterations=config.MAX_ITERATIONS,
            tolerance=config.TOLERANCE,
            seed=config.RANDOM_SEED,
        )
        labels, evidence = generate_persona_labels(
            df, selected_fields, result.independent_memberships
        )
        memberships_by_k[str(k)] = result.independent_memberships
        models[str(k)] = {
            "labels": labels,
            "descriptions": ["Deterministic fallback label based on representative values."] * len(labels),
            "evidence": evidence,
            "memberships": [
                [round(float(value), 6) for value in row]
                for row in result.independent_memberships
            ],
            "representativeAccounts": build_representative_accounts(
                df,
                result.independent_memberships,
                selected_fields,
                dataset.display_fields,
            ),
        }
    persona_inputs = build_persona_model_input(
        df, list(selected_fields), memberships_by_k
    )
    fallback_summary_maps: dict[str, dict[str, str]] = {}
    for k_text, model in models.items():
        for cluster, summaries in enumerate(model["evidence"]):
            fallback_summary_maps[f"K{k_text}-C{cluster}"] = dict(summaries)
    fallback_grounding = build_fallback_grounding(
        persona_inputs, fallback_summary_maps
    )
    for k_text, model in models.items():
        model["personaGrounding"] = [
            fallback_grounding[f"K{k_text}-C{cluster}"]
            for cluster in range(len(model["labels"]))
        ]
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    persona_trace: dict[str, object] = {
        "id": "persona-description-generation",
        "kind": "model",
        "label": "Generate persona candidate descriptions",
        "status": "fallback",
        "model": model_name,
        "input": "Bounded representative values from every selected CSV column",
        "output": "Natural persona names, descriptions, and per-field summaries",
    }
    if openai_key:
        try:
            generated, resolved_model = generate_persona_descriptions(
                openai_key, df, list(selected_fields), memberships_by_k, model=model_name
            )
            for k_text, candidates in generated.items():
                models[k_text]["labels"] = [str(candidate["name"]) for candidate in candidates]
                models[k_text]["descriptions"] = [
                    str(candidate["description"]) for candidate in candidates
                ]
                models[k_text]["evidence"] = [
                    {
                        str(summary["field"]): str(summary["summary"])
                        for summary in candidate["field_summaries"]
                    }
                    for candidate in candidates
                ]
                models[k_text]["personaGrounding"] = [
                    {
                        "candidateId": str(candidate["candidate_id"]),
                        "descriptionEvidence": candidate["description_evidence"],
                        "fieldSummaries": candidate["field_summaries"],
                        "sourceEvidence": candidate["source_evidence"],
                    }
                    for candidate in candidates
                ]
            persona_trace.update({"status": "completed", "model": resolved_model})
        except Exception as exc:
            persona_trace["detail"] = f"Model call failed; using deterministic labels: {exc}"
    else:
        persona_trace["detail"] = "OPENAI_API_KEY is missing; using deterministic labels."

    records = dataset.client_profile()["records"]
    for index, point in enumerate(points):
        records[index]["x"] = round(float(point[0]), 6)
        records[index]["y"] = round(float(point[1]), 6)
    return {
        "status": "ok",
        "datasetId": dataset.dataset_id,
        "datasetName": dataset.filename,
        "rowCount": len(df),
        "selectedFields": list(selected_fields),
        "embeddingBackend": embedding.backend,
        "embeddingMetadata": embedding.metadata,
        "minimumK": minimum_k,
        "maximumK": maximum_k,
        "defaultK": min(6, maximum_k),
        "records": records,
        "points": [
            {"x": round(float(point[0]), 6), "y": round(float(point[1]), 6)}
            for point in points
        ],
        "models": models,
        "modelTrace": [embedding_trace, persona_trace],
    }


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(config.OUTPUT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self, max_bytes: int = 65536) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > max_bytes:
            raise ValueError("Invalid request size")
        payload = json.loads(self.rfile.read(content_length))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/clustering_explorer.html")
            self.end_headers()
            return
        if self.path == "/health":
            load_local_env(config.PROJECT_DIR / ".env")
            self._send_json(
                200,
                {
                    "status": "ok",
                    "newsapi_key_configured": bool(os.getenv("NEWSAPI_KEY")),
                    "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
                    "openai_model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
                    "openai_embedding_model": os.getenv(
                        "OPENAI_EMBEDDING_MODEL", config.OPENAI_EMBEDDING_MODEL
                    ),
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/datasets":
            try:
                payload = self._read_json_body(max_bytes=9 * 1024 * 1024)
                filename = str(payload.get("filename") or "customers.csv")
                csv_text = payload.get("csvText")
                if not isinstance(csv_text, str):
                    raise ValueError("csvText must be a string")
                dataset = register_uploaded_dataset(filename, csv_text)
                profile = dataset.client_profile()
                profile.update({
                    "status": "ok",
                    "minimumK": 3 if len(dataset.dataframe) >= 4 else 2,
                    "maximumK": min(max(config.K_RANGE), len(dataset.dataframe) - 1),
                })
                self._send_json(200, profile)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"status": "error", "message": str(exc)})
            except Exception as exc:
                self._send_json(500, {"status": "error", "message": f"Could not load CSV: {exc}"})
            return

        if self.path == "/api/clusters":
            try:
                payload = self._read_json_body()
                dataset_id = str(payload.get("dataset_id") or "demo")
                dataset = get_dataset(dataset_id)
                requested_fields = payload.get("fields")
                if not isinstance(requested_fields, list):
                    raise ValueError("fields must be a JSON array")
                if any(not isinstance(field, str) for field in requested_fields):
                    raise ValueError("Every clustering field must be a string")
                unknown = [
                    str(field) for field in requested_fields
                    if field not in dataset.clusterable_fields
                ]
                if unknown:
                    raise ValueError(f"Unsupported clustering fields: {', '.join(unknown)}")
                requested = set(requested_fields)
                selected_fields = tuple(
                    field for field in dataset.clusterable_fields if field in requested
                )
                if not selected_fields:
                    raise ValueError("Select at least one clustering field")
                self._send_json(200, build_clustering_payload(dataset_id, selected_fields))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"status": "error", "message": str(exc)})
            except Exception as exc:
                self._send_json(
                    500,
                    {"status": "error", "message": f"Could not recompute clustering: {exc}"},
                )
            return

        if self.path != "/api/news":
            self._send_json(404, {"status": "error", "message": "Not found"})
            return
        load_local_env(config.PROJECT_DIR / ".env")
        api_key = os.getenv("NEWSAPI_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._send_json(
                503,
                {
                    "status": "error",
                    "code": "missingApiKey",
                    "message": "NEWSAPI_KEY is not configured in the project .env file.",
                },
            )
            return
        if not openai_api_key:
            self._send_json(
                503,
                {
                    "status": "error",
                    "code": "missingOpenAIKey",
                    "message": "OPENAI_API_KEY is not configured in the project .env file.",
                },
            )
            return
        try:
            payload = self._read_json_body()
            persona_descriptions = payload.get("persona_descriptions")
            if not isinstance(persona_descriptions, list) or not persona_descriptions:
                raise ValueError("At least one persona description is required")
            if any(not isinstance(value, str) or not value.strip() for value in persona_descriptions):
                raise ValueError("Every persona description must be a non-empty string")
            if len(persona_descriptions) > 7 or any(len(value) > 4000 for value in persona_descriptions):
                raise ValueError("Persona descriptions exceed the supported request size")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"status": "error", "message": str(exc)})
            return

        model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        try:
            query_spec, query_model = generate_news_query(
                openai_api_key,
                [value.strip() for value in persona_descriptions],
                model=model_name,
            )
            query = str(query_spec["q"])
        except urllib.error.HTTPError as exc:
            try:
                openai_error = json.loads(exc.read().decode("utf-8"))
                message = str(openai_error.get("error", {}).get("message") or "OpenAI query generation failed")
            except Exception:
                message = f"OpenAI returned HTTP {exc.code}"
            self._send_json(502, {"status": "error", "code": "openaiQueryError", "message": message})
            return
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(502, {"status": "error", "code": "openaiQueryError", "message": f"Could not generate the NewsAPI query: {exc}"})
            return

        from_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
        try:
            all_articles: list[dict[str, object]] = []
            total_results = 0
            unique_articles: list[dict[str, object]] = []
            duplicates_removed = 0
            scanned_results = 0
            pages_fetched = 0
            deduplication_diagnostics: dict[str, object] = {}
            for page in range(1, MAX_PAGES + 1):
                upstream = fetch_news_page(api_key, query, from_date, page)
                if upstream.get("status") == "error":
                    self._send_json(502, upstream)
                    return
                page_articles = upstream.get("articles") or []
                if not isinstance(page_articles, list):
                    page_articles = []
                all_articles.extend(page_articles)
                total_results = int(upstream.get("totalResults") or 0)
                pages_fetched = page
                deduplication = deduplicate_news_events(
                    all_articles,
                    RESULT_LIMIT,
                    config=DedupeConfig(
                        api_key=openai_api_key,
                        embedding_backend="openai",
                        embedding_model=os.getenv(
                            "OPENAI_EMBEDDING_MODEL", config.OPENAI_EMBEDDING_MODEL
                        ),
                        cache_dir=config.EMBEDDING_CACHE_DIR,
                    ),
                )
                unique_articles = list(deduplication.articles)
                duplicates_removed = deduplication.diagnostics.removed_duplicates
                scanned_results = deduplication.diagnostics.scanned_articles
                deduplication_diagnostics = diagnostics_to_dict(
                    deduplication.diagnostics
                )
                if len(unique_articles) >= RESULT_LIMIT:
                    break
                if not page_articles or len(all_articles) >= total_results:
                    break
            try:
                analysis = analyze_pain_points(
                    openai_api_key,
                    [value.strip() for value in persona_descriptions],
                    unique_articles,
                    model=model_name,
                )
            except urllib.error.HTTPError as exc:
                try:
                    openai_error = json.loads(exc.read().decode("utf-8"))
                    message = str(openai_error.get("error", {}).get("message") or "OpenAI analysis failed")
                except Exception:
                    message = f"OpenAI returned HTTP {exc.code}"
                self._send_json(
                    502,
                    {
                        "status": "error",
                        "code": "openaiAnalysisError",
                        "stage": "pain-point-analysis",
                        "message": message,
                        "query": query_spec,
                        "modelTrace": [{
                            "id": "news-query-generation",
                            "status": "completed",
                            "model": query_model,
                        }],
                    },
                )
                return
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(
                    502,
                    {
                        "status": "error",
                        "code": "openaiAnalysisError",
                        "stage": "pain-point-analysis",
                        "message": f"Could not complete OpenAI analysis: {exc}",
                        "query": query_spec,
                        "modelTrace": [{
                            "id": "news-query-generation",
                            "status": "completed",
                            "model": query_model,
                        }],
                    },
                )
                return
            self._send_json(
                200,
                {
                    "status": "ok",
                    "totalResults": total_results,
                    "articles": unique_articles,
                    "returnedResults": len(unique_articles),
                    "duplicatesRemoved": duplicates_removed,
                    "scannedResults": scanned_results,
                    "deduplication": deduplication_diagnostics,
                    "pagesFetched": pages_fetched,
                    "lookbackDays": LOOKBACK_DAYS,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "query": query_spec,
                    "analysis": analysis,
                    "modelTrace": [
                        {
                            "id": "news-query-generation",
                            "kind": "model",
                            "label": "Generate NewsAPI query",
                            "status": "completed",
                            "model": query_model,
                            "input": "Only the visible text of selected persona candidates",
                            "output": "NewsAPI q plus market, event, and functional terms",
                        },
                        {
                            "id": "news-event-deduplication",
                            "kind": "model + local",
                            "label": "Deduplicate news events",
                            "status": "completed",
                            "model": deduplication_diagnostics.get("model")
                            or deduplication_diagnostics.get("backend"),
                            "input": "NewsAPI titles and descriptions",
                            "output": "Up to 25 newest-first event candidates with merge provenance",
                            "detail": (
                                "Lexical fallback used"
                                if deduplication_diagnostics.get("fallback_used")
                                else "Cached semantic vectors plus local event guards"
                            ),
                        },
                        {
                            "id": "pain-point-analysis",
                            "kind": "model",
                            "label": "Analyze operational pain hypotheses",
                            "status": "completed",
                            "model": analysis.get("model", model_name),
                            "input": "Selected persona text plus deduplicated article titles and descriptions",
                            "output": "Ranked pain hypotheses, evidence ledger, and per-article audit",
                        },
                    ],
                },
            )
        except urllib.error.HTTPError as exc:
            try:
                upstream_error = json.loads(exc.read().decode("utf-8"))
            except Exception:
                upstream_error = {"status": "error", "message": f"NewsAPI returned HTTP {exc.code}"}
            upstream_error["stage"] = "news-retrieval"
            upstream_error["query"] = query_spec
            upstream_error["modelTrace"] = [{
                "id": "news-query-generation",
                "status": "completed",
                "model": query_model,
            }]
            self._send_json(exc.code, upstream_error)
        except (urllib.error.URLError, TimeoutError) as exc:
            self._send_json(
                502,
                {"status": "error", "message": f"Could not reach NewsAPI: {exc.reason if hasattr(exc, 'reason') else exc}"},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the clustering explorer and NewsAPI proxy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    load_local_env(config.PROJECT_DIR / ".env")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Clustering explorer: http://{args.host}:{args.port}/clustering_explorer.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
