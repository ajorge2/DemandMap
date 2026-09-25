from __future__ import annotations

import json
import urllib.request
from typing import Any

from .pain_point_analysis import DEFAULT_OPENAI_MODEL, OPENAI_RESPONSES_URL, extract_response_text


INITIAL_MAX_OUTPUT_TOKENS = 6000
RETRY_MAX_OUTPUT_TOKENS = 12000
NEWSAPI_QUERY_LIMIT = 500
NEWSAPI_QUERY_TARGET = 420


QUERY_INSTRUCTIONS = """Convert one or more B2B customer persona descriptions into one NewsAPI /v2/everything q query.

The purpose is to retrieve recent news about events, changes, pressures, and business circumstances that could create operational problems for the supplied personas, not merely articles mentioning job titles. Infer the industries or markets, relevant technologies or business functions, core responsibilities, and external events that could materially affect those responsibilities.

Search for the environment surrounding the personas. Useful events may include regulation, compliance changes, acquisitions, expansion, AI adoption, cybersecurity incidents, breaches, product launches, partnerships, hiring, layoffs, cost reduction, migrations, outages, fraud, reporting changes, funding, or rapid growth. Positive developments can create operational pain too.

If multiple personas are supplied, build one combined query: merge shared market and event themes, preserve meaningful differences with OR, and do not require every persona characteristic in the same article. Generally aim for (market terms) AND (event/change terms) AND (functional terms), but avoid too many required conditions. Use AND, OR, NOT, parentheses, and quoted phrases. Target 350 characters or fewer for q, and never exceed 420. Keep every quote and parenthesis balanced; simplify the term lists instead of ending mid-expression.

Use only the supplied persona_descriptions as persona-specific context. Do not inject fixed assumptions such as B2B, AI, SaaS, startup, or marketing technology unless visible in those descriptions. Return only data matching the JSON schema."""

QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "q": {"type": "string", "maxLength": NEWSAPI_QUERY_TARGET},
        "market_terms": {
            "type": "array", "maxItems": 12, "items": {"type": "string"}
        },
        "event_terms": {
            "type": "array", "maxItems": 12, "items": {"type": "string"}
        },
        "functional_terms": {
            "type": "array", "maxItems": 12, "items": {"type": "string"}
        },
        "reasoning": {"type": "string"},
    },
    "required": ["q", "market_terms", "event_terms", "functional_terms", "reasoning"],
}


def _query_syntax_error(query: str) -> str | None:
    if not query.strip():
        return "query is empty"
    if len(query) > NEWSAPI_QUERY_LIMIT:
        return f"query exceeds {NEWSAPI_QUERY_LIMIT} characters"
    depth = 0
    in_quote = False
    escaped = False
    for character in query:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_quote = not in_quote
        elif not in_quote and character == "(":
            depth += 1
        elif not in_quote and character == ")":
            depth -= 1
            if depth < 0:
                return "query has an unmatched closing parenthesis"
    if in_quote:
        return "query has an unmatched double quote"
    if depth:
        return "query has an unmatched opening parenthesis"
    return None


def _safe_term(value: object) -> str:
    term = str(value or "").replace("&", " and ").replace("/", " ")
    term = " ".join(term.replace('"', "").replace("(", "").replace(")", "").split())
    if not term:
        return ""
    return f'"{term}"' if " " in term else term


def _build_balanced_query(query_spec: dict[str, object]) -> str:
    group_names = ["market_terms", "event_terms", "functional_terms"]
    groups: list[list[str]] = []
    for name in group_names:
        raw_terms = query_spec.get(name)
        if not isinstance(raw_terms, list):
            groups.append([])
            continue
        terms: list[str] = []
        for value in raw_terms:
            term = _safe_term(value)
            if term and term.casefold() not in {existing.casefold() for existing in terms}:
                terms.append(term)
            if len(terms) >= 8:
                break
        groups.append(terms)

    def render() -> str:
        return " AND ".join(f"({' OR '.join(group)})" for group in groups if group)

    rebuilt = render()
    while len(rebuilt) > NEWSAPI_QUERY_TARGET:
        removable = [index for index, group in enumerate(groups) if len(group) > 1]
        if not removable:
            break
        largest = max(removable, key=lambda index: len(" OR ".join(groups[index])))
        groups[largest].pop()
        rebuilt = render()
    if _query_syntax_error(rebuilt):
        raise ValueError("Could not construct a valid NewsAPI query from the generated terms")
    return rebuilt


def generate_news_query(
    api_key: str,
    persona_descriptions: list[str],
    model: str = DEFAULT_OPENAI_MODEL,
) -> tuple[dict[str, object], str]:
    request_payload: dict[str, object] = {
        "model": model,
        "store": False,
        "instructions": QUERY_INSTRUCTIONS,
        "input": json.dumps(
            {"persona_descriptions": persona_descriptions}, ensure_ascii=False
        ),
        "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "newsapi_query",
                "strict": True,
                "schema": QUERY_SCHEMA,
            }
        },
    }
    def send_request() -> dict[str, object]:
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "fuzzy-persona-news-prototype/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    upstream = send_request()
    if upstream.get("status") == "incomplete":
        details = upstream.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "max_output_tokens":
            request_payload["max_output_tokens"] = RETRY_MAX_OUTPUT_TOKENS
            upstream = send_request()

    if upstream.get("status") not in {None, "completed"}:
        details = upstream.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        usage = upstream.get("usage")
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        detail_parts = [f"status: {upstream.get('status')}"]
        if reason:
            detail_parts.append(f"reason: {reason}")
        if output_tokens is not None:
            detail_parts.append(f"output tokens used: {output_tokens}")
        detail_parts.append(f"token budget: {request_payload['max_output_tokens']}")
        raise ValueError(f"OpenAI response was incomplete ({'; '.join(detail_parts)})")
    query = json.loads(extract_response_text(upstream))
    if not isinstance(query, dict) or not str(query.get("q") or "").strip():
        raise ValueError("OpenAI query generation did not return a q value")
    query_text = str(query["q"])
    if _query_syntax_error(query_text):
        query["q"] = _build_balanced_query(query)
    return query, str(upstream.get("model") or model)
