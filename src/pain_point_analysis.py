from __future__ import annotations

import json
import urllib.request
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"

ANALYSIS_INSTRUCTIONS = """You analyze recent news articles to identify potential operational problems affecting one or more B2B customer personas.

Process the supplied articles one at a time and in input order. Assign identifiers A1, A2, A3, and so on. For each article, identify the event and infer zero or more plausible operational consequences for the supplied personas. Do not simply summarize the news. Ask what operational work, complexity, risk, bottleneck, or new requirement the event could create. Positive, negative, and neutral developments may all create operational problems.

Evidence constraint: use only each supplied article's title and description as evidence. Do not introduce outside facts. Clearly separate what the article states from the operational consequence you infer. Never present an inference as an observed fact, never cite an article for a problem it did not contribute to, and do not force an article to produce a problem.

For each candidate problem, assign confidence from 0.00 to 1.00. Use 0.90-1.00 when the consequence follows very directly, 0.70-0.89 for a strong reasonable inference, 0.50-0.69 when plausible but requiring inference, 0.30-0.49 when speculative, and omit anything below 0.30.

Maintain a running problem ledger. Merge semantically equivalent problems under one concrete canonical name, but preserve meaningful distinctions. Track supporting article IDs, confidence from each article, article tally, distinct-event tally, total confidence, average confidence, and a short evidence summary. Near-duplicate coverage may be cited but must not count as fully independent evidence.

Rank problems primarily by weighted_support, defined as the sum of confidence scores across distinct supporting events. Break ties by distinct event support count and then average confidence. Do not rank an isolated inference highly merely because its confidence is high. Return the strongest 3-7 problems when evidence supports them, or fewer when evidence is thin.

For every top problem, explain why it may be emerging, cite the supporting article IDs, explain the specific persona impact, report evidence metrics, and assign an overall High, Moderate, or Tentative confidence label reflecting strength and recurrence. Produce a compact evidence ledger and a compact record for every article. Explicitly say when evidence is thin. Prefer concrete operational problems over vague phrases such as growth challenges, business pressure, or a need to innovate. These are evidence-backed hypotheses, not claims that the personas definitely have the problems.

Return only data that conforms to the supplied JSON schema. Article IDs must use the A1, A2, ... identifiers in input order. All supporting_article_ids must refer to supplied articles."""


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evidence_strength_note": {"type": "string"},
        "top_problems": {
            "type": "array",
            "maxItems": 7,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rank": {"type": "integer", "minimum": 1, "maximum": 7},
                    "problem_name": {"type": "string"},
                    "why_emerging": {"type": "string"},
                    "supporting_article_ids": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^A[1-9][0-9]*$"},
                    },
                    "persona_impact": {"type": "string"},
                    "weighted_support": {"type": "number", "minimum": 0},
                    "distinct_supporting_events": {"type": "integer", "minimum": 0},
                    "supporting_articles": {"type": "integer", "minimum": 0},
                    "average_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence_label": {
                        "type": "string",
                        "enum": ["High", "Moderate", "Tentative"],
                    },
                },
                "required": [
                    "rank", "problem_name", "why_emerging", "supporting_article_ids",
                    "persona_impact", "weighted_support", "distinct_supporting_events",
                    "supporting_articles", "average_confidence", "confidence_label",
                ],
            },
        },
        "evidence_ledger": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "problem": {"type": "string"},
                    "supporting_article_ids": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^A[1-9][0-9]*$"},
                    },
                    "distinct_events": {"type": "integer", "minimum": 0},
                    "average_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "weighted_support": {"type": "number", "minimum": 0},
                },
                "required": [
                    "problem", "supporting_article_ids", "distinct_events",
                    "average_confidence", "weighted_support",
                ],
            },
        },
        "article_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "article_id": {"type": "string", "pattern": "^A[1-9][0-9]*$"},
                    "title": {"type": "string"},
                    "event": {"type": "string"},
                    "no_supported_inference": {"type": "boolean"},
                    "problems": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "problem": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.3, "maximum": 1},
                                "inference": {"type": "string"},
                            },
                            "required": ["problem", "confidence", "inference"],
                        },
                    },
                },
                "required": [
                    "article_id", "title", "event", "no_supported_inference", "problems",
                ],
            },
        },
    },
    "required": [
        "evidence_strength_note", "top_problems", "evidence_ledger", "article_analysis",
    ],
}


def _article_input(article: dict[str, object], index: int) -> dict[str, object]:
    source = article.get("source")
    source_name = source.get("name", "") if isinstance(source, dict) else str(source or "")
    return {
        "article_id": f"A{index + 1}",
        "title": str(article.get("title") or ""),
        "description": str(article.get("description") or ""),
        "source": str(source_name or ""),
        "publishedAt": str(article.get("publishedAt") or ""),
        "url": str(article.get("url") or ""),
    }


def extract_response_text(response: dict[str, object]) -> str:
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    raise ValueError("OpenAI response did not contain output text")


def validate_analysis(
    analysis: dict[str, object],
    prepared_articles: list[dict[str, object]],
) -> dict[str, object]:
    """Validate cross-references that JSON Schema cannot express dynamically."""
    expected_ids = [str(article["article_id"]) for article in prepared_articles]
    expected_id_set = set(expected_ids)
    article_by_id = {str(article["article_id"]): article for article in prepared_articles}

    article_analysis = analysis.get("article_analysis")
    if not isinstance(article_analysis, list):
        raise ValueError("OpenAI analysis did not include article_analysis")
    returned_ids = [
        str(item.get("article_id"))
        for item in article_analysis
        if isinstance(item, dict)
    ]
    if len(returned_ids) != len(article_analysis) or set(returned_ids) != expected_id_set:
        raise ValueError("OpenAI analysis did not account for every supplied article exactly once")
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("OpenAI analysis returned a duplicate article ID")

    for item in article_analysis:
        article_id = str(item["article_id"])
        item["title"] = str(article_by_id[article_id].get("title") or "")
        problems = item.get("problems")
        if item.get("no_supported_inference") and problems:
            raise ValueError(f"{article_id} claims no inference but includes inferred problems")

    def validate_support_ids(rows: object, label: str) -> None:
        if not isinstance(rows, list):
            raise ValueError(f"OpenAI analysis did not include {label}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"OpenAI analysis contained an invalid {label} row")
            ids = row.get("supporting_article_ids")
            if not isinstance(ids, list) or any(str(value) not in expected_id_set for value in ids):
                raise ValueError(f"OpenAI analysis cited an unknown article in {label}")

    validate_support_ids(analysis.get("top_problems"), "top_problems")
    validate_support_ids(analysis.get("evidence_ledger"), "evidence_ledger")
    analysis["article_analysis"] = sorted(
        article_analysis,
        key=lambda item: expected_ids.index(str(item["article_id"])),
    )
    return analysis


def analyze_pain_points(
    api_key: str,
    persona_descriptions: list[str],
    articles: list[dict[str, object]],
    model: str = DEFAULT_OPENAI_MODEL,
) -> dict[str, object]:
    prepared_articles = [_article_input(article, index) for index, article in enumerate(articles)]
    if not prepared_articles:
        return {
            "evidence_strength_note": "No recent articles were available to analyze.",
            "top_problems": [],
            "evidence_ledger": [],
            "article_analysis": [],
            "model": model,
            "articles_reviewed": 0,
        }
    input_payload = {
        "persona_descriptions": persona_descriptions,
        "articles": prepared_articles,
    }
    request_payload = {
        "model": model,
        "store": False,
        "instructions": ANALYSIS_INSTRUCTIONS,
        "input": json.dumps(input_payload, ensure_ascii=False),
        "max_output_tokens": 12000,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "operational_pain_analysis",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            }
        },
    }
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
    with urllib.request.urlopen(request, timeout=300) as response:
        upstream = json.loads(response.read().decode("utf-8"))
    if upstream.get("status") not in {None, "completed"}:
        raise ValueError(f"OpenAI response status was {upstream.get('status')}")
    analysis = json.loads(extract_response_text(upstream))
    if not isinstance(analysis, dict):
        raise ValueError("OpenAI analysis was not a JSON object")
    analysis = validate_analysis(analysis, prepared_articles)
    analysis["model"] = str(upstream.get("model") or model)
    analysis["articles_reviewed"] = len(prepared_articles)
    return analysis
