from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .pain_point_analysis import DEFAULT_OPENAI_MODEL, OPENAI_RESPONSES_URL, extract_response_text


PERSONA_INSTRUCTIONS = """You create clear, natural B2B customer persona candidates from fuzzy semantic clusters.

Use only the supplied representative CSV evidence. Produce one candidate for every candidate_id. A candidate name should read like a natural customer segment, not a bag of keywords or a cluster label. Its description should be one or two concise sentences explaining who these customers are, their organizational context, and the themes visible in the data.

Every selected CSV field must materially inform either the name or description and must receive one field summary. Cite source support only by returning the supplied evidence_id values; never copy or invent a quotation. Every populated-field summary must cite at least one evidence item from that same field. A field with no supplied values must use evidence_status "no_evidence", explicitly say that no representative evidence was available, and return no evidence IDs. Each overall description must cite at least one supplied evidence item when any evidence exists.

Do not ignore a field merely because it is unfamiliar. When a field is sparse or ambiguous, acknowledge that cautiously instead of inventing a conclusion. Do not invent company sizes, industries, responsibilities, technologies, or needs that are not supported by the evidence. Do not mention clustering mechanics, prototypes, source rows, evidence IDs, or candidate IDs in the prose. Keep overlapping candidates meaningfully distinct when the evidence supports a distinction.

Return exactly one result per candidate_id and exactly one field summary per selected field. Return only data matching the JSON schema."""

PERSONA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "description_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "field_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "field": {"type": "string"},
                                "summary": {"type": "string"},
                                "evidence_status": {
                                    "type": "string",
                                    "enum": ["supported", "no_evidence"],
                                },
                                "evidence_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "field", "summary", "evidence_status", "evidence_ids"
                            ],
                        },
                    },
                },
                "required": [
                    "candidate_id", "name", "description",
                    "description_evidence_ids", "field_summaries",
                ],
            },
        }
    },
    "required": ["candidates"],
}

MAX_EVIDENCE_VALUE_CHARACTERS = 320
MAX_VALUES_PER_FIELD = 4


def _clean(value: object, limit: int = MAX_EVIDENCE_VALUE_CHARACTERS) -> str:
    if pd.isna(value):
        return ""
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"


def _evidence_id(candidate_id: str, row_id: str, field_position: int) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}\x1f{row_id}\x1f{field_position}".encode("utf-8")
    ).hexdigest()[:16]
    return f"ev_{digest}"


def _representative_rows(
    df: pd.DataFrame,
    fields: list[str],
    memberships: np.ndarray,
    cluster: int,
    candidate_id: str,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, str]]]:
    count = min(max(6, int(np.ceil(len(df) / memberships.shape[1] * 1.3))), 10, len(df))
    indices = np.argsort(-memberships[:, cluster], kind="stable")[:count]
    rows = [
        {
            "row_id": f"source_row_{int(index)}",
            "values": {field: _clean(df.iloc[int(index)].get(field)) for field in fields},
        }
        for index in indices
    ]
    evidence_items: list[dict[str, str]] = []
    field_evidence: dict[str, object] = {}
    for field_position, field in enumerate(fields):
        seen_values: set[str] = set()
        field_items: list[dict[str, str]] = []
        for index in indices:
            value = _clean(df.iloc[int(index)].get(field))
            folded = value.casefold()
            if not value or folded in seen_values:
                continue
            seen_values.add(folded)
            row_id = f"source_row_{int(index)}"
            item = {
                "evidence_id": _evidence_id(candidate_id, row_id, field_position),
                "candidate_id": candidate_id,
                "row_id": row_id,
                "field": field,
                "value": value,
            }
            field_items.append(item)
            evidence_items.append(item)
            if len(field_items) >= MAX_VALUES_PER_FIELD:
                break
        populated = sum(bool(_clean(df.iloc[int(index)].get(field))) for index in indices)
        field_evidence[field] = {
            "evidence_ids": [item["evidence_id"] for item in field_items],
            "coverage": f"{populated}/{len(indices)} representative rows",
        }
    return rows, field_evidence, evidence_items


def build_persona_model_input(
    df: pd.DataFrame,
    selected_fields: list[str],
    memberships_by_k: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for k_text, memberships in memberships_by_k.items():
        for cluster in range(memberships.shape[1]):
            candidate_id = f"K{k_text}-C{cluster}"
            rows, field_evidence, evidence_items = _representative_rows(
                df, selected_fields, memberships, cluster, candidate_id
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "selected_fields": list(selected_fields),
                    "representative_rows": rows,
                    "field_evidence": field_evidence,
                    "evidence_items": evidence_items,
                }
            )
    return candidates


def _resolved_item(item: dict[str, str]) -> dict[str, str]:
    return {
        "evidenceId": item["evidence_id"],
        "rowId": item["row_id"],
        "field": item["field"],
        "value": item["value"],
    }


def _check_reference_list(
    references: object,
    *,
    candidate_id: str,
    candidate_map: dict[str, dict[str, str]],
    global_map: dict[str, dict[str, str]],
    context: str,
    expected_field: str | None = None,
) -> list[dict[str, str]]:
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        raise ValueError(f"{candidate_id} returned invalid evidence references for {context}")
    if len(references) != len(set(references)):
        raise ValueError(f"{candidate_id} returned duplicate evidence references for {context}")
    resolved: list[dict[str, str]] = []
    for evidence_id in references:
        item = global_map.get(evidence_id)
        if item is None:
            raise ValueError(f"{candidate_id} referenced unknown evidence ID {evidence_id}")
        if evidence_id not in candidate_map:
            raise ValueError(f"{candidate_id} cross-referenced evidence from another candidate")
        if expected_field is not None and item["field"] != expected_field:
            raise ValueError(
                f"{candidate_id} field {expected_field} cited wrong-field evidence from {item['field']}"
            )
        resolved.append(_resolved_item(item))
    return resolved


def validate_and_resolve_persona_output(
    returned: object,
    candidates: list[dict[str, object]],
    selected_fields: Iterable[str],
) -> dict[str, dict[str, object]]:
    """Validate model-selected evidence IDs and resolve them from server-owned values."""
    if not isinstance(returned, list):
        raise ValueError("Persona generation did not return candidates")
    expected_fields = list(selected_fields)
    expected_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    global_map: dict[str, dict[str, str]] = {}
    candidate_maps: dict[str, dict[str, dict[str, str]]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        scoped: dict[str, dict[str, str]] = {}
        for raw_item in candidate.get("evidence_items", []):
            if not isinstance(raw_item, dict):
                raise ValueError(f"{candidate_id} has malformed server evidence")
            item = {str(key): str(value) for key, value in raw_item.items()}
            evidence_id = item.get("evidence_id", "")
            if not evidence_id or evidence_id in global_map:
                raise ValueError("Server evidence IDs must be present and globally unique")
            scoped[evidence_id] = item
            global_map[evidence_id] = item
        candidate_maps[candidate_id] = scoped

    by_id: dict[str, dict[str, object]] = {}
    for candidate in returned:
        if not isinstance(candidate, dict):
            raise ValueError("Persona generation returned an invalid candidate")
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id not in expected_ids or candidate_id in by_id:
            raise ValueError("Persona generation returned an unknown or duplicate candidate ID")
        if not str(candidate.get("name") or "").strip() or not str(candidate.get("description") or "").strip():
            raise ValueError(f"{candidate_id} returned an empty name or description")
        scoped_map = candidate_maps[candidate_id]
        description_evidence = _check_reference_list(
            candidate.get("description_evidence_ids"),
            candidate_id=candidate_id,
            candidate_map=scoped_map,
            global_map=global_map,
            context="description",
        )
        if scoped_map and not description_evidence:
            raise ValueError(f"{candidate_id} description has no supporting evidence")

        summaries = candidate.get("field_summaries")
        if not isinstance(summaries, list):
            raise ValueError(f"{candidate_id} is missing field summaries")
        summary_fields = [str(item.get("field")) for item in summaries if isinstance(item, dict)]
        if len(summary_fields) != len(summaries):
            raise ValueError(f"{candidate_id} returned an invalid field summary")
        if len(summary_fields) != len(set(summary_fields)) or set(summary_fields) != set(expected_fields):
            raise ValueError(f"{candidate_id} did not account for every selected CSV field")

        resolved_summaries: list[dict[str, object]] = []
        for summary in summaries:
            field = str(summary["field"])
            text = str(summary.get("summary") or "").strip()
            if not text:
                raise ValueError(f"{candidate_id} returned an empty summary for {field}")
            status = str(summary.get("evidence_status") or "")
            populated = any(item["field"] == field for item in scoped_map.values())
            evidence = _check_reference_list(
                summary.get("evidence_ids"),
                candidate_id=candidate_id,
                candidate_map=scoped_map,
                global_map=global_map,
                context=f"field {field}",
                expected_field=field,
            )
            if populated and (status != "supported" or not evidence):
                raise ValueError(f"{candidate_id} populated field {field} has no valid evidence")
            if not populated and (status != "no_evidence" or evidence):
                raise ValueError(f"{candidate_id} sparse field {field} must be marked no_evidence")
            resolved_summaries.append(
                {
                    "field": field,
                    "summary": text,
                    "evidenceStatus": status,
                    "evidence": evidence,
                }
            )

        by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "name": str(candidate["name"]).strip(),
            "description": str(candidate["description"]).strip(),
            "description_evidence": description_evidence,
            "field_summaries": resolved_summaries,
            "source_evidence": [_resolved_item(item) for item in scoped_map.values()],
        }
    if set(by_id) != expected_ids:
        raise ValueError("Persona generation did not return every candidate")
    return by_id


def build_fallback_grounding(
    candidates: list[dict[str, object]],
    summary_maps: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, object]]:
    """Build an auditable grounding payload for deterministic fallback personas."""
    result: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        items = [item for item in candidate.get("evidence_items", []) if isinstance(item, dict)]
        resolved_items = [_resolved_item(item) for item in items]
        field_summaries: list[dict[str, object]] = []
        for field in candidate.get("selected_fields", []):
            field = str(field)
            field_items = [item for item in resolved_items if item["field"] == field]
            summary = (summary_maps or {}).get(candidate_id, {}).get(field)
            if not summary:
                summary = (
                    "Deterministic summary based on the exact CSV values below."
                    if field_items else "No populated values in the representative rows."
                )
            field_summaries.append(
                {
                    "field": field,
                    "summary": summary,
                    "evidenceStatus": "supported" if field_items else "no_evidence",
                    "evidence": field_items,
                }
            )
        result[candidate_id] = {
            "candidateId": candidate_id,
            "descriptionEvidence": resolved_items[: min(4, len(resolved_items))],
            "fieldSummaries": field_summaries,
            "sourceEvidence": resolved_items,
        }
    return result


def generate_persona_descriptions(
    api_key: str,
    df: pd.DataFrame,
    selected_fields: list[str],
    memberships_by_k: dict[str, np.ndarray],
    model: str = DEFAULT_OPENAI_MODEL,
) -> tuple[dict[str, list[dict[str, object]]], str]:
    candidates = build_persona_model_input(df, selected_fields, memberships_by_k)
    request_payload = {
        "model": model,
        "store": False,
        "instructions": PERSONA_INSTRUCTIONS,
        "input": json.dumps({"candidates": candidates}, ensure_ascii=False),
        "max_output_tokens": 12000,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "persona_candidates",
                "strict": True,
                "schema": PERSONA_SCHEMA,
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
    with urllib.request.urlopen(request, timeout=180) as response:
        upstream = json.loads(response.read().decode("utf-8"))
    if upstream.get("status") not in {None, "completed"}:
        raise ValueError(f"OpenAI response status was {upstream.get('status')}")
    output = json.loads(extract_response_text(upstream))
    returned = output.get("candidates") if isinstance(output, dict) else None
    by_id = validate_and_resolve_persona_output(returned, candidates, selected_fields)

    grouped: dict[str, list[dict[str, object]]] = {}
    for k_text, memberships in memberships_by_k.items():
        grouped[k_text] = [by_id[f"K{k_text}-C{cluster}"] for cluster in range(memberships.shape[1])]
    return grouped, str(upstream.get("model") or model)
