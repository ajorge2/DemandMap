from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MISSING_VALUE = "Not provided"
DEFAULT_EXAMPLE_LIMIT = 3
DEFAULT_VALUE_LIMIT = 180


def _bounded_value(value: object, *, max_characters: int) -> str:
    if pd.isna(value) or not str(value).strip():
        return MISSING_VALUE
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= max_characters:
        return cleaned
    return f"{cleaned[: max_characters - 1].rstrip()}…"


def build_representative_accounts(
    dataframe: pd.DataFrame,
    memberships: np.ndarray,
    active_fields: Sequence[str],
    display_fields: Sequence[str],
    *,
    limit: int = DEFAULT_EXAMPLE_LIMIT,
    max_value_characters: int = DEFAULT_VALUE_LIMIT,
) -> list[list[dict[str, object]]]:
    """Return the highest-affinity source records for every persona.

    Examples are ranked independently for each persona. Equal scores retain
    source-row order, so results are stable across runs. A record can therefore
    appear for multiple overlapping personas.
    """
    scores = np.asarray(memberships, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(dataframe):
        raise ValueError("memberships must contain one row per dataframe record")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if max_value_characters < 2:
        raise ValueError("max_value_characters must be at least 2")

    selected_fields = list(dict.fromkeys(str(field) for field in active_fields))
    identity_fields = list(dict.fromkeys(str(field) for field in display_fields if field))
    missing = [field for field in (*selected_fields, *identity_fields) if field not in dataframe.columns]
    if missing:
        raise ValueError(f"Unknown dataframe fields: {', '.join(dict.fromkeys(missing))}")

    representatives: list[list[dict[str, object]]] = []
    example_count = min(limit, len(dataframe))
    for persona_index in range(scores.shape[1]):
        ranked_indices = sorted(
            range(len(dataframe)),
            key=lambda row_index: (-float(scores[row_index, persona_index]), row_index),
        )[:example_count]
        examples: list[dict[str, object]] = []
        for row_index in ranked_indices:
            row = dataframe.iloc[row_index]
            examples.append(
                {
                    "index": int(row_index),
                    "rowId": f"source_row_{row_index}",
                    "score": round(float(scores[row_index, persona_index]), 6),
                    "displayValues": {
                        field: _bounded_value(row.get(field), max_characters=max_value_characters)
                        for field in identity_fields
                    },
                    "fieldValues": {
                        field: _bounded_value(row.get(field), max_characters=max_value_characters)
                        for field in selected_fields
                    },
                }
            )
        representatives.append(examples)
    return representatives
