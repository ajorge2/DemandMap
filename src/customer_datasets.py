from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ROWS = 500
MAX_COLUMNS = 80
MAX_CELL_CHARACTERS = 4000

IDENTIFIER_NAME = re.compile(
    r"(^|[ _-])(id|uuid|guid|url|uri|email|e-mail|phone|mobile|linkedin|twitter|"
    r"facebook|instagram|website|domain|profile|created at|updated at|timestamp|"
    r"first name|last name|full name|contact name|person name)([ _-]|$)",
    re.I,
)
NUMERIC_NAME = re.compile(
    r"(^|[ _-])(count|number|connections|employees|revenue|amount|score|rank|age|year)([ _-]|$)",
    re.I,
)


@dataclass
class CustomerDataset:
    dataset_id: str
    filename: str
    dataframe: pd.DataFrame
    clusterable_fields: tuple[str, ...]
    default_fields: tuple[str, ...]
    display_fields: tuple[str, ...]
    is_demo: bool = False

    def client_profile(self) -> dict[str, object]:
        return {
            "datasetId": self.dataset_id,
            "datasetName": self.filename,
            "rowCount": len(self.dataframe),
            "clusterableFields": list(self.clusterable_fields),
            "defaultSelectedFields": list(self.default_fields),
            "displayFields": list(self.display_fields),
            "isDemo": self.is_demo,
            "records": dataset_records(self),
        }


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()[:MAX_CELL_CHARACTERS]


def _looks_numeric(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return True
    parsed = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
    return float(parsed.notna().mean()) >= 0.92


def _looks_like_links(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False
    link_like = values.str.match(r"^(?:https?://|www\.)", case=False) | values.str.contains(
        r"^[^\s@]+@[^\s@]+\.[^\s@]+$", regex=True
    )
    return float(link_like.mean()) >= 0.55


def infer_clusterable_fields(df: pd.DataFrame) -> tuple[str, ...]:
    fields: list[str] = []
    for column in df.columns:
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed:"):
            continue
        series = df[column]
        populated = series.notna() & series.astype(str).str.strip().ne("")
        if int(populated.sum()) < max(2, int(len(df) * 0.08)):
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if IDENTIFIER_NAME.search(name) or NUMERIC_NAME.search(name):
            continue
        if _looks_numeric(series) or _looks_like_links(series):
            continue
        fields.append(name)
    return tuple(fields)


def _field_priority(field: str) -> int:
    name = field.casefold()
    priorities = (
        (r"job.?title|\brole\b|position|seniority|function", 100),
        (r"industry|market|sector|segment|vertical|category", 95),
        (r"headline|tagline", 90),
        (r"summary|description|about|bio|overview|notes", 85),
        (r"company|organization|organisation|employer|account", 72),
        (r"department|team|specialt|expertise|skill|technology", 68),
        (r"country|state|province|region|city|location|geograph", 60),
    )
    return next((score for pattern, score in priorities if re.search(pattern, name)), 40)


def choose_default_fields(fields: tuple[str, ...]) -> tuple[str, ...]:
    ranked = sorted(enumerate(fields), key=lambda item: (-_field_priority(item[1]), item[0]))
    high_signal = [field for _, field in ranked if _field_priority(field) >= 68]
    selected = (high_signal or [field for _, field in ranked])[:4]
    return tuple(selected)


def choose_display_fields(df: pd.DataFrame, clusterable: tuple[str, ...]) -> tuple[str, ...]:
    columns = [str(column) for column in df.columns]

    def first_matching(pattern: str, excluded: set[str]) -> str | None:
        return next(
            (column for column in columns if column not in excluded and re.search(pattern, column, re.I)),
            None,
        )

    chosen: list[str] = []
    for pattern in (
        r"^(full name|customer(?: name)?|contact(?: name)?|person(?: name)?|account name|company name|company|organization)$",
        r"job.?title|\brole\b|position|headline",
        r"company|organization|employer|industry|market|sector",
    ):
        match = first_matching(pattern, set(chosen))
        if match:
            chosen.append(match)
    for field in (*clusterable, *columns):
        if field not in chosen:
            chosen.append(field)
        if len(chosen) >= 3:
            break
    return tuple(chosen[:3])


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The CSV does not contain any data rows")
    if len(df) > MAX_ROWS:
        raise ValueError(f"This prototype supports up to {MAX_ROWS} rows per upload")
    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"This prototype supports up to {MAX_COLUMNS} columns per upload")
    normalized = df.dropna(how="all").copy()
    normalized.columns = [str(column).strip() or f"Column {index + 1}" for index, column in enumerate(normalized.columns)]
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError("CSV column names must be unique")
    if len(normalized) < 3:
        raise ValueError("The CSV needs at least 3 non-empty customer rows")
    for column in normalized.columns:
        if not pd.api.types.is_numeric_dtype(normalized[column]):
            normalized[column] = normalized[column].map(_clean_cell)
    return normalized.reset_index(drop=True)


def parse_uploaded_csv(csv_text: str) -> pd.DataFrame:
    if not csv_text.strip():
        raise ValueError("The uploaded CSV is empty")
    if len(csv_text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError(f"CSV uploads are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        df = pd.read_csv(io.StringIO(csv_text), sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not parse the CSV: {exc}") from exc
    return normalize_dataframe(df)


def make_customer_dataset(
    dataset_id: str,
    filename: str,
    df: pd.DataFrame,
    *,
    is_demo: bool = False,
) -> CustomerDataset:
    normalized = normalize_dataframe(df)
    clusterable = infer_clusterable_fields(normalized)
    if not clusterable:
        raise ValueError(
            "No descriptive text columns were found. Include fields such as role, industry, company, location, headline, or summary."
        )
    safe_filename = Path(filename or "customers.csv").name[:180]
    return CustomerDataset(
        dataset_id=dataset_id,
        filename=safe_filename,
        dataframe=normalized,
        clusterable_fields=clusterable,
        default_fields=choose_default_fields(clusterable),
        display_fields=choose_display_fields(normalized, clusterable),
        is_demo=is_demo,
    )


def dataset_records(dataset: CustomerDataset) -> list[dict[str, object]]:
    fields = list(dataset.display_fields)
    while len(fields) < 3:
        fields.append("")
    records: list[dict[str, object]] = []
    for index, row in dataset.dataframe.iterrows():
        records.append(
            {
                "index": int(index),
                "rowId": f"source_row_{index}",
                "name": _clean_cell(row.get(fields[0])) if fields[0] else f"Row {index + 1}",
                "title": _clean_cell(row.get(fields[1])) if fields[1] else "",
                "company": _clean_cell(row.get(fields[2])) if fields[2] else "",
            }
        )
    return records
