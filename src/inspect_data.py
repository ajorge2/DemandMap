from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTIFIER_HINTS = ("id", "url", "profile", "email", "name")
TEXT_HINTS = ("title", "headline", "summary", "description", "about", "bio")


def infer_column_role(series: pd.Series, name: str) -> str:
    non_null = series.dropna()
    lower = name.lower()
    if non_null.empty:
        return "empty"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    mean_length = non_null.astype(str).str.len().mean()
    string_values = non_null.astype(str)
    url_ratio = float(string_values.str.match(r"https?://", case=False).mean())
    if url_ratio > 0.5:
        return "identifier"
    if any(token in lower for token in TEXT_HINTS) or mean_length >= 55:
        return "free-form semantic text"
    if any(token in lower for token in IDENTIFIER_HINTS) and unique_ratio > 0.75:
        return "identifier"
    if unique_ratio <= 0.35:
        return "categorical"
    return "categorical / short text"


def inspect_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for name in df.columns:
        series = df[name]
        values = [str(v).replace("\n", " ")[:180] for v in series.dropna().unique()[:4]]
        columns.append(
            {
                "column": name,
                "role": infer_column_role(series, name),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "unique_non_null": int(series.nunique(dropna=True)),
                "samples": values,
            }
        )
    return {"row_count": int(len(df)), "column_count": int(len(df.columns)), "columns": columns}


def semantic_view_candidates(inspection: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for col in inspection["columns"]:
        if col["role"] != "free-form semantic text":
            continue
        coverage = 1.0 - col["missing_rate"]
        diversity = col["unique_non_null"] / max(inspection["row_count"], 1)
        score = 0.65 * coverage + 0.35 * diversity
        candidates.append(
            {
                "column": col["column"],
                "coverage": coverage,
                "diversity": diversity,
                "view_score": score,
            }
        )
    return sorted(candidates, key=lambda x: x["view_score"], reverse=True)


def write_inspection_report(inspection: dict[str, Any], path: Path) -> None:
    candidates = semantic_view_candidates(inspection)
    lines = [
        "# Data inspection",
        "",
        f"- Rows: {inspection['row_count']}",
        f"- Columns: {inspection['column_count']}",
        "",
        "## Column profile",
        "",
        "| Column | Inferred role | Missing | Unique | Sample values |",
        "|---|---|---:|---:|---|",
    ]
    for col in inspection["columns"]:
        samples = "; ".join(value.replace("|", "\\|") for value in col["samples"])
        lines.append(
            f"| {col['column']} | {col['role']} | {col['missing_rate']:.0%} | "
            f"{col['unique_non_null']} | {samples} |"
        )
    lines.extend(
        [
            "",
            "## Candidate semantic views",
            "",
            "| Rank | Field | Coverage | Diversity |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, candidate in enumerate(candidates, 1):
        lines.append(
            f"| {rank} | {candidate['column']} | {candidate['coverage']:.0%} | "
            f"{candidate['diversity']:.0%} |"
        )
    lines.extend(
        [
            "",
            "Numerical fields are kept as source attributes and are not embedded. "
            "The all-null first column is preserved in the output but excluded from modeling.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    inspection = inspect_dataframe(df)
    print(json.dumps(inspection, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(inspection, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        write_inspection_report(inspection, args.markdown)


if __name__ == "__main__":
    main()
