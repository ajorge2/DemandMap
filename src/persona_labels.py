from __future__ import annotations

import html
import re
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd


LONG_TEXT_FIELDS = {"Headline", "Summary", "Summarize LinkedIn profile"}
LOCATION_FIELDS = ("City", "State or Province", "Country")
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "been", "being", "but",
    "can", "como", "com", "company", "currently", "das", "dos", "for", "from",
    "has", "have", "into", "its", "mais", "not", "our", "over", "para", "que",
    "she", "her", "hers", "he", "him", "his", "that", "the", "their", "them", "they",
    "this", "through", "uma", "was", "were", "which", "who", "with", "you", "your",
    "years", "world", "first", "one", "most", "more", "than", "current", "work", "working",
    "home", "took",
    "founder", "cofounder",
    "chief", "executive", "advisor", "investor", "president", "chairman", "member",
}

THEME_ALIASES = {
    "author": "publishing",
    "authors": "publishing",
    "linkedin": "professional networking",
    "economy": "creator economy",
    "creator": "creator economy",
    "entrepreneur": "entrepreneurship",
    "entrepreneurs": "entrepreneurship",
    "investment": "investing",
    "investments": "investing",
    "ventures": "venture investing",
    "podcaster": "media",
}

THEME_GROUPS = (
    ({"ai", "artificial", "intelligence", "software", "technology", "data", "digital"}, "AI and technology"),
    ({"investment", "investments", "investing", "fund", "funds", "capital", "venture", "portfolio", "shareholders"}, "investing and capital"),
    ({"entrepreneur", "entrepreneurs", "entrepreneurship", "startup", "startups", "growth", "sales", "revenue", "companies"}, "entrepreneurship and growth"),
    ({"media", "content", "creator", "creators", "author", "publishing", "podcast", "podcaster", "speaker"}, "media and content"),
    ({"leadership", "leader", "leaders", "board", "advisor", "advisory", "management"}, "leadership and advisory"),
    ({"linkedin", "networking", "community", "communities"}, "professional networking"),
    ({"finance", "financial", "fintech", "payments", "bank", "banking", "wealth"}, "finance and fintech"),
    ({"health", "healthcare", "medical", "wellness", "patient", "patients"}, "health and wellness"),
    ({"marketing", "brand", "brands", "audience", "advertising"}, "marketing and brand growth"),
)

ROLE_RULES = (
    (re.compile(r"\b(founder|co[ -]?founder|fundador|fundadora|ceo|chief executive)\b", re.I), "Founders and CEOs"),
    (re.compile(r"\b(investor|venture|angel|seed|capital|scout)\b", re.I), "Investors and venture partners"),
    (re.compile(r"\b(advisor|advisory|board|mentor|mentora|chair|chairman|chairwoman)\b", re.I), "Advisors and board members"),
    (re.compile(r"\b(human resources|people|talent|recruit|hr)\b", re.I), "People and talent leaders"),
    (re.compile(r"\b(data|ai|technology|technical|engineering|product|information|scientific|medical)\b", re.I), "Technical and product leaders"),
    (re.compile(r"\b(marketing|growth|brand|creator|content|evangelist|influencer|influenciadora|ambassador|embaixadora)\b", re.I), "Marketing and creator leaders"),
    (re.compile(r"\b(sales|revenue|business development|go.to.market)\b", re.I), "Sales and revenue leaders"),
    (re.compile(r"\b(chief|officer|vice president|president|director|head|partner)\b", re.I), "Functional executives and partners"),
)


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _format_list(values: Sequence[str]) -> str:
    items = [value for value in values if value]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _representative_indices(memberships: np.ndarray, cluster: int) -> np.ndarray:
    size = max(8, int(np.ceil(len(memberships) / memberships.shape[1] * 1.6)))
    return np.argsort(-memberships[:, cluster])[: min(size, len(memberships))]


def _representative_values(
    df: pd.DataFrame,
    field: str,
    indices: Sequence[int],
    weights: np.ndarray,
    limit: int,
) -> list[str]:
    scores: Counter[str] = Counter()
    display: dict[str, str] = {}
    for index in indices:
        value = _clean(df.iloc[int(index)].get(field))
        if not value:
            continue
        key = value.casefold()
        scores[key] += float(weights[int(index)])
        display.setdefault(key, value)
    return [display[key] for key, _ in scores.most_common(limit)]


def _excluded_name_tokens(df: pd.DataFrame) -> set[str]:
    excluded: set[str] = set()
    for field in ("Full Name", "First Name", "Last Name"):
        if field not in df:
            continue
        for value in df[field].dropna().astype(str):
            excluded.update(re.findall(r"[a-zà-öø-ÿ]{3,}", value.lower()))
    return excluded


def _field_themes(
    df: pd.DataFrame,
    field: str,
    indices: Sequence[int],
    weights: np.ndarray,
    excluded: set[str],
    limit: int = 3,
) -> list[str]:
    documents = [_clean(value).lower() for value in df[field].tolist()]
    document_frequency: Counter[str] = Counter()
    tokenized: list[list[str]] = []
    for document in documents:
        tokens = [
            token for token in re.findall(r"[a-zà-öø-ÿ][a-zà-öø-ÿ0-9-]{2,}", document)
            if token not in STOPWORDS and token not in excluded
        ]
        tokenized.append(tokens)
        document_frequency.update(set(tokens))
    scores: Counter[str] = Counter()
    for index in indices:
        for token in sorted(set(tokenized[int(index)])):
            rarity = np.log((1.0 + len(df)) / (1.0 + document_frequency[token])) + 1.0
            scores[token] += float(weights[int(index)]) * float(rarity)
    grouped_scores: list[tuple[float, str]] = []
    for keywords, label in THEME_GROUPS:
        score = sum(scores[token] for token in keywords)
        if score > 0:
            grouped_scores.append((float(score), label))
    grouped_scores.sort(key=lambda item: (-item[0], item[1]))
    if grouped_scores:
        return [label for _, label in grouped_scores[:limit]]
    themes: list[str] = []
    for token, _ in scores.most_common(limit * 3):
        theme = THEME_ALIASES.get(token, token)
        if theme not in themes:
            themes.append(theme)
        if len(themes) >= limit:
            break
    return themes


def _role_name(
    df: pd.DataFrame,
    indices: Sequence[int],
    weights: np.ndarray,
) -> str:
    scores: Counter[str] = Counter()
    for index in indices:
        title = _clean(df.iloc[int(index)].get("Job Title"))
        for pattern, label in ROLE_RULES:
            if pattern.search(title):
                scores[label] += float(weights[int(index)])
    if not scores:
        return "Business professionals"
    ranked = scores.most_common(2)
    primary = ranked[0][0]
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * 0.72:
        secondary = ranked[1][0].lower()
        return f"{primary} with strong overlap among {secondary}"
    return primary


def generate_persona_labels(
    df: pd.DataFrame,
    selected_fields: Sequence[str],
    memberships: np.ndarray,
) -> tuple[list[str], list[dict[str, str]]]:
    """Compose natural persona names and expose per-field supporting evidence."""
    labels: list[str] = []
    evidence_rows: list[dict[str, str]] = []
    disambiguators: list[str] = []
    excluded = _excluded_name_tokens(df)
    for cluster in range(memberships.shape[1]):
        indices = _representative_indices(memberships, cluster)
        weights = memberships[:, cluster]
        evidence: dict[str, str] = {}
        raw_values: dict[str, list[str]] = {}
        field_themes: dict[str, list[str]] = {}
        field_coverage: dict[str, float] = {}
        for field in selected_fields:
            if field in LONG_TEXT_FIELDS:
                themes = _field_themes(df, field, indices, weights, excluded)
                field_themes[field] = themes
                populated = sum(bool(_clean(df.iloc[int(index)].get(field))) for index in indices)
                field_coverage[field] = populated / max(len(indices), 1)
                evidence[field] = (
                    f"{_format_list(themes)} ({populated}/{len(indices)} representative rows)"
                    if themes else "No populated values in representative rows"
                )
            else:
                limit = 3 if field == "Job Title" else 2
                values = _representative_values(df, field, indices, weights, limit)
                raw_values[field] = values
                evidence[field] = _format_list(values) or "No populated values in representative rows"

        label = _role_name(df, indices, weights) if "Job Title" in selected_fields else "Business professionals"

        location_values: list[str] = []
        for field in LOCATION_FIELDS:
            if field in selected_fields and raw_values.get(field):
                location_values.extend(raw_values[field][:1])
        if location_values:
            label += f" based in {_format_list(location_values)}"

        companies = raw_values.get("Company", [])
        if companies:
            label += f" at organizations including {_format_list(companies[:2])}"

        themes: list[str] = []
        has_non_long_field = any(field not in LONG_TEXT_FIELDS for field in selected_fields)
        for field in selected_fields:
            if field not in LONG_TEXT_FIELDS:
                continue
            if has_non_long_field and field_coverage.get(field, 0.0) < 0.30:
                continue
            for theme in field_themes.get(field, []):
                if theme not in themes:
                    themes.append(theme)
        if themes:
            label += f", focused on {_format_list(themes[:3])}"

        if label == "Business professionals":
            fallback_values: list[str] = []
            for field in selected_fields:
                fallback_values.extend(raw_values.get(field, [])[:1])
            if fallback_values:
                label += f" associated with {_format_list(fallback_values[:2])}"

        labels.append(label)
        evidence_rows.append(evidence)
        disambiguator = ""
        for field in ("Job Title", "Company", *selected_fields):
            values = raw_values.get(field, [])
            if values:
                disambiguator = " / ".join(values[:2])
                break
        if not disambiguator:
            disambiguator = next(
                (theme for field in selected_fields for theme in field_themes.get(field, [])),
                f"cluster {cluster + 1}",
            )
        disambiguators.append(disambiguator)

    label_counts = Counter(labels)
    used_labels: Counter[str] = Counter()
    for index, label in enumerate(labels):
        if label_counts[label] < 2:
            continue
        candidate = f"{label} — {disambiguators[index]} profiles"
        used_labels[candidate] += 1
        labels[index] = (
            candidate if used_labels[candidate] == 1
            else f"{candidate} {used_labels[candidate]}"
        )
    return labels, evidence_rows
