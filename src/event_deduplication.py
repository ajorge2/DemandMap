from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .embeddings import EmbeddingResult, generate_embeddings


DEFAULT_THRESHOLD = 0.80
DEFAULT_WEIGHTS = {
    "semantic": 0.76,
    "token_overlap": 0.12,
    "entity_overlap": 0.07,
    "title_similarity": 0.05,
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "of", "on", "or", "says", "the", "to",
    "with", "after", "amid", "over", "new", "latest", "report", "reports",
}
_ENTITY_STOPWORDS = {
    "The", "A", "An", "After", "Amid", "New", "Report", "Reports", "Exclusive",
}
_EVENT_ALIASES = {
    "acquires": "acquisition", "acquired": "acquisition", "acquire": "acquisition",
    "acquisition": "acquisition", "buys": "acquisition", "bought": "acquisition",
    "raises": "funding", "raised": "funding", "funding": "funding",
    "financing": "funding", "investment": "funding", "secures": "funding",
    "layoffs": "layoff", "layoff": "layoff", "cuts": "layoff", "redundancies": "layoff",
    "launches": "launch", "launched": "launch", "unveils": "launch", "introduces": "launch",
    "partners": "partnership", "partnership": "partnership", "alliance": "partnership",
    "breached": "breach", "breaches": "breach", "breach": "breach", "hacked": "breach",
    "cyberattack": "breach", "outages": "outage", "outage": "outage", "downtime": "outage",
    "regulations": "regulation", "regulation": "regulation", "rules": "regulation",
    "expands": "expansion", "expanded": "expansion", "expansion": "expansion",
    "hires": "hiring", "hired": "hiring", "hiring": "hiring",
}


@dataclass(frozen=True)
class DedupeConfig:
    threshold: float = DEFAULT_THRESHOLD
    semantic_floor: float = 0.72
    max_publication_gap_days: int = 21
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    cache_dir: Path | None = None
    api_key: str = ""
    allow_lexical_fallback: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("The deduplication threshold must be between 0 and 1")
        if not 0.0 <= self.semantic_floor <= 1.0:
            raise ValueError("The semantic floor must be between 0 and 1")
        if self.max_publication_gap_days < 0:
            raise ValueError("max_publication_gap_days cannot be negative")
        missing = set(DEFAULT_WEIGHTS) - set(self.weights)
        if missing:
            raise ValueError(f"Missing score weights: {', '.join(sorted(missing))}")
        if not math.isclose(sum(float(value) for value in self.weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Deduplication score weights must sum to 1")


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: np.ndarray
    backend: str
    semantic_active: bool
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PairEvidence:
    score: float
    duplicate: bool
    semantic_similarity: float
    token_overlap: float
    entity_overlap: float
    title_similarity: float
    publication_gap_days: int | None
    shared_entities: tuple[str, ...]
    left_event_types: tuple[str, ...]
    right_event_types: tuple[str, ...]
    left_amounts: tuple[str, ...]
    right_amounts: tuple[str, ...]
    blockers: tuple[str, ...]
    decision_reason: str


@dataclass(frozen=True)
class MergeProvenance:
    representative_index: int
    duplicate_index: int
    representative_title: str
    duplicate_title: str
    evidence: PairEvidence


@dataclass(frozen=True)
class DeduplicationDiagnostics:
    scanned_articles: int
    usable_articles: int
    retained_articles: int
    removed_duplicates: int
    limit: int
    threshold: float
    backend: str
    model: str | None
    semantic_active: bool
    fallback_used: bool
    fallback_reason: str | None
    weights: Mapping[str, float]
    merges: tuple[MergeProvenance, ...]


@dataclass(frozen=True)
class DeduplicationResult:
    articles: tuple[dict[str, object], ...]
    diagnostics: DeduplicationDiagnostics


@dataclass(frozen=True)
class EvaluationResult:
    fixture_version: str
    backend: str
    threshold: float
    pair_count: int
    positive_pairs: int
    negative_pairs: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_merges: tuple[dict[str, object], ...]
    missed_merges: tuple[dict[str, object], ...]


EmbeddingProvider = Callable[[Sequence[str]], EmbeddingBatch | EmbeddingResult | np.ndarray]


def _clean(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def article_text(article: Mapping[str, object]) -> str:
    title = _clean(article.get("title"))
    description = _clean(article.get("description"))
    if description:
        return f"News event title: {title}\nEvent description: {description}"
    return f"News event title: {title}"


def _normalized_tokens(value: object) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [_EVENT_ALIASES.get(token, token) for token in tokens if token not in _STOPWORDS]


def _entities(article: Mapping[str, object]) -> set[str]:
    text = f"{_clean(article.get('title'))} {_clean(article.get('description'))}"
    entities: set[str] = set()
    for match in re.findall(r"\b(?:[A-Z][A-Za-z0-9&.-]*)(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3}\b", text):
        cleaned = match.strip(" .-")
        if cleaned and cleaned not in _ENTITY_STOPWORDS and len(cleaned) > 1:
            entities.add(cleaned.casefold())
    return entities


def _amounts(article: Mapping[str, object]) -> set[str]:
    text = f"{_clean(article.get('title'))} {_clean(article.get('description'))}".casefold()
    amounts: set[str] = set()
    pattern = re.compile(
        r"(?P<currency>[$€£])?\s*(?P<number>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>billion|million|bn|mn|b|m)\b"
    )
    unit_aliases = {"billion": "b", "bn": "b", "million": "m", "mn": "m"}
    for match in pattern.finditer(text):
        unit = unit_aliases.get(match.group("unit"), match.group("unit"))
        # Publishers often omit the currency symbol in rewrites of the same round.
        # Preserve amount and magnitude for conflict detection without treating
        # "$40 million" and "40m" as different events.
        amounts.add(f"{match.group('number')}{unit}")
    return amounts


def _event_types(article: Mapping[str, object]) -> set[str]:
    tokens = _normalized_tokens(f"{article.get('title', '')} {article.get('description', '')}")
    known = set(_EVENT_ALIASES.values())
    return {token for token in tokens if token in known}


def _published(article: Mapping[str, object]) -> datetime | None:
    value = _clean(article.get("publishedAt"))
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _sequence_similarity(left: str, right: str) -> float:
    # A compact deterministic edit similarity avoids importing a second NLP stack.
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _coerce_embedding_batch(
    raw: EmbeddingBatch | EmbeddingResult | np.ndarray,
    expected_rows: int,
) -> EmbeddingBatch:
    if isinstance(raw, EmbeddingBatch):
        batch = raw
    elif isinstance(raw, EmbeddingResult):
        semantic_active = raw.backend not in {"tfidf-lsa", "lexical-fallback"}
        batch = EmbeddingBatch(raw.vectors, raw.backend, semantic_active, raw.metadata)
    else:
        batch = EmbeddingBatch(np.asarray(raw, dtype=np.float64), "injected", True, {})
    vectors = np.asarray(batch.vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows or not np.isfinite(vectors).all():
        raise ValueError("Embedding provider returned an invalid article-vector matrix")
    return EmbeddingBatch(vectors, batch.backend, batch.semantic_active, dict(batch.metadata))


def _embed_articles(
    texts: Sequence[str],
    config: DedupeConfig,
    embedding_provider: EmbeddingProvider | None,
) -> tuple[EmbeddingBatch, bool, str | None]:
    try:
        if embedding_provider is not None:
            batch = _coerce_embedding_batch(embedding_provider(texts), len(texts))
        else:
            result = generate_embeddings(
                texts,
                backend=config.embedding_backend,
                model_name=config.embedding_model,
                api_key=config.api_key,
                cache_dir=config.cache_dir,
                lsa_dimensions=48,
            )
            batch = _coerce_embedding_batch(result, len(texts))
        fallback_used = not batch.semantic_active
        fallback_reason = None
        if fallback_used:
            fallback_reason = str(batch.metadata.get("fallback_reason") or (
                f"{batch.backend} is a lexical/latent-text fallback, not a pretrained semantic model"
            ))
        return batch, fallback_used, fallback_reason
    except Exception as exc:
        if not config.allow_lexical_fallback:
            raise RuntimeError(f"Semantic event embeddings failed: {exc}") from exc
        fallback = generate_embeddings(texts, backend="tfidf-lsa", lsa_dimensions=48)
        batch = _coerce_embedding_batch(fallback, len(texts))
        return batch, True, f"{type(exc).__name__}: {exc}"


def compare_articles(
    left: Mapping[str, object],
    right: Mapping[str, object],
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    *,
    config: DedupeConfig,
    semantic_active: bool,
) -> PairEvidence:
    semantic = max(0.0, _cosine(left_vector, right_vector))
    left_tokens = set(_normalized_tokens(article_text(left)))
    right_tokens = set(_normalized_tokens(article_text(right)))
    token_overlap = _jaccard(left_tokens, right_tokens)
    left_entities = _entities(left)
    right_entities = _entities(right)
    entity_overlap = _jaccard(left_entities, right_entities) if left_entities and right_entities else 0.0
    shared_entities = tuple(sorted(left_entities & right_entities))
    left_title = " ".join(_normalized_tokens(left.get("title")))
    right_title = " ".join(_normalized_tokens(right.get("title")))
    title_similarity = _sequence_similarity(left_title, right_title)
    left_amounts = _amounts(left)
    right_amounts = _amounts(right)
    left_types = _event_types(left)
    right_types = _event_types(right)
    left_date = _published(left)
    right_date = _published(right)
    gap = abs((left_date - right_date).days) if left_date and right_date else None

    blockers: list[str] = []
    if left_amounts and right_amounts and not left_amounts.intersection(right_amounts):
        blockers.append("conflicting_amounts")
    if left_types and right_types and left_types.isdisjoint(right_types):
        blockers.append("conflicting_event_types")
    if gap is not None and gap > config.max_publication_gap_days:
        blockers.append("publication_gap")

    if semantic_active:
        score = (
            float(config.weights["semantic"]) * semantic
            + float(config.weights["token_overlap"]) * token_overlap
            + float(config.weights["entity_overlap"]) * entity_overlap
            + float(config.weights["title_similarity"]) * title_similarity
        )
        has_anchor = bool(shared_entities) or token_overlap >= 0.22 or title_similarity >= 0.62
        duplicate = (
            not blockers
            and semantic >= config.semantic_floor
            and score >= config.threshold
            and has_anchor
        )
        reason = "semantic_score_and_event_anchors" if duplicate else (
            "blocked:" + ",".join(blockers) if blockers else "below_semantic_threshold"
        )
    else:
        # Degraded behavior is intentionally conservative and is labeled as lexical.
        score = 0.55 * title_similarity + 0.30 * token_overlap + 0.15 * entity_overlap
        duplicate = not blockers and bool(shared_entities) and (
            title_similarity >= 0.88 or (title_similarity >= 0.76 and token_overlap >= 0.48)
        )
        reason = "lexical_fallback_match" if duplicate else (
            "blocked:" + ",".join(blockers) if blockers else "below_lexical_fallback_threshold"
        )

    return PairEvidence(
        score=round(float(score), 6),
        duplicate=duplicate,
        semantic_similarity=round(semantic, 6),
        token_overlap=round(token_overlap, 6),
        entity_overlap=round(entity_overlap, 6),
        title_similarity=round(title_similarity, 6),
        publication_gap_days=gap,
        shared_entities=shared_entities,
        left_event_types=tuple(sorted(left_types)),
        right_event_types=tuple(sorted(right_types)),
        left_amounts=tuple(sorted(left_amounts)),
        right_amounts=tuple(sorted(right_amounts)),
        blockers=tuple(blockers),
        decision_reason=reason,
    )


def deduplicate_articles(
    articles: Sequence[Mapping[str, object]],
    limit: int = 25,
    *,
    config: DedupeConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> DeduplicationResult:
    if limit < 1:
        raise ValueError("The retained event limit must be at least 1")
    config = config or DedupeConfig()
    indexed = [
        (index, dict(article))
        for index, article in enumerate(articles)
        if isinstance(article, Mapping)
        if _clean(article.get("title")) not in {"", "[Removed]"}
    ]
    indexed.sort(
        key=lambda item: (_published(item[1]) or datetime.min.replace(tzinfo=timezone.utc), -item[0]),
        reverse=True,
    )
    if not indexed:
        diagnostics = DeduplicationDiagnostics(
            scanned_articles=0, usable_articles=0, retained_articles=0,
            removed_duplicates=0, limit=limit, threshold=config.threshold,
            backend="not-run", model=None, semantic_active=False,
            fallback_used=False, fallback_reason=None, weights=dict(config.weights), merges=(),
        )
        return DeduplicationResult((), diagnostics)

    texts = [article_text(article) for _, article in indexed]
    batch, fallback_used, fallback_reason = _embed_articles(texts, config, embedding_provider)
    kept_positions: list[int] = []
    merges: list[MergeProvenance] = []
    scanned = 0
    for position, (original_index, article) in enumerate(indexed):
        scanned += 1
        merge: MergeProvenance | None = None
        for prior_position in kept_positions:
            prior_index, prior = indexed[prior_position]
            evidence = compare_articles(
                prior, article, batch.vectors[prior_position], batch.vectors[position],
                config=config, semantic_active=batch.semantic_active,
            )
            if evidence.duplicate:
                merge = MergeProvenance(
                    representative_index=prior_index,
                    duplicate_index=original_index,
                    representative_title=_clean(prior.get("title")),
                    duplicate_title=_clean(article.get("title")),
                    evidence=evidence,
                )
                break
        if merge is not None:
            merges.append(merge)
            continue
        kept_positions.append(position)
        if len(kept_positions) >= limit:
            break

    kept = tuple(dict(indexed[position][1]) for position in kept_positions)
    diagnostics = DeduplicationDiagnostics(
        scanned_articles=scanned,
        usable_articles=len(indexed),
        retained_articles=len(kept),
        removed_duplicates=len(merges),
        limit=limit,
        threshold=config.threshold,
        backend=batch.backend,
        model=str(batch.metadata.get("model")) if batch.metadata.get("model") else None,
        semantic_active=batch.semantic_active,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        weights=dict(config.weights),
        merges=tuple(merges),
    )
    return DeduplicationResult(kept, diagnostics)


def evaluate_fixture(path: Path) -> EvaluationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("version") or "unknown")
    articles = payload.get("articles")
    pairs = payload.get("pairs")
    if not isinstance(articles, list) or not isinstance(pairs, list):
        raise ValueError("Evaluation fixture must contain article and pair lists")
    by_id = {str(article["id"]): article for article in articles}
    vectors = {
        str(article["id"]): np.asarray(article["semantic_vector"], dtype=np.float64)
        for article in articles
    }
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    config = DedupeConfig(
        threshold=float(settings.get("threshold", DEFAULT_THRESHOLD)),
        semantic_floor=float(settings.get("semantic_floor", 0.72)),
        max_publication_gap_days=int(settings.get("max_publication_gap_days", 21)),
    )
    false_merges: list[dict[str, object]] = []
    missed_merges: list[dict[str, object]] = []
    tp = fp = tn = fn = 0
    positive = 0
    for pair in pairs:
        left_id = str(pair["left"])
        right_id = str(pair["right"])
        expected = bool(pair["same_event"])
        positive += int(expected)
        evidence = compare_articles(
            by_id[left_id], by_id[right_id], vectors[left_id], vectors[right_id],
            config=config, semantic_active=True,
        )
        case = {
            "left": left_id,
            "right": right_id,
            "case": str(pair.get("case") or ""),
            "score": evidence.score,
            "reason": evidence.decision_reason,
        }
        if expected and evidence.duplicate:
            tp += 1
        elif expected:
            fn += 1
            missed_merges.append(case)
        elif evidence.duplicate:
            fp += 1
            false_merges.append(case)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvaluationResult(
        fixture_version=version,
        backend=str(payload.get("embedding_backend") or "fixture-vectors"),
        threshold=config.threshold,
        pair_count=len(pairs),
        positive_pairs=positive,
        negative_pairs=len(pairs) - positive,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        false_merges=tuple(false_merges),
        missed_merges=tuple(missed_merges),
    )


def diagnostics_to_dict(diagnostics: DeduplicationDiagnostics) -> dict[str, object]:
    return asdict(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event-level news deduplication")
    parser.add_argument("--evaluate", type=Path, required=True, help="Path to a labeled JSON fixture")
    args = parser.parse_args()
    print(json.dumps(asdict(evaluate_fixture(args.evaluate)), indent=2))


if __name__ == "__main__":
    main()
