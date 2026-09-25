from __future__ import annotations

import html
import hashlib
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence
from uuid import uuid4

import numpy as np


OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_CACHE_LOCK = Lock()

ROLE_ALIASES = {
    "co founder": "cofounder founder",
    "cofounder": "cofounder founder",
    "fundador": "founder",
    "fundadora": "founder",
    "embaixadora": "ambassador",
    "influenciadora": "influencer",
    "mentora": "mentor advisor",
    "consultora de recursos humanos": "human resources consultant",
}


def normalize_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/|–—-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    for source, target in ROLE_ALIASES.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    normalized_texts: list[str]
    backend: str
    metadata: dict[str, object]


def _clean_embedding_text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def _cache_path(cache_dir: Path, model_name: str, text: str) -> Path:
    model_key = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    digest = hashlib.sha256(f"{model_name}\0{text}".encode("utf-8")).hexdigest()
    return cache_dir / model_key / f"{digest}.npy"


def _read_cached_vector(path: Path) -> np.ndarray | None:
    try:
        vector = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        return None
    return np.asarray(vector, dtype=np.float64)


def _write_cached_vector(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npy")
    np.save(temporary, np.asarray(vector, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _request_openai_embeddings(
    api_key: str,
    texts: list[str],
    model_name: str,
) -> list[np.ndarray]:
    request = urllib.request.Request(
        OPENAI_EMBEDDINGS_URL,
        data=json.dumps(
            {"model": model_name, "input": texts, "encoding_format": "float"},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "fuzzy-persona-clustering/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            message = str(error_payload.get("error", {}).get("message") or "Embedding request failed")
        except Exception:
            message = f"OpenAI embeddings returned HTTP {exc.code}"
        raise RuntimeError(message) from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError("OpenAI embeddings returned an unexpected number of vectors")
    ordered: list[np.ndarray | None] = [None] * len(texts)
    for position, item in enumerate(data):
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise ValueError("OpenAI embeddings returned an invalid vector")
        index = int(item.get("index", position))
        if index < 0 or index >= len(ordered) or ordered[index] is not None:
            raise ValueError("OpenAI embeddings returned invalid vector ordering")
        vector = np.asarray(item["embedding"], dtype=np.float64)
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
            raise ValueError("OpenAI embeddings returned a malformed vector")
        ordered[index] = vector
    if any(vector is None for vector in ordered):
        raise ValueError("OpenAI embeddings omitted a requested vector")
    return [vector for vector in ordered if vector is not None]


def _openai_embeddings(
    values: Sequence[object],
    api_key: str,
    model_name: str,
    cache_dir: Path,
    batch_size: int = 64,
) -> EmbeddingResult:
    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
    texts = [_clean_embedding_text(value) for value in values]
    unique_texts = list(dict.fromkeys(text for text in texts if text))
    by_text: dict[str, np.ndarray] = {}
    cache_hits = 0
    api_inputs = 0
    api_requests = 0

    with _CACHE_LOCK:
        missing: list[str] = []
        for text in unique_texts:
            cached = _read_cached_vector(_cache_path(cache_dir, model_name, text))
            if cached is None:
                missing.append(text)
            else:
                by_text[text] = cached
                cache_hits += 1
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = _request_openai_embeddings(api_key, batch, model_name)
            api_requests += 1
            api_inputs += len(batch)
            for text, vector in zip(batch, vectors):
                by_text[text] = vector
                _write_cached_vector(_cache_path(cache_dir, model_name, text), vector)

    dimensions = next((len(vector) for vector in by_text.values()), 0)
    if not dimensions:
        raise ValueError("No nonblank text was available to embed")
    if any(len(vector) != dimensions for vector in by_text.values()):
        raise ValueError("Cached OpenAI embedding dimensions do not match")
    matrix = np.zeros((len(texts), dimensions), dtype=np.float64)
    for index, text in enumerate(texts):
        if text:
            matrix[index] = by_text[text]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return EmbeddingResult(
        vectors=matrix,
        normalized_texts=[normalize_text(value) for value in values],
        backend="openai",
        metadata={
            "model": model_name,
            "dimensions": int(dimensions),
            "cache_hits": int(cache_hits),
            "api_inputs": int(api_inputs),
            "api_requests": int(api_requests),
            "blank_inputs": int(sum(not text for text in texts)),
        },
    )


def generate_combined_field_embeddings(
    field_values: dict[str, Sequence[object]],
    api_key: str,
    model_name: str,
    cache_dir: Path,
) -> EmbeddingResult:
    if not field_values:
        raise ValueError("Select at least one field to embed")
    row_count = len(next(iter(field_values.values())))
    if any(len(values) != row_count for values in field_values.values()):
        raise ValueError("Selected embedding fields have inconsistent row counts")

    matrices: list[np.ndarray] = []
    populated_masks: list[np.ndarray] = []
    field_metadata: dict[str, dict[str, object]] = {}
    combined_texts = ["" for _ in range(row_count)]
    for field, values in field_values.items():
        cleaned = [_clean_embedding_text(value) for value in values]
        prefixed = [f"{field}: {value}" if value else "" for value in cleaned]
        result = _openai_embeddings(prefixed, api_key, model_name, cache_dir)
        matrices.append(result.vectors)
        populated_masks.append(np.asarray([bool(value) for value in cleaned], dtype=np.float64))
        field_metadata[field] = dict(result.metadata)
        for index, value in enumerate(cleaned):
            if value:
                combined_texts[index] = " | ".join(
                    part for part in (combined_texts[index], f"{field}: {value}") if part
                )

    dimensions = matrices[0].shape[1]
    if any(matrix.shape != (row_count, dimensions) for matrix in matrices):
        raise ValueError("Selected fields returned incompatible embedding dimensions")
    combined = np.zeros((row_count, dimensions), dtype=np.float64)
    populated_count = np.zeros((row_count, 1), dtype=np.float64)
    for matrix, mask in zip(matrices, populated_masks):
        combined += matrix * mask[:, None]
        populated_count += mask[:, None]
    combined /= np.maximum(populated_count, 1.0)
    combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)

    cache_hits = sum(int(metadata.get("cache_hits", 0)) for metadata in field_metadata.values())
    api_inputs = sum(int(metadata.get("api_inputs", 0)) for metadata in field_metadata.values())
    api_requests = sum(int(metadata.get("api_requests", 0)) for metadata in field_metadata.values())
    return EmbeddingResult(
        vectors=combined,
        normalized_texts=[normalize_text(text) for text in combined_texts],
        backend="openai",
        metadata={
            "model": model_name,
            "dimensions": int(dimensions),
            "combination": "Equal-weight mean of nonblank selected-field embeddings, then L2 normalization",
            "cache_hits": cache_hits,
            "api_inputs": api_inputs,
            "api_requests": api_requests,
            "fields": field_metadata,
        },
    )


def _word_features(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text)
    features = [f"w:{token}" for token in tokens]
    features.extend(f"b:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return features


def _char_features(text: str) -> list[str]:
    compact = f" {text} "
    features: list[str] = []
    for n in (3, 4, 5):
        features.extend(f"c:{compact[i:i+n]}" for i in range(max(0, len(compact) - n + 1)))
    return features


def _tfidf_lsa(texts: Sequence[str], dimensions: int) -> EmbeddingResult:
    documents = []
    document_frequency: Counter[str] = Counter()
    for text in texts:
        counts = Counter(_word_features(text))
        char_counts = Counter(_char_features(text))
        for key, value in char_counts.items():
            counts[key] += 0.25 * value
        documents.append(counts)
        document_frequency.update(counts.keys())

    vocabulary = sorted(
        document_frequency,
        key=lambda term: (document_frequency[term], term),
        reverse=True,
    )[:4000]
    index = {term: i for i, term in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    n_docs = len(texts)
    for row, counts in enumerate(documents):
        for term, count in counts.items():
            col = index.get(term)
            if col is None:
                continue
            tf = 1.0 + np.log(float(count))
            idf = np.log((1.0 + n_docs) / (1.0 + document_frequency[term])) + 1.0
            matrix[row, col] = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    usable = min(dimensions, max(2, len(texts) - 1), len(singular_values))
    vectors = u[:, :usable] * singular_values[:usable]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    explained = singular_values[:usable] ** 2
    total = np.sum(singular_values**2)
    return EmbeddingResult(
        vectors=vectors.astype(np.float64),
        normalized_texts=list(texts),
        backend="tfidf-lsa",
        metadata={
            "dimensions": int(usable),
            "feature_count": int(len(vocabulary)),
            "explained_energy": float(explained.sum() / max(total, 1e-12)),
            "note": (
                "Offline fallback: word/bigram and character n-gram TF-IDF projected "
                "with latent semantic analysis. Weaker than a pretrained sentence model."
            ),
        },
    )


def generate_embeddings(
    values: Sequence[object],
    backend: str = "auto",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    lsa_dimensions: int = 24,
    api_key: str = "",
    cache_dir: Path | None = None,
) -> EmbeddingResult:
    if backend == "openai":
        if cache_dir is None:
            raise ValueError("A local cache directory is required for OpenAI embeddings")
        return _openai_embeddings(values, api_key, model_name, cache_dir)
    normalized = [normalize_text(value) for value in values]
    if backend in {"auto", "sentence-transformers"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, local_files_only=(backend == "auto"))
            vectors = model.encode(
                normalized,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return EmbeddingResult(
                vectors=np.asarray(vectors, dtype=np.float64),
                normalized_texts=normalized,
                backend="sentence-transformers",
                metadata={"model": model_name, "dimensions": int(vectors.shape[1])},
            )
        except Exception as exc:
            if backend == "sentence-transformers":
                raise RuntimeError(
                    "Sentence Transformers backend requested but the package/model is unavailable."
                ) from exc
            fallback = _tfidf_lsa(normalized, lsa_dimensions)
            fallback.metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return fallback
    if backend == "tfidf-lsa":
        return _tfidf_lsa(normalized, lsa_dimensions)
    raise ValueError(f"Unknown embedding backend: {backend}")
