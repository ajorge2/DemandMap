from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FuzzyResult:
    k: int
    centers: np.ndarray
    native_memberships: np.ndarray
    independent_memberships: np.ndarray
    objective: float
    iterations: int
    diagnostics: dict[str, float]


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _fit_once(
    x: np.ndarray,
    k: int,
    fuzziness: float,
    max_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    rng = np.random.default_rng(seed)
    # K-means++ style seeding prevents the common high-dimensional FCM failure
    # mode where all centroids collapse near the global mean.
    chosen = [int(rng.integers(len(x)))]
    while len(chosen) < k:
        current = x[chosen]
        nearest = np.min(_squared_distances(x, current), axis=1)
        nearest[chosen] = 0.0
        total = float(nearest.sum())
        if total <= 1e-12:
            remaining = [i for i in range(len(x)) if i not in chosen]
            chosen.append(int(rng.choice(remaining)))
        else:
            chosen.append(int(rng.choice(len(x), p=nearest / total)))
    initial_centers = x[chosen]
    initial_distances2 = np.maximum(_squared_distances(x, initial_centers), 1e-12)
    inverse = initial_distances2 ** (-1.0 / (fuzziness - 1.0))
    memberships = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    previous_objective = np.inf
    for iteration in range(1, max_iterations + 1):
        powered = memberships**fuzziness
        centers = (powered.T @ x) / np.maximum(powered.sum(axis=0)[:, None], 1e-12)
        distances2 = np.maximum(_squared_distances(x, centers), 1e-12)
        exponent = -1.0 / (fuzziness - 1.0)
        inverse = distances2**exponent
        memberships = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
        objective = float(np.sum((memberships**fuzziness) * distances2))
        if abs(previous_objective - objective) <= tolerance * max(1.0, previous_objective):
            break
        previous_objective = objective
    return centers, memberships, objective, iteration


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    distances = np.sqrt(np.maximum(_squared_distances(x, x), 0.0))
    values = []
    for i, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != i]
        if len(same) == 0:
            values.append(0.0)
            continue
        a = float(distances[i, same].mean())
        b = min(float(distances[i, labels == other].mean()) for other in unique if other != label)
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values))


def independent_prototype_memberships(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Convert centroid distances to independent, non-simplex affinities.

    Fuzzy C-Means learns the prototypes, but its native memberships sum to one.
    Distances are converted to tie-aware within-prototype percentiles and then
    sharpened. This avoids the distance-concentration problem in sparse text
    embeddings while preserving each prototype's nearest-to-farthest ordering.
    The output can sum above one, which is useful for explicitly overlapping
    semantic traits. These are affinities, not calibrated probabilities.
    """
    distances = np.sqrt(np.maximum(_squared_distances(x, centers), 0.0))
    scores = np.zeros_like(distances)
    n = len(x)
    for cluster in range(centers.shape[0]):
        values = distances[:, cluster]
        order = np.argsort(values)
        ranks = np.empty(n, dtype=float)
        start = 0
        while start < n:
            end = start + 1
            while end < n and np.isclose(values[order[end]], values[order[start]], rtol=1e-8, atol=1e-10):
                end += 1
            average_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = average_rank
            start = end
        closeness_percentile = 1.0 - ranks / max(n - 1, 1)
        scores[:, cluster] = closeness_percentile**3.0
    return np.clip(scores, 0.0, 1.0)


def fit_fuzzy_cmeans(
    x: np.ndarray,
    k: int,
    fuzziness: float = 1.8,
    n_initializations: int = 20,
    max_iterations: int = 500,
    tolerance: float = 1e-7,
    seed: int = 42,
) -> FuzzyResult:
    best = None
    for offset in range(n_initializations):
        candidate = _fit_once(
            x,
            k,
            fuzziness,
            max_iterations,
            tolerance,
            seed + 1009 * offset + 37 * k,
        )
        if best is None or candidate[2] < best[2]:
            best = candidate
    assert best is not None
    centers, native, objective, iterations = best
    labels = native.argmax(axis=1)
    center_distances2 = _squared_distances(centers, centers)
    np.fill_diagonal(center_distances2, np.inf)
    min_center_distance2 = float(np.min(center_distances2))
    diagnostics = {
        "partition_coefficient": float(np.mean(np.sum(native**2, axis=1))),
        "partition_entropy": float(-np.mean(np.sum(native * np.log(native + 1e-12), axis=1))),
        "xie_beni": float(objective / max(len(x) * min_center_distance2, 1e-12)),
        "silhouette_argmax": _silhouette(x, labels),
        "minimum_centroid_distance": float(np.sqrt(min_center_distance2)),
        "objective": float(objective),
    }
    independent = independent_prototype_memberships(x, centers)
    return FuzzyResult(k, centers, native, independent, objective, iterations, diagnostics)


def _rank(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    order = np.argsort(values if higher_is_better else -values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks if higher_is_better else 1.0 - ranks


def select_k(results: list[FuzzyResult]) -> tuple[FuzzyResult, list[dict[str, float]]]:
    silhouette = np.array([r.diagnostics["silhouette_argmax"] for r in results])
    fpc = np.array([r.diagnostics["partition_coefficient"] for r in results])
    xb = np.array([r.diagnostics["xie_beni"] for r in results])
    separation = np.array([r.diagnostics["minimum_centroid_distance"] for r in results])
    composite = (
        0.40 * _rank(silhouette, True)
        + 0.25 * _rank(fpc, True)
        + 0.20 * _rank(xb, False)
        + 0.15 * _rank(separation, True)
    )
    rows = []
    for result, score in zip(results, composite):
        row = {"k": float(result.k), "selection_score": float(score), **result.diagnostics}
        rows.append(row)
    selected = results[int(np.argmax(composite))]
    return selected, rows
