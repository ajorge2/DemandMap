from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


STOPWORDS = {
    "and", "of", "the", "to", "a", "de", "member", "chief", "executive",
    "cofounder", "founding", "special", "strategic",
}

LABEL_RULES = [
    ({"investor", "venture", "angel", "seed", "partner", "scout"}, "Investors & Venture Partners"),
    ({"advisor", "advisory", "board", "mentor"}, "Advisors & Board Members"),
    ({"founder", "ceo", "chairman", "chairwoman", "president"}, "Founders & Executive Leaders"),
    ({"marketing", "growth", "brand", "creator", "content", "evangelist", "influencer", "ambassador"}, "Marketing, Creator & Brand Leaders"),
    ({"talent", "human", "resources", "people", "recruiting", "acquisition"}, "People & Talent Leaders"),
    ({"information", "technology", "data", "ai", "product", "engineering", "scientific", "medical"}, "Technical & Specialist Leaders"),
    ({"sales", "revenue", "business"}, "Sales & Revenue Leaders"),
]


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"\b\w+\b", text.lower()) if token not in STOPWORDS and len(token) > 2]


def propose_labels(texts: Sequence[str], memberships: np.ndarray) -> tuple[list[str], list[list[str]]]:
    labels = []
    top_terms_by_cluster = []
    global_counts = Counter(token for text in texts for token in set(_tokens(text)))
    for cluster in range(memberships.shape[1]):
        scores: Counter[str] = Counter()
        core_size = max(8, int(np.ceil(len(texts) / memberships.shape[1] * 1.5)))
        core_indices = np.argsort(-memberships[:, cluster])[:core_size]
        for index in core_indices:
            text = texts[index]
            other_best = float(np.max(np.delete(memberships[index], cluster)))
            contrast = max(float(memberships[index, cluster]) - other_best, 0.0)
            weight = float(memberships[index, cluster]) * (0.25 + contrast)
            for token in set(_tokens(text)):
                scores[token] += float(weight) / np.sqrt(max(global_counts[token], 1))
        top_terms = [term for term, _ in scores.most_common(8)]
        top_terms_by_cluster.append(top_terms)
        best_label = None
        best_score = 0.0
        for keywords, label in LABEL_RULES:
            score = sum(scores[word] for word in keywords)
            if score > best_score:
                best_score = score
                best_label = label
        labels.append(best_label or " / ".join(term.title() for term in top_terms[:3]))
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    for base_label, indices in grouped.items():
        if len(indices) < 2:
            continue
        used_terms: set[str] = set()
        for index in indices:
            descriptor = next(
                (term for term in top_terms_by_cluster[index] if term not in used_terms),
                f"cluster {index}",
            )
            used_terms.add(descriptor)
            labels[index] = f"{base_label} — {descriptor.title()} focus"
    return labels, top_terms_by_cluster


def overlap_counts(memberships: np.ndarray, threshold: float) -> dict[str, int]:
    counts = (memberships >= threshold).sum(axis=1)
    return {
        "zero": int(np.sum(counts == 0)),
        "one": int(np.sum(counts == 1)),
        "two": int(np.sum(counts == 2)),
        "three_or_more": int(np.sum(counts >= 3)),
    }


def write_cluster_report(
    df: pd.DataFrame,
    texts: Sequence[str],
    memberships: np.ndarray,
    native_memberships: np.ndarray,
    labels: Sequence[str],
    top_terms: Sequence[Sequence[str]],
    selected_k: int,
    diagnostics_rows: list[dict[str, float]],
    threshold: float,
    thresholds: Sequence[float],
    semantic_field: str,
    embedding_backend: str,
    embedding_metadata: dict[str, object],
    path: Path,
) -> None:
    lines = [
        "# Fuzzy semantic clustering report",
        "",
        "## Experiment setup",
        "",
        f"- Semantic view: `{semantic_field}` ({len(texts)}/{len(df)} usable rows)",
        f"- Embedding backend used: `{embedding_backend}`",
        f"- Selected clusters: {selected_k}",
        f"- Primary membership cutoff: {threshold:.2f}",
        "- Clustering: Fuzzy C-Means prototypes plus independently calibrated prototype affinities.",
        "",
        "The final `cluster_*_membership` scores are independent affinities and do not sum to one. "
        "This allows one prospect to clear the cutoff for several semantic prototypes. The native "
        "Fuzzy C-Means memberships are also exported for auditability; those do sum to one.",
        "",
    ]
    if embedding_backend == "tfidf-lsa":
        lines.extend(
            [
                "> The requested pretrained sentence model was not available locally. This run used the "
                "offline TF-IDF + LSA fallback. It is reproducible and useful for plumbing and threshold "
                "evaluation, but its semantic generalization is weaker, especially for short titles and synonyms.",
                "",
            ]
        )
    lines.extend(
        [
            "## K diagnostics",
            "",
            "The selection score combines argmax silhouette (diagnostic only), fuzzy partition coefficient, "
            "Xie-Beni compactness/separation, and minimum centroid separation. No hard label is used in the final output.",
            "",
            "| K | Selection score | Silhouette | Partition coefficient | Partition entropy | Xie-Beni |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics_rows:
        marker = " **selected**" if int(row["k"]) == selected_k else ""
        lines.append(
            f"| {int(row['k'])}{marker} | {row['selection_score']:.3f} | "
            f"{row['silhouette_argmax']:.3f} | {row['partition_coefficient']:.3f} | "
            f"{row['partition_entropy']:.3f} | {row['xie_beni']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cutoff sensitivity",
            "",
            "| Cutoff | Zero clusters | Exactly one | Two | Three or more |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for value in thresholds:
        stats = overlap_counts(memberships, value)
        lines.append(
            f"| {value:.2f} | {stats['zero']} | {stats['one']} | {stats['two']} | {stats['three_or_more']} |"
        )

    flags = memberships >= threshold
    lines.extend(["", f"## Cluster overlap at cutoff {threshold:.2f}", ""])
    pair_rows = []
    for left, right in combinations(range(selected_k), 2):
        shared = int(np.sum(flags[:, left] & flags[:, right]))
        union = int(np.sum(flags[:, left] | flags[:, right]))
        if shared:
            pair_rows.append((shared, shared / max(union, 1), left, right))
    if pair_rows:
        lines.extend(["| Cluster pair | Shared rows | Jaccard overlap |", "|---|---:|---:|"])
        for shared, jaccard, left, right in sorted(pair_rows, reverse=True):
            lines.append(
                f"| {left}: {labels[left]} + {right}: {labels[right]} | {shared} | {jaccard:.2f} |"
            )
    else:
        lines.append("No cluster pair shares a row at this cutoff.")

    title_values = df[semantic_field].fillna("").astype(str).tolist()
    name_values = df["Full Name"].fillna("").astype(str).tolist() if "Full Name" in df else [""] * len(df)
    for cluster in range(selected_k):
        scores = memberships[:, cluster]
        member_indices = np.where(flags[:, cluster])[0]
        strong = np.argsort(-scores)[: min(6, len(scores))]
        borderline = np.argsort(np.abs(scores - threshold))[: min(6, len(scores))]
        overlap = [i for i in member_indices if flags[i].sum() >= 2]
        overlap = sorted(overlap, key=lambda i: -scores[i])[:6]
        lines.extend(
            [
                "",
                f"## Cluster {cluster} — {labels[cluster]}",
                "",
                f"- Members above cutoff: {len(member_indices)}",
                f"- Representative terms: {', '.join(top_terms[cluster][:6])}",
                "",
                "Strong members:",
                "",
            ]
        )
        for i in strong:
            lines.append(f"- {title_values[i]} ({name_values[i]}) — {scores[i]:.3f}")
        lines.extend(["", "Borderline examples nearest the cutoff:", ""])
        for i in borderline:
            direction = "in" if scores[i] >= threshold else "out"
            lines.append(f"- {title_values[i]} ({name_values[i]}) — {scores[i]:.3f} ({direction})")
        lines.extend(["", "Shared with other clusters:", ""])
        if not overlap:
            lines.append("- None at this cutoff.")
        for i in overlap:
            memberships_text = ", ".join(
                f"{other}: {memberships[i, other]:.2f}"
                for other in np.where(flags[i])[0]
            )
            lines.append(f"- {title_values[i]} ({name_values[i]}) — {memberships_text}")

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is a 50-row exploratory sample dominated by founders, investors, and advisors.",
            "- Labels summarize examples; they do not determine membership.",
        "- Independent affinities are sharpened within-prototype distance percentiles, not probabilities of persona truth.",
            "- The selected cutoff changes assignment counts without recomputing embeddings or centroids.",
            "- Future views such as company description or industry should be embedded and clustered separately before cross-view comparison.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
