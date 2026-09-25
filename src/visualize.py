from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

import numpy as np


PALETTE = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2", "#DB2777", "#4B5563"]


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * singular_values[:2]


def write_svg(
    vectors: np.ndarray,
    titles: Sequence[str],
    names: Sequence[str],
    memberships: np.ndarray,
    labels: Sequence[str],
    threshold: float,
    path: Path,
) -> None:
    points = pca_2d(vectors)
    width, height = 1200, 820
    left, right, top, bottom = 80, 330, 90, 80
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x = left + (points[:, 0] - x_min) / max(x_max - x_min, 1e-12) * (width - left - right)
    y = top + (y_max - points[:, 1]) / max(y_max - y_min, 1e-12) * (height - top - bottom)
    flags = memberships >= threshold
    primary = memberships.argmax(axis=1)
    center_xy = []
    for cluster in range(memberships.shape[1]):
        weights = memberships[:, cluster]
        center_xy.append((float(np.average(x, weights=weights)), float(np.average(y, weights=weights))))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="80" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Job-title embedding projection</text>',
        f'<text x="80" y="67" font-family="Arial" font-size="13" fill="#4B5563">PCA view; fill is strongest affinity, dark outline means 2+ memberships at cutoff {threshold:.2f}</text>',
        f'<rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" fill="#F9FAFB" stroke="#D1D5DB"/>',
    ]
    for cluster, (cx, cy) in enumerate(center_xy):
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-family="Arial" '
            f'font-size="12" font-weight="700" fill="{PALETTE[cluster % len(PALETTE)]}" opacity="0.85">C{cluster}</text>'
        )
    for i, (px, py) in enumerate(zip(x, y)):
        count = int(flags[i].sum())
        fill = PALETTE[int(primary[i]) % len(PALETTE)] if count else "#9CA3AF"
        stroke = "#111827" if count >= 2 else "#FFFFFF"
        stroke_width = 2.6 if count >= 2 else 1.2
        detail = ", ".join(f"C{j}={memberships[i, j]:.2f}" for j in np.argsort(-memberships[i])[:3])
        tooltip = html.escape(f"{names[i]} | {titles[i]} | memberships: {count} | {detail}")
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{fill}" fill-opacity="0.78" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"><title>{tooltip}</title></circle>'
        )
    legend_x = width - right + 35
    parts.append(f'<text x="{legend_x}" y="105" font-family="Arial" font-size="15" font-weight="700" fill="#111827">Clusters</text>')
    for cluster, label in enumerate(labels):
        ly = 137 + cluster * 54
        parts.append(f'<circle cx="{legend_x + 7}" cy="{ly - 5}" r="7" fill="{PALETTE[cluster % len(PALETTE)]}"/>')
        safe_label = html.escape(label)
        parts.append(f'<text x="{legend_x + 24}" y="{ly}" font-family="Arial" font-size="12" fill="#111827">C{cluster}: {safe_label}</text>')
    parts.extend(
        [
            f'<text x="{left + (width-left-right)/2:.1f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="12" fill="#6B7280">PCA component 1</text>',
            f'<text x="22" y="{top + (height-top-bottom)/2:.1f}" transform="rotate(-90 22 {top + (height-top-bottom)/2:.1f})" text-anchor="middle" font-family="Arial" font-size="12" fill="#6B7280">PCA component 2</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")

