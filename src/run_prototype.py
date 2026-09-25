from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .analyze_clusters import propose_labels, write_cluster_report
from .embeddings import generate_combined_field_embeddings, generate_embeddings
from .fuzzy_cluster import fit_fuzzy_cmeans, select_k
from .inspect_data import inspect_dataframe, write_inspection_report
from .visualize import write_svg


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def _save_run_state(
    df: pd.DataFrame,
    vectors: np.ndarray,
    centers: np.ndarray,
    memberships: np.ndarray,
    native: np.ndarray,
    labels: list[str],
    top_terms: list[list[str]],
    diagnostics: list[dict[str, float]],
    normalized_texts: list[str],
    embedding_backend: str,
    embedding_metadata: dict[str, object],
    candidate_results: list,
) -> None:
    state_arrays = {
        "vectors": vectors,
        "centers": centers,
        "memberships": memberships,
        "native_memberships": native,
    }
    candidate_metadata = {}
    for candidate in candidate_results:
        candidate_labels, candidate_terms = propose_labels(
            normalized_texts, candidate.independent_memberships
        )
        if candidate.k == memberships.shape[1]:
            candidate_labels = labels
            candidate_terms = top_terms
        state_arrays[f"centers_k{candidate.k}"] = candidate.centers
        state_arrays[f"memberships_k{candidate.k}"] = candidate.independent_memberships
        state_arrays[f"native_memberships_k{candidate.k}"] = candidate.native_memberships
        candidate_metadata[str(candidate.k)] = {
            "labels": candidate_labels,
            "top_terms": candidate_terms,
            "diagnostics": candidate.diagnostics,
        }
    np.savez_compressed(
        config.OUTPUT_DIR / "run_state.npz",
        **state_arrays,
    )
    metadata = {
        "labels": labels,
        "top_terms": top_terms,
        "diagnostics": diagnostics,
        "normalized_texts": normalized_texts,
        "embedding_backend": embedding_backend,
        "embedding_metadata": embedding_metadata,
        "selected_k": int(memberships.shape[1]),
        "source_row_count": int(len(df)),
        "models": candidate_metadata,
    }
    (config.OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _load_local_env(config.PROJECT_DIR / ".env")
    df = pd.read_csv(config.INPUT_CSV)
    inspection = inspect_dataframe(df)
    write_inspection_report(inspection, config.OUTPUT_DIR / "data_inspection.md")

    usable = df[config.SEMANTIC_FIELD].notna() & df[config.SEMANTIC_FIELD].astype(str).str.strip().ne("")
    if not usable.all():
        raise ValueError("This prototype currently expects the selected semantic view to cover every row.")

    if config.EMBEDDING_BACKEND == "openai":
        embedding = generate_combined_field_embeddings(
            {config.SEMANTIC_FIELD: df.loc[usable, config.SEMANTIC_FIELD].tolist()},
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model_name=os.getenv(
                "OPENAI_EMBEDDING_MODEL", config.OPENAI_EMBEDDING_MODEL
            ).strip(),
            cache_dir=config.EMBEDDING_CACHE_DIR,
        )
    else:
        embedding = generate_embeddings(
            df.loc[usable, config.SEMANTIC_FIELD].tolist(),
            backend=config.EMBEDDING_BACKEND,
            model_name=config.SENTENCE_TRANSFORMER_MODEL,
            lsa_dimensions=config.LSA_DIMENSIONS,
        )
    results = [
        fit_fuzzy_cmeans(
            embedding.vectors,
            k,
            fuzziness=config.FUZZINESS,
            n_initializations=config.N_INITIALIZATIONS,
            max_iterations=config.MAX_ITERATIONS,
            tolerance=config.TOLERANCE,
            seed=config.RANDOM_SEED,
        )
        for k in config.K_RANGE
        if k < len(df)
    ]
    selected, diagnostic_rows = select_k(results)
    labels, top_terms = propose_labels(embedding.normalized_texts, selected.independent_memberships)
    labels = [config.CLUSTER_LABEL_OVERRIDES.get(i, label) for i, label in enumerate(labels)]
    _save_run_state(
        df,
        embedding.vectors,
        selected.centers,
        selected.independent_memberships,
        selected.native_memberships,
        labels,
        top_terms,
        diagnostic_rows,
        embedding.normalized_texts,
        embedding.backend,
        embedding.metadata,
        results,
    )

    from .analyze_from_saved import regenerate_threshold_outputs

    regenerate_threshold_outputs()
    summary = {
        "rows": len(df),
        "semantic_field": config.SEMANTIC_FIELD,
        "embedding_backend": embedding.backend,
        "selected_k": selected.k,
        "threshold": config.MEMBERSHIP_THRESHOLD,
        "outputs": str(config.OUTPUT_DIR),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
