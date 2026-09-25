from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .analyze_clusters import write_cluster_report
from .visualize import write_svg


def regenerate_threshold_outputs() -> None:
    state = np.load(config.OUTPUT_DIR / "run_state.npz")
    metadata = json.loads((config.OUTPUT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(config.INPUT_CSV)
    memberships = state["memberships"]
    native = state["native_memberships"]
    threshold = config.MEMBERSHIP_THRESHOLD
    flags = memberships >= threshold

    output = df.copy()
    output.insert(0, "row_id", [f"source_row_{i}" for i in range(len(df))])
    output.insert(1, "selected_semantic_view", config.SEMANTIC_FIELD)
    output.insert(2, "selected_text_normalized", metadata["normalized_texts"])
    for cluster in range(memberships.shape[1]):
        output[f"cluster_{cluster}_label"] = metadata["labels"][cluster]
        output[f"cluster_{cluster}_membership"] = np.round(memberships[:, cluster], 6)
        output[f"cluster_{cluster}_native_fcm"] = np.round(native[:, cluster], 6)
        output[f"cluster_{cluster}_above_{threshold:.2f}"] = flags[:, cluster]
    output["membership_count"] = flags.sum(axis=1)
    output["clusters_above_threshold"] = [
        "; ".join(str(i) for i in np.where(row)[0]) for row in flags
    ]
    output.to_csv(config.OUTPUT_DIR / "memberships.csv", index=False)

    write_cluster_report(
        df=df,
        texts=metadata["normalized_texts"],
        memberships=memberships,
        native_memberships=native,
        labels=metadata["labels"],
        top_terms=metadata["top_terms"],
        selected_k=int(metadata["selected_k"]),
        diagnostics_rows=metadata["diagnostics"],
        threshold=threshold,
        thresholds=config.THRESHOLDS_TO_COMPARE,
        semantic_field=config.SEMANTIC_FIELD,
        embedding_backend=metadata["embedding_backend"],
        embedding_metadata=metadata["embedding_metadata"],
        path=config.OUTPUT_DIR / "cluster_report.md",
    )
    write_svg(
        vectors=state["vectors"],
        titles=df[config.SEMANTIC_FIELD].fillna("").astype(str).tolist(),
        names=df["Full Name"].fillna("").astype(str).tolist(),
        memberships=memberships,
        labels=metadata["labels"],
        threshold=threshold,
        path=config.OUTPUT_DIR / "embedding_projection.svg",
    )


def main() -> None:
    regenerate_threshold_outputs()
    print(
        f"Regenerated threshold-dependent outputs at cutoff "
        f"{config.MEMBERSHIP_THRESHOLD:.2f} without recomputing embeddings."
    )


if __name__ == "__main__":
    main()

