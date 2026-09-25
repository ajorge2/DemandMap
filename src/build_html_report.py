from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from . import config
from .persona_generation import build_fallback_grounding, build_persona_model_input
from .persona_labels import generate_persona_labels
from .representative_accounts import build_representative_accounts
from .visualize import pca_2d


def build_report() -> None:
    state = np.load(config.OUTPUT_DIR / "run_state.npz")
    metadata = json.loads((config.OUTPUT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(config.INPUT_CSV)
    memberships = state["memberships"]
    points = pca_2d(state["vectors"])

    records = []
    for index, row in df.iterrows():
        records.append(
            {
                "index": int(index),
                "rowId": f"source_row_{index}",
                "name": str(row.get("Full Name", "")),
                "title": str(row.get(config.SEMANTIC_FIELD, "")),
                "company": str(row.get("Company", "")),
                "x": round(float(points[index, 0]), 6),
                "y": round(float(points[index, 1]), 6),
            }
        )

    models = {}
    for k_text, model_metadata in metadata["models"].items():
        candidate_memberships = state[f"memberships_k{k_text}"]
        labels, evidence = generate_persona_labels(
            df, [config.SEMANTIC_FIELD], candidate_memberships
        )
        persona_inputs = build_persona_model_input(
            df, [config.SEMANTIC_FIELD], {k_text: candidate_memberships}
        )
        summary_maps = {
            f"K{k_text}-C{cluster}": dict(summaries)
            for cluster, summaries in enumerate(evidence)
        }
        grounding = build_fallback_grounding(persona_inputs, summary_maps)
        models[k_text] = {
            "labels": labels,
            "descriptions": [
                f"Representative roles include {row.get(config.SEMANTIC_FIELD, 'the roles shown below')}."
                for row in evidence
            ],
            "evidence": evidence,
            "personaGrounding": [
                grounding[f"K{k_text}-C{cluster}"]
                for cluster in range(candidate_memberships.shape[1])
            ],
            "memberships": [
                [round(float(value), 6) for value in row]
                for row in candidate_memberships
            ],
            "representativeAccounts": build_representative_accounts(
                df,
                candidate_memberships,
                [config.SEMANTIC_FIELD],
                ["Full Name", "Job Title", "Company"],
            ),
        }

    payload = {
        "datasetId": "demo",
        "datasetName": "B2B AI Startups Marketing Tech Export.csv",
        "isDemo": True,
        "records": records,
        "models": models,
        "defaultK": metadata["selected_k"],
        "minimumK": min(int(value) for value in models),
        "maximumK": max(int(value) for value in models),
        "defaultThreshold": config.MEMBERSHIP_THRESHOLD,
        "embeddingBackend": metadata["embedding_backend"],
        "embeddingMetadata": metadata.get("embedding_metadata", {}),
        "semanticField": config.SEMANTIC_FIELD,
        "clusterableFields": list(config.CLUSTERABLE_FIELDS),
        "defaultSelectedFields": [config.SEMANTIC_FIELD],
        "displayFields": ["Full Name", "Job Title", "Company"],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = "Fuzzy semantic clustering explorer"
    document = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: light-dark(#f5f7fb, #0d1117);
      --surface: light-dark(#ffffff, #161b22);
      --surface-2: light-dark(#f8fafc, #1c232d);
      --text: light-dark(#162033, #edf2f7);
      --muted: light-dark(#617087, #9ba7b7);
      --border: light-dark(#dce3ec, #303946);
      --shadow: light-dark(0 16px 42px rgba(31, 45, 61, .08), 0 16px 42px rgba(0, 0, 0, .28));
      --c0: #3b82f6;
      --c1: #ef4444;
      --c2: #10b981;
      --c3: #8b5cf6;
      --c4: #f59e0b;
      --c5: #06b6d4;
      --c6: #ec4899;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1220px, calc(100% - 32px)); margin: 32px auto 64px; }}
    header {{ display: grid; gap: 14px; margin-bottom: 24px; }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: clamp(30px, 4vw, 48px); letter-spacing: -.04em; line-height: 1.04; }}
    h2 {{ font-size: 19px; letter-spacing: -.015em; }}
    .lede {{ max-width: 780px; color: var(--muted); line-height: 1.6; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .tag {{ padding: 7px 10px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface); color: var(--muted); font-size: 13px; }}
    .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow); }}
    .control-panel {{ padding: 22px; margin-bottom: 18px; }}
    .dataset-panel {{ display: grid; grid-template-columns: 1fr auto; gap: 20px; align-items: center; padding: 20px 22px; margin-bottom: 18px; }}
    .dataset-copy {{ display: grid; gap: 6px; }}
    .dataset-copy p {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .dataset-actions {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 9px; }}
    .upload-button, .secondary-button {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 10px; padding: 10px 14px; font: inherit; font-size: 13px; font-weight: 700; cursor: pointer; }}
    .upload-button {{ border: 0; background: var(--text); color: var(--surface); }}
    .secondary-button {{ border: 1px solid var(--border); background: var(--surface-2); color: var(--text); }}
    .upload-button input {{ position: absolute; inline-size: 1px; block-size: 1px; opacity: 0; pointer-events: none; }}
    .dataset-status {{ grid-column: 1 / -1; min-height: 18px; color: var(--muted); font-size: 13px; }}
    .dataset-status.is-error {{ color: #dc2626; }}
    .control-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }}
    .control-copy {{ display: grid; gap: 8px; margin-bottom: 10px; }}
    .control-copy p {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
    .slider-row {{ display: grid; grid-template-columns: 1fr 70px; gap: 14px; align-items: center; }}
    .field-control {{ margin-top: 24px; padding-top: 22px; border-top: 1px solid var(--border); }}
    .field-toggle-list {{ display: flex; flex-wrap: wrap; gap: 9px; }}
    .field-toggle {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); color: var(--muted); font-size: 13px; cursor: pointer; user-select: none; }}
    .field-toggle:has(input:checked) {{ border-color: var(--c0); color: var(--text); box-shadow: inset 0 0 0 1px var(--c0); }}
    .field-toggle input {{ width: 15px; height: 15px; margin: 0; accent-color: var(--c0); cursor: pointer; }}
    .field-status {{ min-height: 20px; margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .field-status.is-error {{ color: #dc2626; }}
    input[type="range"] {{ width: 100%; accent-color: var(--c0); }}
    output {{ font-variant-numeric: tabular-nums; font-size: 19px; font-weight: 700; text-align: right; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }}
    .stat {{ padding: 18px; }}
    .stat-label {{ color: var(--muted); font-size: 13px; }}
    .stat-value {{ margin-top: 8px; font-size: 30px; font-weight: 750; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }}
    .visual-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; align-items: stretch; }}
    .chart-panel {{ padding: 18px; min-height: 620px; }}
    .panel-heading {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 12px; }}
    .panel-heading p {{ color: var(--muted); font-size: 13px; margin-top: 5px; }}
    #scatter {{ width: 100%; height: 540px; display: block; }}
    .side-panel {{ padding: 18px; }}
    .cluster-list {{ display: grid; grid-template-columns: 1fr; row-gap: 0; margin-top: 16px; }}
    .cluster-list-header {{ display: grid; grid-template-columns: 40px 1fr auto 54px; gap: 10px; align-items: center; padding: 0 0 8px; color: var(--muted); font-size: 12px; font-weight: 650; border-bottom: 1px solid var(--border); }}
    .cluster-list-header span:last-child {{ text-align: right; }}
    .cluster-row {{ display: grid; grid-template-columns: 40px minmax(0, 1fr) auto 54px; gap: 10px; align-items: start; padding: 12px 0; border-bottom: 1px solid var(--border); transition: opacity .15s ease; }}
    .cluster-row.is-unselected {{ opacity: .55; }}
    .cluster-row:last-child {{ border-bottom: 0; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 50%; }}
    .cluster-name {{ min-width: 0; font-size: 14px; line-height: 1.35; }}
    .cluster-name strong {{ display: block; font-weight: 680; }}
    .cluster-evidence {{ display: block; margin-top: 4px; color: var(--muted); font-size: 11px; line-height: 1.45; }}
    .cluster-description {{ display: block; margin-top: 5px; color: var(--text); font-size: 12px; line-height: 1.5; font-weight: 450; }}
    .cluster-evidence b {{ color: var(--text); font-weight: 620; }}
    .cluster-count {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .persona-select {{ justify-self: end; width: 18px; height: 18px; accent-color: var(--c0); cursor: pointer; }}
    .representative-note {{ margin-top: 7px; color: var(--muted); font-size: 12px; line-height: 1.5; }}
    .representative-records {{ margin-top: 9px; border: 1px solid var(--border); border-radius: 9px; background: var(--surface-2); }}
    .representative-records summary {{ padding: 8px 10px; cursor: pointer; color: var(--text); font-size: 11px; font-weight: 700; }}
    .representative-list {{ display: grid; gap: 7px; padding: 0 8px 8px; }}
    .representative-card {{ min-width: 0; padding: 9px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); overflow-wrap: anywhere; }}
    .representative-card-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .representative-card-head strong {{ min-width: 0; font-size: 12px; }}
    .affinity-score {{ flex: 0 0 auto; color: var(--c0); font-size: 11px; font-weight: 750; font-variant-numeric: tabular-nums; }}
    .representative-row-id {{ margin-top: 2px; color: var(--muted); font: 10px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .representative-fields {{ display: grid; gap: 3px; margin-top: 7px; color: var(--muted); font-size: 10px; line-height: 1.45; }}
    .representative-fields b {{ color: var(--text); font-weight: 650; }}
    .representative-section-label {{ margin-top: 7px; color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }}
    .persona-grounding {{ margin-top: 9px; border: 1px solid color-mix(in srgb, var(--c0) 36%, var(--border)); border-radius: 9px; background: var(--surface-2); }}
    .persona-grounding > summary {{ padding: 8px 10px; cursor: pointer; color: var(--text); font-size: 11px; font-weight: 700; }}
    .grounding-body {{ display: grid; gap: 10px; padding: 0 9px 10px; }}
    .grounding-layer {{ padding: 9px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }}
    .grounding-layer-label {{ color: var(--c0); font-size: 9px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }}
    .grounding-layer p {{ margin-top: 5px; color: var(--muted); font-size: 10px; line-height: 1.45; }}
    .grounding-field {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }}
    .grounding-field:first-of-type {{ border-top: 0; }}
    .grounding-field strong {{ display: block; font-size: 10px; }}
    .source-evidence-list {{ display: grid; gap: 6px; margin-top: 7px; }}
    .source-evidence-item {{ padding: 7px; border-left: 3px solid var(--c0); background: var(--surface-2); overflow-wrap: anywhere; }}
    .source-evidence-meta {{ color: var(--muted); font: 9px/1.35 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .source-evidence-value {{ margin-top: 3px; color: var(--text); font-size: 10px; line-height: 1.4; }}
    .no-source-evidence {{ margin-top: 6px; color: var(--muted); font-size: 10px; font-style: italic; }}
    .pain-point-action {{ display: flex; align-items: center; gap: 12px; margin-top: 16px; }}
    .pain-point-button {{ appearance: none; border: 0; border-radius: 10px; padding: 11px 16px; background: var(--text); color: var(--surface); font: inherit; font-weight: 700; cursor: pointer; }}
    .pain-point-button:disabled {{ cursor: not-allowed; opacity: .45; }}
    .pain-point-note {{ color: var(--muted); font-size: 13px; }}
    .pain-point-result {{ margin-top: 16px; padding: 16px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--border); }}
    .pain-point-result h3 {{ margin: 0 0 10px; font-size: 14px; }}
    .pain-point-result pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; color: var(--text); font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .news-status {{ margin-top: 14px; color: var(--muted); font-size: 14px; }}
    .news-status.is-error {{ color: #dc2626; }}
    .article-results {{ margin-top: 18px; }}
    .article-results-heading {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 10px; }}
    .article-results-heading span {{ color: var(--muted); font-size: 13px; }}
    .analysis-note {{ padding: 12px 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-2); color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .analysis-section {{ margin-top: 22px; }}
    .analysis-section > h3 {{ margin-bottom: 10px; font-size: 16px; }}
    .problem-list {{ display: grid; gap: 12px; }}
    .problem-card {{ padding: 16px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface-2); }}
    .problem-title-row {{ display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }}
    .problem-title-row h4 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
    .confidence-badge {{ flex: 0 0 auto; padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); font-size: 11px; font-weight: 700; }}
    .problem-card p {{ margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.55; }}
    .problem-card p strong {{ color: var(--text); }}
    .evidence-citations {{ display: inline-flex; flex-wrap: wrap; gap: 5px; margin-left: 5px; }}
    .evidence-citations a {{ color: var(--c0); font-weight: 700; text-underline-offset: 2px; }}
    .metric-row {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }}
    .metric-chip {{ padding: 5px 8px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
    .ledger-wrap {{ overflow-x: auto; }}
    .article-analysis-list {{ display: grid; gap: 8px; }}
    .article-analysis-item {{ border: 1px solid var(--border); border-radius: 10px; background: var(--surface-2); }}
    .article-analysis-item summary {{ cursor: pointer; padding: 11px 13px; font-size: 13px; font-weight: 680; line-height: 1.4; }}
    .article-analysis-body {{ padding: 0 13px 13px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .article-analysis-body p {{ margin: 6px 0 0; }}
    .article-analysis-body ul {{ margin: 8px 0 0; padding-left: 20px; }}
    .article-analysis-body a {{ color: var(--c0); }}
    .overlap-title {{ margin-top: 28px; }}
    .overlap-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 30px; row-gap: 9px; margin-top: 13px; }}
    .overlap-row {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; font-size: 13px; color: var(--muted); }}
    .overlap-row strong {{ color: var(--text); font-variant-numeric: tabular-nums; }}
    .table-panel {{ margin-top: 18px; padding: 18px; }}
    .table-wrap {{ overflow-x: auto; margin-top: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ color: var(--muted); font-weight: 600; text-align: left; border-bottom: 1px solid var(--border); padding: 11px 10px; white-space: nowrap; }}
    td {{ border-bottom: 1px solid var(--border); padding: 11px 10px; vertical-align: top; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .score-list {{ display: flex; flex-wrap: wrap; gap: 5px; min-width: 250px; }}
    .score {{ display: inline-flex; gap: 5px; align-items: center; border-radius: 999px; padding: 4px 7px; background: var(--surface-2); font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .score i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
    .muted {{ color: var(--muted); }}
    .tooltip {{ position: fixed; pointer-events: none; opacity: 0; z-index: 20; max-width: 340px; padding: 10px 12px; border-radius: 10px; background: light-dark(#111827, #f8fafc); color: light-dark(#f8fafc, #111827); font-size: 12px; line-height: 1.45; box-shadow: 0 12px 32px rgba(0,0,0,.24); transition: opacity .12s ease; }}
    .footnote {{ margin-top: 18px; color: var(--muted); font-size: 13px; line-height: 1.55; }}
    .trace-panel {{ padding: 20px 22px; margin-bottom: 18px; }}
    .trace-heading {{ display: flex; justify-content: space-between; gap: 14px; align-items: baseline; margin-bottom: 14px; }}
    .trace-heading p {{ color: var(--muted); font-size: 13px; }}
    .trace-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .trace-step {{ padding: 12px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface-2); }}
    .trace-step-top {{ display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }}
    .trace-step strong {{ font-size: 13px; line-height: 1.35; }}
    .trace-badge {{ flex: 0 0 auto; padding: 3px 6px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }}
    .trace-badge.model {{ border-color: #8b5cf6; color: #8b5cf6; }}
    .trace-step p {{ margin-top: 7px; color: var(--muted); font-size: 11px; line-height: 1.45; }}
    .trace-status {{ display: block; margin-top: 8px; color: var(--muted); font-size: 10px; font-weight: 700; }}
    .trace-status.running {{ color: #d97706; }}
    .trace-status.completed {{ color: #059669; }}
    .trace-status.fallback, .trace-status.error {{ color: #dc2626; }}
    @media (max-width: 900px) {{
      .control-grid, .visual-grid {{ grid-template-columns: 1fr; }}
      .trace-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      .chart-panel {{ min-height: 520px; }}
      #scatter {{ height: 440px; }}
    }}
    @media (max-width: 520px) {{
      main {{ width: min(100% - 20px, 1220px); margin-top: 18px; }}
      .stats {{ grid-template-columns: 1fr 1fr; }}
      .stat {{ padding: 14px; }}
      .stat-value {{ font-size: 25px; }}
      .panel-heading {{ display: grid; }}
      .overlap-list {{ grid-template-columns: 1fr; }}
      .dataset-panel, .trace-grid {{ grid-template-columns: 1fr; }}
      .dataset-actions {{ justify-content: flex-start; }}
      .cluster-list-header, .cluster-row {{ grid-template-columns: 24px minmax(0, 1fr) auto 38px; gap: 6px; }}
      .representative-card-head {{ align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Fuzzy semantic clustering explorer</h1>
      <p class="lede">Explore overlapping customer personas from any CSV. Choose the fields that define similarity, adjust fuzzy membership, and turn selected personas into evidence-backed market pain hypotheses.</p>
      <div class="meta">
        <span class="tag" id="rowMeta">50 prospects</span>
        <span class="tag" id="datasetMeta">Demo CSV</span>
        <span class="tag" id="semanticMeta">Clustering fields: Job Title</span>
        <span class="tag" id="kMeta">6 fuzzy prototypes</span>
        <span class="tag" id="embeddingMeta">Embedding vectors ready</span>
      </div>
    </header>

    <section class="panel dataset-panel" aria-label="Customer CSV source">
      <div class="dataset-copy">
        <h2>Customer CSV</h2>
        <p>CSV parsing stays local. During clustering, nonblank values from the selected descriptive fields are sent to OpenAI's Embeddings API; returned vectors are cached locally without source text. The bundled export stays available as the default demo.</p>
      </div>
      <div class="dataset-actions">
        <label class="upload-button">Upload customer CSV<input id="csvUpload" type="file" accept=".csv,text/csv"></label>
        <button class="secondary-button" id="demoButton" type="button">Use demo CSV</button>
      </div>
      <div class="dataset-status" id="datasetStatus" role="status" aria-live="polite">Showing the bundled demo CSV. Uploads stay in this local server session.</div>
    </section>

    <section class="panel trace-panel" aria-label="Workflow and model call trace">
      <div class="trace-heading"><h2>Workflow trace</h2><p>Shows exactly which stages are local, external APIs, or model calls.</p></div>
      <div class="trace-grid" id="traceGrid"></div>
    </section>

    <section class="panel control-panel" aria-label="Clustering controls">
      <div class="control-grid">
        <div>
          <div class="control-copy">
            <h2>Prototype count (K)</h2>
            <p>Sets the rough number of semantic groups before fuzzy overlap is applied.</p>
          </div>
          <div class="slider-row">
            <input id="clusterCount" type="range" min="{min(config.K_RANGE)}" max="{max(config.K_RANGE)}" step="1" value="{metadata['selected_k']}" aria-label="Prototype count">
            <output id="clusterCountValue" for="clusterCount">{metadata['selected_k']}</output>
          </div>
        </div>
        <div>
          <div class="control-copy">
            <h2>Membership cutoff</h2>
            <p>Counts a prospect in every prototype whose independent affinity clears this value.</p>
          </div>
          <div class="slider-row">
            <input id="threshold" type="range" min="0.10" max="0.90" step="0.01" value="{config.MEMBERSHIP_THRESHOLD:.2f}" aria-label="Membership cutoff">
            <output id="thresholdValue" for="threshold">{config.MEMBERSHIP_THRESHOLD:.2f}</output>
          </div>
        </div>
      </div>
      <div class="field-control">
        <div class="control-copy">
          <h2>CSV columns</h2>
          <p>Choose the descriptive fields that define similarity. The first use of a field sends its nonblank values to OpenAI for embedding; later combinations reuse locally cached vectors. Blank values contribute nothing, and identifiers and numeric fields are excluded.</p>
        </div>
        <div class="field-toggle-list" id="fieldToggleList" aria-label="CSV columns used for clustering"></div>
        <div class="field-status" id="fieldStatus" role="status" aria-live="polite"></div>
      </div>
    </section>

    <section class="stats" aria-label="Overlap summary">
      <article class="panel stat"><div class="stat-label">Zero clusters</div><div class="stat-value" id="zeroCount">0</div></article>
      <article class="panel stat"><div class="stat-label">Exactly one</div><div class="stat-value" id="oneCount">0</div></article>
      <article class="panel stat"><div class="stat-label">Two clusters</div><div class="stat-value" id="twoCount">0</div></article>
      <article class="panel stat"><div class="stat-label">Three or more</div><div class="stat-value" id="threeCount">0</div></article>
    </section>

    <section class="visual-grid">
      <aside class="panel side-panel">
        <h2>Persona candidates</h2>
        <p class="representative-note">Highest-affinity examples are ranked independently of the membership cutoff. An example can appear here without being included in the prospect count.</p>
        <div class="cluster-list-header" aria-hidden="true"><span>Key</span><span>Persona candidate</span><span>Prospects</span><span>Select</span></div>
        <div class="cluster-list" id="clusterList"></div>
        <div class="pain-point-action">
          <button class="pain-point-button" id="painPointButton" type="button" aria-describedby="painPointNote">Search market pain-points</button>
          <span class="pain-point-note" id="painPointNote">Uses only the visible text from selected Persona candidates.</span>
        </div>
        <div class="pain-point-result" id="painPointResult" hidden>
          <h3>NewsAPI query request</h3>
          <pre id="painPointJson"></pre>
        </div>
        <div class="news-status" id="newsStatus" role="status" aria-live="polite" hidden></div>
        <section class="article-results" id="articleResults" aria-labelledby="articleResultsHeading" hidden>
          <div class="article-results-heading">
            <h3 id="articleResultsHeading">Current operational pain hypotheses</h3>
            <span id="articleCount"></span>
          </div>
          <div id="painAnalysis"></div>
        </section>
        <h2 class="overlap-title">Largest intersections</h2>
        <div class="overlap-list" id="overlapList" aria-live="polite"></div>
      </aside>
      <article class="panel chart-panel">
        <div class="panel-heading">
          <div><h2>Embedding projection</h2><p>Fill shows strongest affinity. A dark ring indicates overlapping membership at the current cutoff.</p></div>
        </div>
        <svg id="scatter" role="img" aria-label="PCA projection of the selected CSV-column embeddings"></svg>
      </article>
    </section>

    <section class="panel table-panel">
      <div class="panel-heading">
        <div><h2>Prospect memberships</h2><p>Sorted by the number of clusters cleared, then strongest affinity.</p></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th id="displayHeader0">Prospect</th><th id="displayHeader1">Title</th><th id="displayHeader2">Company</th><th>Persona candidates</th></tr></thead>
          <tbody id="prospectRows"></tbody>
        </table>
      </div>
    </section>
    <p class="footnote">Memberships are independently calibrated prototype affinities, not probabilities. Human-readable cluster labels summarize representative examples and do not determine membership. The native Fuzzy C-Means memberships remain available in the project CSV.</p>
  </main>
  <div class="tooltip" id="tooltip" role="tooltip"></div>
  <script>
    const DATA = {data_json};
    const colors = ['var(--c0)','var(--c1)','var(--c2)','var(--c3)','var(--c4)','var(--c5)','var(--c6)'];
    const clusterCountInput = document.getElementById('clusterCount');
    const clusterCountValue = document.getElementById('clusterCountValue');
    const thresholdInput = document.getElementById('threshold');
    const thresholdValue = document.getElementById('thresholdValue');
    const tooltip = document.getElementById('tooltip');
    const painPointButton = document.getElementById('painPointButton');
    const painPointResult = document.getElementById('painPointResult');
    const painPointJson = document.getElementById('painPointJson');
    const newsStatus = document.getElementById('newsStatus');
    const articleResults = document.getElementById('articleResults');
    const articleCount = document.getElementById('articleCount');
    const painAnalysis = document.getElementById('painAnalysis');
    const fieldToggleList = document.getElementById('fieldToggleList');
    const fieldStatus = document.getElementById('fieldStatus');
    const csvUpload = document.getElementById('csvUpload');
    const demoButton = document.getElementById('demoButton');
    const datasetStatus = document.getElementById('datasetStatus');
    const traceGrid = document.getElementById('traceGrid');
    let isSearching = false;
    let isClustering = false;
    let activeDatasetId = DATA.datasetId;
    let activeDatasetName = DATA.datasetName;
    let activeClusterableFields = [...DATA.clusterableFields];
    let activeDisplayFields = [...DATA.displayFields];
    let activeRecords = DATA.records.map(record => ({{ ...record }}));
    let activeModels = DATA.models;
    let activeFields = [...DATA.defaultSelectedFields];
    let activeEmbeddingMetadata = DATA.embeddingMetadata || {{}};
    let clusteringRequestController = null;
    let reclusterTimer = null;
    const selectedPersonas = {{}};
    const clusteringCache = new Map();
    const traceState = {{
      parse: {{ status: 'completed', detail: 'Bundled demo loaded locally.' }},
      cluster: {{ status: 'completed', detail: 'Precomputed demo embeddings and fuzzy memberships.' }},
      personas: {{ status: 'completed', detail: 'Pre-generated demo descriptions; no call on page load.' }},
      query: {{ status: 'idle', detail: 'Runs after persona selection.' }},
      news: {{ status: 'idle', detail: 'Runs after query generation.' }},
      pains: {{ status: 'idle', detail: 'Runs after article deduplication.' }}
    }};
    function clusteringCacheKey(fields) {{ return `${{activeDatasetId}}::${{fields.join('\u001f')}}`; }}
    clusteringCache.set(clusteringCacheKey(activeFields), {{
      models: DATA.models,
      records: activeRecords,
      embeddingBackend: DATA.embeddingBackend,
      embeddingMetadata: DATA.embeddingMetadata || {{}},
      minimumK: DATA.minimumK,
      maximumK: DATA.maximumK,
      defaultK: DATA.defaultK,
      modelTrace: []
    }});

    function renderTrace() {{
      const steps = [
        ['parse', 'Parse CSV + infer fields', 'local', 'Schema, usable text columns, and display columns.'],
        ['cluster', 'Embed selected fields + fuzzy cluster', 'model + local', 'Selected CSV cells become cached OpenAI vectors; selected-field vectors are combined locally before fuzzy clustering.'],
        ['personas', 'Generate persona descriptions', 'model', 'Representative values from every selected column → names, descriptions, field summaries.'],
        ['query', 'Generate NewsAPI query', 'model', 'Only visible selected-persona text → query JSON under 500 characters.'],
        ['news', 'Retrieve + deduplicate news', 'API + model', 'NewsAPI results → cached title-and-description embeddings → up to 25 event candidates.'],
        ['pains', 'Analyze market pain-points', 'model', 'Persona text plus article titles/descriptions → evidence-backed hypotheses.']
      ];
      traceGrid.innerHTML = steps.map(([id, label, kind, description]) => {{
        const state = traceState[id];
        return `<article class="trace-step"><div class="trace-step-top"><strong>${{escapeHtml(label)}}</strong><span class="trace-badge ${{kind.includes('model') ? 'model' : ''}}">${{escapeHtml(kind)}}</span></div><p>${{escapeHtml(description)}}</p><span class="trace-status ${{escapeHtml(state.status)}}">${{escapeHtml(state.status)}} · ${{escapeHtml(state.detail)}}</span></article>`;
      }}).join('');
    }}
    function setTrace(id, status, detail) {{
      traceState[id] = {{ status, detail }};
      renderTrace();
    }}

    function selectionKey() {{ return `${{activeDatasetId}}::${{activeFields.join('\u001f')}}::${{clusterCountInput.value}}`; }}
    function currentModel() {{ return activeModels[String(clusterCountInput.value)]; }}
    function currentSelection() {{
      const key = selectionKey();
      const model = currentModel();
      if (!selectedPersonas[key] || selectedPersonas[key].length !== model.labels.length) {{
        selectedPersonas[key] = model.labels.map(() => true);
      }}
      return selectedPersonas[key];
    }}
    function scoresFor(record) {{ return currentModel().memberships[record.index]; }}
    function membershipsAt(record, threshold) {{
      return scoresFor(record).map((score, cluster) => ({{ score, cluster }})).filter(item => item.score >= threshold);
    }}

    function renderStats(threshold) {{
      const counts = [0,0,0,0];
      activeRecords.forEach(record => {{
        const count = membershipsAt(record, threshold).length;
        counts[count === 0 ? 0 : count === 1 ? 1 : count === 2 ? 2 : 3]++;
      }});
      document.getElementById('zeroCount').textContent = counts[0];
      document.getElementById('oneCount').textContent = counts[1];
      document.getElementById('twoCount').textContent = counts[2];
      document.getElementById('threeCount').textContent = counts[3];
    }}

    function renderClusters(threshold) {{
      const model = currentModel();
      const selected = currentSelection();
      const counts = model.labels.map((_, cluster) => activeRecords.filter(record => scoresFor(record)[cluster] >= threshold).length);
      document.getElementById('clusterList').innerHTML = model.labels.map((label, cluster) => {{
        const evidence = model.evidence && model.evidence[cluster] ? model.evidence[cluster] : {{}};
        const description = model.descriptions && model.descriptions[cluster] ? model.descriptions[cluster] : '';
        const grounding = model.personaGrounding && model.personaGrounding[cluster] ? model.personaGrounding[cluster] : null;
        const examples = model.representativeAccounts && Array.isArray(model.representativeAccounts[cluster])
          ? model.representativeAccounts[cluster]
          : [];
        const evidenceMarkup = activeFields.map(field => `<span><b>${{escapeHtml(field)}}:</b> ${{escapeHtml(evidence[field] || 'No representative values')}}</span>`).join(' · ');
        const sourceItem = item => `<div class="source-evidence-item"><div class="source-evidence-meta">${{escapeHtml(item.rowId)}} · ${{escapeHtml(item.field)}} · ${{escapeHtml(item.evidenceId)}}</div><div class="source-evidence-value">${{escapeHtml(item.value)}}</div></div>`;
        const groundingMarkup = grounding ? `
          <details class="persona-grounding">
            <summary>Inspect persona evidence (${{Array.isArray(grounding.sourceEvidence) ? grounding.sourceEvidence.length : 0}} source values)</summary>
            <div class="grounding-body">
              <section class="grounding-layer">
                <div class="grounding-layer-label">Generated interpretation</div>
                <p>The persona prose below is model-generated or deterministic interpretation. Each populated field is linked to server-verified CSV evidence.</p>
                <div class="grounding-field"><strong>Description support</strong><div class="source-evidence-list">${{(grounding.descriptionEvidence || []).map(sourceItem).join('')}}</div>${{!(grounding.descriptionEvidence || []).length ? '<div class="no-source-evidence">No source evidence was available for the description.</div>' : ''}}</div>
                ${{(grounding.fieldSummaries || []).map(summary => `<div class="grounding-field"><strong>${{escapeHtml(summary.field)}}</strong><p>${{escapeHtml(summary.summary)}}</p><div class="source-evidence-list">${{(summary.evidence || []).map(sourceItem).join('')}}</div>${{summary.evidenceStatus === 'no_evidence' ? '<div class="no-source-evidence">No populated representative value was supplied for this field.</div>' : ''}}</div>`).join('')}}
              </section>
              <section class="grounding-layer">
                <div class="grounding-layer-label">Exact CSV source evidence</div>
                <p>Values are copied and bounded by the server, not written by the model. Row IDs remain stable for this dataset.</p>
                <div class="source-evidence-list">${{(grounding.sourceEvidence || []).map(sourceItem).join('')}}</div>
              </section>
            </div>
          </details>` : '';
        const examplesMarkup = examples.length ? `
          <details class="representative-records">
            <summary>Highest-affinity examples (${{examples.length}})</summary>
            <div class="representative-list">${{examples.map(example => {{
              const displayEntries = Object.entries(example.displayValues || {{}});
              const fieldEntries = Object.entries(example.fieldValues || {{}});
              const identity = (displayEntries.find(([, value]) => value && value !== 'Not provided') || [null, example.rowId])[1];
              return `<article class="representative-card">
                <div class="representative-card-head"><strong>${{escapeHtml(identity)}}</strong><span class="affinity-score">Affinity ${{Number(example.score).toFixed(3)}}</span></div>
                <div class="representative-row-id">${{escapeHtml(example.rowId)}}</div>
                <div class="representative-section-label">Display identity</div>
                <div class="representative-fields">${{displayEntries.map(([field, value]) => `<span><b>${{escapeHtml(field)}}:</b> ${{escapeHtml(value)}}</span>`).join('')}}</div>
                <div class="representative-section-label">Selected clustering fields</div>
                <div class="representative-fields">${{fieldEntries.map(([field, value]) => `<span><b>${{escapeHtml(field)}}:</b> ${{escapeHtml(value)}}</span>`).join('')}}</div>
              </article>`;
            }}).join('')}}</div>
          </details>` : '';
        return `
        <div class="cluster-row ${{selected[cluster] ? '' : 'is-unselected'}}">
          <span class="swatch" style="background:${{colors[cluster]}}"></span>
          <span class="cluster-name"><strong>C${{cluster}} · ${{escapeHtml(label)}}</strong><small class="cluster-description">${{escapeHtml(description)}}</small><small class="cluster-evidence"><b>Generated field interpretation:</b> ${{evidenceMarkup}}</small>${{groundingMarkup}}${{examplesMarkup}}</span>
          <span class="cluster-count">${{counts[cluster]}}</span>
          <input class="persona-select" type="checkbox" data-cluster="${{cluster}}" ${{selected[cluster] ? 'checked' : ''}} aria-label="Select ${{escapeHtml(label)}} for market pain-point search">
        </div>`;
      }}).join('');
      document.querySelectorAll('.persona-select').forEach(input => {{
        input.addEventListener('change', () => {{
          const cluster = Number(input.dataset.cluster);
          currentSelection()[cluster] = input.checked;
          input.closest('.cluster-row').classList.toggle('is-unselected', !input.checked);
          invalidatePainPointRequest();
          updatePainPointButton();
        }});
      }});

      const pairs = [];
      for (let a = 0; a < model.labels.length; a++) {{
        for (let b = a + 1; b < model.labels.length; b++) {{
          const shared = activeRecords.filter(record => scoresFor(record)[a] >= threshold && scoresFor(record)[b] >= threshold).length;
          if (shared) pairs.push({{ a, b, shared }});
        }}
      }}
      pairs.sort((x, y) => y.shared - x.shared);
      document.getElementById('overlapList').innerHTML = pairs.length
        ? pairs.slice(0, 6).map(pair => `<div class="overlap-row"><span>C${{pair.a}} + C${{pair.b}}</span><strong>${{pair.shared}} shared</strong></div>`).join('')
        : '<div class="muted">No intersections at this cutoff.</div>';
    }}

    function renderTable(threshold) {{
      const rows = activeRecords.map(record => {{
        const active = membershipsAt(record, threshold).sort((a,b) => b.score - a.score);
        return {{ ...record, active, max: Math.max(...scoresFor(record)) }};
      }}).sort((a,b) => b.active.length - a.active.length || b.max - a.max);
      document.getElementById('prospectRows').innerHTML = rows.map(record => `
        <tr>
          <td><strong>${{escapeHtml(record.name)}}</strong><br><span class="muted">${{record.rowId}}</span></td>
          <td>${{escapeHtml(record.title)}}</td>
          <td>${{escapeHtml(record.company)}}</td>
          <td><div class="score-list">${{record.active.length ? record.active.map(item => `<span class="score"><i style="background:${{colors[item.cluster]}}"></i>C${{item.cluster}} ${{item.score.toFixed(2)}}</span>`).join('') : '<span class="muted">No cluster</span>'}}</div></td>
        </tr>`).join('');
    }}

    function renderScatter(threshold) {{
      const svg = document.getElementById('scatter');
      const rect = svg.getBoundingClientRect();
      const width = Math.max(320, rect.width || 760);
      const height = Math.max(360, rect.height || 540);
      const pad = {{ left: 54, right: 24, top: 22, bottom: 46 }};
      const xs = activeRecords.map(d => d.x), ys = activeRecords.map(d => d.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
      const sx = value => pad.left + (value - xMin) / (xMax - xMin || 1) * (width - pad.left - pad.right);
      const sy = value => pad.top + (yMax - value) / (yMax - yMin || 1) * (height - pad.top - pad.bottom);
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      const frameColor = getComputedStyle(document.documentElement).getPropertyValue('--border').trim();
      const muted = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
      const surface2 = getComputedStyle(document.documentElement).getPropertyValue('--surface-2').trim();
      let markup = `<rect x="${{pad.left}}" y="${{pad.top}}" width="${{width-pad.left-pad.right}}" height="${{height-pad.top-pad.bottom}}" rx="10" fill="${{surface2}}" stroke="${{frameColor}}"/>`;
      markup += `<text x="${{width/2}}" y="${{height-10}}" text-anchor="middle" fill="${{muted}}" font-size="12">PCA component 1</text>`;
      markup += `<text x="14" y="${{height/2}}" text-anchor="middle" fill="${{muted}}" font-size="12" transform="rotate(-90 14 ${{height/2}})">PCA component 2</text>`;
      activeRecords.forEach((record, index) => {{
        const active = membershipsAt(record, threshold);
        const scores = scoresFor(record);
        const primary = scores.indexOf(Math.max(...scores));
        const fill = active.length ? colors[primary] : muted;
        const stroke = active.length >= 2 ? 'var(--text)' : 'var(--surface)';
        const radius = active.length >= 3 ? 7.5 : 6.5;
        markup += `<circle data-index="${{index}}" cx="${{sx(record.x)}}" cy="${{sy(record.y)}}" r="${{radius}}" fill="${{fill}}" fill-opacity=".82" stroke="${{stroke}}" stroke-width="${{active.length >= 2 ? 2.4 : 1.2}}"/>`;
      }});
      svg.innerHTML = markup;
      svg.querySelectorAll('circle[data-index]').forEach(circle => {{
        circle.addEventListener('pointerenter', event => showTooltip(event, activeRecords[Number(circle.dataset.index)], threshold));
        circle.addEventListener('pointermove', moveTooltip);
        circle.addEventListener('pointerleave', hideTooltip);
      }});
    }}

    function showTooltip(event, record, threshold) {{
      const active = membershipsAt(record, threshold).sort((a,b) => b.score - a.score);
      tooltip.innerHTML = `<strong>${{escapeHtml(record.name)}}</strong><br>${{escapeHtml(record.title)}} · ${{escapeHtml(record.company)}}<br>${{active.length ? active.map(item => `C${{item.cluster}} ${{item.score.toFixed(2)}}`).join(' · ') : 'No cluster at this cutoff'}}`;
      tooltip.style.opacity = '1';
      moveTooltip(event);
    }}
    function moveTooltip(event) {{
      tooltip.style.left = `${{Math.min(window.innerWidth - 360, event.clientX + 14)}}px`;
      tooltip.style.top = `${{Math.max(12, event.clientY - 16)}}px`;
    }}
    function hideTooltip() {{ tooltip.style.opacity = '0'; }}
    function selectedPersonaDescriptions() {{
      const model = currentModel();
      const selected = currentSelection();
      return model.labels.flatMap((label, cluster) => {{
        if (!selected[cluster]) return [];
        const evidence = model.evidence && model.evidence[cluster] ? model.evidence[cluster] : {{}};
        const description = model.descriptions && model.descriptions[cluster] ? model.descriptions[cluster] : '';
        const visibleEvidence = activeFields.map(field => `${{field}}: ${{evidence[field] || 'No representative values'}}`);
        return [`${{label}}. ${{description}} ${{visibleEvidence.join('. ')}}.`.replace(/\\s+/g, ' ').trim()];
      }});
    }}
    function buildPainPointRequest() {{
      const personaDescriptions = selectedPersonaDescriptions();
      if (!personaDescriptions.length) return;
      return {{ persona_descriptions: personaDescriptions }};
    }}
    function invalidatePainPointRequest() {{
      painPointResult.hidden = true;
      painPointJson.textContent = '';
      newsStatus.hidden = true;
      newsStatus.textContent = '';
      newsStatus.classList.remove('is-error');
      articleResults.hidden = true;
      painAnalysis.innerHTML = '';
      articleCount.textContent = '';
    }}
    function updatePainPointButton() {{
      const selected = currentSelection();
      const count = selected.filter(Boolean).length;
      painPointButton.disabled = count === 0 || isSearching || isClustering;
      painPointButton.textContent = isSearching ? 'Searching…' : 'Search market pain-points';
      document.getElementById('painPointNote').textContent = count
        ? `Uses only the visible text from ${{count}} selected persona candidate${{count === 1 ? '' : 's'}} to find up to 25 distinct events from the last 30 days.`
        : 'Select at least one Persona candidate.';
    }}
    function citationLinks(ids, articles) {{
      const articleMap = new Map(articles.map((article, index) => [`A${{index + 1}}`, article]));
      return `<span class="evidence-citations">${{(ids || []).map(id => {{
        const article = articleMap.get(id) || {{}};
        const url = safeArticleUrl(article.url);
        const title = article.title || id;
        return url === '#'
          ? `<span>${{escapeHtml(id)}}</span>`
          : `<a href="${{escapeHtml(url)}}" target="_blank" rel="noopener noreferrer" title="${{escapeHtml(title)}}">${{escapeHtml(id)}}</a>`;
      }}).join('')}}</span>`;
    }}
    function renderPainAnalysis(payload) {{
      const articles = Array.isArray(payload.articles) ? payload.articles.slice(0, 25) : [];
      const duplicates = Number(payload.duplicatesRemoved || 0);
      const analysis = payload.analysis || {{}};
      const topProblems = Array.isArray(analysis.top_problems) ? analysis.top_problems : [];
      const ledger = Array.isArray(analysis.evidence_ledger) ? analysis.evidence_ledger : [];
      const perArticle = Array.isArray(analysis.article_analysis) ? analysis.article_analysis : [];
      articleCount.textContent = `${{articles.length}} distinct event${{articles.length === 1 ? '' : 's'}} reviewed · ${{duplicates}} duplicate${{duplicates === 1 ? '' : 's'}} skipped · ${{Number(payload.lookbackDays || 30)}}-day window`;

      const problemMarkup = topProblems.length ? topProblems.map(problem => `
        <article class="problem-card">
          <div class="problem-title-row">
            <h4>${{escapeHtml(problem.rank)}}. ${{escapeHtml(problem.problem_name)}}</h4>
            <span class="confidence-badge">${{escapeHtml(problem.confidence_label)}}</span>
          </div>
          <p><strong>Why it may be emerging:</strong> ${{escapeHtml(problem.why_emerging)}} ${{citationLinks(problem.supporting_article_ids, articles)}}</p>
          <p><strong>Persona impact:</strong> ${{escapeHtml(problem.persona_impact)}} ${{citationLinks(problem.supporting_article_ids, articles)}}</p>
          <div class="metric-row">
            <span class="metric-chip">Weighted support ${{Number(problem.weighted_support || 0).toFixed(2)}}</span>
            <span class="metric-chip">${{Number(problem.distinct_supporting_events || 0)}} distinct events</span>
            <span class="metric-chip">${{Number(problem.supporting_articles || 0)}} articles</span>
            <span class="metric-chip">Avg confidence ${{Number(problem.average_confidence || 0).toFixed(2)}}</span>
          </div>
        </article>`).join('') : '<p class="muted">The supplied articles did not support a strong operational pain hypothesis.</p>';

      const ledgerMarkup = ledger.length ? `
        <div class="ledger-wrap"><table>
          <thead><tr><th>Problem</th><th>Supporting articles</th><th>Distinct events</th><th>Avg confidence</th><th>Weighted support</th></tr></thead>
          <tbody>${{ledger.map(row => `<tr>
            <td>${{escapeHtml(row.problem)}}</td>
            <td>${{citationLinks(row.supporting_article_ids, articles)}}</td>
            <td>${{Number(row.distinct_events || 0)}}</td>
            <td>${{Number(row.average_confidence || 0).toFixed(2)}}</td>
            <td>${{Number(row.weighted_support || 0).toFixed(2)}}</td>
          </tr>`).join('')}}</tbody>
        </table></div>` : '<p class="muted">No evidence ledger entries.</p>';

      const articleMarkup = perArticle.length ? perArticle.map((item, index) => {{
        const article = articles[index] || {{}};
        const id = item.article_id || `A${{index + 1}}`;
        const title = article.title || item.title || 'Untitled article';
        const contributions = Array.isArray(item.problems) ? item.problems : [];
        const body = item.no_supported_inference || !contributions.length
          ? '<p>No sufficiently supported operational inference.</p>'
          : `<ul>${{contributions.map(problem => `<li><strong>${{escapeHtml(problem.problem)}} — ${{Number(problem.confidence || 0).toFixed(2)}}</strong><br>${{escapeHtml(problem.inference)}}</li>`).join('')}}</ul>`;
        return `<details class="article-analysis-item">
          <summary>${{citationLinks([id], articles)}} ${{escapeHtml(title)}}</summary>
          <div class="article-analysis-body"><p><strong>Event:</strong> ${{escapeHtml(item.event || 'No material event identified')}}</p>${{body}}</div>
        </details>`;
      }}).join('') : '<p class="muted">No article-by-article analysis returned.</p>';

      painAnalysis.innerHTML = `
        <div class="analysis-note">${{escapeHtml(analysis.evidence_strength_note || 'Evidence strength was not summarized.')}}${{analysis.model ? ` · Analyzed with ${{escapeHtml(analysis.model)}}` : ''}}</div>
        <section class="analysis-section"><h3>Top Current Operational Problems</h3><div class="problem-list">${{problemMarkup}}</div></section>
        <section class="analysis-section"><h3>Evidence Ledger</h3>${{ledgerMarkup}}</section>
        <section class="analysis-section"><h3>Article-by-Article Analysis</h3><div class="article-analysis-list">${{articleMarkup}}</div></section>`;
      articleResults.hidden = false;
    }}
    async function searchMarketPainPoints() {{
      const request = buildPainPointRequest();
      if (!request) return;
      if (window.location.protocol === 'file:') {{
        newsStatus.textContent = 'Open this dashboard through the local server to search NewsAPI.';
        newsStatus.classList.add('is-error');
        newsStatus.hidden = false;
        return;
      }}
      isSearching = true;
      updatePainPointButton();
      setTrace('query', 'running', 'Creating a query from only the selected visible persona text.');
      setTrace('news', 'idle', 'Waiting for query generation.');
      setTrace('pains', 'idle', 'Waiting for deduplicated articles.');
      newsStatus.textContent = 'Generating the NewsAPI query, searching recent news, and analyzing operational pain hypotheses…';
      newsStatus.classList.remove('is-error');
      newsStatus.hidden = false;
      articleResults.hidden = true;
      try {{
        const response = await fetch('/api/news', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(request)
        }});
        const payload = await response.json();
        if (!response.ok || payload.status === 'error') {{
          if (payload.query) {{
            painPointJson.textContent = JSON.stringify({{ ...payload.query, persona_descriptions: request.persona_descriptions }}, null, 2);
            painPointResult.hidden = false;
          }}
          if (payload.stage === 'pain-point-analysis') {{
            const queryTrace = Array.isArray(payload.modelTrace) ? payload.modelTrace.find(item => item.id === 'news-query-generation') : null;
            setTrace('query', 'completed', `${{queryTrace && queryTrace.model ? queryTrace.model : 'OpenAI'}} · visible persona text only.`);
            setTrace('news', 'completed', 'News retrieval and event-level semantic deduplication completed.');
            setTrace('pains', 'error', payload.message || 'Pain-point analysis failed.');
          }}
          throw new Error(payload.message || `NewsAPI request failed (${{response.status}})`);
        }}
        const queryDisplay = {{ ...payload.query, persona_descriptions: request.persona_descriptions }};
        painPointJson.textContent = JSON.stringify(queryDisplay, null, 2);
        painPointResult.hidden = false;
        const traces = Array.isArray(payload.modelTrace) ? payload.modelTrace : [];
        const queryTrace = traces.find(item => item.id === 'news-query-generation');
        const painTrace = traces.find(item => item.id === 'pain-point-analysis');
        setTrace('query', 'completed', `${{queryTrace && queryTrace.model ? queryTrace.model : 'OpenAI'}} · visible persona text only.`);
        const dedupe = payload.deduplication || {{}};
        setTrace('news', 'completed', `${{Number(payload.returnedResults || 0)}} distinct events retained; ${{Number(payload.duplicatesRemoved || 0)}} duplicates removed · ${{dedupe.backend || 'semantic model'}}${{dedupe.fallback_used ? ' fallback' : ''}}.`);
        setTrace('pains', 'completed', `${{painTrace && painTrace.model ? painTrace.model : 'OpenAI'}} · structured evidence synthesis.`);
        const returned = Number(payload.returnedResults || 0);
        const problemCount = payload.analysis && Array.isArray(payload.analysis.top_problems) ? payload.analysis.top_problems.length : 0;
        newsStatus.textContent = `Analysis complete: ${{problemCount}} operational pain hypothes${{problemCount === 1 ? 'is' : 'es'}} from ${{returned}} distinct news event${{returned === 1 ? '' : 's'}}.`;
        renderPainAnalysis(payload);
      }} catch (error) {{
        newsStatus.textContent = error.message || 'The NewsAPI request failed.';
        newsStatus.classList.add('is-error');
        if (traceState.pains.status === 'error') {{ /* Stage-specific error already shown. */ }}
        else if (traceState.query.status === 'running') setTrace('query', 'error', error.message || 'Query generation failed.');
        else if (traceState.news.status !== 'completed') setTrace('news', 'error', error.message || 'News retrieval failed.');
        else setTrace('pains', 'error', error.message || 'Pain-point analysis failed.');
      }} finally {{
        isSearching = false;
        updatePainPointButton();
      }}
    }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>'"]/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}})[char]);
    }}
    function safeArticleUrl(value) {{
      try {{
        const url = new URL(String(value));
        return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : '#';
      }} catch {{
        return '#';
      }}
    }}
    function selectedFieldsFromControls() {{
      return [...fieldToggleList.querySelectorAll('input:checked')].map(input => input.value);
    }}
    function syncFieldControls() {{
      fieldToggleList.querySelectorAll('input').forEach(input => {{
        input.checked = activeFields.includes(input.value);
      }});
    }}
    function embeddingLabel(backend, metadata) {{
      if (backend === 'openai') return `OpenAI ${{metadata && metadata.model ? metadata.model : 'embeddings'}}`;
      if (backend === 'tfidf-lsa') return 'offline TF-IDF + LSA';
      return backend || 'embeddings';
    }}
    function applyClusteringState(fields, state) {{
      activeFields = [...fields];
      activeModels = state.models;
      activeRecords = state.records;
      activeEmbeddingMetadata = state.embeddingMetadata || {{}};
      const minimumK = Number(state.minimumK || Math.min(...Object.keys(activeModels).map(Number)));
      const maximumK = Number(state.maximumK || Math.max(...Object.keys(activeModels).map(Number)));
      clusterCountInput.min = String(minimumK);
      clusterCountInput.max = String(maximumK);
      if (!activeModels[String(clusterCountInput.value)]) {{
        clusterCountInput.value = String(state.defaultK || Math.min(6, maximumK));
      }}
      document.getElementById('semanticMeta').textContent = `Clustering fields: ${{activeFields.join(' + ')}}`;
      const embeddingName = embeddingLabel(state.embeddingBackend, activeEmbeddingMetadata);
      fieldStatus.textContent = `Using ${{activeFields.length}} CSV column${{activeFields.length === 1 ? '' : 's'}} · ${{embeddingName}}`;
      fieldStatus.classList.remove('is-error');
      const embeddingTrace = Array.isArray(state.modelTrace) ? state.modelTrace.find(item => item.id === 'embedding-generation') : null;
      const personaTrace = Array.isArray(state.modelTrace) ? state.modelTrace.find(item => item.id === 'persona-description-generation') : null;
      setTrace(
        'cluster',
        embeddingTrace ? embeddingTrace.status : 'completed',
        embeddingTrace && embeddingTrace.detail
          ? embeddingTrace.detail
          : `${{activeRecords.length}} rows · ${{embeddingName}}.`
      );
      setTrace(
        'personas',
        personaTrace ? personaTrace.status : 'completed',
        personaTrace && (personaTrace.detail || personaTrace.model)
          ? (personaTrace.detail || `${{personaTrace.model}} · every selected field accounted for.`)
          : activeDatasetId === 'demo'
            ? 'Pre-generated demo descriptions; no call on page load.'
            : 'Descriptions ready.'
      );
      document.getElementById('rowMeta').textContent = `${{activeRecords.length}} prospects`;
      document.getElementById('datasetMeta').textContent = activeDatasetId === 'demo' ? 'Demo CSV' : activeDatasetName;
      document.getElementById('embeddingMeta').textContent = embeddingName;
      document.getElementById('displayHeader0').textContent = activeDisplayFields[0] || 'Record';
      document.getElementById('displayHeader1').textContent = activeDisplayFields[1] || 'Detail 1';
      document.getElementById('displayHeader2').textContent = activeDisplayFields[2] || 'Detail 2';
      invalidatePainPointRequest();
      update();
    }}
    async function recluster(fields) {{
      const key = clusteringCacheKey(fields);
      const cached = clusteringCache.get(key);
      if (cached) {{
        if (clusteringRequestController) clusteringRequestController.abort();
        isClustering = false;
        applyClusteringState(fields, cached);
        return;
      }}
      if (window.location.protocol === 'file:') {{
        fieldStatus.textContent = 'Open this dashboard through the local server to change clustering columns.';
        fieldStatus.classList.add('is-error');
        syncFieldControls();
        return;
      }}
      if (clusteringRequestController) clusteringRequestController.abort();
      const controller = new AbortController();
      clusteringRequestController = controller;
      isClustering = true;
      fieldStatus.textContent = `Reclustering with ${{fields.join(' + ')}}…`;
      fieldStatus.classList.remove('is-error');
      setTrace('cluster', 'running', `Loading cached vectors and embedding new values for ${{fields.length}} selected column${{fields.length === 1 ? '' : 's'}}.`);
      setTrace('personas', 'running', 'Runs immediately after fuzzy memberships are ready.');
      updatePainPointButton();
      try {{
        const response = await fetch('/api/clusters', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ dataset_id: activeDatasetId, fields }}),
          signal: controller.signal
        }});
        const payload = await response.json();
        if (!response.ok || payload.status === 'error') {{
          throw new Error(payload.message || `Clustering request failed (${{response.status}})`);
        }}
        const state = {{
          models: payload.models,
          records: payload.records,
          embeddingBackend: payload.embeddingBackend,
          embeddingMetadata: payload.embeddingMetadata || {{}},
          minimumK: payload.minimumK,
          maximumK: payload.maximumK,
          defaultK: payload.defaultK,
          modelTrace: payload.modelTrace
        }};
        clusteringCache.set(key, state);
        applyClusteringState(payload.selectedFields, state);
      }} catch (error) {{
        if (error.name === 'AbortError') return;
        fieldStatus.textContent = error.message || 'Could not recompute the clustering.';
        fieldStatus.classList.add('is-error');
        setTrace('cluster', 'error', error.message || 'Could not recompute the clustering.');
        setTrace('personas', 'idle', 'Waiting for a successful clustering run.');
        syncFieldControls();
      }} finally {{
        if (clusteringRequestController === controller) {{
          clusteringRequestController = null;
          isClustering = false;
          updatePainPointButton();
        }}
      }}
    }}
    function scheduleRecluster(fields) {{
      window.clearTimeout(reclusterTimer);
      reclusterTimer = window.setTimeout(() => recluster(fields), 180);
    }}
    function renderFieldToggles() {{
      fieldToggleList.innerHTML = activeClusterableFields.map(field => `
        <label class="field-toggle">
          <input type="checkbox" value="${{escapeHtml(field)}}" ${{activeFields.includes(field) ? 'checked' : ''}}>
          <span>${{escapeHtml(field)}}</span>
        </label>`).join('');
      fieldToggleList.querySelectorAll('input').forEach(input => {{
        input.addEventListener('change', () => {{
          const fields = selectedFieldsFromControls();
          if (!fields.length) {{
            input.checked = true;
            fieldStatus.textContent = 'Keep at least one CSV column selected.';
            fieldStatus.classList.add('is-error');
            return;
          }}
          scheduleRecluster(fields);
        }});
      }});
      fieldStatus.textContent = `Ready to cluster with ${{activeFields.length}} selected CSV column${{activeFields.length === 1 ? '' : 's'}}.`;
    }}

    async function uploadCustomerCsv(file) {{
      if (!file) return;
      if (window.location.protocol === 'file:') {{
        datasetStatus.textContent = 'Open this dashboard through the local server to upload a CSV.';
        datasetStatus.classList.add('is-error');
        return;
      }}
      datasetStatus.textContent = `Reading ${{file.name}}…`;
      datasetStatus.classList.remove('is-error');
      setTrace('parse', 'running', `Reading ${{file.name}} in the local dashboard server.`);
      try {{
        const csvText = await file.text();
        const response = await fetch('/api/datasets', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ filename: file.name, csvText }})
        }});
        const profile = await response.json();
        if (!response.ok || profile.status === 'error') throw new Error(profile.message || 'Could not load the CSV.');
        activeDatasetId = profile.datasetId;
        activeDatasetName = profile.datasetName;
        activeClusterableFields = [...profile.clusterableFields];
        activeDisplayFields = [...profile.displayFields];
        activeFields = [...profile.defaultSelectedFields];
        activeRecords = profile.records.map(record => ({{ ...record }}));
        Object.keys(selectedPersonas).forEach(key => delete selectedPersonas[key]);
        renderFieldToggles();
        document.getElementById('rowMeta').textContent = `${{profile.rowCount}} prospects`;
        document.getElementById('datasetMeta').textContent = profile.datasetName;
        datasetStatus.textContent = `${{profile.datasetName}} parsed locally · ${{profile.rowCount}} rows · ${{profile.clusterableFields.length}} descriptive columns found. Selected clustering values are sent to OpenAI for embedding.`;
        setTrace('parse', 'completed', `${{profile.rowCount}} rows; ${{profile.clusterableFields.length}} descriptive columns inferred locally.`);
        setTrace('query', 'idle', 'Runs after persona selection.');
        setTrace('news', 'idle', 'Runs after query generation.');
        setTrace('pains', 'idle', 'Runs after article deduplication.');
        await recluster(activeFields);
      }} catch (error) {{
        datasetStatus.textContent = error.message || 'Could not load the CSV.';
        datasetStatus.classList.add('is-error');
        setTrace('parse', 'error', error.message || 'Could not parse the CSV.');
      }} finally {{
        csvUpload.value = '';
      }}
    }}

    function useDemoDataset() {{
      if (clusteringRequestController) clusteringRequestController.abort();
      activeDatasetId = DATA.datasetId;
      activeDatasetName = DATA.datasetName;
      activeClusterableFields = [...DATA.clusterableFields];
      activeDisplayFields = [...DATA.displayFields];
      activeFields = [...DATA.defaultSelectedFields];
      activeRecords = DATA.records.map(record => ({{ ...record }}));
      activeModels = DATA.models;
      clusterCountInput.min = String(DATA.minimumK);
      clusterCountInput.max = String(DATA.maximumK);
      clusterCountInput.value = String(DATA.defaultK);
      Object.keys(selectedPersonas).forEach(key => delete selectedPersonas[key]);
      renderFieldToggles();
      datasetStatus.textContent = 'Showing the bundled demo CSV. Uploads stay in this local server session.';
      datasetStatus.classList.remove('is-error');
      setTrace('parse', 'completed', 'Bundled demo loaded locally.');
      setTrace('cluster', 'completed', 'Precomputed demo embeddings and fuzzy memberships.');
      setTrace('personas', 'completed', 'Pre-generated demo descriptions; no call on page load.');
      setTrace('query', 'idle', 'Runs after persona selection.');
      setTrace('news', 'idle', 'Runs after query generation.');
      setTrace('pains', 'idle', 'Runs after article deduplication.');
      applyClusteringState(activeFields, clusteringCache.get(clusteringCacheKey(activeFields)));
    }}
    function update() {{
      const threshold = Number(thresholdInput.value);
      const k = Number(clusterCountInput.value);
      thresholdValue.textContent = threshold.toFixed(2);
      clusterCountValue.textContent = k;
      document.getElementById('kMeta').textContent = `${{k}} fuzzy prototypes`;
      document.getElementById('semanticMeta').textContent = `Clustering fields: ${{activeFields.join(' + ')}}`;
      renderStats(threshold);
      renderClusters(threshold);
      renderTable(threshold);
      renderScatter(threshold);
      updatePainPointButton();
    }}
    thresholdInput.addEventListener('input', () => {{ invalidatePainPointRequest(); update(); }});
    clusterCountInput.addEventListener('input', () => {{ invalidatePainPointRequest(); update(); }});
    painPointButton.addEventListener('click', searchMarketPainPoints);
    csvUpload.addEventListener('change', () => uploadCustomerCsv(csvUpload.files && csvUpload.files[0]));
    demoButton.addEventListener('click', useDemoDataset);
    new ResizeObserver(() => renderScatter(Number(thresholdInput.value))).observe(document.getElementById('scatter'));
    document.getElementById('embeddingMeta').textContent = embeddingLabel(DATA.embeddingBackend, DATA.embeddingMetadata || {{}});
    traceState.cluster.detail = `${{embeddingLabel(DATA.embeddingBackend, DATA.embeddingMetadata || {{}})}} · precomputed demo vectors.`;
    renderFieldToggles();
    renderTrace();
    update();
  </script>
</body>
</html>'''
    (config.OUTPUT_DIR / "clustering_explorer.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    build_report()
