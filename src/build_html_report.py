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
    title = "DemandMap — Customer personas and market needs"
    document = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #d7d0bd;
      --surface: #f4e8c8;
      --surface-2: #fbf3dc;
      --surface-3: #e8dcc0;
      --text: #27281f;
      --muted: #6f6b5d;
      --border: #aea58f;
      --line: rgba(60, 60, 46, .18);
      --shadow: 10px 14px 0 rgba(48, 42, 28, .09), 0 22px 44px rgba(48, 42, 28, .12);
      --brand: #596242;
      --brand-strong: #4e5737;
      --brand-soft: #d9ddc5;
      --lake: #526f9f;
      --cream: #f4e8c8;
      --olive: #596242;
      --vermilion: #f0440d;
      --success: #596242;
      --warning: #bb6a18;
      --fill-cool: linear-gradient(115deg, #667dad, #526f9f 58%, #435c87);
      --c0: #526f9f;
      --c1: #f0440d;
      --c2: #596242;
      --c3: #b88d32;
      --c4: #cb6b45;
      --c5: #738c8d;
      --c6: #765366;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background-color: var(--bg); background-image: radial-gradient(rgba(59,55,42,.18) .7px, transparent .8px), radial-gradient(rgba(255,255,255,.28) .7px, transparent .8px); background-position: 0 0, 3px 3px; background-size: 6px 6px; color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }}
    main {{ width: min(1440px, 100%); margin: 0 auto; padding: 0 14px 36px; }}
    header {{ position: relative; isolation: isolate; overflow: hidden; min-height: 86px; display: flex; align-items: center; gap: 18px; margin: 0 -14px 18px; padding: 14px 24px; border: 0; border-radius: 0 0 76px 18px; background: var(--lake); color: #fff8e6; box-shadow: 0 14px 0 rgba(39,40,31,.10); }}
    header::after {{ content: ""; position: absolute; z-index: -1; width: 280px; height: 130px; right: 17%; top: -72px; border-radius: 42% 58% 64% 36% / 62% 40% 60% 38%; background: rgba(244,232,200,.22); transform: rotate(-7deg); }}
    header > * {{ position: relative; z-index: 1; }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font: 850 23px/1.1 Inter, ui-sans-serif, system-ui, sans-serif; letter-spacing: -.045em; white-space: nowrap; }}
    h2 {{ font-size: 16px; letter-spacing: -.01em; }}
    .brand-lockup {{ min-width: 210px; padding-right: 18px; border-right: 1px solid rgba(255,255,255,.9); }}
    .lede {{ margin-top: 2px; color: rgba(255,248,230,.76); font-size: 12px; }}
    .product-name {{ display: none; }}
    .journey {{ display: flex; align-items: center; gap: 4px; }}
    .journey-step {{ padding: 8px 12px; border: 0; border-radius: 22px 8px 24px 10px; background: rgba(244,232,200,.92); color: var(--text); font-size: 12px; white-space: nowrap; box-shadow: 3px 4px 0 rgba(25,28,22,.12); }}
    .journey-step:nth-child(even) {{ border-radius: 9px 24px 10px 22px; transform: translateY(3px); }}
    .journey-step span {{ color: var(--vermilion); font-weight: 900; }}
    .journey-step strong {{ margin-left: 4px; font-weight: 600; }}
    .meta {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; margin-left: auto; }}
    .tag {{ padding: 6px 9px; border: 0; border-radius: 18px 7px 18px 8px; background: rgba(244,232,200,.9); color: #45483a; font-size: 11px; }}
    .app-layout {{ display: grid; grid-template-columns: 330px minmax(0, 1fr); gap: 18px; align-items: start; }}
    .setup-rail, .workspace {{ display: grid; gap: 12px; min-width: 0; }}
    .panel {{ background: var(--surface); border: 0; border-radius: 46px 18px 58px 22px; box-shadow: var(--shadow); }}
    .control-panel {{ padding: 16px; margin: 0; }}
    .step-kicker {{ margin-bottom: 4px; color: var(--brand-strong); font-size: 10px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }}
    .section-intro {{ display: grid; gap: 4px; margin-bottom: 12px; }}
    .section-intro p {{ color: var(--muted); font-size: 12px; }}
    .dataset-panel {{ display: grid; gap: 12px; align-items: center; padding: 16px; margin: 0; }}
    .dataset-copy {{ display: grid; gap: 6px; }}
    .dataset-copy p {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .dataset-actions {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .upload-button, .secondary-button {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 10px; padding: 10px 14px; font: inherit; font-size: 13px; font-weight: 700; cursor: pointer; }}
    .upload-button {{ border: 0; background: var(--fill-cool); color: #fff; }}
    .secondary-button {{ border: 1px solid var(--border); background: var(--surface-2); color: var(--text); }}
    .upload-button input {{ position: absolute; inline-size: 1px; block-size: 1px; opacity: 0; pointer-events: none; }}
    .dataset-status {{ min-height: 16px; color: var(--muted); font-size: 11px; }}
    .dataset-status.is-error {{ color: #dc2626; }}
    .control-grid {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    .control-card {{ padding: 12px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface-2); }}
    .control-copy {{ display: grid; gap: 8px; margin-bottom: 10px; }}
    .control-copy p {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
    .slider-row {{ display: grid; grid-template-columns: 1fr 70px; gap: 14px; align-items: center; }}
    .field-control {{ margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }}
    .field-toggle-list {{ display: flex; flex-wrap: wrap; gap: 9px; }}
    .field-toggle {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface-2); color: var(--muted); font-size: 13px; cursor: pointer; user-select: none; }}
    .field-toggle:has(input:checked) {{ border-color: var(--c0); color: var(--text); box-shadow: inset 0 0 0 1px var(--c0); }}
    .field-toggle input {{ width: 15px; height: 15px; margin: 0; accent-color: var(--c0); cursor: pointer; }}
    .field-status {{ min-height: 20px; margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .field-status.is-error {{ color: #dc2626; }}
    input[type="range"] {{ width: 100%; accent-color: var(--c0); }}
    output {{ color: var(--brand-strong); font-variant-numeric: tabular-nums; font-size: 15px; font-weight: 800; text-align: right; }}
    .recommended {{ display: inline-flex; width: max-content; padding: 3px 7px; border-radius: 999px; background: var(--brand-soft); color: var(--brand-strong); font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }}
    .distribution-heading {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin: 2px 0 10px; }}
    .distribution-heading p {{ color: var(--muted); font-size: 13px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0; }}
    .stat {{ padding: 12px; box-shadow: none; background: var(--surface-2); }}
    .stat-label {{ color: var(--muted); font-size: 13px; }}
    .stat-value {{ margin-top: 3px; font-size: 23px; font-weight: 750; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }}
    .visual-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; align-items: stretch; }}
    .chart-panel {{ padding: 18px; min-height: 620px; border: 0; box-shadow: none; }}
    .panel-heading {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 12px; }}
    .panel-heading p {{ color: var(--muted); font-size: 13px; margin-top: 5px; }}
    #scatter {{ width: 100%; height: 540px; display: block; }}
    .side-panel {{ padding: 16px; }}
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
    .cluster-count {{ color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .persona-id {{ margin-right: 6px; color: var(--muted); font-size: 10px; font-weight: 700; }}
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
    .pain-point-button {{ appearance: none; border: 0; border-radius: 9px; padding: 11px 15px; background: var(--fill-cool); color: #fff; font: inherit; font-weight: 750; cursor: pointer; }}
    .pain-point-button:disabled {{ cursor: not-allowed; opacity: .45; }}
    .pain-point-note {{ color: var(--muted); font-size: 13px; }}
    .pain-point-result {{ margin-top: 16px; padding: 0; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--border); }}
    .pain-point-result > summary {{ padding: 12px 14px; cursor: pointer; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .pain-point-result > div {{ padding: 0 14px 14px; }}
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
    .disclosure-panel {{ margin: 0; overflow: clip; }}
    .disclosure-panel > summary {{ padding: 12px 14px; cursor: pointer; font-size: 12px; font-weight: 700; list-style-position: inside; }}
    .disclosure-panel > summary small {{ display: block; margin: 4px 0 0 20px; color: var(--muted); font-size: 12px; font-weight: 450; }}
    .disclosure-panel[open] > summary {{ border-bottom: 1px solid var(--border); }}
    .technical-note {{ padding: 14px 18px; color: var(--muted); font-size: 12px; line-height: 1.55; }}
    :where(button, input, summary, .upload-button):focus-visible {{ outline: 3px solid color-mix(in srgb, var(--brand) 45%, transparent); outline-offset: 3px; }}
    .trace-panel {{ padding: 12px; margin: 0; }}
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

    /* Sculptural geometry: calm utility with a few deliberate surprises. */
    .dataset-panel {{ padding: 22px 18px 24px; border-radius: 70px 22px 38px 86px; background: var(--surface-2); }}
    .control-panel {{ padding: 20px 17px 24px; border-radius: 28px 82px 30px 52px; }}
    .setup-rail .disclosure-panel {{ border-radius: 18px 48px 20px 34px; background: var(--olive); color: #fff8e6; box-shadow: 7px 9px 0 rgba(48,42,28,.11); }}
    .setup-rail .disclosure-panel > summary {{ color: #fff8e6; }}
    .setup-rail .disclosure-panel > summary small,
    .setup-rail .trace-heading p,
    .setup-rail .trace-step p,
    .setup-rail .trace-status {{ color: rgba(255,248,230,.7); }}
    .setup-rail .trace-step {{ border: 0; border-radius: 24px 9px 26px 11px; background: rgba(255,248,230,.11); }}
    .setup-rail .trace-badge {{ border-color: rgba(255,248,230,.35); color: #fff8e6; }}
    .upload-button, .secondary-button, .pain-point-button {{ border-radius: 24px 9px 26px 11px; transition: transform .16s ease, box-shadow .16s ease, background .16s ease; }}
    .upload-button {{ padding: 11px 16px; background: var(--vermilion); color: #fff8e6; box-shadow: 4px 5px 0 rgba(87,35,15,.16); }}
    .secondary-button {{ border: 0; background: var(--brand-soft); color: var(--brand-strong); }}
    .upload-button:hover, .secondary-button:hover, .pain-point-button:hover:not(:disabled) {{ transform: translate(-1px, -2px); }}
    .upload-button:active, .secondary-button:active, .pain-point-button:active:not(:disabled) {{ transform: translate(1px, 1px); box-shadow: none; }}
    .control-grid {{ gap: 10px; }}
    .control-card {{ padding: 14px 13px 15px; border: 0; border-radius: 30px 11px 34px 14px; background: var(--brand-soft); }}
    .control-card:nth-child(even) {{ border-radius: 12px 36px 14px 30px; }}
    .control-copy p {{ font-size: 12px; }}
    input[type="range"] {{ accent-color: var(--olive); }}
    output {{ color: var(--olive); }}
    .recommended {{ border-radius: 12px 5px 14px 6px; background: var(--surface-2); }}
    .field-toggle {{ padding: 8px 11px; border: 0; border-radius: 18px 7px 20px 9px; background: var(--surface-2); }}
    .field-toggle:nth-child(even) {{ border-radius: 7px 20px 9px 18px; }}
    .field-toggle:has(input:checked) {{ background: var(--olive); color: #fff8e6; box-shadow: 3px 4px 0 rgba(48,42,28,.12); }}
    .field-toggle input {{ accent-color: var(--vermilion); }}
    .distribution-heading {{ margin: 5px 8px 9px; }}
    .stats {{ gap: 11px; align-items: start; }}
    .stat {{ min-height: 92px; display: grid; place-content: center; padding: 14px; border-radius: 46% 54% 34% 30% / 28% 32% 50% 54%; background: var(--olive); color: #fff8e6; text-align: center; box-shadow: 6px 8px 0 rgba(48,42,28,.12); }}
    .stat:nth-child(1) {{ border-radius: 54% 46% 30% 38% / 34% 28% 50% 46%; background: var(--vermilion); transform: translateY(7px); }}
    .stat:nth-child(2) {{ border-radius: 48% 52% 34% 28% / 26% 36% 50% 54%; background: var(--lake); }}
    .stat:nth-child(3) {{ border-radius: 56% 44% 27% 38% / 34% 24% 55% 46%; transform: translateY(5px); }}
    .stat:nth-child(4) {{ border-radius: 43% 57% 38% 27% / 25% 38% 48% 58%; background: #7a724b; }}
    .stat-label {{ color: rgba(255,248,230,.72); font-size: 11px; font-weight: 700; letter-spacing: .02em; }}
    .stat-value {{ margin-top: 0; font-size: 28px; font-weight: 850; }}
    .side-panel {{ position: relative; overflow: hidden; padding: 22px 20px 24px; border-radius: 74px 24px 88px 28px; background: var(--surface-2); }}
    .side-panel::before {{ content: ""; position: absolute; width: 190px; height: 150px; right: -92px; top: -82px; border-radius: 48% 52% 38% 62% / 56% 42% 58% 44%; background: var(--brand-soft); transform: rotate(18deg); }}
    .side-panel > * {{ position: relative; }}
    .cluster-list-header {{ display: none; }}
    .cluster-list {{ gap: 8px; margin-top: 12px; }}
    .cluster-row {{ grid-template-columns: 24px minmax(0,1fr) auto 34px; gap: 9px; padding: 14px 12px; border: 0; border-radius: 34px 12px 38px 14px; background: var(--cream); box-shadow: 3px 4px 0 rgba(48,42,28,.07); }}
    .cluster-row:nth-child(even) {{ border-radius: 13px 40px 15px 32px; }}
    .cluster-row.is-unselected {{ opacity: .47; filter: saturate(.55); }}
    .swatch {{ width: 9px; height: 26px; border-radius: 8px 4px 9px 5px; }}
    .persona-select {{ width: 19px; height: 19px; accent-color: var(--vermilion); }}
    .representative-records, .persona-grounding, .pain-point-result {{ border: 0; border-radius: 24px 9px 28px 11px; background: var(--surface-3); }}
    .representative-card, .grounding-layer {{ border: 0; border-radius: 20px 8px 22px 10px; background: var(--surface-2); }}
    .pain-point-action {{ align-items: center; padding: 5px 3px 2px; }}
    .pain-point-button {{ padding: 12px 17px; background: var(--vermilion); color: #fff8e6; box-shadow: 4px 5px 0 rgba(87,35,15,.16); }}
    .pain-point-note {{ font-size: 11px; }}
    .analysis-note, .problem-card, .article-analysis-item {{ border: 0; border-radius: 28px 11px 32px 13px; background: var(--cream); }}
    .problem-card:nth-child(even), .article-analysis-item:nth-child(even) {{ border-radius: 12px 32px 14px 28px; }}
    .workspace > .disclosure-panel {{ border-radius: 30px 72px 34px 42px; background: var(--surface-2); }}
    .workspace > .disclosure-panel:nth-of-type(even) {{ border-radius: 68px 26px 48px 30px; }}
    .workspace > .disclosure-panel > summary {{ padding: 15px 19px; }}
    .table-panel {{ margin-top: 0; }}
    .score, .metric-chip, .confidence-badge {{ border: 0; background: var(--brand-soft); }}
    .tooltip {{ border-radius: 22px 8px 24px 10px; background: var(--olive); color: #fff8e6; }}
    @media (max-width: 900px) {{
      header {{ flex-wrap: wrap; }}
      .app-layout {{ grid-template-columns: 1fr; }}
      .control-grid, .visual-grid {{ grid-template-columns: 1fr; }}
      .journey {{ order: 3; width: 100%; overflow-x: auto; }}
      .trace-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      .chart-panel {{ min-height: 520px; }}
      #scatter {{ height: 440px; }}
    }}
    @media (max-width: 520px) {{
      main {{ padding: 0 8px 28px; }}
      header {{ margin-inline: -8px; padding: 10px 12px; }}
      .meta {{ display: none; }}
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
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{ scroll-behavior: auto !important; transition: none !important; animation: none !important; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="brand-lockup"><h1>DemandMap</h1><p class="lede">Customer personas + market signals</p></div>
      <div class="journey" aria-label="DemandMap workflow">
        <div class="journey-step"><span>1</span><strong>Data</strong></div>
        <div class="journey-step"><span>2</span><strong>Personas</strong></div>
        <div class="journey-step"><span>3</span><strong>Select</strong></div>
        <div class="journey-step"><span>4</span><strong>Research</strong></div>
      </div>
      <div class="meta">
        <span class="tag" id="rowMeta">50 customers</span>
        <span class="tag" id="datasetMeta">Demo CSV</span>
        <span class="tag" id="semanticMeta" hidden>Personas shaped by Job Title</span>
        <span class="tag" id="kMeta">6 persona options</span>
        <span class="tag" id="embeddingMeta" hidden>Analysis ready</span>
      </div>
    </header>

    <div class="app-layout">
    <aside class="setup-rail" aria-label="DemandMap setup">
    <section class="panel dataset-panel" aria-label="Customer CSV source">
      <div class="dataset-copy">
        <div class="step-kicker">1 · Data</div>
        <h2>Customer CSV</h2>
        <p>Use the demo or upload your own.</p>
      </div>
      <div class="dataset-actions">
        <label class="upload-button">Upload customer CSV<input id="csvUpload" type="file" accept=".csv,text/csv"></label>
        <button class="secondary-button" id="demoButton" type="button">Use demo CSV</button>
      </div>
      <div class="dataset-status" id="datasetStatus" role="status" aria-live="polite">Demo ready · 50 customers</div>
    </section>

    <details class="panel disclosure-panel" aria-label="How DemandMap uses data">
      <summary>Data & model activity</summary>
      <div class="trace-panel">
        <div class="trace-heading"><h2>Processing and model activity</h2><p>CSV parsing is local. Selected nonblank values are sent to OpenAI for semantic analysis; market research uses NewsAPI.</p></div>
        <div class="trace-grid" id="traceGrid"></div>
      </div>
    </details>

    <section class="panel control-panel" aria-label="Clustering controls">
      <div class="section-intro">
        <div class="step-kicker">2 · Personas</div>
        <h2>Shape the groups</h2>
      </div>
      <div class="control-grid">
        <div class="control-card">
          <div class="control-copy">
            <h2>Persona options <span class="recommended">6 recommended</span></h2>
            <p>Fewer = broader. More = narrower.</p>
          </div>
          <div class="slider-row">
            <input id="clusterCount" type="range" min="{min(config.K_RANGE)}" max="{max(config.K_RANGE)}" step="1" value="{metadata['selected_k']}" aria-label="Number of persona options">
            <output id="clusterCountValue" for="clusterCount">{metadata['selected_k']} personas</output>
          </div>
        </div>
        <div class="control-card">
          <div class="control-copy">
            <h2>Overlap <span class="recommended">Balanced</span></h2>
            <p>Broad = more shared customers.</p>
          </div>
          <div class="slider-row">
            <input id="threshold" type="range" min="0.10" max="0.90" step="0.01" value="{config.MEMBERSHIP_THRESHOLD:.2f}" aria-label="Persona overlap from broad to focused">
            <output id="thresholdValue" for="threshold">Balanced</output>
          </div>
        </div>
      </div>
      <div class="field-control">
        <div class="control-copy">
          <h2>Traits</h2>
          <p>What should make customers similar?</p>
        </div>
        <div class="field-toggle-list" id="fieldToggleList" aria-label="Customer traits used to shape personas"></div>
        <div class="field-status" id="fieldStatus" role="status" aria-live="polite"></div>
      </div>
      <details class="footnote">
        <summary>Method</summary>
        <p class="technical-note">DemandMap creates semantic vectors for selected fields, combines them by customer, and applies Fuzzy C-Means. “Persona options” controls the prototype count (K). “Persona overlap” adjusts the independent membership cutoff; the underlying scores are affinities, not probabilities.</p>
      </details>
    </section>
    </aside>

    <section class="workspace" aria-label="DemandMap results">

    <div class="distribution-heading">
      <div><div class="step-kicker">Overview</div><h2>Customer distribution</h2></div>
    </div>
    <section class="stats" aria-label="Overlap summary">
      <article class="panel stat"><div class="stat-label">Not matched</div><div class="stat-value" id="zeroCount">0</div></article>
      <article class="panel stat"><div class="stat-label">One persona</div><div class="stat-value" id="oneCount">0</div></article>
      <article class="panel stat"><div class="stat-label">Two personas</div><div class="stat-value" id="twoCount">0</div></article>
      <article class="panel stat"><div class="stat-label">Three or more</div><div class="stat-value" id="threeCount">0</div></article>
    </section>

    <section class="visual-grid">
      <aside class="panel side-panel">
        <div class="section-intro">
          <div class="step-kicker">3 · Select</div>
          <h2>Persona candidates</h2>
        </div>
        <div class="cluster-list-header" aria-hidden="true"><span></span><span>Persona</span><span>Customers</span><span>Research</span></div>
        <div class="cluster-list" id="clusterList"></div>
        <div class="pain-point-action">
          <button class="pain-point-button" id="painPointButton" type="button" aria-describedby="painPointNote">Find current needs</button>
          <span class="pain-point-note" id="painPointNote">Select a persona to research.</span>
        </div>
        <div class="news-status" id="newsStatus" role="status" aria-live="polite" hidden></div>
        <section class="article-results" id="articleResults" aria-labelledby="articleResultsHeading" hidden>
          <div class="article-results-heading">
            <div><div class="step-kicker">4 · Research</div><h3 id="articleResultsHeading">Current needs</h3></div>
            <span id="articleCount"></span>
          </div>
          <div id="painAnalysis"></div>
        </section>
        <details class="pain-point-result" id="painPointResult" hidden>
          <summary>View market-search details</summary>
          <div><pre id="painPointJson"></pre></div>
        </details>
        <h2 class="overlap-title">Where personas overlap most</h2>
        <div class="overlap-list" id="overlapList" aria-live="polite"></div>
      </aside>
    </section>

    <details class="panel disclosure-panel">
      <summary>Customer similarity map</summary>
      <div class="chart-panel">
          <div class="panel-heading"><div><h2>Customer similarity map</h2><p>Color shows the strongest persona match. A ring means the customer fits more than one persona at the current overlap setting.</p></div></div>
          <svg id="scatter" role="img" aria-label="Visual map of customer similarity based on selected customer traits"></svg>
      </div>
    </details>

    <details class="panel disclosure-panel">
      <summary>All customer matches</summary>
      <div class="table-panel">
        <div class="table-wrap"><table>
          <thead><tr><th id="displayHeader0">Customer</th><th id="displayHeader1">Role</th><th id="displayHeader2">Company</th><th>Persona matches</th></tr></thead>
          <tbody id="prospectRows"></tbody>
        </table></div>
      </div>
    </details>
    <details class="footnote"><summary>About these personas</summary><p class="technical-note">Generated from representative rows in the selected fields. Use them as research starting points, not verified facts about every customer.</p></details>
    </section>
    </div>
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
        ['parse', 'Prepare the customer file', 'local', 'Identify usable customer traits and display columns without changing the CSV.'],
        ['cluster', 'Find patterns in selected traits', 'model + local', 'OpenAI creates reusable semantic vectors; grouping and overlap are calculated locally.'],
        ['personas', 'Describe persona options', 'model', 'Representative values from every selected trait become names, descriptions, and evidence links.'],
        ['query', 'Plan the market search', 'model', 'Only visible text from selected personas becomes a NewsAPI query.'],
        ['news', 'Collect distinct market events', 'API + model', 'Recent NewsAPI coverage is compared so repeated reporting of one event is counted once.'],
        ['pains', 'Summarize possible customer needs', 'model', 'Persona descriptions and recent events become evidence-backed hypotheses.']
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
        const sourceItem = item => `<div class="source-evidence-item"><div class="source-evidence-meta">${{escapeHtml(item.rowId)}} · ${{escapeHtml(item.field)}} · ${{escapeHtml(item.evidenceId)}}</div><div class="source-evidence-value">${{escapeHtml(item.value)}}</div></div>`;
        const groundingMarkup = grounding ? `
          <details class="persona-grounding">
            <summary>View supporting customer data (${{Array.isArray(grounding.sourceEvidence) ? grounding.sourceEvidence.length : 0}} values)</summary>
            <div class="grounding-body">
              <section class="grounding-layer">
                <div class="grounding-layer-label">DemandMap interpretation</div>
                <p>This persona summary is generated from the source values below. It should be reviewed as an informed starting point, not a verified fact about every customer.</p>
                <div class="grounding-field"><strong>Values supporting the description</strong><div class="source-evidence-list">${{(grounding.descriptionEvidence || []).map(sourceItem).join('')}}</div>${{!(grounding.descriptionEvidence || []).length ? '<div class="no-source-evidence">No source values were available for this description.</div>' : ''}}</div>
                ${{(grounding.fieldSummaries || []).map(summary => `<div class="grounding-field"><strong>${{escapeHtml(summary.field)}}</strong><p>${{escapeHtml(summary.summary)}}</p><div class="source-evidence-list">${{(summary.evidence || []).map(sourceItem).join('')}}</div>${{summary.evidenceStatus === 'no_evidence' ? '<div class="no-source-evidence">No populated representative value was supplied for this field.</div>' : ''}}</div>`).join('')}}
              </section>
              <section class="grounding-layer">
                <div class="grounding-layer-label">Exact values from your CSV</div>
                <p>These values are copied by the server rather than written by the model. Row IDs stay stable for this dataset.</p>
                <div class="source-evidence-list">${{(grounding.sourceEvidence || []).map(sourceItem).join('')}}</div>
              </section>
            </div>
          </details>` : '';
        const examplesMarkup = examples.length ? `
          <details class="representative-records">
            <summary>View example customers (${{examples.length}})</summary>
            <div class="representative-list">${{examples.map(example => {{
              const displayEntries = Object.entries(example.displayValues || {{}});
              const fieldEntries = Object.entries(example.fieldValues || {{}});
              const identity = (displayEntries.find(([, value]) => value && value !== 'Not provided') || [null, example.rowId])[1];
              return `<article class="representative-card">
                <div class="representative-card-head"><strong>${{escapeHtml(identity)}}</strong><span class="affinity-score">Match strength ${{Number(example.score).toFixed(2)}}</span></div>
                <div class="representative-row-id">${{escapeHtml(example.rowId)}}</div>
                <div class="representative-section-label">Customer</div>
                <div class="representative-fields">${{displayEntries.map(([field, value]) => `<span><b>${{escapeHtml(field)}}:</b> ${{escapeHtml(value)}}</span>`).join('')}}</div>
                <div class="representative-section-label">Traits shaping this persona</div>
                <div class="representative-fields">${{fieldEntries.map(([field, value]) => `<span><b>${{escapeHtml(field)}}:</b> ${{escapeHtml(value)}}</span>`).join('')}}</div>
              </article>`;
            }}).join('')}}</div>
          </details>` : '';
        return `
        <div class="cluster-row ${{selected[cluster] ? '' : 'is-unselected'}}">
          <span class="swatch" style="background:${{colors[cluster]}}"></span>
          <span class="cluster-name"><strong><span class="persona-id">P${{cluster + 1}}</span>${{escapeHtml(label)}}</strong><small class="cluster-description">${{escapeHtml(description)}}</small>${{groundingMarkup}}${{examplesMarkup}}</span>
          <span class="cluster-count">${{counts[cluster]}} customers</span>
          <input class="persona-select" type="checkbox" data-cluster="${{cluster}}" ${{selected[cluster] ? 'checked' : ''}} aria-label="Include ${{escapeHtml(label)}} in market research">
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
        ? pairs.slice(0, 6).map(pair => `<div class="overlap-row"><span>${{escapeHtml(model.labels[pair.a])}} + ${{escapeHtml(model.labels[pair.b])}}</span><strong>${{pair.shared}} shared</strong></div>`).join('')
        : '<div class="muted">No customers currently match more than one persona.</div>';
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
          <td><div class="score-list">${{record.active.length ? record.active.map(item => `<span class="score"><i style="background:${{colors[item.cluster]}}"></i>${{escapeHtml(currentModel().labels[item.cluster])}} · ${{item.score.toFixed(2)}}</span>`).join('') : '<span class="muted">Not matched at this setting</span>'}}</div></td>
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
      markup += `<text x="${{width/2}}" y="${{height-10}}" text-anchor="middle" fill="${{muted}}" font-size="12">Similarity dimension 1</text>`;
      markup += `<text x="14" y="${{height/2}}" text-anchor="middle" fill="${{muted}}" font-size="12" transform="rotate(-90 14 ${{height/2}})">Similarity dimension 2</text>`;
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
      tooltip.innerHTML = `<strong>${{escapeHtml(record.name)}}</strong><br>${{escapeHtml(record.title)}} · ${{escapeHtml(record.company)}}<br>${{active.length ? active.map(item => `${{escapeHtml(currentModel().labels[item.cluster])}} · ${{item.score.toFixed(2)}}`).join('<br>') : 'Not matched at this setting'}}`;
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
      painPointButton.textContent = isSearching ? 'Researching current needs…' : count ? `Find current needs for ${{count}} persona${{count === 1 ? '' : 's'}}` : 'Find current needs';
      document.getElementById('painPointNote').textContent = count
        ? `${{count}} selected · 30-day market scan`
        : 'Select a persona to research.';
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
          <p><strong>Why this may be happening now:</strong> ${{escapeHtml(problem.why_emerging)}} ${{citationLinks(problem.supporting_article_ids, articles)}}</p>
          <p><strong>What it could mean for this audience:</strong> ${{escapeHtml(problem.persona_impact)}} ${{citationLinks(problem.supporting_article_ids, articles)}}</p>
          <div class="metric-row">
            <span class="metric-chip">${{Number(problem.distinct_supporting_events || 0)}} distinct event${{Number(problem.distinct_supporting_events || 0) === 1 ? '' : 's'}}</span>
            <span class="metric-chip">${{Number(problem.supporting_articles || 0)}} supporting article${{Number(problem.supporting_articles || 0) === 1 ? '' : 's'}}</span>
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
        <section class="analysis-section"><h3>Most supported needs</h3><div class="problem-list">${{problemMarkup}}</div></section>
        <details class="analysis-section"><summary><strong>View evidence summary</strong></summary>${{ledgerMarkup}}</details>
        <details class="analysis-section"><summary><strong>Review every article</strong></summary><div class="article-analysis-list">${{articleMarkup}}</div></details>`;
      articleResults.hidden = false;
    }}
    async function searchMarketPainPoints() {{
      const request = buildPainPointRequest();
      if (!request) return;
      if (window.location.protocol === 'file:') {{
        newsStatus.textContent = 'Market research is available when DemandMap is opened through its server.';
        newsStatus.classList.add('is-error');
        newsStatus.hidden = false;
        return;
      }}
      isSearching = true;
      updatePainPointButton();
      setTrace('query', 'running', 'Creating a query from only the selected visible persona text.');
      setTrace('news', 'idle', 'Waiting for query generation.');
      setTrace('pains', 'idle', 'Waiting for deduplicated articles.');
      newsStatus.textContent = 'Reviewing recent market events and connecting them to the selected personas. This may take a moment…';
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
        newsStatus.textContent = `Research complete: ${{problemCount}} supported customer-need hypothes${{problemCount === 1 ? 'is' : 'es'}} from ${{returned}} distinct market event${{returned === 1 ? '' : 's'}}.`;
        renderPainAnalysis(payload);
      }} catch (error) {{
        newsStatus.textContent = error.message || 'DemandMap could not complete the market research. Your persona selections are still available; try again shortly.';
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
      document.getElementById('semanticMeta').textContent = `Personas shaped by ${{activeFields.join(' + ')}}`;
      const embeddingName = embeddingLabel(state.embeddingBackend, activeEmbeddingMetadata);
      fieldStatus.textContent = `${{activeFields.length}} trait${{activeFields.length === 1 ? '' : 's'}} selected · personas ready`;
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
      document.getElementById('rowMeta').textContent = `${{activeRecords.length}} customers`;
      document.getElementById('datasetMeta').textContent = activeDatasetId === 'demo' ? 'Demo CSV' : activeDatasetName;
      document.getElementById('embeddingMeta').textContent = 'Analysis ready';
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
        fieldStatus.textContent = 'Open DemandMap through its server to change the customer traits used for personas.';
        fieldStatus.classList.add('is-error');
        syncFieldControls();
        return;
      }}
      if (clusteringRequestController) clusteringRequestController.abort();
      const controller = new AbortController();
      clusteringRequestController = controller;
      isClustering = true;
      fieldStatus.textContent = `Updating personas using ${{fields.join(' + ')}}…`;
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
        fieldStatus.textContent = error.message || 'DemandMap could not update the personas. Your previous selection is still available; try again.';
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
      fieldStatus.textContent = `${{activeFields.length}} trait${{activeFields.length === 1 ? '' : 's'}} selected`;
    }}

    async function uploadCustomerCsv(file) {{
      if (!file) return;
      if (window.location.protocol === 'file:') {{
        datasetStatus.textContent = 'Open DemandMap through its server to upload a customer CSV.';
        datasetStatus.classList.add('is-error');
        return;
      }}
      datasetStatus.textContent = `Preparing ${{file.name}}…`;
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
        document.getElementById('rowMeta').textContent = `${{profile.rowCount}} customers`;
        document.getElementById('datasetMeta').textContent = profile.datasetName;
        datasetStatus.textContent = `${{profile.datasetName}} · ${{profile.rowCount}} customers · ${{profile.clusterableFields.length}} usable traits`;
        setTrace('parse', 'completed', `${{profile.rowCount}} rows; ${{profile.clusterableFields.length}} descriptive columns inferred locally.`);
        setTrace('query', 'idle', 'Runs after persona selection.');
        setTrace('news', 'idle', 'Runs after query generation.');
        setTrace('pains', 'idle', 'Runs after article deduplication.');
        await recluster(activeFields);
      }} catch (error) {{
        datasetStatus.textContent = error.message || 'DemandMap could not read this CSV. Check that it has column headings and at least three populated customer rows, then try again.';
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
      datasetStatus.textContent = `Demo ready · ${{DATA.records.length}} customers`;
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
      const overlapLabel = threshold <= 0.30 ? 'Broad' : threshold <= 0.50 ? 'Balanced' : threshold <= 0.70 ? 'Focused' : 'Very focused';
      thresholdValue.textContent = overlapLabel;
      thresholdInput.setAttribute('aria-valuetext', `${{overlapLabel}} overlap, technical cutoff ${{threshold.toFixed(2)}}`);
      clusterCountValue.textContent = `${{k}} personas`;
      clusterCountInput.setAttribute('aria-valuetext', `${{k}} persona options`);
      document.getElementById('kMeta').textContent = `${{k}} persona options`;
      document.getElementById('semanticMeta').textContent = `Personas shaped by ${{activeFields.join(' + ')}}`;
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
    document.getElementById('embeddingMeta').textContent = 'Analysis ready';
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
