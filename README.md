# Fuzzy semantic clustering prototype

This project clusters the supplied B2B prospect export using one semantic view at a time. The first experiment uses `Job Title` because it is populated for all 50 rows and directly captures prospect role. `Headline` and `Summary` are richer but only populated for six rows.

## Method

1. Inspect the schema, missingness, uniqueness, and candidate semantic views.
2. Embed each selected CSV field value with OpenAI `text-embedding-3-small`, caching vectors locally by a hash of model plus source text.
3. Combine the currently selected fields with an equal-weight vector mean and L2 normalization. Changing to a combination of already embedded fields stays local and does not make another embedding request.
4. Explore K=3 through K=7 using Fuzzy C-Means. Compare argmax silhouette as a diagnostic only, fuzzy partition coefficient, partition entropy, Xie-Beni compactness/separation, and centroid separation.
5. Convert distances to each learned fuzzy prototype into tie-aware, within-prototype percentile affinities and sharpen them to avoid high-dimensional distance concentration. These scores do not sum to one, so a row can belong to several clusters at the same cutoff. Native Fuzzy C-Means memberships are retained in the CSV for auditability.
6. Generate a report, cutoff sensitivity table, overlap pairs, examples, and a PCA SVG.

The independent affinities are prototype scores, not calibrated probabilities of persona truth. They are appropriate for evaluating overlap behavior, but a production model should validate calibration against human-labeled examples.

## Run

From the project directory, using a Python environment with NumPy and pandas:

```bash
python -m src.run_prototype
```

The clustering and reporting code does not use a generative model to assign memberships. OpenAI supplies semantic vectors; Fuzzy C-Means and the independently calibrated prototype affinities run locally.

## News search and operational pain-point analysis

Copy `.env.example` to `.env`, then add both keys without quotes:

```text
NEWSAPI_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

`OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` are optional; the local server defaults to `gpt-5-mini` and `text-embedding-3-small` respectively.

The repository-level `.gitignore` excludes `.env` files. Run the local dashboard server with:

```bash
python -m src.serve_dashboard
```

Open `http://127.0.0.1:8765/clustering_explorer.html`. The browser sends the generated query and the visible text of the selected persona rows to the local proxy. The proxy authenticates with the `X-Api-Key` header and searches English-language coverage from the last 30 days, newest-first. It fetches additional result pages when needed and retains up to 25 distinct events by comparing cached title-and-description embeddings plus local entity, event-type, amount, date, token, and title signals. Conflicting event evidence blocks merges, and every accepted merge retains auditable pair-level provenance.

### Heroku-buildpack deployment

The repository root is ready for a Python Heroku buildpack. `requirements.txt` triggers Python detection, `.python-version` selects Python 3.13, and `Procfile` starts the web process on the platform-provided `PORT`:

```text
web: python -m src.serve_dashboard --host 0.0.0.0 --port $PORT
```

On Northflank, select its Heroku/buildpack build type with this directory as the build context. Expose the same injected application port and use `/health` for the HTTP health check. Configure `OPENAI_API_KEY` and `NEWSAPI_KEY` as secrets; `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` remain optional.

The proxy then sends only those visible persona descriptions and the retained articles' titles, descriptions, source names, dates, and URLs to the OpenAI Responses API. Structured output is used to produce ranked operational pain hypotheses, an evidence ledger, and a compact article-by-article audit. The dashboard presents that synthesis instead of a raw article feed. Both API keys stay in the server-side `.env` file and are never embedded in the HTML; OpenAI response storage is disabled for these requests.

The NewsAPI query builder receives persona-specific context only from the selected candidates' visible names and per-column evidence. It does not inject fixed market assumptions such as B2B, AI, SaaS, startup, or martech. Its generic event vocabulary is inferred from role and theme words that actually appear in that visible text, and the generated JSON includes the exact `persona_descriptions` used for auditability.

The demo dashboard's CSV-column controls can combine `Job Title`, company, location, headline, summary, and enriched profile summary text. Changing a column selection asks the same local server to combine cached field embeddings, recompute the K=3–7 fuzzy models, generate persona names, and refresh memberships and projection. A newly selected field makes an embedding request only for values missing from the local vector cache. Model-generated persona descriptions are validated to include a summary for every selected field. Each populated summary and overall description must cite deterministic, candidate-scoped evidence IDs; the server rejects unknown, duplicate, wrong-field, and cross-candidate references before resolving them to bounded exact CSV values. The dashboard keeps generated interpretation visually separate from inspectable source row IDs, fields, and values. Identifier, URL, numeric, duplicate enrichment, and all-null fields are excluded from these controls.

## Upload another customer CSV

Use **Upload customer CSV** at the top of the dashboard. The bundled B2B prospect export remains the default demo and can be restored instantly with **Use demo CSV**.

Uploads are parsed in memory by the local server and support 3–500 non-empty rows, up to 80 columns, and files up to 8 MB. The server infers descriptive text columns while excluding likely identifiers, names, URLs, email addresses, and numeric measures. It also chooses initial high-signal fields and three display columns; every inferred descriptive field remains available as a checkbox.

After OpenAI embeddings and local fuzzy memberships are computed, one batched generative call receives bounded representative values—not the complete CSV—and generates natural persona names, descriptions, and a summary for every selected field. If that call fails, clustering remains usable with deterministic fallback labels. Uploaded source datasets live only in the current server process and are not written to the project; the vector cache contains hashes and numeric vectors rather than source text.

The dashboard's **Workflow trace** makes all external boundaries visible. The complete call-site and data-flow contract is documented in [`MODEL_CALLS.md`](MODEL_CALLS.md): customer-field embedding, persona description generation, NewsAPI query generation, news-event embedding, and pain-point synthesis are model-assisted boundaries; CSV parsing, field inference, vector combination, fuzzy clustering, merge decisions, and validation are local.

Event deduplication has a versioned labeled calibration fixture and a deterministic evaluator in `tests/fixtures/event_deduplication_v1.json` and `src/event_deduplication.py`. Its reported metrics describe that deliberately small fixture only, not broad production accuracy.

## Change the cutoff without recomputing embeddings

Edit the obvious setting in `src/config.py`:

```python
MEMBERSHIP_THRESHOLD = 0.40
```

Then run:

```bash
python -m src.analyze_from_saved
```

That command reuses `outputs/run_state.npz` and refreshes `memberships.csv`, `cluster_report.md`, and `embedding_projection.svg`.

## Outputs

- `outputs/data_inspection.md`: schema, inferred roles, missingness, samples, and semantic-view candidates.
- `outputs/memberships.csv`: original rows plus independent memberships, native FCM memberships, threshold flags, and stable source row IDs.
- `outputs/cluster_report.md`: K diagnostics, cutoff sensitivity, overlap pairs, and per-cluster examples.
- `outputs/embedding_projection.svg`: PCA projection with strong-cluster color and overlap outlines.
- `outputs/run_state.npz` and `outputs/run_metadata.json`: saved embeddings, centroids, memberships, and metadata for threshold-only reruns.

## Modular extension

Keep each future semantic view separate. Change `SEMANTIC_FIELD`, write its results to a distinct output directory, and compare memberships later by `row_id`. Do not concatenate employee count or other numeric attributes into embedding text. Numeric and categorical attributes can be analyzed after clustering as cluster descriptors.
