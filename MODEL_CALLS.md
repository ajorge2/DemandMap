# Model-call map

The dashboard has five model-assisted boundaries. CSV parsing, field inference, selected-field vector combination, fuzzy clustering, projection, and the final event-merge decisions are deterministic local code; customer and news-event semantics use cached embedding-model vectors.

## 1. Customer-field embeddings

- **When:** After a CSV is loaded or the user selects a CSV column whose values are not already cached.
- **Code:** `generate_combined_field_embeddings()` in `src/embeddings.py`, called by `build_clustering_payload()` in `src/serve_dashboard.py`.
- **Sent:** Each nonblank selected field cell independently, prefixed only with its visible CSV column name.
- **Returned:** One `text-embedding-3-small` vector for each unique field value.
- **Cache:** Vectors are stored locally by a SHA-256 hash of model plus input text. Source text is not written to the vector cache. Selecting a new combination of already embedded fields requires no new embedding call.
- **Combination:** Selected field vectors are averaged with equal weight per populated field and L2-normalized locally before fuzzy clustering.

## 2. Persona candidate descriptions

- **When:** After local embeddings and memberships are computed for every available K value.
- **Code:** `generate_persona_descriptions()` in `src/persona_generation.py`, called by `build_clustering_payload()` in `src/serve_dashboard.py`.
- **Sent:** A bounded set of representative rows and representative values for every user-selected CSV column. The server assigns each value a deterministic evidence ID before the request; the complete CSV is not sent.
- **Returned:** One natural persona name and description per candidate, one summary for every selected field, and references to server-issued evidence IDs. The model never supplies the source quotations displayed in the dashboard.
- **Validation:** Candidate IDs must exactly match the local clusters, every selected field must appear exactly once, populated summaries must cite same-field evidence, and unknown, duplicate, wrong-field, or cross-candidate references reject the entire response.
- **Audit:** The server resolves accepted IDs to its own bounded `{row ID, field, exact value}` map. The dashboard separately labels generated interpretation and exact CSV evidence.
- **Fallback:** If the call fails or no OpenAI key is configured, the dashboard retains deterministic labels and exposes the fallback in the workflow trace.

## 3. NewsAPI query generation

- **When:** After the user selects persona rows and clicks **Search market pain-points**, before NewsAPI is called.
- **Code:** `generate_news_query()` in `src/news_query.py`, called by the `/api/news` handler.
- **Sent:** Only the persona names, descriptions, and per-field evidence visible in the selected rows.
- **Returned:** A NewsAPI-compatible `q` under 500 characters, market terms, event terms, functional terms, and concise reasoning.
- **Validation:** A non-empty query is required and its length is checked again server-side.

## 4. News-event embeddings for deduplication

- **When:** After each NewsAPI page is added to the 30-day result set.
- **Code:** `deduplicate_articles()` in `src/event_deduplication.py`, called by the `/api/news` handler.
- **Sent:** Each usable article's title and NewsAPI description, prefixed as event title and description.
- **Returned:** One semantic vector per unique article text. Vectors are cached by model plus a SHA-256 hash of the text.
- **Local decision:** Cached vectors are combined with token, entity, event-type, amount, date, and title signals. Conflicting event evidence blocks merges; the retained representative is newest-first. Merge provenance and fallback status are returned for audit.
- **Fallback:** If semantic embeddings fail, the server exposes that it used a conservative lexical/LSA fallback rather than silently labeling the result semantic.

## 5. Operational pain-point synthesis

- **When:** After NewsAPI retrieval and event-level semantic deduplication.
- **Code:** `analyze_pain_points()` in `src/pain_point_analysis.py`, called by the `/api/news` handler.
- **Sent:** Selected visible persona text plus the retained articles' titles, descriptions, source names, dates, and URLs.
- **Returned:** Ranked operational pain hypotheses, an evidence ledger, and an entry for every retained article.
- **Validation:** Every article must be accounted for exactly once, citations can reference only supplied article IDs, and displayed titles are restored from the NewsAPI payload.

The three generative calls use the OpenAI Responses API with strict JSON Schema structured outputs and `store: false`. Customer-field and news-event vectors use the OpenAI Embeddings API and the same hash-addressed local cache pattern. The browser-visible workflow trace reports which stages are local, API calls, or model calls and updates their runtime status, including embedding cache hits, newly embedded values, retained events, and removed duplicates.
