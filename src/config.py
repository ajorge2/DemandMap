from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "data" / "prospects.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Initial semantic view selected after inspecting the supplied export.
SEMANTIC_FIELD = "Job Title"

# Descriptive CSV fields that can be combined interactively in the dashboard.
# Names, profile URLs, numeric columns, duplicate enrichment fields, and the
# all-null export label are intentionally excluded from semantic clustering.
CLUSTERABLE_FIELDS = (
    "Job Title",
    "Company",
    "City",
    "State or Province",
    "Country",
    "Headline",
    "Summary",
    "Summarize LinkedIn profile",
)

# Customer clustering uses cached OpenAI embeddings. Individual CSV-field
# vectors are cached locally and combined when the user changes field toggles.
EMBEDDING_BACKEND = "openai"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CACHE_DIR = PROJECT_DIR / ".cache" / "openai_embeddings"
SENTENCE_TRANSFORMER_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LSA_DIMENSIONS = 24

# Change this one value and rerun only analyze_from_saved.py to refresh
# threshold-dependent columns, the report, and the SVG. Embeddings are reused.
MEMBERSHIP_THRESHOLD = 0.40
THRESHOLDS_TO_COMPARE = (0.25, 0.40, 0.55, 0.70)

K_RANGE = (3, 4, 5, 6, 7)
FUZZINESS = 1.5
N_INITIALIZATIONS = 24
MAX_ITERATIONS = 500
TOLERANCE = 1e-7
RANDOM_SEED = 42

# Optional human review layer. Keys are deterministic cluster IDs from the
# selected run; values replace the heuristic label in reports and charts.
CLUSTER_LABEL_OVERRIDES = {
    0: "Board & Advisory Roles",
    1: "C-Suite & Functional Executives",
    2: "Investors",
    3: "Co-Founders & Chairs",
    4: "Founders & CEOs",
    5: "Partners, Mentors & Advisors",
}
