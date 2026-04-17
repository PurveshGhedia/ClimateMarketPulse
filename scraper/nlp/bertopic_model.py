"""
scraper/nlp/bertopic_model.py

BERTopic pipeline — Step 7 of the NLP pipeline.
Runs topic modeling on articles with relevance_score >= 0.30,
writes topic assignments back to articles.db, and exports:
  - data/processed/topic_info.csv          (topic summaries)
  - data/processed/monthly_topic_prevalence.csv  (exogenous var for ARIMAX)

Usage (from project root, nlp conda env):
    python scraper/nlp/bertopic_model.py

Requirements:
    pip install bertopic umap-learn hdbscan

Model config:
  - Embeddings : all-MiniLM-L6-v2 (reuses relevance scoring model)
  - UMAP       : n_components=5, n_neighbors=15, metric=cosine
  - HDBSCAN    : min_cluster_size=MIN_TOPIC_SIZE, metric=euclidean
  - MIN_TOPIC_SIZE = 10
  - BERTopic threshold for ARIMAX: relevance_score >= 0.35 (noted in output)
"""

import os
import sys
import sqlite3
import warnings
warnings.filterwarnings("ignore")   # suppress UMAP/numba warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH        = os.path.join(PROJECT_ROOT, "data", "articles.db")
PROCESSED_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
TOPIC_INFO_CSV = os.path.join(PROCESSED_DIR, "topic_info.csv")
PREVALENCE_CSV = os.path.join(PROCESSED_DIR, "monthly_topic_prevalence.csv")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "all-MiniLM-L6-v2"
BERTOPIC_THRESHOLD = 0.30    # articles fed to BERTopic
ARIMAX_THRESHOLD   = 0.35    # noted in output for reference
MIN_TOPIC_SIZE   = 10
MAX_TEXT_CHARS   = 1100      # same truncation as relevance_score.py

# UMAP: low n_components keeps HDBSCAN clustering stable;
# cosine metric suits sentence embeddings
UMAP_CONFIG = dict(
    n_components = 5,
    n_neighbors  = 15,
    min_dist     = 0.0,
    metric       = "cosine",
    random_state = 42,
    low_memory   = True,     # important on Mac — avoids OOM on large corpora
)

# HDBSCAN
HDBSCAN_CONFIG = dict(
    min_cluster_size    = MIN_TOPIC_SIZE,
    min_samples         = 5,
    metric              = "euclidean",
    cluster_selection_method = "eom",
    prediction_data     = True,   # needed for soft clustering probs
)

# CountVectorizer: strip common English stopwords + domain noise words
# that appear in nearly every article and hurt topic labels
CUSTOM_STOP_WORDS = [
    "said", "also", "would", "could", "one", "two", "three",
    "year", "years", "month", "months", "cent", "per", "rs",
    "lakh", "crore", "rupee", "rupees", "india", "indian",
    "government", "minister", "ministry", "official", "told",
    "according", "new", "last", "first", "time", "day", "week",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_articles(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT article_id, headline, full_text,
               publish_year, publish_month, relevance_score
        FROM   articles
        WHERE  is_duplicate = 0
          AND  relevance_score >= ?
        ORDER  BY publish_year, publish_month
    """, conn, params=(BERTOPIC_THRESHOLD,))
    return df


def build_text(row: pd.Series) -> str:
    headline  = (row["headline"]  or "").strip()
    full_text = (row["full_text"] or "").strip()
    return f"{headline} {full_text}".strip()[:MAX_TEXT_CHARS]


def write_topic_assignments(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Write topic_id, topic_label, topic_prob back to articles.db."""
    updates = list(zip(
        df["topic_id"].tolist(),
        df["topic_label"].tolist(),
        df["topic_prob"].tolist(),
        df["article_id"].tolist(),
    ))
    conn.executemany("""
        UPDATE articles
        SET topic_id    = ?,
            topic_label = ?,
            topic_prob  = ?
        WHERE article_id = ?
    """, updates)
    conn.commit()


def build_monthly_prevalence(df: pd.DataFrame, topic_info: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly topic prevalence table for ARIMAX.

    For each (year, month) × topic, compute:
      - article_count  : number of articles assigned to that topic
      - mean_prob      : average topic probability (confidence weight)
      - weighted_count : sum of topic probabilities (softer count)

    Excludes topic_id = -1 (outliers).
    Only includes topics from articles with relevance_score >= ARIMAX_THRESHOLD
    so the prevalence signal is clean for the econometric model.
    """
    # Filter to ARIMAX threshold for the prevalence table
    arimax_df = df[
        (df["relevance_score"] >= ARIMAX_THRESHOLD) &
        (df["topic_id"] != -1)
    ].copy()

    if arimax_df.empty:
        print("  [WARN] No articles above ARIMAX threshold — prevalence table will be empty.")
        return pd.DataFrame()

    grouped = arimax_df.groupby(["publish_year", "publish_month", "topic_id", "topic_label"])

    prevalence = grouped.agg(
        article_count  = ("article_id",   "count"),
        mean_prob      = ("topic_prob",    "mean"),
        weighted_count = ("topic_prob",    "sum"),
    ).reset_index()

    prevalence = prevalence.sort_values(
        ["publish_year", "publish_month", "topic_id"]
    ).reset_index(drop=True)

    return prevalence


# ── Main ──────────────────────────────────────────────────────────────────────
def run_bertopic(db_path: str) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("ClimateMarketPulse — BERTopic (Step 7)")
    print("=" * 60)
    print(f"BERTopic threshold  : relevance_score >= {BERTOPIC_THRESHOLD}")
    print(f"ARIMAX threshold    : relevance_score >= {ARIMAX_THRESHOLD} (prevalence table)")
    print(f"MIN_TOPIC_SIZE      : {MIN_TOPIC_SIZE}")
    print(f"DB path             : {db_path}")
    print(f"Output dir          : {PROCESSED_DIR}")
    print("-" * 60)

    # ── Load articles ─────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    df   = load_articles(conn)
    conn.close()

    total = len(df)
    if total == 0:
        print("ERROR: No articles found above threshold. Run relevance_score.py first.")
        sys.exit(1)

    print(f"Articles loaded     : {total}")
    year_range = f"{int(df['publish_year'].min())}–{int(df['publish_year'].max())}"
    print(f"Year range          : {year_range}")

    # ── Build texts ───────────────────────────────────────────────────────
    texts = df.apply(build_text, axis=1).tolist()

    # ── Encode with sentence-transformers ────────────────────────────────
    print("\nEncoding articles...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Device              : {model.device}")

    embeddings = model.encode(
        texts,
        batch_size         = 256,
        show_progress_bar  = True,
        normalize_embeddings = True,
        convert_to_numpy   = True,
    )
    print(f"Embeddings shape    : {embeddings.shape}")

    # ── Build BERTopic ────────────────────────────────────────────────────
    print("\nFitting BERTopic...")

    umap_model   = UMAP(**UMAP_CONFIG)
    hdbscan_model = HDBSCAN(**HDBSCAN_CONFIG)

    vectorizer = CountVectorizer(
        stop_words  = "english",
        ngram_range = (1, 2),
        min_df      = 3,           # ignore terms appearing in < 3 docs
        max_df      = 0.85,        # ignore terms in > 85% of docs
    )
    # Remove domain noise from the vocabulary
    vectorizer.set_params(
        stop_words = list(
            set(CountVectorizer(stop_words="english")
                .get_stop_words()) | set(CUSTOM_STOP_WORDS)
        )
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model      = model,
        umap_model           = umap_model,
        hdbscan_model        = hdbscan_model,
        vectorizer_model     = vectorizer,
        representation_model = representation_model,
        min_topic_size       = MIN_TOPIC_SIZE,
        nr_topics            = "auto",
        calculate_probabilities = True,
        verbose              = True,
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    # ── Extract results ───────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = int((np.array(topics) == -1).sum())

    print(f"\nTopics found        : {n_topics}")
    print(f"Outliers (topic=-1) : {n_outliers}  ({n_outliers/total*100:.1f}%)")

    # ── Build topic labels ────────────────────────────────────────────────
    # Format: "T{id}_{top_word}" e.g. "T3_monsoon_crop"
    topic_label_map: dict[int, str] = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            topic_label_map[-1] = "outlier"
        else:
            # Take top 2 words from Name column (format: "N_word1_word2_word3_word4")
            words = str(row["Name"]).split("_")[1:3]
            label = f"T{tid}_{'_'.join(words)}" if words else f"T{tid}"
            topic_label_map[tid] = label

    # ── Assign back to dataframe ──────────────────────────────────────────
    # probs shape: (n_articles, n_topics) — take max prob per article
    if probs is not None and len(probs.shape) == 2:
        topic_probs = probs[np.arange(len(topics)), np.clip(topics, 0, probs.shape[1]-1)]
        # For outliers, set prob to 0
        topic_probs = np.where(np.array(topics) == -1, 0.0, topic_probs)
    else:
        topic_probs = np.ones(len(topics)) * 0.5

    df["topic_id"]    = topics
    df["topic_label"] = [topic_label_map.get(t, f"T{t}") for t in topics]
    df["topic_prob"]  = topic_probs.tolist()

    # ── Write back to DB ──────────────────────────────────────────────────
    print("\nWriting topic assignments to DB...")
    conn = sqlite3.connect(db_path)

    # Add columns if they don't exist yet (safe migration)
    for col, dtype in [("topic_id", "INTEGER"), ("topic_label", "TEXT"), ("topic_prob", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {dtype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    write_topic_assignments(conn, df)
    conn.close()
    print("DB updated.")

    # ── Save topic_info.csv ───────────────────────────────────────────────
    # Enrich with our label
    topic_info["our_label"] = topic_info["Topic"].map(topic_label_map)
    topic_info.to_csv(TOPIC_INFO_CSV, index=False)
    print(f"Saved: {TOPIC_INFO_CSV}")

    # ── Print top topics ──────────────────────────────────────────────────
    print(f"\nTop 15 topics (excluding outliers):")
    print(f"  {'ID':<6} {'Label':<35} {'Count':>6}  Top words")
    print(f"  {'-'*6}  {'-'*35}  {'-'*6}  {'-'*40}")
    display = topic_info[topic_info["Topic"] != -1].head(15)
    for _, row in display.iterrows():
        tid   = row["Topic"]
        label = topic_label_map.get(tid, "")
        count = row["Count"]
        words = str(row["Name"]).replace("_", " ").strip()
        print(f"  {tid:<6} {label:<35} {count:>6}  {words[:60]}")

    # ── Build and save monthly prevalence ────────────────────────────────
    print("\nBuilding monthly topic prevalence table...")
    prevalence = build_monthly_prevalence(df, topic_info)

    if not prevalence.empty:
        prevalence.to_csv(PREVALENCE_CSV, index=False)
        print(f"Saved: {PREVALENCE_CSV}")
        print(f"Prevalence table    : {len(prevalence)} rows  "
              f"({prevalence['publish_year'].nunique()} years × "
              f"{prevalence['topic_id'].nunique()} topics)")
    else:
        print("Prevalence table empty — check ARIMAX threshold.")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"BERTopic complete.")
    print(f"  Topics (excl. outliers) : {n_topics}")
    print(f"  Outlier articles        : {n_outliers}")
    print(f"  topic_info.csv          : {TOPIC_INFO_CSV}")
    print(f"  monthly_prevalence.csv  : {PREVALENCE_CSV}")
    print(f"\nNext step: Phase 4 — merge price DBs and run Granger/VAR.")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}")
        sys.exit(1)
    run_bertopic(DB_PATH)
