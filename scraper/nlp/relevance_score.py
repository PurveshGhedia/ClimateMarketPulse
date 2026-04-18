"""
scraper/nlp/relevance_score.py

Relevance scoring pipeline — Step 4 of the NLP pipeline.
Computes cosine similarity between each article and a fixed query using
sentence-transformers (all-MiniLM-L6-v2), then writes scores back to
the `relevance_score` column in articles.db.

Query: "climate event impact on food commodity prices India"
Threshold (for downstream use): 0.35

On Mac M-series, sentence-transformers automatically uses MPS acceleration.

Usage (from project root, nlp conda env):
    python scraper/nlp/relevance_score.py

Requirements:
    pip install sentence-transformers

After running, check distribution with:
    SELECT
        ROUND(relevance_score, 1) AS bucket,
        COUNT(*) AS n
    FROM articles
    WHERE is_duplicate = 0
    GROUP BY bucket
    ORDER BY bucket;
"""

import os
import sys
import sqlite3
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH      = os.path.join(PROJECT_ROOT, "data", "articles.db")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "all-MiniLM-L6-v2"
RELEVANCE_QUERY = "climate event impact on food commodity prices India"
THRESHOLD       = 0.35        # used only for reporting; not applied here
ENCODE_BATCH    = 256         # articles encoded per model.encode() call
DB_COMMIT_EVERY = 500         # rows per DB commit
# Max chars of article text fed to model. MiniLM max is 256 tokens ≈ 1200 chars.
# Using headline + first 1000 chars of body gives good signal without truncation loss.
MAX_TEXT_CHARS  = 1100


def load_articles(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """
    Load all non-duplicate articles that haven't been scored yet.
    Returns list of (article_id, text) tuples.
    Text = headline + first MAX_TEXT_CHARS chars of full_text.
    """
    rows = conn.execute("""
        SELECT article_id, headline, full_text
        FROM   articles
        WHERE  is_duplicate = 0
          AND  relevance_score IS NULL
    """).fetchall()
    return rows


def build_text(headline: str | None, full_text: str | None) -> str:
    headline  = (headline  or "").strip()
    full_text = (full_text or "").strip()
    combined  = f"{headline} {full_text}".strip()
    return combined[:MAX_TEXT_CHARS]


def run_scoring(db_path: str) -> None:
    print("=" * 55)
    print("ClimateMarketPulse — Relevance Scoring (Step 4)")
    print("=" * 55)
    print(f"Model               : {MODEL_NAME}")
    print(f"Query               : {RELEVANCE_QUERY}")
    print(f"Threshold (report)  : {THRESHOLD}")
    print(f"DB path             : {db_path}")
    print("-" * 55)

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading sentence-transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Device: {model.device}")

    # ── Encode query ──────────────────────────────────────────────────────
    query_embedding = model.encode(
        RELEVANCE_QUERY,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # ── Load articles ─────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = load_articles(conn)

    total = len(rows)
    if total == 0:
        print("No unscored articles found. relevance_score already populated.")
        conn.close()
        return

    print(f"Articles to score   : {total}")
    print("-" * 55)

    # ── Score in batches ──────────────────────────────────────────────────
    article_ids = [r["article_id"] for r in rows]
    texts       = [build_text(r["headline"], r["full_text"]) for r in rows]

    all_scores: list[float] = []
    processed = 0

    def iter_chunks(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    for chunk_texts in iter_chunks(texts, ENCODE_BATCH):
        embeddings = model.encode(
            chunk_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=ENCODE_BATCH,
        )
        # cos_sim returns a (n, 1) tensor when query is 1D; flatten to list
        scores = cos_sim(query_embedding, embeddings)[0].tolist()
        all_scores.extend(scores)
        processed += len(chunk_texts)

        if processed % 1000 == 0:
            pct = processed / total * 100
            print(f"  Encoded: {processed:>5}/{total}  ({pct:.1f}%)")

    print(f"  Encoded: {total}/{total}  (100.0%)")

    # ── Write scores back to DB ───────────────────────────────────────────
    print("Writing scores to DB...")
    updates = list(zip(all_scores, article_ids))

    for i in range(0, len(updates), DB_COMMIT_EVERY):
        batch = updates[i : i + DB_COMMIT_EVERY]
        conn.executemany(
            "UPDATE articles SET relevance_score = ? WHERE article_id = ?",
            batch,
        )
        conn.commit()

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────
    scores_arr = np.array(all_scores)
    above      = int(np.sum(scores_arr >= THRESHOLD))
    below      = total - above

    print(f"\n{'=' * 55}")
    print(f"Scoring complete.")
    print(f"  Total scored        : {total}")
    print(f"  Above threshold     : {above}  ({above/total*100:.1f}%)  [score >= {THRESHOLD}]")
    print(f"  Below threshold     : {below}  ({below/total*100:.1f}%)")
    print(f"\nScore distribution:")
    print(f"  {'Bucket':<10} {'Count':>6}")
    print(f"  {'-'*10}  {'-'*6}")

    buckets: Counter = Counter()
    for s in scores_arr:
        bucket = f"{s:.1f}"
        buckets[bucket] += 1

    for bucket in sorted(buckets.keys()):
        bar = "█" * (buckets[bucket] // 20)
        print(f"  {bucket:<10} {buckets[bucket]:>6}  {bar}")

    print(f"\n  Min   : {scores_arr.min():.4f}")
    print(f"  Max   : {scores_arr.max():.4f}")
    print(f"  Mean  : {scores_arr.mean():.4f}")
    print(f"  Median: {float(np.median(scores_arr)):.4f}")
    print(f"\nBERTopic will run on {above} articles with relevance_score >= {THRESHOLD}.")


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}")
        sys.exit(1)
    run_scoring(DB_PATH)
