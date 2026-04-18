"""
ClimateMarketPulse — Sentiment Scoring (Step 6)
Model  : ProsusAI/finbert  (positive / negative / neutral)
Input  : articles.db  WHERE relevance_score >= 0.30 AND is_duplicate = 0
Text   : headline + first 300 chars of full_text  (truncated to 512 tokens)
Output : sentiment_label + sentiment_score written back to articles.db
         data/processed/monthly_sentiment.csv

Usage:
    cd ClimateMarketPulse
    conda activate nlp
    python scraper/nlp/sentiment_score.py
    python scraper/nlp/sentiment_score.py --batch-size 16   # if MPS runs OOM
    python scraper/nlp/sentiment_score.py --dry-run          # score 50 articles, no DB write
"""

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.parent
ARTICLES_DB = ROOT / "data" / "articles.db"
OUT_CSV     = ROOT / "data" / "processed" / "monthly_sentiment.csv"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"

# FinBERT label order as returned by the model head
# (verified from ProsusAI/finbert config: 0=positive, 1=negative, 2=neutral)
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_columns(con: sqlite3.Connection) -> None:
    """Add sentiment columns if they don't exist — safe to call on re-runs."""
    cur = con.cursor()
    for col, dtype in [("sentiment_label", "TEXT"), ("sentiment_score", "REAL")]:
        try:
            cur.execute(f"ALTER TABLE articles ADD COLUMN {col} {dtype}")
            print(f"  Added column: {col}")
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()


def load_unscored(con: sqlite3.Connection, dry_run: bool) -> pd.DataFrame:
    """
    Load articles that need scoring:
      - relevance_score >= 0.30
      - is_duplicate = 0
      - sentiment_label IS NULL  (resume support)
    """
    limit = "LIMIT 50" if dry_run else ""
    df = pd.read_sql_query(
        f"""SELECT article_id, headline, full_text
            FROM articles
            WHERE relevance_score >= 0.30
              AND is_duplicate = 0
              AND sentiment_label IS NULL
            {limit}""",
        con,
    )
    return df


def write_results(con: sqlite3.Connection, results: list[dict]) -> None:
    """Batch-update sentiment_label and sentiment_score by article_id."""
    cur = con.cursor()
    cur.executemany(
        "UPDATE articles SET sentiment_label=?, sentiment_score=? WHERE article_id=?",
        [(r["label"], r["score"], r["article_id"]) for r in results],
    )
    con.commit()


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def build_text(headline, full_text) -> str:
    """
    headline + first 300 chars of full_text.
    Falls back gracefully if either field is NULL.
    """
    parts = []
    if headline and str(headline).strip():
        parts.append(str(headline).strip())
    if full_text and str(full_text).strip():
        parts.append(str(full_text).strip()[:300])
    return " ".join(parts) if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> list[tuple[str, float]]:
    """
    Returns list of (label, confidence_score) for each text.
    confidence_score = softmax probability of the predicted class.
    """
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits          # (batch, 3)

    probs      = F.softmax(logits, dim=-1)         # (batch, 3)
    pred_ids   = probs.argmax(dim=-1).cpu().tolist()
    pred_probs = probs.max(dim=-1).values.cpu().tolist()

    return [(LABEL_MAP[pid], round(float(pp), 4))
            for pid, pp in zip(pred_ids, pred_probs)]


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE → monthly_sentiment.csv
# ══════════════════════════════════════════════════════════════════════════════

def build_monthly_aggregate(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Pull all scored articles and aggregate by year × month:
      - mean_sentiment_score
      - positive_count, negative_count, neutral_count
      - total_count
      - net_sentiment = (positive - negative) / total  ← useful as ARIMAX feature
    Only includes months where at least 1 article was scored.
    """
    df = pd.read_sql_query(
        """SELECT publish_year AS year, publish_month AS month,
                  sentiment_label, sentiment_score
           FROM articles
           WHERE sentiment_label IS NOT NULL
             AND is_duplicate = 0
             AND relevance_score >= 0.30
             AND publish_year IS NOT NULL
             AND publish_month IS NOT NULL""",
        con,
    )

    if df.empty:
        print("  No scored articles found — aggregate CSV not written.")
        return df

    # numeric sentiment for mean: positive=1, neutral=0, negative=-1
    polarity_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["polarity"] = df["sentiment_label"].map(polarity_map)

    agg = (
        df.groupby(["year", "month"])
        .agg(
            mean_sentiment_score =("sentiment_score", "mean"),
            mean_polarity        =("polarity", "mean"),
            positive_count       =("sentiment_label", lambda x: (x == "positive").sum()),
            negative_count       =("sentiment_label", lambda x: (x == "negative").sum()),
            neutral_count        =("sentiment_label", lambda x: (x == "neutral").sum()),
            total_count          =("sentiment_label", "count"),
        )
        .reset_index()
    )

    agg["net_sentiment"] = (
        (agg["positive_count"] - agg["negative_count"]) / agg["total_count"]
    ).round(4)

    agg = agg.sort_values(["year", "month"]).reset_index(drop=True)

    # round floats
    for col in ["mean_sentiment_score", "mean_polarity"]:
        agg[col] = agg[col].round(4)

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FinBERT sentiment scoring")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default 32; reduce to 16 if OOM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score first 50 unscored articles only; do not write to DB")
    args = parser.parse_args()

    print("=" * 65)
    print("ClimateMarketPulse — Sentiment Scoring (ProsusAI/finbert)")
    print("=" * 65)

    # ── device ────────────────────────────────────────────────────────────────
    device = get_device()
    print(f"Device      : {device}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Dry run     : {args.dry_run}")

    # ── DB connection ─────────────────────────────────────────────────────────
    con = sqlite3.connect(ARTICLES_DB)
    ensure_columns(con)

    df = load_unscored(con, dry_run=args.dry_run)
    total = len(df)
    if total == 0:
        print("\nAll articles already scored. Nothing to do.")
        con.close()
        return
    print(f"Articles to score: {total}\n")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model     = model.to(device)
    model.eval()
    print("Model loaded.\n")

    # ── inference loop ────────────────────────────────────────────────────────
    results     = []
    n_batches   = (total + args.batch_size - 1) // args.batch_size
    t_start     = time.time()

    for batch_i in range(n_batches):
        batch_df = df.iloc[batch_i * args.batch_size : (batch_i + 1) * args.batch_size]
        texts    = [build_text(row.headline, row.full_text)
                    for row in batch_df.itertuples()]

        preds = score_batch(texts, tokenizer, model, device)

        for (article_id,), (label, score) in zip(
            batch_df[["article_id"]].itertuples(index=False), preds
        ):
            results.append({"article_id": article_id,
                             "label": label, "score": score})

        # progress
        done     = min((batch_i + 1) * args.batch_size, total)
        elapsed  = time.time() - t_start
        eta      = (elapsed / done) * (total - done) if done > 0 else 0
        print(f"  [{done:>4}/{total}]  last batch: {label} {score:.3f}"
              f"  |  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

        # write every 10 batches to avoid losing progress on crash
        if not args.dry_run and (batch_i + 1) % 10 == 0:
            write_results(con, results)
            results = []

    # final flush
    if not args.dry_run and results:
        write_results(con, results)
        print(f"\nWrote {total} sentiment scores to articles.db")

    # ── label distribution ────────────────────────────────────────────────────
    all_results = results if args.dry_run else []
    if args.dry_run:
        labels = [r["label"] for r in all_results]
    else:
        label_df = pd.read_sql_query(
            """SELECT sentiment_label FROM articles
               WHERE sentiment_label IS NOT NULL
                 AND is_duplicate = 0
                 AND relevance_score >= 0.30""",
            con,
        )
        labels = label_df["sentiment_label"].tolist()

    if labels:
        from collections import Counter
        dist = Counter(labels)
        total_scored = sum(dist.values())
        print("\nLabel distribution (all scored articles):")
        for lbl in ["positive", "negative", "neutral"]:
            n = dist.get(lbl, 0)
            print(f"  {lbl:<10} {n:>5}  ({n/total_scored*100:.1f}%)")

    # ── monthly aggregate ─────────────────────────────────────────────────────
    if not args.dry_run:
        print("\nBuilding monthly sentiment aggregate...")
        agg = build_monthly_aggregate(con)
        if not agg.empty:
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            agg.to_csv(OUT_CSV, index=False)
            print(f"Saved: {OUT_CSV}  ({len(agg)} rows)")
            print("\nSample (first 5 rows):")
            print(agg.head().to_string(index=False))

    con.close()
    total_time = time.time() - t_start
    print(f"\nDone. Total time: {total_time:.0f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
