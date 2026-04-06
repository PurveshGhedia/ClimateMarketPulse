"""
Export articles from SQLite to CSV and print corpus statistics.
Run this anytime to check progress or inspect data in Excel/Sheets.

Usage:
    python export_csv.py
"""

import sqlite3
import csv
import os
from db import DB_PATH, get_conn


def print_stats(conn):
    print("\n" + "="*60)
    print("  CORPUS STATISTICS")
    print("="*60)

    # Per-outlet breakdown
    print("\n  Articles by outlet:")
    print(f"  {'Outlet':<22} {'Count':>6}  {'Earliest':<12}  {'Latest'}")
    print(f"  {'─'*22} {'─'*6}  {'─'*12}  {'─'*12}")
    for row in conn.execute("""
        SELECT outlet,
               COUNT(*) as n,
               MIN(publish_date) as earliest,
               MAX(publish_date) as latest
        FROM articles
        GROUP BY outlet
        ORDER BY n DESC
    """):
        print(f"  {row['outlet']:<22} {row['n']:>6}  "
              f"{(row['earliest'] or 'unknown'):<12}  {row['latest'] or 'unknown'}")

    # Per-year breakdown
    print("\n  Articles by year:")
    print(f"  {'Year':<8} {'Count':>6}")
    print(f"  {'─'*8} {'─'*6}")
    for row in conn.execute("""
        SELECT publish_year, COUNT(*) as n
        FROM articles
        WHERE publish_year IS NOT NULL
        GROUP BY publish_year
        ORDER BY publish_year
    """):
        print(f"  {row['publish_year']:<8} {row['n']:>6}")

    # Per source_type
    print("\n  Articles by source:")
    for row in conn.execute("""
        SELECT source_type, COUNT(*) as n
        FROM articles GROUP BY source_type ORDER BY n DESC
    """):
        print(f"  {row['source_type']:<20} {row['n']:>6}")

    # Overall totals
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    dupes = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE is_duplicate=1"
    ).fetchone()[0]
    short = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE word_count < 80"
    ).fetchone()[0]

    print(f"\n  Total rows:        {total:>6}")
    print(f"  Duplicates:        {dupes:>6}")
    print(f"  Too short (<80w):  {short:>6}")
    print(f"  Usable articles:   {total - dupes - short:>6}")
    print("="*60 + "\n")


def export_csv(conn, min_words=80, remove_duplicates=True):
    """Export clean articles to CSV."""
    query = "SELECT * FROM articles WHERE word_count >= ?"
    params = [min_words]
    if remove_duplicates:
        query += " AND is_duplicate = 0"
    query += " ORDER BY publish_date"

    rows = conn.execute(query, params).fetchall()

    if not rows:
        print("  No articles to export yet.")
        return None

    out_path = os.path.join(os.path.dirname(DB_PATH), "articles_export.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])

    print(f"  Exported {len(rows)} articles → {out_path}")
    return out_path


def export_summary_csv(conn):
    """Export a lightweight summary CSV (no full_text) for quick inspection."""
    rows = conn.execute("""
        SELECT article_id, outlet, source_type, publish_date,
               publish_year, publish_month, headline,
               word_count, climate_terms_found, commodity_terms_found,
               states_mentioned, relevance_score, is_duplicate
        FROM articles
        ORDER BY publish_date
    """).fetchall()

    if not rows:
        return

    out_path = os.path.join(os.path.dirname(DB_PATH), "articles_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])

    print(f"  Summary CSV (no text) → {out_path}")


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("No database found yet. Run pib_scraper.py or wayback_scraper.py first.")
    else:
        conn = get_conn()
        print_stats(conn)
        export_csv(conn)
        export_summary_csv(conn)
        conn.close()
