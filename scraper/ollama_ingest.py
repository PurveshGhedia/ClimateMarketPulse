"""
scraper/ollama_ingest.py

Ingestion pipeline for separately scraped articles into articles.db.
Uses pure Python keyword matching (keyword_prefilter / pib_filter from
keywords.py) — no LLM required. Fast, deterministic, and consistent with
the filtering applied during the original scraping phase.

Accepts a folder of JSON or CSV files. Each file must contain articles
with at minimum: url, full_text. All other fields are optional but
recommended: headline, publish_date, outlet, author, source_type.

Supported input formats
-----------------------
JSON  : Either a JSON array  [ {...}, {...} ]
        or JSONL (one JSON object per line)
CSV   : Standard CSV with a header row. Column names must match field names.

Usage (from project root, nlp conda env):
    python scraper/ollama_ingest.py --input_dir path/to/articles/
    python scraper/ollama_ingest.py --input_dir path/to/articles/ --source_type pib
    python scraper/ollama_ingest.py --input_dir path/to/articles/ --dry_run

Arguments
---------
--input_dir   : Folder containing .json / .jsonl / .csv files  [required]
--source_type : Override source_type for ALL ingested articles  [default: direct]
--outlet      : Override outlet for ALL ingested articles       [default: per-file or "unknown"]
--dry_run     : Parse + filter but do NOT write to DB           [flag]
--verbose     : Print every article decision (pass/fail/skip)   [flag]
"""

from keywords import keyword_prefilter, pib_filter
from db import get_conn, insert_article, make_content_hash
import os
import sys
import json
import csv
import argparse
import sqlite3
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# This script lives at scraper/ollama_ingest.py
# db.py and keywords.py are in the same scraper/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, SCRIPT_DIR)   # so we can import db and keywords directly


DB_PATH = os.path.join(PROJECT_ROOT, "data", "articles.db")

# ── Required and optional fields ─────────────────────────────────────────────
REQUIRED_FIELDS = {"url", "full_text"}
OPTIONAL_FIELDS = {
    "headline", "publish_date", "author",
    "outlet", "source_type", "archive_url",
    "crawl_id", "extraction_method",
}


# ── File readers ──────────────────────────────────────────────────────────────
def read_json_file(path: Path) -> list[dict]:
    """Reads a JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    # Try JSON array first
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Fall back to JSONL
    records = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(
                f"  [WARN] Skipping malformed JSON on line {i} in {path.name}: {e}")
    return records


def read_csv_file(path: Path) -> list[dict]:
    """Reads a CSV file with header row."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from all values
            records.append({k: (v.strip() if isinstance(v, str) else v)
                            for k, v in row.items()})
    return records


def load_all_files(input_dir: str) -> list[tuple[str, dict]]:
    """
    Walks input_dir, reads all .json / .jsonl / .csv files.
    Returns list of (filename, record_dict) tuples.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"ERROR: {input_dir} is not a valid directory.")
        sys.exit(1)

    all_records: list[tuple[str, dict]] = []
    suffixes = {".json", ".jsonl", ".csv"}

    files = sorted(input_path.glob("*"))
    eligible = [f for f in files if f.suffix.lower() in suffixes]

    if not eligible:
        print(f"No .json / .jsonl / .csv files found in {input_dir}")
        sys.exit(0)

    for fpath in eligible:
        if fpath.suffix.lower() in {".json", ".jsonl"}:
            records = read_json_file(fpath)
        else:
            records = read_csv_file(fpath)

        for rec in records:
            all_records.append((fpath.name, rec))

    print(f"Files scanned       : {len(eligible)}")
    print(f"Total candidates    : {len(all_records)}")
    return all_records


# ── Dedup check ───────────────────────────────────────────────────────────────
def load_existing_hashes(conn: sqlite3.Connection) -> set[str]:
    """Load all content_hashes currently in DB into a set for O(1) lookup."""
    rows = conn.execute("SELECT content_hash FROM articles").fetchall()
    return {row[0] for row in rows}


# ── Keyword filtering ─────────────────────────────────────────────────────────
def apply_keyword_filter(record: dict, source_type: str) -> tuple[bool, list, list]:
    """
    Apply the appropriate keyword filter based on source_type.
    PIB articles use pib_filter (relaxed: ≥1 OR ≥1).
    All others use keyword_prefilter (strict: ≥2 AND ≥2).
    Returns (is_relevant, climate_hits, commodity_hits).
    """
    # Build text to filter: headline + full_text
    text = " ".join(filter(None, [
        record.get("headline", "") or "",
        record.get("full_text", "") or "",
    ]))

    if source_type == "pib":
        return pib_filter(text)
    else:
        return keyword_prefilter(text)


# ── Record builder ────────────────────────────────────────────────────────────
def build_record(
    raw: dict,
    source_type_override: str,
    outlet_override: str | None,
    climate_hits: list,
    commodity_hits: list,
) -> dict:
    """
    Build the record dict for insert_article().
    Never set article_id, content_hash, word_count, scraped_at —
    insert_article() auto-derives all of these.
    """
    record = {
        "url":          raw.get("url", "").strip(),
        "source_type":  raw.get("source_type", source_type_override) or source_type_override,
        "outlet":       outlet_override or raw.get("outlet", "unknown") or "unknown",
        "headline":     raw.get("headline") or None,
        "author":       raw.get("author") or None,
        "full_text":    raw.get("full_text") or None,
        "publish_date": raw.get("publish_date") or None,
        "archive_url":  raw.get("archive_url") or None,
        "crawl_id":     raw.get("crawl_id") or None,
        "extraction_method": raw.get("extraction_method") or "manual_ingest",

        # Populate keyword columns from filter results
        "climate_terms_found":   "|".join(climate_hits) if climate_hits else None,
        "commodity_terms_found": "|".join(commodity_hits) if commodity_hits else None,

        # NLP columns — populated later by ner_states.py / relevance_score.py etc.
        "states_mentioned": None,
        "relevance_score":  None,
        "is_duplicate":     0,
    }
    return record


# ── Main ──────────────────────────────────────────────────────────────────────
def run_ingestion(
    input_dir:           str,
    source_type_default: str = "direct",
    outlet_override:     str | None = None,
    dry_run:             bool = False,
    verbose:             bool = False,
) -> None:

    print("=" * 55)
    print("ClimateMarketPulse — Article Ingestion Pipeline")
    print("=" * 55)
    print(f"Input dir           : {input_dir}")
    print(f"DB path             : {DB_PATH}")
    print(f"Default source_type : {source_type_default}")
    print(f"Dry run             : {dry_run}")
    print("-" * 55)

    # Load all candidate records from files
    all_records = load_all_files(input_dir)

    # Connect to DB and load existing hashes
    conn = None if dry_run else sqlite3.connect(DB_PATH)
    existing_hashes = load_existing_hashes(conn) if conn else set()
    print(f"Existing DB hashes  : {len(existing_hashes)}")
    print("-" * 55)

    # Counters
    n_total = len(all_records)
    n_missing_url = 0
    n_missing_text = 0
    n_dedup_skipped = 0
    n_failed_filter = 0
    n_inserted = 0
    n_insert_ignore = 0  # INSERT OR IGNORE fired (article_id collision)

    for fname, raw in all_records:

        # ── Validate required fields ──────────────────────────────────────
        url = (raw.get("url") or "").strip()
        full_text = (raw.get("full_text") or "").strip()

        if not url:
            n_missing_url += 1
            if verbose:
                print(f"  [SKIP-NO-URL]  {fname}")
            continue

        if not full_text:
            n_missing_text += 1
            if verbose:
                print(f"  [SKIP-NO-TEXT] {url[:80]}")
            continue

        # ── Dedup check (content_hash) ────────────────────────────────────
        content_hash = make_content_hash(full_text)
        if content_hash in existing_hashes:
            n_dedup_skipped += 1
            if verbose:
                print(f"  [DEDUP]        {url[:80]}")
            continue

        # ── Keyword filter ────────────────────────────────────────────────
        # Determine effective source_type for filter selection
        effective_source_type = raw.get("source_type") or source_type_default

        is_relevant, climate_hits, commodity_hits = apply_keyword_filter(
            raw, effective_source_type
        )

        if not is_relevant:
            n_failed_filter += 1
            if verbose:
                print(
                    f"  [FILTERED]     {url[:80]}\n"
                    f"              climate={len(climate_hits)} "
                    f"commodity={len(commodity_hits)}"
                )
            continue

        # ── Build record and insert ───────────────────────────────────────
        record = build_record(
            raw,
            source_type_override=source_type_default,
            outlet_override=outlet_override,
            climate_hits=climate_hits,
            commodity_hits=commodity_hits,
        )

        if dry_run:
            n_inserted += 1  # count as "would insert"
            if verbose:
                print(
                    f"  [DRY-RUN OK]   {url[:80]}\n"
                    f"              climate={climate_hits[:3]} "
                    f"commodity={commodity_hits[:3]}"
                )
        else:
            inserted = insert_article(conn, record)
            if inserted:
                n_inserted += 1
                # Add to local set so intra-batch duplicates are caught
                existing_hashes.add(content_hash)
                if verbose:
                    print(f"  [INSERTED]     {url[:80]}")
            else:
                n_insert_ignore += 1
                if verbose:
                    print(
                        f"  [IGNORE]       {url[:80]}  (article_id collision)")

    if conn:
        conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Ingestion complete.")
    print(f"  Total candidates    : {n_total}")
    print(f"  Skipped (no URL)    : {n_missing_url}")
    print(f"  Skipped (no text)   : {n_missing_text}")
    print(f"  Dedup skipped       : {n_dedup_skipped}")
    print(f"  Failed keyword filter: {n_failed_filter}")
    if not dry_run:
        print(f"  Inserted            : {n_inserted}")
        print(
            f"  INSERT OR IGNORE    : {n_insert_ignore}  (article_id collision)")
    else:
        print(f"  Would insert (dry)  : {n_inserted}")
    print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest separately scraped articles into articles.db"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing .json / .jsonl / .csv article files",
    )
    parser.add_argument(
        "--source_type",
        default="direct",
        help="source_type value for all ingested articles (default: direct)",
    )
    parser.add_argument(
        "--outlet",
        default=None,
        help="Override outlet for all articles (default: taken from file or 'unknown')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and filter but do NOT write to DB",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every article decision",
    )

    args = parser.parse_args()

    run_ingestion(
        input_dir=args.input_dir,
        source_type_default=args.source_type,
        outlet_override=args.outlet,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
