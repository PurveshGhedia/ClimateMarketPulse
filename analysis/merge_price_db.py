"""
analysis/merge_price_db.py

Merges per-year price DBs (price_data_2020.db … price_data_2024.db)
into a single price_data.db for analysis.

Usage (from project root, nlp conda env):
    python analysis/merge_price_db.py

Output:
    data/processed/price_data.db   — unified DB with cfpi_item table
    data/processed/price_long.csv  — full long-format CSV (all items, all states)
"""

import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DB        = os.path.join(PROCESSED_DIR, "price_data.db")
OUT_CSV       = os.path.join(PROCESSED_DIR, "price_long.csv")

YEARS = [2020, 2021, 2022, 2023, 2024]

CREATE_CFPI = """
CREATE TABLE IF NOT EXISTS cfpi_item (
    item_code      TEXT,
    item_name      TEXT,
    subgroup       TEXT,
    state_code     TEXT,
    year           INTEGER,
    month          INTEGER,
    index_value    REAL,
    inflation_yoy  REAL,
    PRIMARY KEY (item_code, state_code, year, month)
);
CREATE INDEX IF NOT EXISTS idx_price_ym
    ON cfpi_item(year, month);
CREATE INDEX IF NOT EXISTS idx_price_state
    ON cfpi_item(state_code, year, month);
CREATE INDEX IF NOT EXISTS idx_price_item
    ON cfpi_item(item_name, state_code, year, month);
"""

def merge():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 55)
    print("ClimateMarketPulse — Price DB Merge")
    print("=" * 55)

    # Init output DB
    out_conn = sqlite3.connect(OUT_DB)
    out_conn.executescript(CREATE_CFPI)
    out_conn.commit()

    total_inserted = 0
    total_skipped  = 0

    all_frames = []

    for year in YEARS:
        db_path = os.path.join(RAW_DIR, f"price_data_{year}.db")
        if not os.path.exists(db_path):
            print(f"  [WARN] Missing: {db_path} — skipping year {year}")
            continue

        src_conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT item_code, item_name, subgroup, state_code, "
            "       year, month, index_value, inflation_yoy "
            "FROM cfpi_item",
            src_conn
        )
        src_conn.close()

        if df.empty:
            print(f"  [WARN] {year}: DB is empty")
            continue

        rows = df.shape[0]
        print(f"  {year}: {rows:>7} rows loaded from price_data_{year}.db")

        # INSERT OR IGNORE into unified DB
        inserted = 0
        for _, row in df.iterrows():
            cursor = out_conn.execute(
                "INSERT OR IGNORE INTO cfpi_item "
                "(item_code, item_name, subgroup, state_code, year, month, "
                " index_value, inflation_yoy) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (row.item_code, row.item_name, row.subgroup, str(row.state_code),
                 int(row.year), int(row.month),
                 row.index_value if pd.notna(row.index_value) else None,
                 row.inflation_yoy if pd.notna(row.inflation_yoy) else None)
            )
            inserted += cursor.rowcount

        out_conn.commit()
        total_inserted += inserted
        total_skipped  += (rows - inserted)
        all_frames.append(df)
        print(f"         {inserted:>7} inserted, {rows-inserted} skipped (duplicates)")

    # ── Validation ────────────────────────────────────────────────────────
    total_in_db = out_conn.execute("SELECT COUNT(*) FROM cfpi_item").fetchone()[0]
    year_check  = out_conn.execute(
        "SELECT year, COUNT(*) FROM cfpi_item GROUP BY year ORDER BY year"
    ).fetchall()

    print(f"\nMerge complete.")
    print(f"  Total rows in merged DB : {total_in_db}")
    print(f"  Year distribution:")
    for yr, cnt in year_check:
        print(f"    {yr}: {cnt:>8} rows")

    # ── State coverage check ──────────────────────────────────────────────
    state_count = out_conn.execute(
        "SELECT COUNT(DISTINCT state_code) FROM cfpi_item"
    ).fetchone()[0]
    item_count = out_conn.execute(
        "SELECT COUNT(DISTINCT item_code) FROM cfpi_item"
    ).fetchone()[0]
    print(f"  Distinct states         : {state_count}")
    print(f"  Distinct items          : {item_count}")

    # ── Export long CSV ───────────────────────────────────────────────────
    if all_frames:
        full_df = pd.concat(all_frames, ignore_index=True)
        full_df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved: {OUT_CSV}  ({len(full_df)} rows)")

    out_conn.close()
    print(f"Saved: {OUT_DB}")
    print("=" * 55)
    print("\nNext: python analysis/align_features.py")


if __name__ == "__main__":
    merge()
