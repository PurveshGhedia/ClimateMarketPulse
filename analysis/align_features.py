"""
analysis/align_features.py

Aligns NLP-derived features (BERTopic monthly prevalence) with CPI price
data at All-India level (state_code='99') for the 8 focus commodities.

Produces data/processed/analysis_panel.csv — the input to causal_analysis.py

Panel structure:
    rows    : (commodity, year, month)  →  60 months × 8 commodities = 480 rows
    columns : index_value, inflation_yoy, T0_weighted … T{n}_weighted,
              T0_lag1 … lag features, event dummies

Usage (from project root):
    python analysis/align_features.py

Requires:
    data/processed/price_data.db          (from merge_price_db.py)
    data/processed/monthly_topic_prevalence.csv  (from bertopic_model.py)
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PRICE_DB      = os.path.join(PROCESSED_DIR, "price_data.db")
PREVALENCE_CSV = os.path.join(PROCESSED_DIR, "monthly_topic_prevalence.csv")
OUT_CSV       = os.path.join(PROCESSED_DIR, "analysis_panel.csv")

# ── Focus commodities ─────────────────────────────────────────────────────────
# item_code → short label used as commodity column identifier
FOCUS_COMMODITIES = {
    "1.1.07.3.1.01.0": "Tomato",
    "1.1.07.1.1.02.0": "Onion",
    "1.1.07.1.1.01.0": "Potato",
    "1.1.01.1.1.02.X": "Rice",
    "1.1.01.1.1.08.X": "Wheat",
    "1.1.08.1.1.01.0": "Tur_Dal",
    "1.1.05.1.1.01.0": "Mustard_Oil",
    "1.1.06.1.1.11.0": "Mango",
}

# ── Key BERTopic topics for analysis ─────────────────────────────────────────
# Based on BERTopic output — these are the topics with direct climate-commodity signal
KEY_TOPICS = [0, 1, 2, 3, 6]   # kharif/rainfall, veg prices, CPI, climate, COVID
# Label map for readable column names
TOPIC_LABEL_MAP = {
    0: "T0_kharif_rainfall",
    1: "T1_veg_prices",
    2: "T2_cpi_inflation",
    3: "T3_climate_change",
    6: "T6_covid",
}

# ── Lag config ────────────────────────────────────────────────────────────────
LAG_MONTHS = [1, 3, 6]

# ── Event dummies ─────────────────────────────────────────────────────────────
# 2022 heatwave: March–May 2022 (exceptionally early and intense)
# 2023 El Niño: June–October 2023 (IMD declared El Niño in June 2023)
EVENTS = {
    "event_heatwave_2022": lambda y, m: (y == 2022) and (m in [3, 4, 5]),
    "event_elnino_2023":   lambda y, m: (y == 2023) and (m in [6, 7, 8, 9, 10]),
    "event_covid_2020":    lambda y, m: (y == 2020) and (m in [3, 4, 5, 6]),
}


def load_price_data() -> pd.DataFrame:
    """Load All-India price data for focus commodities."""
    if not os.path.exists(PRICE_DB):
        raise FileNotFoundError(
            f"price_data.db not found at {PRICE_DB}\n"
            "Run: python analysis/merge_price_db.py"
        )

    conn = sqlite3.connect(PRICE_DB)
    codes = list(FOCUS_COMMODITIES.keys())
    placeholders = ",".join("?" * len(codes))

    df = pd.read_sql_query(f"""
        SELECT item_code, item_name, year, month, index_value, inflation_yoy
        FROM   cfpi_item
        WHERE  state_code = '99'
          AND  item_code  IN ({placeholders})
          AND  year BETWEEN 2020 AND 2024
        ORDER  BY item_code, year, month
    """, conn, params=codes)
    conn.close()

    # Map item_code → short label
    df["commodity"] = df["item_code"].map(FOCUS_COMMODITIES)
    return df


def load_prevalence() -> pd.DataFrame:
    """Load and pivot monthly topic prevalence to wide format."""
    if not os.path.exists(PREVALENCE_CSV):
        raise FileNotFoundError(
            f"monthly_topic_prevalence.csv not found at {PREVALENCE_CSV}\n"
            "Run: python scraper/nlp/bertopic_model.py"
        )

    prev = pd.read_csv(PREVALENCE_CSV)

    # Filter to key topics only and 2020-2024
    prev = prev[
        (prev["topic_id"].isin(KEY_TOPICS)) &
        (prev["publish_year"].between(2020, 2024))
    ].copy()

    # Pivot: one column per topic
    wide = prev.pivot_table(
        index   = ["publish_year", "publish_month"],
        columns = "topic_id",
        values  = "weighted_count",
        aggfunc = "sum",
    ).reset_index()

    # Rename columns
    wide.columns.name = None
    rename = {"publish_year": "year", "publish_month": "month"}
    for tid in KEY_TOPICS:
        if tid in wide.columns:
            rename[tid] = TOPIC_LABEL_MAP.get(tid, f"T{tid}_weighted")
    wide = wide.rename(columns=rename)

    # Fill missing months with 0 (no articles that month for that topic)
    topic_cols = [TOPIC_LABEL_MAP.get(t, f"T{t}_weighted")
                  for t in KEY_TOPICS if TOPIC_LABEL_MAP.get(t, f"T{t}_weighted") in wide.columns]
    wide[topic_cols] = wide[topic_cols].fillna(0)

    return wide, topic_cols


def build_full_time_index() -> pd.DataFrame:
    """Build a complete 2020-01 → 2024-12 time index (60 months)."""
    records = []
    for y in range(2020, 2025):
        for m in range(1, 13):
            records.append({"year": y, "month": m})
    return pd.DataFrame(records)


def add_lags(df: pd.DataFrame, topic_cols: list[str]) -> pd.DataFrame:
    """Add lag features per commodity × topic."""
    df = df.sort_values(["commodity", "year", "month"]).copy()
    lag_cols_added = []

    for col in topic_cols:
        for lag in LAG_MONTHS:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby("commodity")[col].shift(lag)
            lag_cols_added.append(lag_col)

    # Also lag the price itself for autoregressive component
    for lag in [1, 2, 3]:
        df[f"inflation_lag{lag}"] = df.groupby("commodity")["inflation_yoy"].shift(lag)

    return df, lag_cols_added


def add_event_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary event dummy columns."""
    for event_name, condition in EVENTS.items():
        df[event_name] = df.apply(
            lambda r: 1 if condition(r["year"], r["month"]) else 0, axis=1
        )
    return df


def run_alignment():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("ClimateMarketPulse — Feature Alignment (Phase 4)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading price data (All India, state_code=99)...")
    price_df = load_price_data()
    print(f"  Price rows loaded   : {len(price_df)}")
    print(f"  Commodities found   : {price_df['commodity'].nunique()}")

    found = price_df["commodity"].unique().tolist()
    missing = set(FOCUS_COMMODITIES.values()) - set(found)
    if missing:
        print(f"  [WARN] Missing commodities in price DB: {missing}")

    print("\nLoading monthly topic prevalence...")
    prevalence_df, topic_cols = load_prevalence()
    print(f"  Prevalence rows     : {len(prevalence_df)}")
    print(f"  Topic columns       : {topic_cols}")

    # ── Build complete time spine ─────────────────────────────────────────
    time_index = build_full_time_index()

    # ── Merge prevalence onto time spine (fill gaps with 0) ───────────────
    time_prev = time_index.merge(prevalence_df, on=["year", "month"], how="left")
    time_prev[topic_cols] = time_prev[topic_cols].fillna(0)

    # ── Cross-join with commodities → panel ───────────────────────────────
    commodities = price_df[["commodity"]].drop_duplicates()
    # Cartesian product: each commodity × each month
    panel_spine = time_index.assign(key=1).merge(
        commodities.assign(key=1), on="key"
    ).drop(columns="key")

    # ── Merge price data ──────────────────────────────────────────────────
    panel = panel_spine.merge(
        price_df[["commodity", "year", "month", "index_value", "inflation_yoy"]],
        on=["commodity", "year", "month"],
        how="left",
    )

    # ── Merge topic prevalence ─────────────────────────────────────────────
    panel = panel.merge(time_prev, on=["year", "month"], how="left")

    # ── Compute inflation_yoy if missing (from index_value) ───────────────
    # Some months may have index_value but null inflation_yoy
    panel = panel.sort_values(["commodity", "year", "month"]).reset_index(drop=True)
    mask = panel["inflation_yoy"].isna() & panel["index_value"].notna()
    if mask.any():
        panel["index_lag12"] = panel.groupby("commodity")["index_value"].shift(12)
        computed = ((panel["index_value"] - panel["index_lag12"])
                    / panel["index_lag12"] * 100)
        panel.loc[mask, "inflation_yoy"] = computed[mask]
        panel = panel.drop(columns=["index_lag12"])

    # ── Add lag features ──────────────────────────────────────────────────
    panel, lag_cols = add_lags(panel, topic_cols)

    # ── Add event dummies ─────────────────────────────────────────────────
    panel = add_event_dummies(panel)

    # ── Add year_month string for readability ─────────────────────────────
    panel["year_month"] = (panel["year"].astype(str) + "-" +
                           panel["month"].apply(lambda m: f"{m:02d}"))

    # ── Reorder columns ───────────────────────────────────────────────────
    base_cols = ["commodity", "year", "month", "year_month",
                 "index_value", "inflation_yoy"]
    event_cols = list(EVENTS.keys())
    infl_lag_cols = [f"inflation_lag{i}" for i in [1, 2, 3]]
    col_order = (base_cols + topic_cols + lag_cols +
                 infl_lag_cols + event_cols)
    # Keep any extra columns at the end
    extra = [c for c in panel.columns if c not in col_order]
    panel = panel[col_order + extra]

    # ── Save ──────────────────────────────────────────────────────────────
    panel.to_csv(OUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nPanel built successfully.")
    print(f"  Shape               : {panel.shape}")
    print(f"  Commodities         : {panel['commodity'].nunique()}")
    print(f"  Time range          : {panel['year_month'].min()} → {panel['year_month'].max()}")

    null_pct = panel["inflation_yoy"].isna().mean() * 100
    print(f"  inflation_yoy nulls : {null_pct:.1f}%")

    print(f"\n  Columns:")
    for col in panel.columns:
        nulls = panel[col].isna().sum()
        print(f"    {col:<35} nulls={nulls}")

    print(f"\nSaved: {OUT_CSV}")
    print("=" * 60)
    print("\nNext: python analysis/causal_analysis.py")


if __name__ == "__main__":
    run_alignment()
