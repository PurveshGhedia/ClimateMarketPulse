"""
ClimateMarketPulse — Demo Dashboard
NLP Project · April 2026

Usage:
    cd ClimateMarketPulse
    pip install gradio plotly pandas
    python app/dashboard.py
"""

import sqlite3
import re
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ARTICLES_DB = ROOT / "data" / "articles.db"
PRICE_DB = ROOT / "data" / "processed" / "price_data.db"
GRANGER_CSV = ROOT / "data" / "processed" / "results" / "granger_results.csv"
EVENT_CSV = ROOT / "data" / "processed" / "results" / "event_study.csv"
ARIMAX_TXT = ROOT / "data" / "processed" / "results" / "arimax_summary.txt"
TOPIC_CSV = ROOT / "data" / "processed" / "topic_info.csv"
PANEL_CSV = ROOT / "data" / "processed" / "analysis_panel.csv"
SENTIMENT_CSV = ROOT / "data" / "processed" / "monthly_sentiment.csv"

# ── Colours ───────────────────────────────────────────────────────────────────
TEAL = "#1D9E75"
TEAL_L = "#5DCAA5"
AMBER = "#EF9F27"
RED = "#E24B4A"
BLUE = "#378ADD"
GRAY = "#888780"
LIGHT = "#F1EFE8"
BG = "#FAFAF8"

# Focus commodity → item_code (All India state_code='99')
COMMODITY_CODES = {
    "Tomato":      "1.1.07.3.1.01.0",
    "Onion":       "1.1.07.1.1.02.0",
    "Potato":      "1.1.07.1.1.01.0",
    "Rice":        "1.1.01.1.1.02.X",
    "Wheat":       "1.1.01.1.1.08.X",
    "Tur_Dal":     "1.1.08.1.1.01.0",
    "Mustard_Oil": "1.1.05.1.1.01.0",
    "Mango":       "1.1.06.1.1.11.0",
}

PLOT_H = 460

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════


def _qdb(db_path: Path, sql: str, params=()) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    return df


def load_corpus_stats() -> dict:
    stats = {}

    stats["sources"] = _qdb(
        ARTICLES_DB,
        "SELECT source_type, COUNT(*) AS n FROM articles GROUP BY source_type"
    )
    stats["total"] = int(_qdb(
        ARTICLES_DB, "SELECT COUNT(*) AS n FROM articles WHERE is_duplicate=0"
    )["n"].iloc[0])
    stats["topic_modeled"] = int(_qdb(
        ARTICLES_DB,
        "SELECT COUNT(*) AS n FROM articles WHERE relevance_score >= 0.30 AND is_duplicate=0"
    )["n"].iloc[0])
    stats["arimax_set"] = int(_qdb(
        ARTICLES_DB,
        "SELECT COUNT(*) AS n FROM articles WHERE relevance_score >= 0.35 AND is_duplicate=0"
    )["n"].iloc[0])
    stats["state_covered"] = int(_qdb(
        ARTICLES_DB,
        "SELECT COUNT(*) AS n FROM articles WHERE states_mentioned != '' AND is_duplicate=0"
    )["n"].iloc[0])

    rel_df = _qdb(
        ARTICLES_DB,
        "SELECT relevance_score FROM articles WHERE relevance_score IS NOT NULL AND is_duplicate=0"
    )
    stats["rel_scores"] = rel_df["relevance_score"].tolist()

    state_rows = _qdb(
        ARTICLES_DB,
        "SELECT states_mentioned FROM articles WHERE states_mentioned != '' AND is_duplicate=0"
    )
    from collections import Counter
    counter: Counter = Counter()
    for val in state_rows["states_mentioned"]:
        for s in str(val).split("|"):
            s = s.strip()
            if s:
                counter[s] += 1
    stats["top_states"] = counter.most_common(8)

    return stats


def load_topic_info() -> pd.DataFrame:
    if not TOPIC_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TOPIC_CSV)
    cols = [c for c in ["Topic", "Name", "Count"] if c in df.columns]
    df = df[cols].copy()
    df = df[df["Topic"] != -1].sort_values("Count", ascending=False).head(15)
    return df


def load_panel_meta() -> dict:
    if not PANEL_CSV.exists():
        return {}
    df = pd.read_csv(PANEL_CSV)
    return {
        "rows":        len(df),
        "commodities": df["commodity"].nunique() if "commodity" in df.columns else "—",
        "date_min":    df["date"].min() if "date" in df.columns else "—",
        "date_max":    df["date"].max() if "date" in df.columns else "—",
    }


def load_granger() -> pd.DataFrame:
    df = pd.read_csv(GRANGER_CSV)  # Removed sep="\t"
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["f_stat"] = pd.to_numeric(df["f_stat"],  errors="coerce")
    return df


def load_event_study() -> pd.DataFrame:
    df = pd.read_csv(EVENT_CSV)  # Removed sep="\t"
    for c in ["coef", "p_value", "ci_lower", "ci_upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_arimax_txt() -> dict:
    """
    Parse arimax_summary.txt into a dict keyed by commodity name.
    Each value is either:
      { skipped: True, reason: str }
    or:
      { skipped: False, model, aic, bic, n, dw, dw_note,
        rows: [{ var, coef, pval, sig }] }
    """
    if not ARIMAX_TXT.exists():
        return {}
    text = ARIMAX_TXT.read_text(encoding="utf-8")

    block_re = re.compile(r"^\s{2}(\w+):\s*$", re.MULTILINE)
    splits = list(block_re.finditer(text))
    results = {}

    for i, m in enumerate(splits):
        name = m.group(1)
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        block = text[start:end]

        skip_m = re.search(r"\[SKIP\].*", block)
        if skip_m:
            results[name] = {"skipped": True,
                             "reason": skip_m.group(0).strip()}
            continue

        hdr = re.search(
            r"ARIMAX\([^)]+\)\s+for\s+\w+\s+AIC=([\d.]+)\s+BIC=([\d.]+)\s+n=(\d+)",
            block
        )
        model_m = re.search(r"(ARIMAX\([^)]+\))", block)
        aic = float(hdr.group(1)) if hdr else None
        bic = float(hdr.group(2)) if hdr else None
        n = int(hdr.group(3)) if hdr else None

        var_re = re.compile(
            r"^\s{2}(\S+)\s+([-\d.]+)\s+([\d.]+)\s*(✓?)\s*$", re.MULTILINE
        )
        rows = [
            {"var": vm.group(1), "coef": float(vm.group(2)),
             "pval": float(vm.group(3)), "sig": bool(vm.group(4).strip())}
            for vm in var_re.finditer(block)
        ]

        dw_m = re.search(r"Durbin-Watson:\s*([\d.]+)\s*\(([^)]+)\)", block)
        dw = float(dw_m.group(1)) if dw_m else None
        dw_note = dw_m.group(2) if dw_m else ""

        results[name] = {
            "skipped": False,
            "model":   model_m.group(1) if model_m else "ARIMAX",
            "aic": aic, "bic": bic, "n": n,
            "dw": dw, "dw_note": dw_note,
            "rows": rows,
        }

    return results


def load_price_series() -> pd.DataFrame:
    codes = list(COMMODITY_CODES.values())
    placeholders = ",".join("?" * len(codes))
    df = _qdb(
        PRICE_DB,
        f"""SELECT item_name, item_code, year, month, inflation_yoy
            FROM cfpi_item
            WHERE state_code='99' AND item_code IN ({placeholders})
            ORDER BY item_code, year, month""",
        params=codes,
    )
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
    code_to_name = {v: k for k, v in COMMODITY_CODES.items()}
    df["commodity"] = df["item_code"].map(code_to_name).fillna(df["item_name"])
    return df


def load_sentiment() -> pd.DataFrame:
    if not SENTIMENT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SENTIMENT_CSV)
    for c in ["net_sentiment", "mean_polarity", "mean_sentiment_score",
              "positive_count", "negative_count", "neutral_count", "total_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def load_sentiment_dist() -> dict:
    """Overall positive/negative/neutral counts from articles.db."""
    try:
        df = _qdb(
            ARTICLES_DB,
            """SELECT sentiment_label, COUNT(*) AS n
               FROM articles
               WHERE sentiment_label IS NOT NULL
                 AND is_duplicate = 0
                 AND relevance_score >= 0.30
               GROUP BY sentiment_label"""
        )
        return dict(zip(df["sentiment_label"], df["n"]))
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _layout(fig, title="", height=PLOT_H):
    fig.update_layout(
        height=height, title=title, title_font_size=14,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="sans-serif", size=12, color="#2C2C2A"),
        margin=dict(l=60, r=30, t=55, b=70),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor="#E8E6DF", linecolor="#CCCAC2")
    fig.update_yaxes(gridcolor="#E8E6DF", linecolor="#CCCAC2")
    return fig


def _no_data_fig(msg="Data not available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font_size=14)
    return _layout(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_sources(stats):
    df = stats["sources"]
    src_colors = {"pib": BLUE, "direct": TEAL, "wayback": AMBER,
                  "selenium": GRAY, "manual_ingest": GRAY}
    colors = [src_colors.get(s, GRAY) for s in df["source_type"]]
    fig = go.Figure(go.Bar(
        x=df["source_type"], y=df["n"], marker_color=colors,
        text=df["n"], textposition="outside",
        hovertemplate="%{x}: %{y} articles<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Article count", showlegend=False)
    return _layout(fig, "Articles by source type", height=340)


def fig_relevance_hist(stats):
    fig = go.Figure(go.Histogram(
        x=stats["rel_scores"], nbinsx=30,
        marker_color=TEAL, opacity=0.8,
        hovertemplate="Score %{x:.2f}: %{y} articles<extra></extra>",
    ))
    fig.add_vline(x=0.30, line_dash="dash", line_color=AMBER,
                  annotation_text="0.30 BERTopic", annotation_position="top right")
    fig.add_vline(x=0.35, line_dash="dash", line_color=RED,
                  annotation_text="0.35 ARIMAX", annotation_position="top left")
    fig.update_layout(xaxis_title="Relevance score",
                      yaxis_title="Article count", showlegend=False)
    return _layout(fig, "Relevance score distribution (all-MiniLM-L6-v2)", height=340)


def fig_top_states(stats):
    if not stats["top_states"]:
        return _no_data_fig("No state data found")
    states, counts = zip(*stats["top_states"])
    fig = go.Figure(go.Bar(
        y=list(states), x=list(counts), orientation="h",
        marker_color=TEAL, text=list(counts), textposition="outside",
        hovertemplate="%{y}: %{x} articles<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Article mentions",
                      yaxis=dict(autorange="reversed"), showlegend=False)
    return _layout(fig, "Top states by article mentions (spaCy + EntityRuler)", height=380)


def fig_topics(topic_df):
    if topic_df.empty:
        return _no_data_fig("topic_info.csv not found")
    name_col = "Name" if "Name" in topic_df.columns else topic_df.columns[1]
    count_col = "Count" if "Count" in topic_df.columns else topic_df.columns[2]
    labels = topic_df[name_col].str[:42].tolist()
    counts = topic_df[count_col].tolist()
    colors = [TEAL if i < 2 else (RED if "covid" in str(labels[i]).lower() else GRAY)
              for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        y=labels, x=counts, orientation="h",
        marker_color=colors, text=counts, textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Article count",
                      yaxis=dict(autorange="reversed"), showlegend=False)
    return _layout(fig, "Top BERTopic clusters (32 total · 16.4% outliers)", height=460)


def fig_price_series(price_df, selected):
    if price_df.empty:
        return _no_data_fig("price_data.db not found")
    palette = [TEAL, BLUE, AMBER, RED, GRAY, TEAL_L, "#D4537E", "#639922"]
    fig = go.Figure()
    for i, comm in enumerate(selected):
        sub = price_df[price_df["commodity"] ==
                       comm].dropna(subset=["inflation_yoy"])
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["inflation_yoy"], mode="lines", name=comm,
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"{comm}<br>%{{x|%b %Y}}: %{{y:.1f}}%<extra></extra>",
        ))
    for x0, x1, label, fill in [
        ("2020-03-01", "2020-06-30", "COVID-19",     "rgba(231,76,60,0.08)"),
        ("2022-03-01", "2022-05-31", "Heatwave 2022", "rgba(239,159,39,0.10)"),
        ("2023-06-01", "2023-10-31", "El Niño 2023", "rgba(29,158,117,0.10)"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=10)
    fig.add_hline(y=0, line_color="#CCCAC2", line_width=1)
    fig.update_layout(xaxis_title="", yaxis_title="Inflation YoY (%)",
                      legend=dict(orientation="h", y=-0.18))
    return _layout(fig, "CPI inflation YoY — All India (state_code=99)")


def fig_granger_summary(gdf):
    sig = (gdf[gdf["p_value"] < 0.10]
           .groupby("commodity").size()
           .reindex(gdf["commodity"].unique(), fill_value=0)
           .sort_values(ascending=False))
    fig = go.Figure(go.Bar(
        x=sig.index, y=sig.values,
        marker_color=[TEAL if v > 0 else GRAY for v in sig.values],
        text=sig.values, textposition="outside",
        hovertemplate="%{x}: %{y} significant Granger hits (p<0.10)<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="Significant hits (p < 0.10)", showlegend=False)
    return _layout(fig, "Granger hits summary — all topics × lags (p < 0.10)", height=360)


def fig_granger_heatmap(gdf, topic_filter):
    sub = gdf[gdf["cause"] == topic_filter].copy()
    commodities = sorted(sub["commodity"].unique())
    lags = sorted(sub["lag"].unique())

    z, text, hover = [], [], []
    for c in commodities:
        row_z, row_t, row_h = [], [], []
        for lag in lags:
            cell = sub[(sub["commodity"] == c) & (sub["lag"] == lag)]
            if cell.empty or pd.isna(cell["p_value"].iloc[0]):
                row_z.append(1.0)
                row_t.append("N/A")
                row_h.append(f"{c} lag-{lag}: N/A")
            else:
                p = float(cell["p_value"].iloc[0])
                row_z.append(p)
                row_t.append(
                    f"{p:.3f}{'*' if p < .05 else ('†' if p < .10 else '')}")
                sig = "✓ p<0.05" if p < .05 else (
                    "† p<0.10" if p < .10 else "n.s.")
                row_h.append(f"{c} → lag-{lag}<br>p={p:.4f}  {sig}")
        z.append(row_z)
        text.append(row_t)
        hover.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Lag {l}" for l in lags], y=commodities,
        text=text, texttemplate="%{text}", textfont_size=12,
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=[
            [0.00, TEAL], [0.05, TEAL_L], [0.10, "#C0DD97"],
            [0.20, LIGHT], [1.00, "#F1EFE8"],
        ],
        zmin=0, zmax=1,
        colorbar=dict(title="p-value", thickness=14, len=0.7,
                      tickvals=[0, .05, .10, .5, 1],
                      ticktext=["0", "0.05*", "0.10†", "0.50", "1.0"]),
    ))
    return _layout(fig, f"Granger heatmap — {topic_filter}  (* p<0.05  † p<0.10)")


def fig_arimax(arimax_data, commodity):
    d = arimax_data.get(commodity)
    if d is None:
        return _no_data_fig(f"No ARIMAX data found for {commodity}")
    if d.get("skipped"):
        return _no_data_fig(d.get("reason", "Skipped"))

    rows = d["rows"]
    vnames = [r["var"] for r in rows]
    coefs = [r["coef"] for r in rows]
    pvals = [r["pval"] for r in rows]
    colors = [TEAL if p < .05 else (AMBER if p < .10 else GRAY) for p in pvals]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                        subplot_titles=["Coefficients", "p-values"])
    fig.add_trace(go.Bar(
        x=vnames, y=coefs, marker_color=colors,
        text=[f"{c:.3f}" for c in coefs], textposition="outside",
        hovertemplate="%{x}<br>coef=%{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="#CCCAC2", row=1, col=1)

    fig.add_trace(go.Bar(
        x=vnames, y=pvals,
        marker_color=[TEAL if p < .05 else (
            AMBER if p < .10 else GRAY) for p in pvals],
        text=[f"{p:.3f}" for p in pvals], textposition="outside",
        hovertemplate="%{x}<br>p=%{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=0.05, line_dash="dash", line_color=RED,
                  annotation_text="p=0.05", row=1, col=2)
    fig.add_hline(y=0.10, line_dash="dot", line_color=AMBER,
                  annotation_text="p=0.10", row=1, col=2)
    fig.update_xaxes(tickangle=40)
    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            x=0.5, y=1.10, xref="paper", yref="paper", showarrow=False,
            font_size=11, font_color=GRAY,
            text=(f"{d['model']}  |  AIC={d['aic']}  BIC={d['bic']}  "
                  f"n={d['n']}  DW={d['dw']} ({d['dw_note']})"),
        )] + list(fig.layout.annotations),
    )
    return _layout(fig, f"ARIMAX — {commodity}", height=480)


def fig_event_bars(edf):
    commodities = [c for c in edf["commodity"].unique() if c != "Mango"]
    events = edf["event"].unique().tolist()
    event_colors = {"event_heatwave_2022": RED,
                    "event_elnino_2023":   AMBER,
                    "event_covid_2020":    BLUE}
    event_labels = {"event_heatwave_2022": "Heatwave 2022 (Mar–May)",
                    "event_elnino_2023":   "El Niño 2023 (Jun–Oct)",
                    "event_covid_2020":    "COVID 2020 (Mar–Jun)"}
    fig = make_subplots(rows=1, cols=len(events),
                        subplot_titles=[event_labels.get(e, e) for e in events])
    for col_i, event in enumerate(events, start=1):
        sub = edf[(edf["event"] == event) & (
            edf["commodity"].isin(commodities))]
        is_sig = sub["significant"].astype(str).str.upper() == "TRUE"
        bcolors = [event_colors.get(event, GRAY)
                   if s else GRAY for s in is_sig]
        fig.add_trace(go.Bar(
            x=sub["commodity"], y=sub["coef"], marker_color=bcolors,
            error_y=dict(
                type="data",
                array=(sub["ci_upper"] - sub["coef"]).tolist(),
                arrayminus=(sub["coef"] - sub["ci_lower"]).tolist(),
                color="#CCCAC2", thickness=1.5, width=4,
            ),
            text=[f"p={p:.3f}" for p in sub["p_value"]],
            textposition="outside", showlegend=False,
            hovertemplate="%{x}<br>coef=%{y:.3f}<br>%{text}<extra></extra>",
        ), row=1, col=col_i)
        fig.add_hline(y=0, line_color="#CCCAC2", row=1, col=col_i)
    fig.update_xaxes(tickangle=30)
    return _layout(fig, "Event study OLS coefficients with 95% CI  (tinted = significant)",
                   height=460)


def fig_forest(edf):
    sig = edf[edf["significant"].astype(str).str.upper() == "TRUE"].copy()
    if sig.empty:
        return _no_data_fig("No significant event study results")
    sig["label"] = (sig["commodity"] + " / "
                    + sig["event"].str.replace("event_", "").str.replace("_", " "))
    fig = go.Figure()
    for _, row in sig.iterrows():
        color = TEAL if row["coef"] > 0 else RED
        fig.add_trace(go.Scatter(
            x=[row["coef"]], y=[row["label"]],
            error_x=dict(type="data", symmetric=False,
                         array=[row["ci_upper"] - row["coef"]],
                         arrayminus=[row["coef"] - row["ci_lower"]],
                         color=color),
            mode="markers", marker=dict(color=color, size=10),
            showlegend=False,
            hovertemplate=(f"{row['label']}<br>coef={row['coef']:.3f}  "
                           f"CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]  "
                           f"p={row['p_value']:.4f}<extra></extra>"),
        ))
    fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
    fig.update_layout(
        xaxis_title="Coefficient (inflation pp)", showlegend=False)
    return _layout(fig, "Forest plot — significant results only (p < 0.05)", height=320)


def fig_placebo(gdf):
    topics = gdf["cause"].unique().tolist()
    comparisons = [
        ("Tur_Dal (key finding)", "Tur_Dal", TEAL, "solid"),
        ("Tomato (comparison)",   "Tomato",  BLUE, "dot"),
        ("Mango (placebo)",       "Mango",   RED,  "dash"),
    ]
    fig = go.Figure()
    for label, comm, color, dash in comparisons:
        sub = gdf[gdf["commodity"] == comm]
        min_p = sub.groupby("cause")["p_value"].min().reindex(topics)
        fig.add_trace(go.Scatter(
            x=topics, y=min_p.values, mode="lines+markers", name=label,
            line=dict(color=color, dash=dash, width=2), marker=dict(size=9),
            hovertemplate="%{x}<br>min-p=%{y:.4f}<extra></extra>",
        ))
    fig.add_hline(y=0.05, line_dash="dash", line_color=GRAY,
                  annotation_text="p=0.05", annotation_position="top right")
    fig.update_layout(xaxis_title="Topic", yaxis_title="Min p-value across lags",
                      yaxis_range=[0, 1.0])
    return _layout(fig, "Placebo test — Mango should NOT respond to climate topics")


def fig_sentiment_series(sdf: pd.DataFrame) -> go.Figure:
    if sdf.empty:
        return _no_data_fig("Run sentiment_score.py first — monthly_sentiment.csv not found")

    # 3-month rolling average
    sdf = sdf.sort_values("date").copy()
    sdf["rolling_net"] = sdf["net_sentiment"].rolling(
        3, min_periods=1).mean().round(4)

    fig = go.Figure()

    # zero baseline
    fig.add_hline(y=0, line_color="#CCCAC2", line_width=1)

    # event shading
    for x0, x1, label, fill in [
        ("2020-03-01", "2020-06-30", "COVID-19",      "rgba(231,76,60,0.08)"),
        ("2022-03-01", "2022-05-31", "Heatwave 2022", "rgba(239,159,39,0.10)"),
        ("2023-06-01", "2023-10-31", "El Niño 2023",  "rgba(29,158,117,0.10)"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=10)

    # monthly net sentiment bars (colour by sign)
    bar_colors = [TEAL if v >= 0 else RED for v in sdf["net_sentiment"]]
    fig.add_trace(go.Bar(
        x=sdf["date"], y=sdf["net_sentiment"],
        marker_color=bar_colors, opacity=0.45,
        name="Monthly net sentiment",
        hovertemplate="%{x|%b %Y}<br>net sentiment: %{y:.3f}<extra></extra>",
    ))

    # 3-month rolling line
    fig.add_trace(go.Scatter(
        x=sdf["date"], y=sdf["rolling_net"],
        mode="lines", name="3-month rolling avg",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="%{x|%b %Y}<br>3m avg: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Net sentiment  (positive − negative) / total",
        legend=dict(orientation="h", y=-0.18),
        barmode="relative",
    )
    return _layout(fig, "Monthly climate-news sentiment index (FinBERT · relevance ≥ 0.30)")


def fig_sentiment_stack(sdf: pd.DataFrame) -> go.Figure:
    if sdf.empty:
        return _no_data_fig("Run sentiment_score.py first — monthly_sentiment.csv not found")

    sdf = sdf.sort_values("date").copy()
    fig = go.Figure()
    for col, color, label in [
        ("positive_count", TEAL, "Positive"),
        ("neutral_count",  GRAY, "Neutral"),
        ("negative_count", RED,  "Negative"),
    ]:
        fig.add_trace(go.Bar(
            x=sdf["date"], y=sdf[col],
            name=label, marker_color=color,
            hovertemplate=f"{label}<br>%{{x|%b %Y}}: %{{y}} articles<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="",
        yaxis_title="Article count",
        legend=dict(orientation="h", y=-0.18),
    )
    return _layout(fig, "Monthly article volume by sentiment label", height=320)


def fig_sentiment_dist(dist: dict) -> go.Figure:
    if not dist:
        return _no_data_fig("No scored articles in articles.db yet")
    labels = ["positive", "neutral", "negative"]
    values = [dist.get(l, 0) for l in labels]
    colors = [TEAL, GRAY, RED]
    total = sum(values)
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v}<br>({v/total*100:.1f}%)" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y} articles<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Article count", showlegend=False)
    return _layout(
        fig,
        f"Overall sentiment distribution — {total:,} scored articles (FinBERT)",
        height=320,
    )


def tbl_extreme_months(sdf: pd.DataFrame) -> str:
    """Returns an HTML table of the 5 most negative and 5 most positive months."""
    if sdf.empty:
        return ""
    sdf = sdf.sort_values("date").copy()
    sdf["month_label"] = sdf["date"].dt.strftime("%b %Y")

    top_neg = sdf.nsmallest(5, "net_sentiment")[
        ["month_label", "net_sentiment", "negative_count",
            "positive_count", "total_count"]
    ]
    top_pos = sdf.nlargest(5, "net_sentiment")[
        ["month_label", "net_sentiment", "negative_count",
            "positive_count", "total_count"]
    ]

    def rows(df, color):
        out = ""
        for _, r in df.iterrows():
            out += (f"<tr>"
                    f"<td style='padding:7px 12px;'>{r['month_label']}</td>"
                    f"<td style='padding:7px 12px;text-align:center;"
                    f"color:{color};font-weight:500;'>{r['net_sentiment']:+.3f}</td>"
                    f"<td style='padding:7px 12px;text-align:center;'>{int(r['negative_count'])}</td>"
                    f"<td style='padding:7px 12px;text-align:center;'>{int(r['positive_count'])}</td>"
                    f"<td style='padding:7px 12px;text-align:center;'>{int(r['total_count'])}</td>"
                    f"</tr>")
        return out

    header = ("<tr style='background:var(--color-background-secondary,#F1EFE8);'>"
              "<th style='padding:7px 12px;text-align:left;'>Month</th>"
              "<th style='padding:7px 12px;'>Net sentiment</th>"
              "<th style='padding:7px 12px;'>Negative</th>"
              "<th style='padding:7px 12px;'>Positive</th>"
              "<th style='padding:7px 12px;'>Total</th></tr>")

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;font-size:13px;">
      <div>
        <p style="font-size:11px;color:#5F5E5A;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:6px;">5 most negative months</p>
        <table style="width:100%;border-collapse:collapse;">
          {header}{rows(top_neg, '#E24B4A')}
        </table>
      </div>
      <div>
        <p style="font-size:11px;color:#5F5E5A;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:6px;">5 most positive months</p>
        <table style="width:100%;border-collapse:collapse;">
          {header}{rows(top_pos, '#1D9E75')}
        </table>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — load everything once
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data from databases and result files...")
corpus_stats = load_corpus_stats()
topic_df = load_topic_info()
panel_meta = load_panel_meta()
granger_df = load_granger()
event_df = load_event_study()
arimax_data = parse_arimax_txt()
price_df = load_price_series()
sentiment_df = load_sentiment()
sentiment_dist = load_sentiment_dist()

all_commodities = sorted(price_df["commodity"].unique().tolist()) \
    if not price_df.empty else list(COMMODITY_CODES.keys())
all_topics = sorted(granger_df["cause"].unique().tolist())
arimax_comms = sorted(arimax_data.keys())
default_arimax = "Tur_Dal" if "Tur_Dal" in arimax_comms else arimax_comms[0]
print("Ready.")

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO APP
# ══════════════════════════════════════════════════════════════════════════════

state_pct = (corpus_stats["state_covered"] / corpus_stats["total"] * 100
             if corpus_stats["total"] else 0)

CSS = """
.gradio-container { max-width: 1200px !important; }
.metric-row { display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; }
.metric-box { background:#F1EFE8; border-radius:10px; padding:14px 18px;
              text-align:center; flex:1; min-width:120px; }
.metric-val { font-size:1.7rem; font-weight:600; color:#1D9E75; }
.metric-lbl { font-size:0.75rem; color:#5F5E5A; margin-top:3px; }
.callout-green { background:#EAF3DE; border-left:4px solid #1D9E75;
                 padding:12px 16px; border-radius:0 8px 8px 0;
                 font-size:13px; margin-top:10px; line-height:1.6; }
.callout-amber { background:#FAEEDA; border-left:4px solid #EF9F27;
                 padding:12px 16px; border-radius:0 8px 8px 0;
                 font-size:13px; margin-top:10px; line-height:1.6; }

"""

with gr.Blocks(css=CSS, title="ClimateMarketPulse") as demo:

    gr.HTML("""
    <div style="padding:20px 0 6px;">
      <h1 style="margin:0;font-size:1.6rem;font-weight:500;">ClimateMarketPulse</h1>
      <p style="color:#5F5E5A;margin:4px 0 0;font-size:13px;">
        NLP &nbsp;·&nbsp; Climate events × Indian food commodity prices
        &nbsp;·&nbsp; 2020–2024
      </p>
    </div>""")

    gr.HTML(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-val">{corpus_stats['total']:,}</div>
        <div class="metric-lbl">Total articles</div></div>
      <div class="metric-box">
        <div class="metric-val">{corpus_stats['topic_modeled']:,}</div>
        <div class="metric-lbl">Topic-modeled (≥0.30)</div></div>
      <div class="metric-box">
        <div class="metric-val">{corpus_stats['arimax_set']:,}</div>
        <div class="metric-lbl">ARIMAX set (≥0.35)</div></div>
      <div class="metric-box">
        <div class="metric-val">{state_pct:.1f}%</div>
        <div class="metric-lbl">State mention coverage</div></div>
      <div class="metric-box">
        <div class="metric-val">{panel_meta.get('rows', '—')}</div>
        <div class="metric-lbl">Panel rows</div></div>
      <div class="metric-box">
        <div class="metric-val">{panel_meta.get('commodities', '—')}</div>
        <div class="metric-lbl">Commodities</div></div>
    </div>
    <hr style="border:none;border-top:1px solid #E8E6DF;margin:0 0 8px;">
    """)

    with gr.Tabs():

        with gr.Tab("Corpus & pipeline", elem_id="pipeline-tab"):
            with gr.Row():
                gr.Plot(fig_sources(corpus_stats))
                gr.Plot(fig_relevance_hist(corpus_stats))
            with gr.Row():
                gr.Plot(fig_top_states(corpus_stats))
                gr.Plot(fig_topics(topic_df))
            gr.HTML("""
            <div style="color: black; background-color: #e6f4ea; padding: 15px; border-radius: 5px; margin-top: 10px;">
              <strong style="color: black;">Pipeline:</strong>
              Playwright + BeautifulSoup scraping →
              MD5 deduplication →
              spaCy NER + EntityRuler (Indian state/UT aliases) →
              all-MiniLM-L6-v2 relevance scoring →
              BERTopic (UMAP + HDBSCAN, 32 topics) →
              MoSPI CPI alignment (127 items × 37 states × 60 months) →
              Granger / VAR / ARIMAX / Event study
            </div>""")

        with gr.Tab("Price trends"):
            gr.Markdown(
                "Select commodities to overlay. Shaded regions mark climate event windows.")
            comm_select = gr.CheckboxGroup(
                choices=all_commodities,
                value=["Tomato", "Onion", "Rice", "Tur_Dal"],
                label="Commodities",
            )
            price_plot = gr.Plot(fig_price_series(
                price_df, ["Tomato", "Onion", "Rice", "Tur_Dal"]))
            comm_select.change(
                fn=lambda sel: fig_price_series(price_df, sel),
                inputs=comm_select, outputs=price_plot,
            )

        with gr.Tab("Granger causality"):
            gr.Markdown(
                "**Null hypothesis:** topic prevalence does NOT Granger-cause commodity inflation.  \n"
                "Lower p = stronger predictive evidence.  &nbsp; ✓ p<0.05 &nbsp; † p<0.10"
            )
            gr.Plot(fig_granger_summary(granger_df))
            topic_dd = gr.Dropdown(
                choices=all_topics, value=all_topics[0],
                label="Select topic for heatmap",
            )
            granger_plot = gr.Plot(
                fig_granger_heatmap(granger_df, all_topics[0]))
            topic_dd.change(
                fn=lambda t: fig_granger_heatmap(granger_df, t),
                inputs=topic_dd, outputs=granger_plot,
            )
            gr.HTML("""
            <div class="callout-amber" style="color: black;">
              <strong style="color: black;">Key finding:</strong>
              <strong style="color: black;">Tur Dal</strong> is the most climate-sensitive commodity —
              T0 kharif/rainfall Granger-causes inflation at lag 1 (p=0.017) and lag 3 (p=0.043).
              T1 veg prices also significant at lag 1 (p=0.003) and lag 3 (p=0.016).
              Matches the kharif pathway: monsoon variability → pulse harvest
              → price transmission within 1–3 months.
            </div>""")

        with gr.Tab("ARIMAX models"):
            gr.Markdown(
                "ARIMAX(1,d,1) with lagged topic scores + climate event dummies.  \n"
                "Lags avoid contemporaneous endogeneity. HC3 robust standard errors."
            )
            arimax_dd = gr.Dropdown(
                choices=arimax_comms, value=default_arimax, label="Commodity")
            arimax_plot = gr.Plot(fig_arimax(arimax_data, default_arimax))
            arimax_dd.change(
                fn=lambda c: fig_arimax(arimax_data, c),
                inputs=arimax_dd, outputs=arimax_plot,
            )
            gr.HTML("""
            <div class="callout-green" style="color: black;">
              <strong style="color: black;">Tur Dal highlights:</strong>
              T1_veg_prices_lag1 coef=0.364 (p=0.003) ✓ &nbsp;|&nbsp;
              T6_covid_lag1 coef=−0.826 (p=0.026) ✓ &nbsp;|&nbsp;
              El Niño 2023 dummy = +5.35pp (p=0.002) ✓ &nbsp;|&nbsp;
              DW=1.829 (no autocorrelation) &nbsp;|&nbsp; AIC=256.81
            </div>""")

        with gr.Tab("Event study"):
            gr.Markdown(
                "OLS with AR(1) control. Coefficient = inflation pp deviation during event window."
            )
            gr.Plot(fig_event_bars(event_df))
            gr.Plot(fig_forest(event_df))

        with gr.Tab("Placebo test"):
            gr.Markdown(
                "If T0 Granger-causes **Tur Dal** but NOT **Mango**, "
                "the result is commodity-specific — not spurious corpus-wide noise."
            )
            gr.Plot(fig_placebo(granger_df))
            gr.HTML("""
            <div class="callout-amber" style="color: black;">
              <strong style="color: black;">Result — partially passes:</strong>
              Tur Dal T0 min-p = 0.017 at lag-1 vs Mango T0 min-p = 0.35 — placebo holds.
              Tomato T0 marginal at lag-3 (p=0.054) — report Tur Dal vs Mango as the
              clean comparison; flag Tomato as exploratory.
            </div>""")

        with gr.Tab("Sentiment"):
            gr.Markdown(
                "**ProsusAI/finbert** scored on headline + first 300 chars of body text. "
                "Articles with `relevance_score ≥ 0.30` only.  \n"
                "`net_sentiment = (positive − negative) / total`  · ranges −1 to +1"
            )
            gr.Plot(fig_sentiment_series(sentiment_df))
            gr.Plot(fig_sentiment_stack(sentiment_df))
            with gr.Row():
                gr.Plot(fig_sentiment_dist(sentiment_dist))
                gr.HTML(tbl_extreme_months(sentiment_df))
            gr.HTML("""
            <div class="callout-amber" style="color: black;">
            <strong style="color: black;">Note:</strong> Negative net sentiment during climate event windows
            (Heatwave 2022, El Niño 2023) would support the hypothesis that media tone
            leads commodity price increases by 1–2 months. Compare the index troughs
            against the price trend spikes in the <em style="color: black;">Price trends</em> tab.
            </div>""")

        with gr.Tab("Findings summary"):
            gr.HTML("""
            <div style="max-width:820px;line-height:1.75;font-size:14px;padding-top:8px;">
            <h3 style="color:#1D9E75;font-weight:500;">Key findings</h3>
            <ol>
              <li><strong>Tur Dal is the most climate-sensitive commodity.</strong>
                T0 Granger-causes inflation at lag 1 (p=0.017) and lag 3 (p=0.043).
                ARIMAX T1 coefficient = 0.364 (p=0.003). El Niño 2023 = +6.9pp (p=0.001).
                Three-method triangulation (Granger + ARIMAX + event study) converges.</li>
              <li><strong>Onion responds strongly to El Niño.</strong>
                Event study: +14pp during Jun–Oct 2023 (p=0.001, CI=[5.6, 22.4]).
                Granger causality weaker — supply shock rather than news-anticipation.</li>
              <li><strong>Rice shows COVID and heatwave effects via ARIMAX.</strong>
                COVID_2020 dummy = +2.47pp (p=0.002). Heatwave_2022 = −0.58pp (p&lt;0.001) —
                negative sign likely reflects export bans and MSP procurement suppressing prices.</li>
              <li><strong>Mustard Oil: heatwave-driven uptick.</strong>
                ARIMAX heatwave dummy = +2.1pp (p=0.002).
                Oilseed Granger pathway not significant — may operate via input cost channel.</li>
              <li><strong>Tomato and Potato: high volatility obscures NLP signal.</strong>
                Wide CIs in event study; extreme perishable seasonality dominates the series.</li>
            </ol>
            <h3 style="color:#EF9F27;font-weight:500;">Limitations</h3>
            <ul>
              <li>Sentiment scoring (Step 6) skipped — future work with ProsusAI/finbert.</li>
              <li>National-level only (state_code=99); state panel is future work.</li>
              <li>Small effective T for Mango VAR (n=21 post price-join).</li>
              <li>Placebo inconclusive for Tomato; Tur Dal vs Mango is the clean comparison.</li>
            </ul>
            </div>""")

    gr.HTML("""
    <div style="text-align:center;color:#888780;font-size:12px;
                padding:14px 0 6px;border-top:1px solid #E8E6DF;margin-top:14px;">
      ClimateMarketPulse &nbsp;·&nbsp; NLP 2026 &nbsp;·&nbsp;
      spaCy + sentence-transformers + BERTopic + statsmodels
    </div>""")

if __name__ == "__main__":
    demo.launch(share=False, server_port=7800)
