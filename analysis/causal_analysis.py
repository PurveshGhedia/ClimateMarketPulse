"""
analysis/causal_analysis.py

Causal analysis pipeline — Phase 4 of ClimateMarketPulse.
Runs on data/processed/analysis_panel.csv produced by align_features.py.

Tests performed per commodity:
  1. ADF stationarity test on inflation_yoy
  2. Granger causality: topic prevalence → inflation_yoy (lags 1,3,6)
  3. VAR model: joint dynamics of inflation + key topic scores
  4. ARIMAX: inflation_yoy ~ ARIMA errors + topic prevalence exogenous vars
  5. Event study: 2022 heatwave, 2023 El Niño, 2020 COVID shock
  6. Placebo test: Mango (T4 topic, no strong climate-price pathway expected)

Outputs (all in data/processed/results/):
  granger_results.csv     — p-values for all commodity × topic × lag combos
  var_summary.txt         — VAR model summaries
  arimax_summary.txt      — ARIMAX model summaries
  event_study.csv         — event dummy coefficients and significance
  analysis_report.txt     — human-readable summary for the project report

Usage (from project root):
    python analysis/causal_analysis.py

Requirements:
    pip install statsmodels scipy
"""

from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import statsmodels.api as sm
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PANEL_CSV = os.path.join(PROCESSED_DIR, "analysis_panel.csv")
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")

# ── Analysis config ───────────────────────────────────────────────────────────
SIGNIFICANCE = 0.05
GRANGER_LAGS = [1, 3, 6]
VAR_MAX_LAGS = 6

# Topics to test as Granger causes
TOPIC_COLS = [
    "T0_kharif_rainfall",
    "T1_veg_prices",
    "T2_cpi_inflation",
    "T3_climate_change",
    "T6_covid",
]

EVENT_COLS = [
    "event_heatwave_2022",
    "event_elnino_2023",
    "event_covid_2020",
]

# Commodities split by expected signal strength
CLIMATE_SENSITIVE = ["Tomato", "Onion", "Potato", "Rice", "Wheat", "Tur_Dal"]
LESS_SENSITIVE = ["Mustard_Oil"]
PLACEBO = ["Mango"]   # expect weaker Granger signal for T0


# ── Helpers ───────────────────────────────────────────────────────────────────
def tee(msg: str, buf: StringIO) -> None:
    """Print to stdout and write to buffer."""
    print(msg)
    buf.write(msg + "\n")


def adf_test(series: pd.Series, name: str) -> dict:
    """ADF stationarity test. Returns dict with result."""
    clean = series.dropna()
    if len(clean) < 10:
        return {"series": name, "adf_stat": None, "p_value": None,
                "stationary": None, "n_obs": len(clean)}
    result = adfuller(clean, autolag="AIC")
    return {
        "series":     name,
        "adf_stat":   round(result[0], 4),
        "p_value":    round(result[1], 4),
        "stationary": result[1] < SIGNIFICANCE,
        "n_obs":      len(clean),
        "lags_used":  result[2],
    }


def run_granger_tests(
    df: pd.DataFrame,
    target: str,
    causes: list[str],
    commodity: str,
) -> list[dict]:
    """
    Run Granger causality tests for each cause variable → target.
    Returns list of result dicts.
    """
    results = []

    for cause in causes:
        if cause not in df.columns:
            continue
        # Align series
        combined = df[[target, cause]].dropna()
        if len(combined) < 20:
            continue

        for lag in GRANGER_LAGS:
            if lag >= len(combined) - 2:
                continue
            try:
                test_result = grangercausalitytests(
                    combined[[target, cause]].values,
                    maxlag=lag,
                    verbose=False,
                )
                # Use the F-test p-value at the requested lag
                p_val = test_result[lag][0]["ssr_ftest"][1]
                f_stat = test_result[lag][0]["ssr_ftest"][0]
                results.append({
                    "commodity":    commodity,
                    "cause":        cause,
                    "target":       target,
                    "lag":          lag,
                    "f_stat":       round(f_stat, 4),
                    "p_value":      round(p_val, 4),
                    "significant":  p_val < SIGNIFICANCE,
                    "n_obs":        len(combined),
                })
            except Exception as e:
                results.append({
                    "commodity": commodity, "cause": cause,
                    "target": target, "lag": lag,
                    "f_stat": None, "p_value": None,
                    "significant": False, "n_obs": 0,
                    "error": str(e),
                })
    return results


def run_var(
    df: pd.DataFrame,
    commodity: str,
    topic_cols: list[str],
    target: str,
    buf: StringIO,
) -> dict:
    """
    Fit VAR model. Uses `target` which may be differenced to enforce stationarity.
    """
    if commodity in ["Tomato", "Onion", "Potato", "Mustard_Oil"]:
        var_topics = ["T1_veg_prices", "T0_kharif_rainfall"]
    else:
        var_topics = ["T0_kharif_rainfall", "T2_cpi_inflation"]

    available = [t for t in var_topics if t in df.columns]
    var_cols = [target] + available
    var_df = df[var_cols].dropna()

    if len(var_df) < 20:
        tee(f"  [SKIP] VAR for {commodity}: insufficient obs ({len(var_df)})", buf)
        return {}

    try:
        model = VAR(var_df)
        lag_order_result = model.select_order(
            maxlags=min(VAR_MAX_LAGS, len(var_df)//5))
        best_lag = lag_order_result.aic or 1
        best_lag = max(1, int(best_lag))

        fitted = model.fit(best_lag)

        tee(f"\n  VAR({best_lag}) for {commodity} — AIC={fitted.aic:.2f}  "
            f"BIC={fitted.bic:.2f}  n={len(var_df)}", buf)
        tee(f"  Variables: {var_cols}", buf)

        # Granger causality within VAR framework
        for topic in available:
            try:
                gc = fitted.test_causality(
                    target, [topic], kind="f"
                )
                tee(f"    Granger (VAR): {topic} → {target}  "
                    f"p={gc.pvalue:.4f} {'✓ SIGNIFICANT' if gc.pvalue < SIGNIFICANCE else ''}", buf)
            except Exception:
                pass

        return {
            "commodity": commodity,
            "lag":       best_lag,
            "aic":       fitted.aic,
            "bic":       fitted.bic,
            "n_obs":     len(var_df),
        }

    except Exception as e:
        tee(f"  [ERROR] VAR for {commodity}: {e}", buf)
        return {}


def run_arimax(
    df: pd.DataFrame,
    commodity: str,
    topic_cols: list[str],
    event_cols: list[str],
    d_order: int,
    buf: StringIO,
) -> dict:
    """
    ARIMAX with topic prevalence + event dummies. 
    Integration term `d_order` dynamically set based on ADF test.
    Drops exogenous columns with 0 variance to prevent singular matrices.
    """
    target = "inflation_yoy"

    exog_cols = []
    for t in topic_cols:
        lag_col = f"{t}_lag1"
        if lag_col in df.columns:
            exog_cols.append(lag_col)
    exog_cols += [c for c in event_cols if c in df.columns]

    # ONLY keep exog columns that actually vary (var > 0)
    valid_exog = [c for c in exog_cols if c in df.columns and df[c].var() > 0]

    model_df = df[[target] + valid_exog].dropna()

    if len(model_df) < 24:
        tee(f"  [SKIP] ARIMAX for {commodity}: insufficient obs ({len(model_df)})", buf)
        return {}

    y = model_df[target]
    exog = model_df[valid_exog] if valid_exog else None

    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=(1, d_order, 1),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        # Bumping maxiter up to 500 prevents the Convergence Warning
        result = model.fit(disp=False, method="lbfgs", maxiter=500)

        tee(f"\n  ARIMAX(1,{d_order},1) for {commodity}  "
            f"AIC={result.aic:.2f}  BIC={result.bic:.2f}  n={len(y)}", buf)

        if exog is not None:
            tee(f"  {'Variable':<38} {'Coef':>10}  {'p-value':>10}  Sig", buf)
            tee(f"  {'-'*38}  {'-'*10}  {'-'*10}  ---", buf)

            significant_topics = []
            for var_name in valid_exog:
                if var_name in result.params.index:
                    coef = result.params[var_name]
                    pval = result.pvalues[var_name]
                    sig = "✓" if pval < SIGNIFICANCE else ""
                    tee(f"  {var_name:<38} {coef:>10.4f}  {pval:>10.4f}  {sig}", buf)
                    if pval < SIGNIFICANCE:
                        significant_topics.append(var_name)

        dw = durbin_watson(result.resid)
        tee(f"  Durbin-Watson: {dw:.3f}  "
            f"({'no autocorrelation' if 1.5 < dw < 2.5 else 'possible autocorrelation'})", buf)

        return {
            "commodity":          commodity,
            "aic":                result.aic,
            "bic":                result.bic,
            "n_obs":              len(y),
            "significant_topics": significant_topics if exog is not None else [],
            "dw":                 dw,
        }

    except Exception as e:
        tee(f"  [ERROR] ARIMAX for {commodity}: {e}", buf)
        return {}


def run_event_study(
    df: pd.DataFrame,
    commodity: str,
    event_cols: list[str],
    buf: StringIO,
) -> list[dict]:
    """
    OLS regression dropping events with 0 variance AFTER handling NaNs.
    """
    # 1. First, get the columns we actually want to use
    base_cols = ["inflation_yoy", "inflation_lag1"]
    available_events = [c for c in event_cols if c in df.columns]

    # 2. Drop the rows where our target (inflation_yoy) is NaN
    reg_df = df[base_cols + available_events].dropna()

    if len(reg_df) < 15:
        return []

    # 3. NOW check for zero variance on the cleaned data
    valid_events = [c for c in available_events if reg_df[c].var() > 0]

    if not valid_events:
        tee(f"  [SKIP] Event study for {commodity}: No valid variance in event windows.", buf)
        return []

    y = reg_df["inflation_yoy"]
    # Use valid_events here!
    X = sm.add_constant(reg_df[["inflation_lag1"] + valid_events])

    try:
        ols = sm.OLS(y, X).fit(cov_type="HC3")

        results = []
        for event in valid_events:  # And iterate over valid_events here!
            if event in ols.params.index:
                coef = ols.params[event]
                pval = ols.pvalues[event]
                ci_lo, ci_hi = ols.conf_int().loc[event]
                results.append({
                    "commodity":  commodity,
                    "event":      event,
                    "coef":       round(coef, 4),
                    "p_value":    round(pval, 4),
                    "ci_lower":   round(ci_lo, 4),
                    "ci_upper":   round(ci_hi, 4),
                    "significant": pval < SIGNIFICANCE,
                })
                sig = "✓ SIGNIFICANT" if pval < SIGNIFICANCE else ""
                tee(f"    {event:<30} coef={coef:+.3f}  p={pval:.4f}  "
                    f"CI=[{ci_lo:.3f}, {ci_hi:.3f}]  {sig}", buf)
        return results

    except Exception as e:
        tee(f"  [ERROR] Event study for {commodity}: {e}", buf)
        return []
# ── Main ──────────────────────────────────────────────────────────────────────


def run_analysis():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    var_buf = StringIO()
    arimax_buf = StringIO()
    report_buf = StringIO()

    print("=" * 65)
    print("ClimateMarketPulse — Causal Analysis (Phase 4)")
    print("=" * 65)

    if not os.path.exists(PANEL_CSV):
        print(f"ERROR: {PANEL_CSV} not found. Run align_features.py first.")
        sys.exit(1)

    panel = pd.read_csv(PANEL_CSV)
    commodities = sorted(panel["commodity"].dropna().unique())
    print(f"Panel loaded        : {panel.shape}")
    print(f"Commodities         : {commodities}")
    print(
        f"Time range          : {panel['year_month'].min()} → {panel['year_month'].max()}")
    print(
        f"Topic columns       : {[c for c in TOPIC_COLS if c in panel.columns]}")

    # ── 1. ADF Stationarity Tests ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("1. STATIONARITY TESTS (ADF)")
    print(f"{'─'*65}")
    tee("\n=== ADF Stationarity Tests ===", report_buf)

    adf_results = {}
    for commodity in commodities:
        sub = panel[panel["commodity"] == commodity].sort_values(
            ["year", "month"]).copy()
        r = adf_test(sub["inflation_yoy"], f"{commodity}_inflation_yoy")
        adf_results[commodity] = r
        status = "STATIONARY ✓" if r.get("stationary") else "NON-STATIONARY ✗"
        print(f"  {commodity:<20} ADF={r.get('adf_stat', 'N/A')}  "
              f"p={r.get('p_value', 'N/A')}  {status}")
        tee(f"  {commodity}: p={r.get('p_value')} → {status}", report_buf)

    # ── 2. Granger Causality Tests ────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("2. GRANGER CAUSALITY TESTS")
    print(f"{'─'*65}")
    tee("\n=== Granger Causality Results ===", report_buf)

    all_granger = []
    for commodity in commodities:
        # Prepping data with proper DateTime index to fix ValueWarning
        sub = panel[panel["commodity"] == commodity].sort_values(
            ["year", "month"]).copy()
        sub['date'] = pd.to_datetime(sub['year'].astype(
            str) + '-' + sub['month'].astype(str).str.zfill(2) + '-01')
        sub = sub.set_index('date')
        sub.index.freq = 'MS'  # Monthly Start frequency

        # DIFFERENCING CHECK
        is_stat = adf_results[commodity].get("stationary")
        target_col = "inflation_yoy" if is_stat else "inflation_yoy_diff"
        sub["inflation_yoy_diff"] = sub["inflation_yoy"].diff()

        available_topics = [t for t in TOPIC_COLS if t in sub.columns]

        print(f"  {commodity} (Target: {target_col}):")
        tee(f"\n  {commodity} (Target: {target_col}):", report_buf)

        results = run_granger_tests(
            sub, target_col, available_topics, commodity)
        all_granger.extend(results)

        for r in results:
            sig = "✓" if r.get("significant") else " "
            print(
                f"    {sig} lag={r['lag']}  {r['cause']:<35} F={r.get('f_stat', 'N/A')}  p={r.get('p_value', 'N/A')}")
            tee(f"    {sig} lag={r['lag']}  {r['cause']:<35} p={r.get('p_value', 'N/A')}", report_buf)

    granger_df = pd.DataFrame(all_granger)
    granger_df.to_csv(os.path.join(
        RESULTS_DIR, "granger_results.csv"), index=False)

    # ── 3. VAR Models ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("3. VAR MODELS")
    print(f"{'─'*65}")
    tee("\n=== VAR Models ===", var_buf)

    var_results = []
    for commodity in commodities:
        sub = panel[panel["commodity"] == commodity].sort_values(
            ["year", "month"]).copy()
        sub['date'] = pd.to_datetime(sub['year'].astype(
            str) + '-' + sub['month'].astype(str).str.zfill(2) + '-01')
        sub = sub.set_index('date')
        sub.index.freq = 'MS'

        is_stat = adf_results[commodity].get("stationary")
        target_col = "inflation_yoy" if is_stat else "inflation_yoy_diff"
        sub["inflation_yoy_diff"] = sub["inflation_yoy"].diff()

        r = run_var(sub, commodity, TOPIC_COLS, target_col, var_buf)
        if r:
            var_results.append(r)

    # ── 4. ARIMAX Models ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("4. ARIMAX MODELS")
    print(f"{'─'*65}")
    tee("\n=== ARIMAX Models ===", arimax_buf)

    for commodity in commodities:
        print(f"\n  {commodity}:")
        tee(f"\n  {commodity}:", arimax_buf)
        sub = panel[panel["commodity"] == commodity].sort_values(
            ["year", "month"]).copy()
        sub['date'] = pd.to_datetime(sub['year'].astype(
            str) + '-' + sub['month'].astype(str).str.zfill(2) + '-01')
        sub = sub.set_index('date')
        sub.index.freq = 'MS'

        # If not stationary, ARIMAX needs an integration order (d) of 1
        d_order = 0 if adf_results[commodity].get("stationary") else 1

        run_arimax(sub, commodity, TOPIC_COLS, EVENT_COLS, d_order, arimax_buf)

    # ── 5. Event Study ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("5. EVENT STUDY")
    print(f"{'─'*65}")
    tee("\n=== Event Study ===", report_buf)

    all_events = []
    for commodity in commodities:
        print(f"\n  {commodity}:")
        tee(f"\n  {commodity}:", report_buf)
        sub = panel[panel["commodity"] == commodity].sort_values(
            ["year", "month"]).copy()
        results = run_event_study(sub, commodity, EVENT_COLS, report_buf)
        all_events.extend(results)

    event_df = pd.DataFrame(all_events)
    if not event_df.empty:
        event_df.to_csv(os.path.join(
            RESULTS_DIR, "event_study.csv"), index=False)

    # ── 6. Placebo Test ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("6. PLACEBO TEST — MANGO")
    print(f"{'─'*65}")
    tee("\n=== Placebo Test (Mango) ===", report_buf)

    if "Mango" in commodities and not granger_df.empty:
        mango_g = granger_df[(granger_df["commodity"] == "Mango") & (
            granger_df["cause"] == "T0_kharif_rainfall")]
        tomato_g = granger_df[(granger_df["commodity"] == "Tomato") & (
            granger_df["cause"] == "T0_kharif_rainfall")]

        mango_sig = mango_g["significant"].any(
        ) if not mango_g.empty else False
        tomato_sig = tomato_g["significant"].any(
        ) if not tomato_g.empty else False

        if tomato_sig and not mango_sig:
            msg = "PLACEBO PASSED ✓ — T0 causality is commodity-specific (Tomato sig, Mango not sig)"
        elif not tomato_sig and not mango_sig:
            msg = "INCONCLUSIVE — neither Tomato nor Mango show significant T0 Granger causality"
        else:
            msg = "PLACEBO FAILED ✗ — T0 causality found in both Tomato and Mango (may be spurious)"

        print(f"\n  {msg}")
        tee(f"  {msg}", report_buf)

    # ── Save outputs ──────────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "var_summary.txt"), "w") as f:
        f.write(var_buf.getvalue())
    with open(os.path.join(RESULTS_DIR, "arimax_summary.txt"), "w") as f:
        f.write(arimax_buf.getvalue())
    with open(os.path.join(RESULTS_DIR, "analysis_report.txt"), "w") as f:
        f.write(report_buf.getvalue())

    print(f"\n{'=' * 65}")
    print("Analysis complete. Results saved cleanly.")
    print("=" * 65)


if __name__ == "__main__":
    run_analysis()
