"""
TalentPoint — Streamlit Frontend
Quant-assisted crypto investment decision-support tool.

Run:  streamlit run app.py
"""
import sys
import os
import pickle
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.service import list_backtest_assets, run_historical_backtest
from config import ASSET_UNIVERSE, SCORING_WEIGHTS, RISK_PROFILES, TRADING_MODES
from data.market_data import fetch_market_overview, fetch_ohlcv, fetch_all_asset_data
from data.news_data import fetch_news
from analysis.technical import score_technical, compute_indicators
from analysis.fundamental import score_fundamental
from analysis.sentiment import analyse_sentiment
from analysis.ml_forecast import forecast_asset
from scoring.engine import (
    compute_final_score,
    derive_trade_setup,
    determine_market_regime,
    generate_reasoning,
    rank_assets,
)
from strategy.entry_exit import compute_levels
from strategy.risk import compute_leverage
from utils.helpers import get_logger

log = get_logger("app")
CHECKPOINT_PATH = Path(__file__).with_name(".talentpoint_run_checkpoint.pkl")
LATEST_RESULTS_PATH = Path(__file__).with_name(".talentpoint_latest_results.pkl")
RISK_OPTIONS = ["conservative", "moderate", "aggressive"]
TRADING_MODE_OPTIONS = list(TRADING_MODES.keys())


def load_run_checkpoint() -> dict | None:
    """Load persisted run state from disk if it exists."""
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        with CHECKPOINT_PATH.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        log.warning("Failed to load run checkpoint: %s", exc)
        return None


def save_run_checkpoint(risk_level: str, trading_mode: str, next_index: int, results: dict):
    """Persist run state after each processed asset."""
    payload = {
        "risk_level": risk_level,
        "trading_mode": trading_mode,
        "next_index": next_index,
        "results": results,
        "saved_at": datetime.utcnow().isoformat(),
    }
    with CHECKPOINT_PATH.open("wb") as fh:
        pickle.dump(payload, fh)
    save_latest_results(risk_level, trading_mode, results)


def load_latest_results() -> dict | None:
    """Load the latest saved results snapshot, if available."""
    if not LATEST_RESULTS_PATH.exists():
        return None
    try:
        with LATEST_RESULTS_PATH.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        log.warning("Failed to load latest results snapshot: %s", exc)
        return None


def save_latest_results(risk_level: str, trading_mode: str, results: dict):
    """Persist the latest results so deep links can reopen reports."""
    payload = {
        "risk_level": risk_level,
        "trading_mode": trading_mode,
        "results": results,
        "saved_at": datetime.utcnow().isoformat(),
    }
    with LATEST_RESULTS_PATH.open("wb") as fh:
        pickle.dump(payload, fh)


def clear_run_checkpoint():
    """Delete any persisted run checkpoint."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


def get_resume_state(risk_level: str, trading_mode: str) -> dict | None:
    """Return resume metadata when an unfinished checkpoint matches the active mode/profile."""
    payload = load_run_checkpoint()
    if not payload:
        return None
    if payload.get("risk_level") != risk_level:
        return None
    if payload.get("trading_mode", "swing") != trading_mode:
        return None
    next_index = int(payload.get("next_index", 0))
    total_assets = len(ASSET_UNIVERSE)
    if next_index <= 0 or next_index >= total_assets:
        return None
    return payload


def mark_analysis_running():
    """Flip the UI into active-run mode."""
    st.session_state["analysis_running"] = True

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentPoint",
    page_icon="TP",
    layout="wide",
    initial_sidebar_state="expanded",
)

query_risk_level = str(st.query_params.get("risk", "moderate")).lower()
if query_risk_level not in RISK_OPTIONS:
    query_risk_level = "moderate"
query_trading_mode = str(st.query_params.get("mode", "swing")).lower()
if query_trading_mode not in TRADING_MODE_OPTIONS:
    query_trading_mode = "swing"
st.session_state.setdefault("analysis_running", False)
button_accent = "#22c55e" if st.session_state.get("analysis_running") else "#38bdf8"
button_accent_soft = "#16a34a" if st.session_state.get("analysis_running") else "#2563eb"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #08111f;
        --panel: rgba(10, 22, 39, 0.82);
        --panel-strong: rgba(9, 18, 32, 0.96);
        --panel-border: rgba(148, 163, 184, 0.18);
        --text-main: #f8fafc;
        --text-soft: #9fb0c7;
        --primary-color: #38bdf8;
        --accent: #f97316;
        --accent-2: #22c55e;
        --accent-3: #38bdf8;
        --danger: #fb7185;
        --warning: #fbbf24;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 26%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.15), transparent 24%),
            linear-gradient(180deg, #08111f 0%, #0b1325 52%, #08111f 100%);
        color: var(--text-main);
        font-family: "IBM Plex Sans", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 17, 31, 0.72);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1260px;
    }

    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: "Space Grotesk", sans-serif;
        letter-spacing: -0.02em;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(8, 17, 31, 0.98), rgba(11, 19, 37, 0.98)),
            linear-gradient(120deg, rgba(56, 189, 248, 0.08), rgba(249, 115, 22, 0.08));
        border-right: 1px solid rgba(148, 163, 184, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        padding: 1.8rem 1.8rem 1.5rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 28px;
        background:
            radial-gradient(circle at 85% 15%, rgba(249, 115, 22, 0.24), transparent 18%),
            radial-gradient(circle at 12% 24%, rgba(56, 189, 248, 0.18), transparent 22%),
            linear-gradient(135deg, rgba(11,19,37,0.96), rgba(9,18,32,0.88));
        box-shadow: 0 24px 70px rgba(2, 6, 23, 0.42);
        margin-bottom: 1rem;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #fbbf24;
        margin-bottom: 0.85rem;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.95), rgba(249, 115, 22, 0.95));
        color: #08111f;
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
        letter-spacing: -0.04em;
        box-shadow: 0 14px 28px rgba(8, 17, 31, 0.22);
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.95rem;
    }

    .brand-wordmark {
        display: flex;
        align-items: baseline;
        gap: 0.2rem;
        font-family: "Space Grotesk", sans-serif;
        font-size: 2.8rem;
        line-height: 0.95;
        letter-spacing: -0.05em;
    }

    .brand-wordmark .talent {
        color: #f8fafc;
        font-weight: 500;
    }

    .brand-wordmark .point {
        color: #f97316;
        font-weight: 700;
    }

    .brand-wordmark .dot {
        color: #38bdf8;
        font-weight: 700;
        margin: 0 0.04em;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 0.95;
        margin: 0 0 0.85rem 0;
        max-width: 8.5em;
    }

    .hero-copy {
        max-width: 58rem;
        font-size: 1rem;
        line-height: 1.7;
        color: var(--text-soft);
        margin: 0;
    }

    .glass-panel {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        box-shadow: 0 20px 55px rgba(2, 6, 23, 0.24);
    }

    .summary-tile {
        padding: 1.15rem 1.2rem;
        min-height: 136px;
        backdrop-filter: blur(14px);
    }

    .summary-label {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        color: var(--text-soft);
        margin-bottom: 0.8rem;
    }

    .summary-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 2.2rem;
        line-height: 1;
        margin-bottom: 0.55rem;
    }

    .summary-note {
        color: var(--text-soft);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .section-kicker {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        color: #fbbf24;
        margin-bottom: 0.45rem;
    }

    .pick-frame {
        padding: 1.25rem;
        margin: 1rem 0 1.25rem 0;
    }

    .pick-header {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        margin-bottom: 1rem;
    }

    .pick-rank {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-soft);
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.5rem;
    }

    .pick-symbol {
        font-size: 2rem;
        margin: 0;
    }

    .pick-subtitle {
        color: var(--text-soft);
        margin-top: 0.35rem;
        font-size: 0.96rem;
    }

    .score-pill {
        display: inline-flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: center;
        min-width: 140px;
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(249, 115, 22, 0.18));
        border: 1px solid rgba(255,255,255,0.1);
    }

    .score-pill strong {
        font-size: 1.7rem;
        font-family: "Space Grotesk", sans-serif;
    }

    .tone-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.22);
        color: var(--text-main);
    }

    .tone-badge.bullish { background: rgba(34, 197, 94, 0.15); color: #86efac; border-color: rgba(34, 197, 94, 0.28); }
    .tone-badge.bearish { background: rgba(251, 113, 133, 0.16); color: #fda4af; border-color: rgba(251, 113, 133, 0.28); }
    .tone-badge.neutral { background: rgba(251, 191, 36, 0.14); color: #fde68a; border-color: rgba(251, 191, 36, 0.26); }

    .micro-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 0.35rem 0;
    }

    .micro-card {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.16);
    }

    .micro-label {
        color: var(--text-soft);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }

    .micro-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.35rem;
    }

    .micro-sub {
        color: var(--text-soft);
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        background: rgba(10, 22, 39, 0.68);
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(10, 22, 39, 0.72);
    }

    .asset-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.7rem;
    }

    .asset-chip {
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: var(--text-main);
        font-size: 0.86rem;
    }

    .asset-spotlight {
        padding: 1.2rem 1.25rem;
        margin-bottom: 1rem;
    }

    .asset-spotlight h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1.7rem;
    }

    .asset-spotlight p {
        color: var(--text-soft);
        margin: 0.35rem 0 0 0;
        line-height: 1.6;
    }

    .alloc-card {
        padding: 0.85rem 0.95rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.14);
        margin-top: 0.65rem;
    }

    .alloc-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
        font-size: 0.92rem;
    }

    .alloc-bar {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.16);
        overflow: hidden;
    }

    .alloc-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.95), rgba(249, 115, 22, 0.95));
    }

    .universe-card {
        padding: 1rem 1rem 0.6rem 1rem;
        margin-bottom: 0.9rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(10, 22, 39, 0.72);
    }

    .universe-card.bullish {
        border-color: rgba(34, 197, 94, 0.28);
        box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.06);
    }

    .universe-card.bearish {
        border-color: rgba(251, 113, 133, 0.28);
        box-shadow: inset 0 0 0 1px rgba(251, 113, 133, 0.06);
    }

    .universe-card.neutral {
        border-color: rgba(251, 191, 36, 0.22);
        box-shadow: inset 0 0 0 1px rgba(251, 191, 36, 0.05);
    }

    .universe-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.9rem;
    }

    .status-card {
        padding: 1rem 1.1rem;
        margin: 0.75rem 0 1rem 0;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(10, 22, 39, 0.72);
    }

    .status-card.resume {
        border-color: rgba(56, 189, 248, 0.28);
        box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.05);
    }

    .status-card.fresh {
        border-color: rgba(249, 115, 22, 0.24);
        box-shadow: inset 0 0 0 1px rgba(249, 115, 22, 0.05);
    }

    [data-testid="stProgressBar"] > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%) !important;
    }

    [data-testid="stDataFrame"] [role="progressbar"] > div {
        background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%) !important;
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 2.35rem; }
        .micro-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] [data-testid="baseButton-primary"],
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {button_accent} 0%, {button_accent_soft} 100%) !important;
            background-color: {button_accent} !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #f8fafc !important;
            box-shadow: 0 12px 24px rgba(2, 6, 23, 0.28) !important;
        }}

        [data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover,
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.05);
            border-color: rgba(255, 255, 255, 0.14) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup" style="margin-bottom:0.55rem;">
            <div class="brand-mark">TP</div>
            <div>
                <div class="brand-wordmark" style="font-size:1.6rem; margin:0;">
                    <span class="talent">Talent</span><span class="dot">•</span><span class="point">Point</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Quant-assisted crypto analysis tool")
    st.caption("Market data: CoinPaprika + Yahoo Finance")

    st.divider()
    risk_level = st.radio(
        "Risk Tolerance",
        RISK_OPTIONS,
        index=RISK_OPTIONS.index(query_risk_level),
        format_func=lambda x: x.capitalize(),
        help="Determines leverage, stop-loss width, and ranking bias.",
    )
    profile = RISK_PROFILES[risk_level]
    st.info(profile["description"])

    trading_mode = st.radio(
        "Trading Mode",
        TRADING_MODE_OPTIONS,
        index=TRADING_MODE_OPTIONS.index(query_trading_mode),
        format_func=lambda key: f"{TRADING_MODES[key]['label']} ({TRADING_MODES[key]['holding_period_label']})",
        help="Controls the bar interval, forecast horizon, and backtesting profile.",
    )
    mode_profile = TRADING_MODES[trading_mode]
    st.caption(mode_profile["description"])
    st.caption(
        f"Data interval: `{mode_profile['yfinance_interval']}` | "
        f"Forecast horizon: `{mode_profile['forecast_horizon_bars']}` bars"
    )

    st.divider()
    st.markdown("**Scoring Weights**")
    for k, v in SCORING_WEIGHTS.items():
        adj = profile["weight_adjustments"].get(k, 1.0)
        effective = v * adj
        st.caption(f"{k.replace('_', ' ').title()}: {effective:.0%}")

    st.divider()
    resume_state = get_resume_state(risk_level, trading_mode)
    if resume_state:
        st.info(
            f"Resume available: {resume_state.get('next_index', 0)}/{len(ASSET_UNIVERSE)} assets processed.",
            icon="⏯️",
        )

    run_btn = st.button(
        "Run / Resume Analysis",
        type="primary",
        use_container_width=True,
        on_click=mark_analysis_running,
    )
    reset_run_btn = st.button("Start Fresh Run", use_container_width=True)

    st.divider()
    st.warning(
        "**Disclaimer:** All investments carry risk, including the possible loss of principal. "
        "Use of this tool is at your own risk. The builders and contributors are not liable "
        "for any losses, damages, or decisions made based on its outputs. Crypto markets are "
        "highly volatile, and leverage can lead to rapid liquidation.",
        icon="⚠️",
    )


# ── Helper: Build price chart ────────────────────────────────────────────────
def build_price_chart(
    df: pd.DataFrame,
    symbol: str,
    forecast_prices: list = None,
    trading_mode: str = "swing",
) -> go.Figure:
    """Create an OHLC chart with indicators and optional forecast overlay."""
    df = df.copy().tail(90)  # last 90 days for readability

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} Price", "RSI", "Volume"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
    ), row=1, col=1)

    # EMAs
    if "ema_short" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_short"], name="EMA 12",
            line=dict(color="#60a5fa", width=1),
        ), row=1, col=1)
    if "ema_long" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_long"], name="EMA 26",
            line=dict(color="#f472b6", width=1),
        ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color="#94a3b8", width=0.5, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Lower",
            line=dict(color="#94a3b8", width=0.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(148,163,184,0.05)",
        ), row=1, col=1)

    # Forecast overlay
    if forecast_prices:
        last_date = df.index[-1]
        interval = TRADING_MODES.get(trading_mode, TRADING_MODES["swing"])["yfinance_interval"]
        freq_map = {"1d": "D", "1h": "H", "15m": "15min"}
        freq = freq_map.get(interval, "D")
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=len(forecast_prices),
            freq=freq,
        )
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_prices, name="Forecast",
            line=dict(color="#a78bfa", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4),
        ), row=1, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#f59e0b", width=1),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # Volume
    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(size=11),
    )
    return fig


def format_price(value: float) -> str:
    """Format price according to magnitude for compact UI display."""
    if value >= 1000:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:,.4f}"
    return f"${value:,.6f}"


def tone_badge(label: str, tone: str) -> str:
    """Render a small pill badge with tone styling."""
    safe_tone = tone if tone in {"bullish", "bearish", "neutral"} else "neutral"
    return f'<span class="tone-badge {safe_tone}">{label}</span>'


def render_hero():
    """Render the landing hero section."""
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-eyebrow">Crypto Intelligence Desk</div>
            <div class="brand-lockup">
                <div class="brand-mark">TP</div>
                <div class="brand-wordmark">
                    <span class="talent">Talent</span><span class="dot">•</span><span class="point">Point</span>
                </div>
            </div>
            <p class="hero-copy">
                Track momentum, market structure, sentiment, and machine-learning forecasts in one place.
                The app ranks the current universe, then translates the strongest setups into entries,
                exits, leverage guidance, and supporting evidence.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_summary_tiles(all_results: dict, top_picks: list[dict], risk_level: str, regime: str):
    """Render high-level overview tiles above the picks."""
    avg_score = np.mean([r["final"]["final_score"] for r in all_results.values()])
    best_symbol = top_picks[0]["symbol"] if top_picks else "--"
    strongest_sent = max(all_results.values(), key=lambda r: r["sentiment"]["score"])["symbol"]
    tile1, tile2, tile3, tile4 = st.columns(4)

    with tile1:
        st.markdown(
            f"""
            <div class="glass-panel summary-tile">
                <div class="summary-label">Market Regime</div>
                <div class="summary-value">{regime.upper()}</div>
                <div class="summary-note">Composite view from technical, sentiment, and ML layers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tile2:
        st.markdown(
            f"""
            <div class="glass-panel summary-tile">
                <div class="summary-label">Top Candidate</div>
                <div class="summary-value">{best_symbol}</div>
                <div class="summary-note">Current leader under the <strong>{risk_level}</strong> profile.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tile3:
        st.markdown(
            f"""
            <div class="glass-panel summary-tile">
                <div class="summary-label">Avg Final Score</div>
                <div class="summary-value">{avg_score:.1f}</div>
                <div class="summary-note">Cross-universe conviction across {len(all_results)} tracked assets.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tile4:
        st.markdown(
            f"""
            <div class="glass-panel summary-tile">
                <div class="summary-label">Strongest Sentiment</div>
                <div class="summary-value">{strongest_sent}</div>
                <div class="summary-note">Highest current sentiment score in the active run.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_health_check(all_results: dict):
    """Render a compact pipeline health summary."""
    ml_statuses = [
        result.get("ml_forecast", {}).get("direction", {}).get("model_status", "unknown")
        for result in all_results.values()
    ]
    ml_ok = sum(status == "ok" for status in ml_statuses)
    total_assets = len(all_results)
    sentiment_articles = sum(
        result.get("sentiment", {}).get("article_count", 0) for result in all_results.values()
    )
    ohlcv_ready = sum(0 if result.get("ohlcv", pd.DataFrame()).empty else 1 for result in all_results.values())
    overall_state = "Healthy" if ml_ok == total_assets and ohlcv_ready == total_assets else "Degraded"
    state_tone = "bullish" if overall_state == "Healthy" else "bearish"

    st.markdown('<div class="section-kicker">System Health</div>', unsafe_allow_html=True)
    st.markdown(
        f"### Pipeline status {tone_badge(overall_state.upper(), state_tone)}",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets Analysed", f"{total_assets}")
    c2.metric("ML Active", f"{ml_ok}/{total_assets}")
    c3.metric("Price Feeds OK", f"{ohlcv_ready}/{total_assets}")
    c4.metric("News Items", f"{sentiment_articles}")

    if ml_ok != total_assets:
        st.error("ML classifier fallback detected for one or more assets. Check the XGBoost runtime.")
    else:
        st.success("ML classifier is active across the analysed asset set.")


def _format_factor_value(value) -> str:
    """Make report values readable in markdown."""
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_asset_review(pick: dict, rank: int | None = None, total_assets: int | None = None) -> dict:
    """Create a recommendation verdict and plain-language strengths/weaknesses."""
    final_score = float(pick.get("final", {}).get("final_score", 0))
    technical = pick.get("technical", {})
    fundamental = pick.get("fundamental", {})
    sentiment = pick.get("sentiment", {})
    ml = pick.get("ml_forecast", {})
    trade_setup = pick.get("trade_setup") or derive_trade_setup(technical, ml, final_score)
    trade_direction = trade_setup.get("direction", "neutral")
    opportunity_score = float(trade_setup.get("opportunity_score", final_score))

    strengths = []
    headwinds = []

    tech_score = float(technical.get("score", 0))
    sent_score = float(sentiment.get("score", 0))
    ml_direction = ml.get("direction", {}).get("direction", "neutral")
    ml_confidence = float(ml.get("direction", {}).get("confidence", 0))
    trend = technical.get("trend", "neutral")

    if trade_direction == "short":
        if tech_score <= 40:
            strengths.append(f"technical structure is weak at {tech_score:.1f}/100, which supports a short")
        elif tech_score > 55:
            headwinds.append(f"technical structure is still fairly strong at {tech_score:.1f}/100")

        if fundamental.get("score", 0) < 45:
            strengths.append(f"fundamentals are soft at {fundamental.get('score', 0):.1f}/100")
        elif fundamental.get("score", 0) >= 60:
            headwinds.append(f"fundamentals remain supportive at {fundamental.get('score', 0):.1f}/100")

        if sent_score < 45:
            strengths.append(f"news sentiment is negative at {sent_score:.1f}/100")
        elif sent_score > 60:
            headwinds.append(f"news sentiment is still constructive at {sent_score:.1f}/100")

        if ml_direction == "bearish":
            strengths.append(f"the ML forecast leans bearish with {ml_confidence:.0f}% confidence")
        else:
            headwinds.append(f"the ML forecast is not decisively bearish ({ml_direction}, {ml_confidence:.0f}% confidence)")

        if trend == "bearish":
            strengths.append("trend regime is currently bearish")
        elif trend == "bullish":
            headwinds.append("trend regime is currently bullish")
    else:
        if tech_score >= 60:
            strengths.append(f"technical structure is supportive at {tech_score:.1f}/100")
        elif tech_score < 45:
            headwinds.append(f"technical structure is weak at {tech_score:.1f}/100")

        if fundamental.get("score", 0) >= 60:
            strengths.append(f"fundamentals are relatively strong at {fundamental.get('score', 0):.1f}/100")
        elif fundamental.get("score", 0) < 45:
            headwinds.append(f"fundamentals are soft at {fundamental.get('score', 0):.1f}/100")

        if sent_score >= 60:
            strengths.append(f"news sentiment is constructive at {sent_score:.1f}/100")
        elif sent_score < 45:
            headwinds.append(f"news sentiment is unsupportive at {sent_score:.1f}/100")

        if ml_direction == "bullish":
            strengths.append(f"the ML forecast leans bullish with {ml_confidence:.0f}% confidence")
        elif ml_direction == "bearish":
            headwinds.append(f"the ML forecast is bearish ({ml_direction}, {ml_confidence:.0f}% confidence)")

        if trend == "bullish":
            strengths.append("trend regime is currently bullish")
        elif trend == "bearish":
            headwinds.append("trend regime is currently bearish")

    if not strengths:
        strengths.append("there are some mixed signals, but none are strong enough to create high conviction")
    if not headwinds:
        headwinds.append("the setup has fewer visible weaknesses, but stronger alternatives ranked ahead")

    if rank is not None and rank <= 3 and trade_direction in {"long", "short"}:
        status = "Short Candidate" if trade_direction == "short" else "Long Candidate"
        verdict = f"{pick['symbol']} is currently a top-3 {trade_direction} setup in this run."
    elif opportunity_score >= 60 and trade_direction in {"long", "short"}:
        status = "Watchlist"
        verdict = f"{pick['symbol']} has a usable {trade_direction} bias, but it did not make the top 3 setups."
    else:
        status = "No Trade"
        verdict = f"{pick['symbol']} is not currently a high-conviction long or short setup."

    rank_text = "N/A"
    if rank is not None and total_assets is not None:
        rank_text = f"{rank}/{total_assets}"
        if rank > 3 and status not in {"Long Candidate", "Short Candidate"}:
            verdict += f" It is currently ranked {rank_text}."

    return {
        "status": status,
        "verdict": verdict,
        "strengths": strengths[:3],
        "headwinds": headwinds[:4],
        "rank_text": rank_text,
        "final_score": final_score,
        "trade_direction": trade_direction,
        "opportunity_score": opportunity_score,
    }


def render_asset_review_summary(review: dict):
    """Render a quick recommendation summary for the selected asset."""
    tone_map = {
        "Long Candidate": "bullish",
        "Short Candidate": "bearish",
        "Watchlist": "neutral",
        "No Trade": "neutral",
    }
    tone = tone_map.get(review["status"], "neutral")
    st.markdown(
        f"### Recommendation {tone_badge(review['status'].upper(), tone)}",
        unsafe_allow_html=True,
    )
    st.markdown(review["verdict"])
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**What is working**")
        for item in review["strengths"]:
            st.caption(f"+ {item}")
    with cols[1]:
        st.markdown("**What is holding it back**")
        for item in review["headwinds"]:
            st.caption(f"- {item}")


def generate_transparency_report(
    pick: dict,
    risk_level: str,
    trading_mode: str,
    rank: int | None = None,
    total_assets: int | None = None,
) -> str:
    """Generate a detailed transparency report for a single asset."""
    symbol = pick["symbol"]
    current_price = float(pick["current_price"])
    technical = pick.get("technical", {})
    fundamental = pick.get("fundamental", {})
    sentiment = pick.get("sentiment", {})
    ml = pick.get("ml_forecast", {})
    final = pick.get("final", {})
    trade_setup = pick.get("trade_setup") or derive_trade_setup(technical, ml, final.get("final_score", 50))
    ohlcv = pick.get("ohlcv", pd.DataFrame())
    levels = compute_levels(
        current_price,
        technical,
        ml,
        risk_level,
        trading_mode,
        trade_setup=trade_setup,
    )
    review = build_asset_review(pick, rank=rank, total_assets=total_assets)

    daily_vol = 0.0
    if not ohlcv.empty:
        daily_vol = float(ohlcv["Close"].pct_change().tail(30).std() * 100)
    leverage_info = compute_leverage(
        risk_level,
        daily_vol,
        ml.get("direction", {}).get("confidence", 50),
    )

    lines = [
        f"# {symbol} Transparency Report",
        "",
        f"- Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Risk profile: {risk_level}",
        f"- Trading mode: {trading_mode} ({TRADING_MODES[trading_mode]['holding_period_label']})",
        f"- Current price: {format_price(current_price)}",
        f"- Final score: {final.get('final_score', 0):.1f}/100",
        f"- Trade direction: {trade_setup.get('direction', 'neutral')}",
        f"- Opportunity score: {trade_setup.get('opportunity_score', 0):.1f}/100",
        f"- Recommendation status: {review['status']}",
        f"- Rank in universe: {review['rank_text']}",
        "",
        "## Data Sources",
        "- OHLCV / price history: Yahoo Finance via yfinance",
        "- Market metadata: CoinPaprika",
        "- News sentiment: editorial RSS, crypto community feeds, market-mood context, plus optional CryptoPanic",
        "- Developer activity: GitHub API where repository metadata is configured",
        "",
        "## Final Score Composition",
    ]

    for name, component in final.get("components", {}).items():
        lines.append(
            f"- {name.replace('_', ' ').title()}: raw {component.get('raw', 0):.1f}, "
            f"weight {component.get('weight', 0):.0%}, weighted contribution {component.get('weighted', 0):.2f}"
        )

    lines.extend([
        "",
        "## Technical Score Detail",
        f"- Technical trend: {technical.get('trend', 'neutral')}",
    ])
    for name, info in technical.get("signals", {}).items():
        lines.append(
            f"- {name.replace('_', ' ').title()}: score {info.get('score', 0):.1f}, signal {info.get('signal', 'neutral')}"
        )

    sr = technical.get("support_resistance", {})
    if sr:
        lines.extend([
            "",
            "### Support / Resistance Snapshot",
            f"- Support: {_format_factor_value(sr.get('support'))}",
            f"- Resistance: {_format_factor_value(sr.get('resistance'))}",
            f"- Pivot: {_format_factor_value(sr.get('pivot'))}",
        ])

    lines.extend([
        "",
        "## Fundamental Score Detail",
    ])
    for name, info in fundamental.get("factors", {}).items():
        lines.append(
            f"- {name.replace('_', ' ').title()}: value {_format_factor_value(info.get('value'))}, "
            f"score {info.get('score', 0):.1f}"
        )

    lines.extend([
        "",
        "## Sentiment Score Detail",
        f"- Sentiment score: {sentiment.get('score', 0):.1f}/100",
        f"- Sentiment trend: {sentiment.get('trend', 'stable')}",
        f"- Article count: {sentiment.get('article_count', 0)}",
        f"- Weighted source count: {sentiment.get('weighted_article_count', 0)}",
        f"- Positive / Negative / Neutral: "
        f"{sentiment.get('positive_pct', 0):.0f}% / "
        f"{sentiment.get('negative_pct', 0):.0f}% / "
        f"{sentiment.get('neutral_pct', 0):.0f}%",
    ])
    for source_type, source_data in sentiment.get("source_breakdown", {}).items():
        lines.append(
            f"- {source_type.replace('_', ' ').title()}: "
            f"{source_data.get('count', 0)} items, weight {source_data.get('weight', 0):.2f}"
        )
    if sentiment.get("top_positive"):
        lines.append("- Top positive headlines:")
        for item in sentiment["top_positive"][:3]:
            lines.append(f"  - {item['title']} ({item['score']:.3f})")
    if sentiment.get("top_negative"):
        lines.append("- Top negative headlines:")
        for item in sentiment["top_negative"][:3]:
            lines.append(f"  - {item['title']} ({item['score']:.3f})")

    direction = ml.get("direction", {})
    forecast = ml.get("forecast", {})
    lines.extend([
        "",
        "## ML Forecast Detail",
        f"- ML score: {ml.get('score', 0):.1f}/100",
        f"- Direction: {direction.get('direction', 'neutral')}",
        f"- Confidence: {direction.get('confidence', 0):.1f}%",
        f"- Cross-validation accuracy: {direction.get('cv_accuracy', 0):.4f}",
        f"- Model status: {direction.get('model_status', 'unknown')}",
        f"- Expected return from forecast: {forecast.get('expected_return_pct', 0):+.2f}%",
        f"- Forecast mean: {_format_factor_value(forecast.get('forecast_mean'))}",
        f"- Forecast low/high: {_format_factor_value(forecast.get('forecast_low'))} / {_format_factor_value(forecast.get('forecast_high'))}",
    ])

    lines.extend([
        "",
        "## Recommendation Verdict",
        f"- Summary: {review['verdict']}",
        "- What is working:",
    ])
    for item in review["strengths"]:
        lines.append(f"  - {item}")
    lines.append("- What is holding it back:")
    for item in review["headwinds"]:
        lines.append(f"  - {item}")

    lines.extend([
        "",
        "## Trade Levels And Why They Were Chosen",
        f"- Trade side: {levels.get('trade_label', levels.get('trade_direction', 'Long Setup'))}",
        f"- Entry price: {format_price(levels['entry_price'])}",
        f"- Stop-loss: {format_price(levels['stop_loss'])}",
        f"- Take-profit / exit: {format_price(levels['exit_price'])}",
        f"- Risk / Reward: {levels['risk_reward_ratio']:.2f}:1",
        f"- Expected return: {levels['expected_return_pct']:+.2f}%",
        f"- Suggested leverage: {leverage_info.get('leverage', 1)}x",
        f"- Entry / exit methodology: {levels.get('methodology', '')}",
        f"- Leverage rationale: {leverage_info.get('rationale', '')}",
        "",
        "## Interpretation Notes",
        "- The score is a weighted synthesis of multiple independent signals rather than a guarantee.",
        "- Trade levels are model-driven suggestions built from current data, not certainty about future price action.",
        "- All outputs depend on live third-party data and may change between runs.",
    ])

    return "\n".join(lines)


def render_transparency_report(
    pick: dict,
    risk_level: str,
    trading_mode: str,
    key_prefix: str,
    rank: int | None = None,
    total_assets: int | None = None,
):
    """Render an openable transparency report plus download link for a given asset."""
    report = generate_transparency_report(
        pick,
        risk_level,
        trading_mode,
        rank=rank,
        total_assets=total_assets,
    )
    expand_report = (
        st.session_state.get("report_focus_symbol") == pick["symbol"]
        and "detail" in key_prefix
    )
    with st.expander("Transparency & Review Report", expanded=expand_report):
        st.markdown(report)
        st.download_button(
            "Download Report",
            data=report,
            file_name=f"{pick['symbol'].lower()}_transparency_report.md",
            mime="text/markdown",
            key=f"{key_prefix}-download-report",
            use_container_width=True,
        )


def render_pick_banner(rank_num: int, symbol: str, trend: str, setup_label: str, opportunity_score: float, reasoning: str):
    """Render a stylized header for each top pick."""
    summary_line = reasoning.splitlines()[0] if reasoning else ""
    st.markdown(
        f"""
        <div class="glass-panel pick-frame">
            <div class="pick-header">
                <div>
                    <div class="pick-rank">Top pick #{rank_num} {tone_badge(trend, trend)}</div>
                    <h3 class="pick-symbol">{symbol}</h3>
                    <div class="pick-subtitle">{summary_line}</div>
                </div>
                <div class="score-pill">
                    <span class="summary-label">{setup_label}</span>
                    <strong>{opportunity_score:.1f}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_strip(levels: dict, leverage_info: dict, ml: dict):
    """Render compact cards for the most actionable numbers."""
    ml_direction = ml.get("direction", {}).get("direction", "neutral").capitalize()
    ml_conf = ml.get("direction", {}).get("confidence", 0)
    trade_label = levels.get("trade_label", "Trade Setup")
    st.markdown(
        f"""
        <div class="micro-grid">
            <div class="micro-card">
                <div class="micro-label">{trade_label}</div>
                <div class="micro-value">{format_price(levels['entry_price'])}</div>
                <div class="micro-sub">Target {format_price(levels['exit_price'])}</div>
            </div>
            <div class="micro-card">
                <div class="micro-label">Protection</div>
                <div class="micro-value">{format_price(levels['stop_loss'])}</div>
                <div class="micro-sub">Risk-reward {levels['risk_reward_ratio']:.1f}:1</div>
            </div>
            <div class="micro-card">
                <div class="micro-label">Positioning</div>
                <div class="micro-value">{leverage_info['leverage']}x</div>
                <div class="micro-sub">Expected return {levels['expected_return_pct']:+.1f}%</div>
            </div>
            <div class="micro-card">
                <div class="micro-label">Model View</div>
                <div class="micro-value">{ml_direction}</div>
                <div class="micro-sub">{ml_conf:.0f}% confidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_asset_universe():
    """Render a chip-style asset universe list on the empty state."""
    chips = "".join(
        f'<span class="asset-chip">{meta["symbol"]} <span style="color:#9fb0c7;">{cid}</span></span>'
        for cid, meta in ASSET_UNIVERSE.items()
    )
    st.markdown(f'<div class="asset-chip-wrap">{chips}</div>', unsafe_allow_html=True)


def build_price_series(ohlcv: pd.DataFrame, points: int = 24) -> list[float]:
    """Return a compact close-price series for sparkline-style rendering."""
    if ohlcv.empty or "Close" not in ohlcv:
        return []
    closes = ohlcv["Close"].tail(points).astype(float)
    return [round(float(v), 6) for v in closes.tolist()]


def score_bar_text(value: float, width: int = 10) -> str:
    """Render a compact text bar to avoid Streamlit's red default progress cells."""
    filled = max(0, min(width, round((value / 100) * width)))
    return f"{'█' * filled}{'░' * (width - filled)} {value:.0f}"


def sparkline_text(values: list[float]) -> str:
    """Render a unicode sparkline for stable color control inside the dataframe."""
    if not values:
        return "N/A"
    bars = "▁▂▃▄▅▆▇█"
    low = min(values)
    high = max(values)
    if high == low:
        return bars[3] * min(len(values), 18)
    scaled = [
        bars[min(len(bars) - 1, int((value - low) / (high - low) * (len(bars) - 1)))]
        for value in values[-18:]
    ]
    return "".join(scaled)


def render_analysis_progress(placeholder, percent: int, text: str):
    """Render a blue analysis progress bar using app-controlled markup."""
    safe_percent = max(0, min(100, int(percent)))
    placeholder.markdown(
        f"""
        <div style="margin:0.5rem 0 1.15rem 0;">
            <div style="color:#f8fafc; font-size:0.95rem; margin-bottom:0.45rem;">{text}</div>
            <div style="width:100%; height:14px; border-radius:999px; background:rgba(148,163,184,0.18); overflow:hidden;">
                <div style="width:{safe_percent}%; height:100%; border-radius:999px; background:linear-gradient(90deg, #38bdf8 0%, #2563eb 100%);"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trend_label(trend: str) -> str:
    """Create a compact, scan-friendly trend label."""
    icons = {"bullish": "▲", "bearish": "▼", "neutral": "●"}
    safe = trend if trend in icons else "neutral"
    return f"{icons[safe]} {safe.capitalize()}"


def sentiment_label(sentiment: dict) -> str:
    """Return a compact sentiment summary string."""
    score = sentiment.get("score", 50)
    if score >= 60:
        return "Positive"
    if score <= 40:
        return "Negative"
    return "Balanced"


def build_asset_report_href(symbol: str, risk_level: str, trading_mode: str) -> str:
    """Build a deep link that opens the selected asset report."""
    return f"?asset={quote(symbol)}&focus=report&risk={quote(risk_level)}&mode={quote(trading_mode)}"


def build_asset_detail_href(symbol: str, risk_level: str, trading_mode: str) -> str:
    """Build a deep link that opens the selected asset detail view."""
    return f"?asset={quote(symbol)}&risk={quote(risk_level)}&mode={quote(trading_mode)}"


def rankings_recommendation(review: dict) -> str:
    """Collapse review status into a scan-friendly trade setup label."""
    status = review.get("status")
    if status == "Long Candidate":
        return "Long"
    if status == "Short Candidate":
        return "Short"
    if status == "Watchlist":
        return "Watchlist"
    return "No Trade"


def render_rankings_table(all_results: dict, risk_level: str, trading_mode: str) -> str | None:
    """Render a custom rankings table with blue score bars and line sparklines."""
    rows = []
    for cid, data in sorted(
        all_results.items(),
        key=lambda x: (
            1 if x[1].get("trade_setup", {}).get("direction") in {"long", "short"} else 0,
            float(x[1].get("trade_setup", {}).get("opportunity_score", x[1]["final"]["final_score"])),
            float(x[1]["final"]["final_score"]),
        ),
        reverse=True,
    ):
        rank = len(rows) + 1
        review = build_asset_review({"coin_id": cid, **data}, rank=rank, total_assets=len(all_results))
        rows.append({
            "Rank": rank,
            "Symbol": data["symbol"],
            "Recommendation": rankings_recommendation(review),
            "Side": data.get("trade_setup", {}).get("direction", "neutral").capitalize(),
            "AssetHref": build_asset_detail_href(data["symbol"], risk_level, trading_mode),
            "Report": build_asset_report_href(data["symbol"], risk_level, trading_mode),
            "Price": float(data["current_price"]),
            "Trend": trend_label(data["technical"]["trend"]),
            "Technical": float(data["technical"]["score"]),
            "Fundamental": float(data["fundamental"]["score"]),
            "Sentiment": float(data["sentiment"]["score"]),
            "ML": float(data["ml_forecast"]["score"]),
            "Final Score": float(data["final"]["final_score"]),
            "Opportunity Score": float(data.get("trade_setup", {}).get("opportunity_score", data["final"]["final_score"])),
            "Narrative": sentiment_label(data["sentiment"]),
            "Sparkline": build_price_series(data.get("ohlcv", pd.DataFrame())),
        })

    def html_escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def score_bar_html(value: float) -> str:
        width = max(4, min(100, int(round(value))))
        return (
            '<div style="display:flex; align-items:center; gap:0.55rem; min-width:150px;">'
            '<div style="flex:1; min-width:96px; height:10px; border-radius:999px; '
            'background:rgba(148,163,184,0.16); overflow:hidden;">'
            f'<div style="width:{width}%; height:100%; border-radius:999px; '
            'background:linear-gradient(90deg, #7dd3fc 0%, #38bdf8 100%);"></div>'
            '</div>'
            f'<span style="color:#7dd3fc; font-weight:700;">{value:.0f}</span>'
            '</div>'
        )

    def sparkline_svg(values: list[float]) -> str:
        if not values:
            return '<span style="color:#9fb0c7;">N/A</span>'
        points = values[-18:]
        low = min(points)
        high = max(points)
        width = 120
        height = 28
        if high == low:
            coords = " ".join(f"{int(i * (width / max(len(points) - 1, 1))):.0f},{height/2:.1f}" for i in range(len(points)))
        else:
            coords = []
            for i, value in enumerate(points):
                x = i * (width / max(len(points) - 1, 1))
                y = height - (((value - low) / (high - low)) * (height - 4) + 2)
                coords.append(f"{x:.1f},{y:.1f}")
            coords = " ".join(coords)
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<polyline fill="none" stroke="#60a5fa" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{coords}" />'
            '</svg>'
        )

    table_rows = []
    for row in rows:
        rec_color = {
            "Long": "#86efac",
            "Short": "#fda4af",
            "Watchlist": "#fde68a",
            "No Trade": "#cbd5e1",
        }.get(row["Recommendation"], "#cbd5e1")
        table_rows.append(
            f"""
            <tr>
                <td>{row['Rank']}</td>
                <td><a href="{row['AssetHref']}" target="_self" style="color:#f8fafc; text-decoration:none; font-weight:700;">{html_escape(row['Symbol'])}</a></td>
                <td><span style="color:{rec_color}; font-weight:700;">{html_escape(row['Recommendation'])}</span></td>
                <td><a href="{row['Report']}" target="_self" style="color:#7dd3fc; text-decoration:none; font-weight:600;">Open report</a></td>
                <td style="color:#e2e8f0;">{format_price(row['Price'])}</td>
                <td style="color:#e2e8f0;">{html_escape(row['Side'])}</td>
                <td style="color:#e2e8f0;">{html_escape(row['Trend'])}</td>
                <td>{score_bar_html(row['Technical'])}</td>
                <td>{score_bar_html(row['Fundamental'])}</td>
                <td>{score_bar_html(row['Sentiment'])}</td>
                <td>{score_bar_html(row['ML'])}</td>
                <td>{score_bar_html(row['Final Score'])}</td>
                <td>{score_bar_html(row['Opportunity Score'])}</td>
                <td style="color:#e2e8f0;">{html_escape(row['Narrative'])}</td>
                <td>{sparkline_svg(row['Sparkline'])}</td>
            </tr>
            """
        )

    table_html = f"""
        <div style="overflow-x:auto; border:1px solid rgba(148,163,184,0.16); border-radius:20px; background:rgba(10,22,39,0.72);">
            <table style="width:100%; border-collapse:collapse; min-width:1180px;">
                <thead>
                    <tr style="background:rgba(15,23,42,0.72); color:#9fb0c7; text-transform:none;">
                        <th style="padding:0.9rem 1rem; text-align:left;">Rank</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Asset</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Recommendation</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Report</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Price</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Side</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Trend</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Technical</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Fundamental</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Sentiment</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">ML</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Final Score</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Opportunity</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Sentiment Tone</th>
                        <th style="padding:0.9rem 1rem; text-align:left;">Price Action</th>
                    </tr>
                </thead>
                <tbody style="color:#f8fafc;">
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
        <style>
            tbody tr:not(:last-child) td {{ border-bottom: 1px solid rgba(148,163,184,0.12); }}
            tbody td {{ padding:0.8rem 1rem; vertical-align:middle; }}
        </style>
    """
    components.html(table_html, height=72 + 56 * max(len(rows), 1), scrolling=True)
    return None


def render_asset_detail_panel(
    pick: dict,
    risk_level: str,
    trading_mode: str,
    rank: int | None = None,
    total_assets: int | None = None,
    panel_key_prefix: str = "detail",
):
    """Render a focused detail panel for a selected asset."""
    symbol = pick["symbol"]
    current_price = pick["current_price"]
    final = pick["final"]
    trade_setup = pick.get("trade_setup") or derive_trade_setup(pick.get("technical", {}), pick.get("ml_forecast", {}), final.get("final_score", 50))
    tech = pick["technical"]
    ml = pick["ml_forecast"]
    ohlcv = pick.get("ohlcv", pd.DataFrame())
    reasoning = generate_reasoning(pick)
    levels = compute_levels(
        current_price,
        tech,
        ml,
        risk_level,
        trading_mode,
        trade_setup=trade_setup,
    )

    daily_vol = 0.0
    if not ohlcv.empty:
        daily_vol = float(ohlcv["Close"].pct_change().tail(30).std() * 100)
    leverage_info = compute_leverage(
        risk_level,
        daily_vol,
        ml.get("direction", {}).get("confidence", 50),
    )

    st.markdown(
        f"""
        <div class="glass-panel asset-spotlight">
            <div class="section-kicker">Selected Asset</div>
            <h3>{symbol} {tone_badge(levels.get('trade_direction', trade_setup.get('bias', 'neutral')).capitalize(), trade_setup.get('bias', 'neutral'))}</h3>
            <p>{reasoning.splitlines()[0] if reasoning else ''}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_transparency_report(
        pick,
        risk_level,
        trading_mode,
        f"{panel_key_prefix}-{symbol}",
        rank=rank,
        total_assets=total_assets,
    )

    render_asset_review_summary(build_asset_review(pick, rank=rank, total_assets=total_assets))
    render_signal_strip(levels, leverage_info, ml)
    a, b, c, d, e = st.columns(5)
    a.metric("Current Price", format_price(current_price))
    b.metric("Trade Side", levels.get("trade_label", levels.get("trade_direction", "Long")).replace(" Setup", ""))
    c.metric("Entry", format_price(levels["entry_price"]))
    d.metric("Stop", format_price(levels["stop_loss"]))
    e.metric("Take Profit", format_price(levels["exit_price"]))

    with st.expander("Selected Asset Chart", expanded=True):
        if not ohlcv.empty:
            forecast_prices = ml.get("forecast", {}).get("forecast_prices", [])
            st.plotly_chart(
                build_price_chart(ohlcv, symbol, forecast_prices, trading_mode=trading_mode),
                use_container_width=True,
                key=f"{panel_key_prefix}-chart-{symbol}-{rank or 'na'}",
            )
        else:
            st.info("No chart data available.")


def render_sidebar_portfolio(top_picks: list[dict], risk_level: str):
    """Render a direction-aware positioning idea in the sidebar."""
    if not top_picks:
        return

    setups = [
        p.get("trade_setup")
        or derive_trade_setup(
            p.get("technical", {}),
            p.get("ml_forecast", {}),
            p.get("final", {}).get("final_score", 50),
        )
        for p in top_picks
    ]
    weights = [
        max(
            setup.get("opportunity_score", pick["final"]["final_score"]),
            1.0,
        )
        for pick, setup in zip(top_picks, setups)
    ]
    total = sum(weights)
    with st.sidebar:
        st.divider()
        st.markdown("### Suggested Positioning")
        st.caption(
            f"Opportunity-weighted mix for the current `{risk_level}` profile. "
            "Long setups imply long exposure; short setups imply hedge or short exposure."
        )
        for pick, weight, setup in zip(top_picks, weights, setups):
            pct = weight / total * 100
            side = setup.get("direction", "neutral").capitalize()
            st.markdown(
                f"""
                <div class="alloc-card">
                    <div class="alloc-head">
                        <strong>{pick['symbol']}</strong>
                        <span>{side} • {pct:.0f}%</span>
                    </div>
                    <div class="alloc-bar">
                        <div class="alloc-fill" style="width: {pct:.0f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def ensure_ui_state():
    """Initialize UI-related session state."""
    st.session_state.setdefault("selected_symbol", None)
    st.session_state.setdefault("report_focus_symbol", None)
    st.session_state.setdefault("watchlist", [])
    st.session_state.setdefault("backtests", {})


def set_selected_symbol(symbol: str | None):
    """Persist the currently focused asset."""
    st.session_state["selected_symbol"] = symbol


def hydrate_selection_from_query_params():
    """Hydrate selected asset and report focus from query parameters."""
    asset = st.query_params.get("asset")
    focus = st.query_params.get("focus")
    if asset:
        set_selected_symbol(str(asset))
        if focus == "report":
            st.session_state["report_focus_symbol"] = str(asset)


def bootstrap_results_for_deep_link(risk_level: str, trading_mode: str):
    """Restore saved results when a deep-linked asset report is requested."""
    if "results" in st.session_state:
        return

    requested_asset = st.session_state.get("selected_symbol")
    if not requested_asset:
        return

    checkpoint = get_resume_state(risk_level, trading_mode)
    if checkpoint and requested_asset in {data.get("symbol") for data in checkpoint.get("results", {}).values()}:
        st.session_state["results"] = checkpoint["results"]
        st.session_state["risk_level"] = checkpoint.get("risk_level", risk_level)
        st.session_state["trading_mode"] = checkpoint.get("trading_mode", trading_mode)
        return

    latest = load_latest_results()
    if latest and requested_asset in {data.get("symbol") for data in latest.get("results", {}).values()}:
        st.session_state["results"] = latest["results"]
        st.session_state["risk_level"] = latest.get("risk_level", risk_level)
        st.session_state["trading_mode"] = latest.get("trading_mode", trading_mode)


def add_to_watchlist(symbol: str):
    """Add an asset to the watchlist if it is not already present."""
    watchlist = st.session_state.setdefault("watchlist", [])
    if symbol not in watchlist:
        watchlist.append(symbol)


def remove_from_watchlist(symbol: str):
    """Remove an asset from the watchlist."""
    watchlist = st.session_state.setdefault("watchlist", [])
    st.session_state["watchlist"] = [item for item in watchlist if item != symbol]

def render_live_runboard(
    all_results: dict,
    total_assets: int,
    risk_level: str,
    trading_mode: str,
    latest_symbol: str | None = None,
    resumed: bool = False,
    processed_count: int | None = None,
):
    """Render live partial results while a scan is still running."""
    st.markdown('<div class="section-kicker">Live Runboard</div>', unsafe_allow_html=True)
    st.subheader("Partial results are available while the scan continues.")

    ranked_count = len(all_results)
    processed = processed_count if processed_count is not None else ranked_count
    skipped_count = max(processed - ranked_count, 0)
    current_leader = "--"
    avg_score = 0.0
    if all_results:
        ranked = rank_assets(all_results, top_n=1)
        current_leader = ranked[0]["symbol"] if ranked else "--"
        avg_score = float(np.mean([r["final"]["final_score"] for r in all_results.values()]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed", f"{processed}/{total_assets}")
    c2.metric("Current Leader", current_leader)
    c3.metric("Latest Attempt", latest_symbol or "--")
    c4.metric("Mode", "Resume" if resumed else "Fresh")

    c5, c6, c7 = st.columns(3)
    c5.metric("Ranked", f"{ranked_count}")
    c6.metric("Skipped / Failed", f"{skipped_count}")
    c7.metric("Avg Score So Far", f"{avg_score:.1f}")

    st.caption(
        "Processed assets include attempts that were skipped or failed. Rolling rankings only include assets with a completed analysis."
    )

    if all_results:
        ranked_assets = rank_assets(all_results, top_n=len(all_results))
        rank_lookup = {item["symbol"]: idx + 1 for idx, item in enumerate(ranked_assets)}
        latest_pick = None
        if latest_symbol:
            for cid, data in all_results.items():
                if data["symbol"] == latest_symbol:
                    latest_pick = {"coin_id": cid, **data}
                    break
        if latest_pick is None:
            latest_pick = ranked_assets[0]

        st.markdown("### Latest Completed Asset")
        render_asset_detail_panel(
            latest_pick,
            risk_level,
            trading_mode,
            rank=rank_lookup.get(latest_pick["symbol"]),
            total_assets=len(all_results),
            panel_key_prefix=f"live-detail-{processed}",
        )

        st.markdown("### Rolling Rankings")
        render_rankings_table(all_results, risk_level, trading_mode)


def render_resume_status_card(risk_level: str, trading_mode: str):
    """Render a visible checkpoint/resume status card."""
    resume_state = get_resume_state(risk_level, trading_mode)
    total_assets = len(ASSET_UNIVERSE)

    if resume_state:
        next_index = int(resume_state.get("next_index", 0))
        completed = next_index
        assets = list(ASSET_UNIVERSE.items())
        next_symbol = assets[next_index][1]["symbol"] if next_index < total_assets else "--"
        saved_at_raw = resume_state.get("saved_at")
        saved_at = saved_at_raw.replace("T", " ")[:19] + " UTC" if saved_at_raw else "Unknown"
        st.markdown(
            f"""
            <div class="status-card resume">
                <div class="section-kicker">Resume Ready</div>
                <strong>{completed}/{total_assets} assets already processed.</strong>
                <p style="margin:0.55rem 0 0 0; color:#9fb0c7;">
                    Next asset: <strong>{next_symbol}</strong><br/>
                    Risk profile: <strong>{risk_level.capitalize()}</strong><br/>
                    Trading mode: <strong>{TRADING_MODES[trading_mode]["label"]}</strong><br/>
                    Saved checkpoint: <strong>{saved_at}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-card fresh">
                <div class="section-kicker">Fresh Run</div>
                <strong>No resumable checkpoint detected for the current risk profile + mode.</strong>
                <p style="margin:0.55rem 0 0 0; color:#9fb0c7;">
                    The next scan will start from asset <strong>1</strong> of <strong>{total_assets}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_watchlist_sidebar(all_results: dict):
    """Render the current watchlist in the sidebar."""
    watchlist = st.session_state.get("watchlist", [])
    if not watchlist:
        return

    symbol_map = {data["symbol"]: data for data in all_results.values()}
    with st.sidebar:
        st.divider()
        st.markdown("### Watchlist")
        for symbol in watchlist:
            data = symbol_map.get(symbol)
            if not data:
                continue
            cols = st.columns([1, 0.42])
            cols[0].markdown(
                f"**{symbol}**  \n"
                f"<span style='color:#9fb0c7;'>Score {data['final']['final_score']:.1f}</span>",
                unsafe_allow_html=True,
            )
            if cols[1].button("Open", key=f"watch-open-{symbol}", use_container_width=True):
                set_selected_symbol(symbol)
            if cols[1].button("Drop", key=f"watch-drop-{symbol}", use_container_width=True):
                remove_from_watchlist(symbol)
                st.rerun()


def render_universe_tab(
    all_results: dict,
    risk_level: str,
    trading_mode: str,
    selected_symbol: str | None = None,
):
    """Render a compact scan of the whole asset universe."""
    st.markdown('<div class="section-kicker">Tracked Assets</div>', unsafe_allow_html=True)
    render_asset_universe()
    st.divider()

    symbols = [data["symbol"] for _, data in sorted(
        all_results.items(),
        key=lambda x: x[1]["final"]["final_score"],
        reverse=True,
    )]
    chosen_symbol = st.selectbox(
        "Open asset detail",
        options=symbols,
        index=symbols.index(selected_symbol) if selected_symbol in symbols else 0,
    )
    if chosen_symbol != st.session_state.get("selected_symbol"):
        set_selected_symbol(chosen_symbol)
    symbol_map = {data["symbol"]: {"coin_id": cid, **data} for cid, data in all_results.items()}
    rank_lookup = {symbol: idx + 1 for idx, symbol in enumerate(symbols)}
    if chosen_symbol in symbol_map:
        top_cols = st.columns([0.76, 0.24])
        if top_cols[1].button(f"Add {chosen_symbol} to watchlist", key=f"watch-{chosen_symbol}", use_container_width=True):
            add_to_watchlist(chosen_symbol)
        render_asset_detail_panel(
            symbol_map[chosen_symbol],
            risk_level,
            trading_mode,
            rank=rank_lookup.get(chosen_symbol),
            total_assets=len(all_results),
            panel_key_prefix="universe-detail",
        )

    st.divider()

    for cid, data in sorted(
        all_results.items(),
        key=lambda x: x[1]["final"]["final_score"],
        reverse=True,
    ):
        trend = data["technical"]["trend"]
        st.markdown(
            f'<div class="universe-card {trend if trend in {"bullish", "bearish"} else "neutral"}">',
            unsafe_allow_html=True,
        )
        left, mid, right = st.columns([1.2, 1, 1.3])
        with left:
            st.markdown(
                f"""
                <div class="universe-head">
                    <div>
                        <h4 style="margin:0;">{data['symbol']}</h4>
                        <div class="pick-subtitle">{cid}</div>
                    </div>
                    <div>{tone_badge(trend.capitalize(), trend)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mid:
            st.metric("Final Score", f"{data['final']['final_score']:.1f}/100")
            st.metric("Price", format_price(data["current_price"]))
        with right:
            st.metric("Technical", f"{data['technical']['score']:.0f}")
            st.metric("Fundamental", f"{data['fundamental']['score']:.0f}")
            st.metric("ML", f"{data['ml_forecast']['score']:.0f}")
            if st.button("Review Asset", key=f"universe-open-{data['symbol']}", use_container_width=True):
                set_selected_symbol(data["symbol"])
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_backtest_chart(backtest_payload: dict, symbol: str, trading_mode: str):
    """Plot equity curve versus buy-and-hold for the selected asset."""
    result = backtest_payload.get("result", {})
    equity_curve = result.get("equity_curve", pd.DataFrame())
    if equity_curve.empty:
        st.info("No backtest equity curve is available yet.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve["timestamp"],
            y=equity_curve["equity"],
            name="Strategy Equity",
            line=dict(color="#38bdf8", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=equity_curve["timestamp"],
            y=equity_curve["asset_normalized"],
            name=f"{symbol} Buy & Hold",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=30, r=20, t=25, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"backtest-chart-{symbol}-{trading_mode}")


def render_backtest_tab(
    risk_level: str,
    default_trading_mode: str,
    selected_symbol: str | None = None,
):
    """Allow the user to run a real historical backtest on demand."""
    assets = list_backtest_assets()
    symbols = [asset["symbol"] for asset in assets]
    asset_lookup = {asset["symbol"]: asset for asset in assets}
    default_symbol = selected_symbol if selected_symbol in asset_lookup else symbols[0]
    chosen_symbol = st.selectbox(
        "Backtest asset",
        options=symbols,
        index=symbols.index(default_symbol),
        key=f"backtest-select-{default_trading_mode}",
    )
    backtest_mode = st.selectbox(
        "Backtest strategy",
        options=TRADING_MODE_OPTIONS,
        index=TRADING_MODE_OPTIONS.index(default_trading_mode),
        format_func=lambda key: f"{TRADING_MODES[key]['label']} ({TRADING_MODES[key]['holding_period_label']})",
        key=f"backtest-mode-{default_trading_mode}",
    )
    mode_profile = TRADING_MODES[backtest_mode]
    initial_cash = st.number_input(
        "Initial capital",
        min_value=1000.0,
        max_value=1_000_000.0,
        value=10_000.0,
        step=1000.0,
        key=f"backtest-capital-{backtest_mode}",
    )
    period_options = {
        "swing": ["2y", "5y", "10y", "max"],
        "day": ["90d", "180d", "365d", "730d"],
        "scalp": ["7d", "30d", "60d"],
    }
    period_values = period_options.get(backtest_mode, [mode_profile.get("backtest_period", mode_profile["yfinance_period"])])
    default_period = mode_profile.get("backtest_period", mode_profile["yfinance_period"])
    history_period = st.selectbox(
        "Historical window",
        options=period_values,
        index=period_values.index(default_period) if default_period in period_values else 0,
        key=f"backtest-period-{backtest_mode}",
    )
    cache_key = f"{chosen_symbol}:{risk_level}:{backtest_mode}:{history_period}:{int(initial_cash)}"
    cached_backtest = st.session_state.setdefault("backtests", {}).get(cache_key)

    st.caption(
        "This module fetches real historical OHLCV for the chosen asset and mode on demand, "
        "then runs a walk-forward simulation over that history."
    )
    st.caption(
        "Disclaimer: historical sentiment is excluded from the backtest because reliable "
        "archival sentiment data is hard to source consistently. The backtest still uses "
        "real historical price data together with the executable strategy logic, including "
        "technical indicators, ML forecasts, mode-specific rules, risk settings, and "
        "fee/slippage assumptions."
    )

    if st.button(
        f"Run {TRADING_MODES[backtest_mode]['label']} backtest for {chosen_symbol}",
        key=f"backtest-run-{cache_key}",
        type="primary",
    ):
        with st.spinner("Fetching history and running walk-forward backtest..."):
            cached_backtest = run_historical_backtest(
                chosen_symbol,
                trading_mode=backtest_mode,
                risk_level=risk_level,
                initial_cash=float(initial_cash),
                period=history_period,
            )
        st.session_state["backtests"][cache_key] = cached_backtest

    if not cached_backtest:
        st.info("Pick an asset and strategy, then run the backtest to see real historical performance.")
        return

    history_meta = cached_backtest.get("history_meta", {})
    result = cached_backtest.get("result", {})
    metrics = result.get("metrics", {})

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Asset", chosen_symbol)
    h2.metric("Mode", TRADING_MODES[backtest_mode]["label"])
    h3.metric("Interval", history_meta.get("interval", mode_profile["yfinance_interval"]))
    h4.metric("Bars", f"{history_meta.get('rows', 0)}")

    if history_meta:
        start_value = history_meta.get("start")
        end_value = history_meta.get("end")
        start_label = pd.to_datetime(start_value).strftime("%Y-%m-%d %H:%M") if start_value is not None else "N/A"
        end_label = pd.to_datetime(end_value).strftime("%Y-%m-%d %H:%M") if end_value is not None else "N/A"
        st.caption(f"History coverage: {start_label} -> {end_label}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy Return", f"{metrics.get('total_return_pct', 0):+.2f}%")
    m2.metric("Buy & Hold", f"{metrics.get('buy_hold_return_pct', 0):+.2f}%")
    m3.metric("Alpha", f"{metrics.get('alpha_vs_buy_hold_pct', 0):+.2f}%")
    m4.metric("Trades", f"{metrics.get('trade_count', 0)}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
    m6.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
    m7.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
    m8.metric("Sortino", f"{metrics.get('sortino_ratio', 0):.2f}")

    m9, m10, m11, m12 = st.columns(4)
    m9.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    m10.metric("Avg Trade", f"{metrics.get('avg_trade_return_pct', 0):+.2f}%")
    m11.metric("Exposure", f"{metrics.get('exposure_pct', 0):.1f}%")
    m12.metric("Ending Equity", format_price(metrics.get("ending_equity", initial_cash)))

    m13, m14, m15 = st.columns(3)
    m13.metric("Annualized Return", f"{metrics.get('annualized_return_pct', 0):+.2f}%")
    m14.metric("Benchmark Annualized", f"{metrics.get('benchmark_annualized_return_pct', 0):+.2f}%")
    m15.metric("Annualized Volatility", f"{metrics.get('annualized_volatility_pct', 0):.2f}%")

    render_backtest_chart(cached_backtest, chosen_symbol, backtest_mode)
    st.caption(result.get("notes", ""))

    trades = result.get("trades", pd.DataFrame())
    if not trades.empty:
        st.markdown("### Executed Trades")
        st.dataframe(
            trades.tail(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("This run did not open any trades under the current mode/profile filters.")


# ── Main Analysis Pipeline ───────────────────────────────────────────────────
def run_analysis(risk_level: str, trading_mode: str) -> dict:
    """Execute the full analysis pipeline for all assets."""
    progress_placeholder = st.empty()
    render_analysis_progress(progress_placeholder, 0, "Fetching market overview...")
    live_placeholder = st.empty()
    mode_profile = TRADING_MODES[trading_mode]

    # 1. Market overview
    try:
        market_df = fetch_market_overview()
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        return {}

    total_assets = len(ASSET_UNIVERSE)
    checkpoint = get_resume_state(risk_level, trading_mode)
    all_results = checkpoint.get("results", {}) if checkpoint else {}
    next_index = int(checkpoint.get("next_index", 0)) if checkpoint else 0
    resumed = checkpoint is not None

    if all_results:
        with live_placeholder.container():
            render_live_runboard(
                all_results,
                total_assets,
                risk_level,
                trading_mode,
                latest_symbol=None,
                resumed=resumed,
                processed_count=next_index,
            )

    assets = list(ASSET_UNIVERSE.items())

    for idx, (coin_id, meta) in enumerate(assets[next_index:], start=next_index):
        symbol = meta["symbol"]
        pct = int((idx + 1) / total_assets * 100)

        def update_status(stage: str):
            render_analysis_progress(
                progress_placeholder,
                pct,
                f"Analysing {symbol} ({idx+1}/{total_assets}) - {stage}...",
            )

        try:
            # Fetch OHLCV
            update_status("fetching price history")
            ohlcv = fetch_ohlcv(
                meta["yf"],
                interval=mode_profile["yfinance_interval"],
                period=mode_profile["yfinance_period"],
            )
            if ohlcv.empty or len(ohlcv) < 30:
                log.warning("Skipping %s — insufficient OHLCV data", symbol)
                continue

            # Market row
            market_row = market_df.loc[market_df["id"] == coin_id]
            if market_row.empty:
                log.warning("Skipping %s — not in market overview", symbol)
                continue
            market_row = market_row.iloc[0]

            # Technical Analysis
            update_status("running technical analysis")
            tech_result = score_technical(ohlcv.copy())

            # Fundamental Analysis
            asset_data = {}
            github_data = {}
            try:
                update_status("fetching fundamentals")
                from data.market_data import fetch_coin_details, fetch_github_activity
                asset_data = fetch_coin_details(coin_id)
                github_data = fetch_github_activity(meta.get("github"))
            except Exception:
                pass

            fund_result = score_fundamental(market_row, market_df, asset_data, github_data)

            # Sentiment Analysis
            update_status("collecting news sentiment")
            news = fetch_news(symbol)
            sent_result = analyse_sentiment(news)

            # ML Forecast
            update_status("running forecast model")
            ml_result = forecast_asset(
                ohlcv.copy(),
                horizon=mode_profile["forecast_horizon_bars"],
                return_threshold=mode_profile["classification_threshold"],
            )

            # Combined Score
            update_status("combining scores")
            final_result = compute_final_score(
                tech_result, fund_result, sent_result, ml_result, risk_level
            )
            trade_setup = derive_trade_setup(
                tech_result,
                ml_result,
                final_result.get("final_score", 50),
            )

            # Store OHLCV with indicators for charting
            ohlcv_with_indicators = compute_indicators(ohlcv.copy())

            all_results[coin_id] = {
                "symbol": symbol,
                "current_price": float(market_row.get("current_price", ohlcv["Close"].iloc[-1])),
                "technical": tech_result,
                "fundamental": fund_result,
                "sentiment": sent_result,
                "ml_forecast": ml_result,
                "final": final_result,
                "trade_setup": trade_setup,
                "ohlcv": ohlcv_with_indicators,
                "market_row": market_row,
            }

        except Exception as e:
            log.error("Error analysing %s: %s", symbol, e, exc_info=True)
            st.toast(f"⚠️ Error on {symbol}: {e}", icon="⚠️")
        finally:
            save_run_checkpoint(risk_level, trading_mode, idx + 1, all_results)
            with live_placeholder.container():
                render_live_runboard(
                    all_results,
                    total_assets,
                    risk_level,
                    trading_mode,
                    latest_symbol=symbol,
                    resumed=resumed,
                    processed_count=idx + 1,
                )

    render_analysis_progress(progress_placeholder, 100, "Analysis complete!")
    clear_run_checkpoint()
    return all_results


# ── Display Results ──────────────────────────────────────────────────────────
def display_results(all_results: dict, risk_level: str, trading_mode: str):
    """Render the top picks and analysis dashboard."""
    if not all_results:
        st.error("No assets could be analysed. Check your internet connection and try again.")
        return

    # Market regime
    regime = determine_market_regime(all_results)
    regime_tone = "neutral"
    if regime == "bullish":
        regime_tone = "bullish"
    elif regime == "bearish":
        regime_tone = "bearish"

    st.markdown('<div class="section-kicker">Run Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f"## Market regime {tone_badge(regime.upper(), regime_tone)}",
        unsafe_allow_html=True,
    )
    st.caption(f"Analysis timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption(
        f"Trading mode: `{TRADING_MODES[trading_mode]['label']}` "
        f"using `{TRADING_MODES[trading_mode]['yfinance_interval']}` bars."
    )

    # Top picks
    top_picks = rank_assets(all_results, top_n=3)
    symbol_map = {data["symbol"]: {"coin_id": cid, **data} for cid, data in all_results.items()}
    ranked_assets = rank_assets(all_results, top_n=len(all_results))
    rank_lookup = {item["symbol"]: idx + 1 for idx, item in enumerate(ranked_assets)}
    selected_symbol = st.session_state.get("selected_symbol")
    if selected_symbol not in symbol_map and top_picks:
        selected_symbol = top_picks[0]["symbol"]
        set_selected_symbol(selected_symbol)
    overview_tab, picks_tab, universe_tab, backtests_tab = st.tabs(["Overview", "Picks", "Universe", "Backtests"])

    with overview_tab:
        render_summary_tiles(all_results, top_picks, risk_level, regime)
        render_health_check(all_results)
        st.markdown('<div class="section-kicker">Live Rankings</div>', unsafe_allow_html=True)
        st.header("Full Asset Rankings")
        newly_selected_symbol = render_rankings_table(all_results, risk_level, trading_mode)
        if newly_selected_symbol:
            selected_symbol = newly_selected_symbol
            set_selected_symbol(selected_symbol)
        if selected_symbol in symbol_map:
            top_cols = st.columns([0.76, 0.24])
            if top_cols[1].button(
                f"Add {selected_symbol} to watchlist",
                key=f"overview-watch-{selected_symbol}",
                use_container_width=True,
            ):
                add_to_watchlist(selected_symbol)
            render_asset_detail_panel(
                symbol_map[selected_symbol],
                risk_level,
                trading_mode,
                rank=rank_lookup.get(selected_symbol),
                total_assets=len(all_results),
                panel_key_prefix="overview-detail",
            )

    with picks_tab:
        st.markdown('<div class="section-kicker">Conviction Board</div>', unsafe_allow_html=True)
        st.header("Top 3 Trade Setups")

        for rank_num, pick in enumerate(top_picks, 1):
            symbol = pick["symbol"]
            current_price = pick["current_price"]
            final = pick["final"]
            trade_setup = pick.get("trade_setup") or derive_trade_setup(
                pick.get("technical", {}),
                pick.get("ml_forecast", {}),
                final.get("final_score", 50),
            )
            tech = pick["technical"]
            ml = pick["ml_forecast"]

            levels = compute_levels(
                current_price,
                tech,
                ml,
                risk_level,
                trading_mode,
                trade_setup=trade_setup,
            )

            ohlcv = pick.get("ohlcv", pd.DataFrame())
            daily_vol = 0.0
            if not ohlcv.empty:
                daily_vol = float(ohlcv["Close"].pct_change().tail(30).std() * 100)
            ml_confidence = ml.get("direction", {}).get("confidence", 50)
            leverage_info = compute_leverage(risk_level, daily_vol, ml_confidence)

            reasoning = generate_reasoning(pick)
            trend = tech.get("trend", "neutral")
            render_pick_banner(
                rank_num,
                symbol,
                trade_setup.get("bias", trend),
                levels.get("trade_label", "Trade Setup"),
                trade_setup.get("opportunity_score", final["final_score"]),
                reasoning,
            )
            render_signal_strip(levels, leverage_info, ml)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Current Price", format_price(current_price))
            col2.metric("Trade Side", levels.get("trade_label", levels.get("trade_direction", "Long")).replace(" Setup", ""))
            col3.metric("Entry Price", format_price(levels["entry_price"]))
            col4.metric("Exit Price", format_price(levels["exit_price"]))
            col5.metric("Stop Loss", format_price(levels["stop_loss"]))
            col6.metric("Opportunity", f"{trade_setup.get('opportunity_score', final['final_score']):.1f}/100")

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Leverage", f"{leverage_info['leverage']}x")
            col_b.metric("Risk/Reward", f"{levels['risk_reward_ratio']:.1f}:1")
            col_c.metric("Expected Return", f"{levels['expected_return_pct']:+.1f}%")
            col_d.metric(
                "ML Direction",
                ml.get("direction", {}).get("direction", "N/A").capitalize(),
                f"{ml.get('direction', {}).get('confidence', 0):.0f}% conf.",
            )

            with st.expander("Score Breakdown"):
                components = final.get("components", {})
                score_cols = st.columns(4)
                for i, (name, data) in enumerate(components.items()):
                    score_cols[i].metric(
                        name.replace("_", " ").title(),
                        f"{data['raw']:.1f}",
                        f"weight: {data['weight']:.0%}",
                    )

            with st.expander("Why Selected"):
                st.markdown(reasoning)
                st.caption(levels["methodology"])

            with st.expander("Leverage & Risk Details"):
                st.markdown(f"**{leverage_info['rationale']}**")
                st.error(leverage_info["liquidation_warning"])

            sent = pick.get("sentiment", {})
            with st.expander(f"Sentiment ({sent.get('article_count', 0)} articles)"):
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.metric("Positive", f"{sent.get('positive_pct', 0):.0f}%")
                s_col2.metric("Negative", f"{sent.get('negative_pct', 0):.0f}%")
                s_col3.metric("Trend", sent.get("trend", "stable").capitalize())

                if sent.get("top_positive"):
                    st.markdown("**Top Positive Headlines:**")
                    for h in sent["top_positive"][:3]:
                        st.caption(f"+ {h['title']} ({h['score']:.3f}) [{h.get('source', 'Unknown')}]")
                if sent.get("top_negative"):
                    st.markdown("**Top Negative Headlines:**")
                    for h in sent["top_negative"][:3]:
                        st.caption(f"- {h['title']} ({h['score']:.3f}) [{h.get('source', 'Unknown')}]")

                source_breakdown = sent.get("source_breakdown", {})
                if source_breakdown:
                    st.markdown("**Source Mix:**")
                    for source_type, source_data in source_breakdown.items():
                        st.caption(
                            f"- {source_type.replace('_', ' ').title()}: "
                            f"{source_data.get('count', 0)} items, "
                            f"weight {source_data.get('weight', 0):.2f}"
                        )

            with st.expander("Price Chart & Indicators", expanded=True):
                if not ohlcv.empty:
                    forecast_prices = ml.get("forecast", {}).get("forecast_prices", [])
                    fig = build_price_chart(ohlcv, symbol, forecast_prices, trading_mode=trading_mode)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"pick-chart-{rank_num}-{symbol}",
                    )
                else:
                    st.info("No chart data available.")

            render_transparency_report(
                pick,
                risk_level,
                trading_mode,
                f"pick-{rank_num}-{symbol}",
                rank=rank_num,
                total_assets=len(all_results),
            )

            st.divider()

    with universe_tab:
        render_universe_tab(all_results, risk_level, trading_mode, selected_symbol)

    with backtests_tab:
        render_backtest_tab(risk_level, trading_mode, selected_symbol)


# ── App Entry Point ──────────────────────────────────────────────────────────
ensure_ui_state()
hydrate_selection_from_query_params()
bootstrap_results_for_deep_link(risk_level, trading_mode)
render_hero()
render_resume_status_card(risk_level, trading_mode)

if reset_run_btn:
    clear_run_checkpoint()
    st.session_state.pop("results", None)
    st.session_state.pop("risk_level", None)
    st.session_state.pop("trading_mode", None)
    st.session_state["analysis_running"] = False
    st.toast("Fresh run state cleared. The next scan will start from asset 1.", icon="♻️")

if run_btn:
    with st.spinner("Running full analysis pipeline..."):
        results = run_analysis(risk_level, trading_mode)
    st.session_state["analysis_running"] = False
    if results:
        st.session_state["results"] = results
        st.session_state["risk_level"] = risk_level
        st.session_state["trading_mode"] = trading_mode

if "results" in st.session_state:
    render_sidebar_portfolio(
        rank_assets(st.session_state["results"], top_n=3),
        st.session_state.get("risk_level", "moderate"),
    )
    render_watchlist_sidebar(st.session_state["results"])
    display_results(
        st.session_state["results"],
        st.session_state.get("risk_level", "moderate"),
        st.session_state.get("trading_mode", trading_mode),
    )
else:
    intro_left, intro_right = st.columns([1.25, 1])

    with intro_left:
        st.markdown('<div class="section-kicker">How It Works</div>', unsafe_allow_html=True)
        st.subheader("Run the scanner and get a ranked, evidence-backed market view.")
        st.markdown(
            """
            The app processes every configured asset through four lenses:
            technical structure, fundamentals, sentiment, and ML forecasting.
            Results are then translated into trade-style levels so you can inspect
            both conviction and risk in one pass.
            """
        )
        st.info("Click **Run Analysis** in the sidebar to start a fresh market scan.", icon="👈")

        resume_state = get_resume_state(risk_level, trading_mode)
        if resume_state:
            st.warning(
                f"A resumable checkpoint is available for this risk profile and trading mode. "
                f"Use **Run / Resume Analysis** to continue from asset {resume_state['next_index'] + 1}.",
                icon="⏯️",
            )

    with intro_right:
        st.markdown(
            """
            <div class="glass-panel summary-tile">
                <div class="summary-label">Current Universe</div>
                <div class="summary-value">15</div>
                <div class="summary-note">
                    Large-cap and actively traded crypto assets with shared
                    scoring rules and risk-profile-aware ranking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker">Tracked Assets</div>', unsafe_allow_html=True)
    with st.expander("Asset Universe"):
        render_asset_universe()

    st.divider()
    st.markdown('<div class="section-kicker">Historical Backtests</div>', unsafe_allow_html=True)
    st.subheader("Run an on-demand backtest without scanning the full universe.")
    render_backtest_tab(risk_level, trading_mode, st.session_state.get("selected_symbol"))
