"""
Crypto Investor MVP — Streamlit Frontend
Quant-assisted crypto investment decision-support tool.

Run:  streamlit run app.py
"""
import sys
import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ASSET_UNIVERSE, SCORING_WEIGHTS, RISK_PROFILES
from data.market_data import fetch_market_overview, fetch_ohlcv, fetch_all_asset_data
from data.news_data import fetch_news
from analysis.technical import score_technical, compute_indicators
from analysis.fundamental import score_fundamental
from analysis.sentiment import analyse_sentiment
from analysis.ml_forecast import forecast_asset
from scoring.engine import compute_final_score, rank_assets, generate_reasoning, determine_market_regime
from strategy.entry_exit import compute_levels
from strategy.risk import compute_leverage
from utils.helpers import get_logger

log = get_logger("app")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Investor MVP",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    @media (max-width: 900px) {
        .hero-title { font-size: 2.35rem; }
        .micro-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Crypto Investor MVP")
    st.caption("Quant-assisted crypto analysis tool")
    st.caption("Market data: CoinPaprika + Yahoo Finance")

    st.divider()
    risk_level = st.radio(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        index=1,
        format_func=lambda x: x.capitalize(),
        help="Determines leverage, stop-loss width, and ranking bias.",
    )
    profile = RISK_PROFILES[risk_level]
    st.info(profile["description"])

    st.divider()
    st.markdown("**Scoring Weights**")
    for k, v in SCORING_WEIGHTS.items():
        adj = profile["weight_adjustments"].get(k, 1.0)
        effective = v * adj
        st.caption(f"{k.replace('_', ' ').title()}: {effective:.0%}")

    st.divider()
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.warning(
        "**Disclaimer:** All investments carry risk, including the possible loss of principal. "
        "Use of this tool is at your own risk. The builders and contributors are not liable "
        "for any losses, damages, or decisions made based on its outputs. Crypto markets are "
        "highly volatile, and leverage can lead to rapid liquidation.",
        icon="⚠️",
    )


# ── Helper: Build price chart ────────────────────────────────────────────────
def build_price_chart(df: pd.DataFrame, symbol: str, forecast_prices: list = None) -> go.Figure:
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
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=len(forecast_prices), freq="D")
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
            <h1 class="hero-title">Sharper crypto signals, packaged like a real dashboard.</h1>
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


def render_pick_banner(rank_num: int, symbol: str, trend: str, final_score: float, reasoning: str):
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
                    <span class="summary-label">Final score</span>
                    <strong>{final_score:.1f}</strong>
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
    st.markdown(
        f"""
        <div class="micro-grid">
            <div class="micro-card">
                <div class="micro-label">Entry / Exit</div>
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


def render_rankings_table(all_results: dict) -> str | None:
    """Render an interactive rankings table with score bars and sparklines."""
    rows = []
    for cid, data in sorted(
        all_results.items(),
        key=lambda x: x[1]["final"]["final_score"],
        reverse=True,
    ):
        rows.append({
            "Rank": len(rows) + 1,
            "Symbol": data["symbol"],
            "Price": float(data["current_price"]),
            "Trend": trend_label(data["technical"]["trend"]),
            "Technical": float(data["technical"]["score"]),
            "Fundamental": float(data["fundamental"]["score"]),
            "Sentiment": float(data["sentiment"]["score"]),
            "ML": float(data["ml_forecast"]["score"]),
            "Final Score": float(data["final"]["final_score"]),
            "Narrative": sentiment_label(data["sentiment"]),
            "Sparkline": build_price_series(data.get("ohlcv", pd.DataFrame())),
        })

    ranking_df = pd.DataFrame(rows)
    event = st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "Symbol": st.column_config.TextColumn("Asset", width="small"),
            "Price": st.column_config.NumberColumn("Price", format="$%.4f"),
            "Trend": st.column_config.TextColumn("Trend", width="small"),
            "Technical": st.column_config.ProgressColumn("Technical", min_value=0, max_value=100, format="%.0f"),
            "Fundamental": st.column_config.ProgressColumn("Fundamental", min_value=0, max_value=100, format="%.0f"),
            "Sentiment": st.column_config.ProgressColumn("Sentiment", min_value=0, max_value=100, format="%.0f"),
            "ML": st.column_config.ProgressColumn("ML", min_value=0, max_value=100, format="%.0f"),
            "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%.1f"),
            "Narrative": st.column_config.TextColumn("Sentiment Tone", width="small"),
            "Sparkline": st.column_config.LineChartColumn("Price Action", y_min=None, y_max=None, width="medium"),
        },
    )
    try:
        rows_selected = event.selection.rows
    except Exception:
        rows_selected = []
    if rows_selected:
        idx = rows_selected[0]
        if 0 <= idx < len(ranking_df):
            return str(ranking_df.iloc[idx]["Symbol"])
    return None


def render_asset_detail_panel(pick: dict, risk_level: str):
    """Render a focused detail panel for a selected asset."""
    symbol = pick["symbol"]
    current_price = pick["current_price"]
    final = pick["final"]
    tech = pick["technical"]
    ml = pick["ml_forecast"]
    ohlcv = pick.get("ohlcv", pd.DataFrame())
    reasoning = generate_reasoning(pick)
    levels = compute_levels(current_price, tech, ml, risk_level)

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
            <h3>{symbol} {tone_badge(tech.get('trend', 'neutral').capitalize(), tech.get('trend', 'neutral'))}</h3>
            <p>{reasoning.splitlines()[0] if reasoning else ''}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_signal_strip(levels, leverage_info, ml)
    a, b, c, d = st.columns(4)
    a.metric("Current Price", format_price(current_price))
    b.metric("Entry", format_price(levels["entry_price"]))
    c.metric("Stop", format_price(levels["stop_loss"]))
    d.metric("Take Profit", format_price(levels["exit_price"]))

    with st.expander("Selected Asset Chart", expanded=True):
        if not ohlcv.empty:
            forecast_prices = ml.get("forecast", {}).get("forecast_prices", [])
            st.plotly_chart(build_price_chart(ohlcv, symbol, forecast_prices), use_container_width=True)
        else:
            st.info("No chart data available.")


def render_sidebar_portfolio(top_picks: list[dict], risk_level: str):
    """Render a simple score-weighted allocation idea in the sidebar."""
    if not top_picks:
        return

    weights = [max(p["final"]["final_score"], 1.0) for p in top_picks]
    total = sum(weights)
    with st.sidebar:
        st.divider()
        st.markdown("### Suggested Allocation")
        st.caption(f"Score-weighted mix for the current `{risk_level}` profile.")
        for pick, weight in zip(top_picks, weights):
            pct = weight / total * 100
            st.markdown(
                f"""
                <div class="alloc-card">
                    <div class="alloc-head">
                        <strong>{pick['symbol']}</strong>
                        <span>{pct:.0f}%</span>
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
    st.session_state.setdefault("watchlist", [])


def set_selected_symbol(symbol: str | None):
    """Persist the currently focused asset."""
    st.session_state["selected_symbol"] = symbol


def add_to_watchlist(symbol: str):
    """Add an asset to the watchlist if it is not already present."""
    watchlist = st.session_state.setdefault("watchlist", [])
    if symbol not in watchlist:
        watchlist.append(symbol)


def remove_from_watchlist(symbol: str):
    """Remove an asset from the watchlist."""
    watchlist = st.session_state.setdefault("watchlist", [])
    st.session_state["watchlist"] = [item for item in watchlist if item != symbol]


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


def render_universe_tab(all_results: dict, risk_level: str, selected_symbol: str | None = None):
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
    if chosen_symbol in symbol_map:
        top_cols = st.columns([0.76, 0.24])
        if top_cols[1].button(f"Add {chosen_symbol} to watchlist", key=f"watch-{chosen_symbol}", use_container_width=True):
            add_to_watchlist(chosen_symbol)
        render_asset_detail_panel(symbol_map[chosen_symbol], risk_level)

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
        st.markdown("</div>", unsafe_allow_html=True)


# ── Main Analysis Pipeline ───────────────────────────────────────────────────
def run_analysis(risk_level: str) -> dict:
    """Execute the full analysis pipeline for all assets."""
    progress = st.progress(0, text="Fetching market overview...")

    # 1. Market overview
    try:
        market_df = fetch_market_overview()
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        return {}

    total_assets = len(ASSET_UNIVERSE)
    all_results = {}

    for idx, (coin_id, meta) in enumerate(ASSET_UNIVERSE.items()):
        symbol = meta["symbol"]
        pct = int((idx + 1) / total_assets * 100)

        def update_status(stage: str):
            progress.progress(
                pct,
                text=f"Analysing {symbol} ({idx+1}/{total_assets}) - {stage}...",
            )

        try:
            # Fetch OHLCV
            update_status("fetching price history")
            ohlcv = fetch_ohlcv(meta["yf"])
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
            ml_result = forecast_asset(ohlcv.copy())

            # Combined Score
            update_status("combining scores")
            final_result = compute_final_score(
                tech_result, fund_result, sent_result, ml_result, risk_level
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
                "ohlcv": ohlcv_with_indicators,
                "market_row": market_row,
            }

        except Exception as e:
            log.error("Error analysing %s: %s", symbol, e, exc_info=True)
            st.toast(f"⚠️ Error on {symbol}: {e}", icon="⚠️")
            continue

    progress.progress(100, text="Analysis complete!")
    return all_results


# ── Display Results ──────────────────────────────────────────────────────────
def display_results(all_results: dict, risk_level: str):
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

    # Top picks
    top_picks = rank_assets(all_results, top_n=3)
    symbol_map = {data["symbol"]: {"coin_id": cid, **data} for cid, data in all_results.items()}
    selected_symbol = st.session_state.get("selected_symbol")
    if selected_symbol not in symbol_map and top_picks:
        selected_symbol = top_picks[0]["symbol"]
        set_selected_symbol(selected_symbol)
    overview_tab, picks_tab, universe_tab = st.tabs(["Overview", "Picks", "Universe"])

    with overview_tab:
        render_summary_tiles(all_results, top_picks, risk_level, regime)
        render_health_check(all_results)
        st.markdown('<div class="section-kicker">Live Rankings</div>', unsafe_allow_html=True)
        st.header("Full Asset Rankings")
        newly_selected_symbol = render_rankings_table(all_results)
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
            render_asset_detail_panel(symbol_map[selected_symbol], risk_level)

    with picks_tab:
        st.markdown('<div class="section-kicker">Conviction Board</div>', unsafe_allow_html=True)
        st.header("Top 3 Investment Picks")

        for rank_num, pick in enumerate(top_picks, 1):
            symbol = pick["symbol"]
            current_price = pick["current_price"]
            final = pick["final"]
            tech = pick["technical"]
            ml = pick["ml_forecast"]

            levels = compute_levels(current_price, tech, ml, risk_level)

            ohlcv = pick.get("ohlcv", pd.DataFrame())
            daily_vol = 0.0
            if not ohlcv.empty:
                daily_vol = float(ohlcv["Close"].pct_change().tail(30).std() * 100)
            ml_confidence = ml.get("direction", {}).get("confidence", 50)
            leverage_info = compute_leverage(risk_level, daily_vol, ml_confidence)

            reasoning = generate_reasoning(pick)
            trend = tech.get("trend", "neutral")
            render_pick_banner(rank_num, symbol, trend, final["final_score"], reasoning)
            render_signal_strip(levels, leverage_info, ml)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Current Price", format_price(current_price))
            col2.metric("Entry Price", format_price(levels["entry_price"]))
            col3.metric("Exit Price", format_price(levels["exit_price"]))
            col4.metric("Stop Loss", format_price(levels["stop_loss"]))
            col5.metric("Score", f"{final['final_score']:.1f}/100")

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
                        st.caption(f"+ {h['title']} ({h['score']:.3f})")
                if sent.get("top_negative"):
                    st.markdown("**Top Negative Headlines:**")
                    for h in sent["top_negative"][:3]:
                        st.caption(f"- {h['title']} ({h['score']:.3f})")

            with st.expander("Price Chart & Indicators", expanded=True):
                if not ohlcv.empty:
                    forecast_prices = ml.get("forecast", {}).get("forecast_prices", [])
                    fig = build_price_chart(ohlcv, symbol, forecast_prices)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No chart data available.")

            st.divider()

    with universe_tab:
        render_universe_tab(all_results, risk_level, selected_symbol)


# ── App Entry Point ──────────────────────────────────────────────────────────
ensure_ui_state()
render_hero()

if run_btn:
    with st.spinner("Running full analysis pipeline..."):
        results = run_analysis(risk_level)
    if results:
        st.session_state["results"] = results
        st.session_state["risk_level"] = risk_level

if "results" in st.session_state:
    render_sidebar_portfolio(
        rank_assets(st.session_state["results"], top_n=3),
        st.session_state.get("risk_level", "moderate"),
    )
    render_watchlist_sidebar(st.session_state["results"])
    display_results(st.session_state["results"], st.session_state.get("risk_level", "moderate"))
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
