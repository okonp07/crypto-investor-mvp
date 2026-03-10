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
    .stMetric { background: #0e1117; border-radius: 8px; padding: 12px; }
    .pick-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .bullish { color: #22c55e; }
    .bearish { color: #ef4444; }
    .neutral { color: #f59e0b; }
    div[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Crypto Investor MVP")
    st.caption("Quant-assisted crypto analysis tool")

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
        "**Disclaimer:** This tool is for educational and research purposes only. "
        "It does NOT constitute financial advice. Crypto markets are highly volatile. "
        "Never invest more than you can afford to lose. Leverage trading carries "
        "extreme risk of liquidation.",
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
        progress.progress(pct, text=f"Analysing {symbol} ({idx+1}/{total_assets})...")

        try:
            # Fetch OHLCV
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
            tech_result = score_technical(ohlcv.copy())

            # Fundamental Analysis
            asset_data = {}
            github_data = {}
            try:
                from data.market_data import fetch_coin_details, fetch_github_activity
                asset_data = fetch_coin_details(coin_id)
                github_data = fetch_github_activity(meta.get("github"))
            except Exception:
                pass

            fund_result = score_fundamental(market_row, market_df, asset_data, github_data)

            # Sentiment Analysis
            news = fetch_news(symbol)
            sent_result = analyse_sentiment(news)

            # ML Forecast
            ml_result = forecast_asset(ohlcv.copy())

            # Combined Score
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
    regime_colors = {"bullish": "green", "bearish": "red", "mixed / transitional": "orange"}
    st.markdown(f"### Market Regime: :{regime_colors.get(regime, 'gray')}[{regime.upper()}]")
    st.caption(f"Analysis timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    st.divider()

    # Top picks
    top_picks = rank_assets(all_results, top_n=3)

    st.header("Top 3 Investment Picks")

    for rank_num, pick in enumerate(top_picks, 1):
        coin_id = pick["coin_id"]
        symbol = pick["symbol"]
        current_price = pick["current_price"]
        final = pick["final"]
        tech = pick["technical"]
        ml = pick["ml_forecast"]

        # Entry/Exit levels
        levels = compute_levels(current_price, tech, ml, risk_level)

        # Leverage
        ohlcv = pick.get("ohlcv", pd.DataFrame())
        daily_vol = 0.0
        if not ohlcv.empty:
            daily_vol = float(ohlcv["Close"].pct_change().tail(30).std() * 100)
        ml_confidence = ml.get("direction", {}).get("confidence", 50)
        leverage_info = compute_leverage(risk_level, daily_vol, ml_confidence)

        # Reasoning
        reasoning = generate_reasoning(pick)

        # ── Card ─────────────────────────────────────────────────────
        trend = tech.get("trend", "neutral")
        trend_class = trend if trend in ("bullish", "bearish") else "neutral"

        st.subheader(f"#{rank_num} — {symbol}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"${current_price:,.4f}")
        col2.metric("Entry Price", f"${levels['entry_price']:,.4f}")
        col3.metric("Exit Price", f"${levels['exit_price']:,.4f}")
        col4.metric("Stop Loss", f"${levels['stop_loss']:,.4f}")
        col5.metric("Score", f"{final['final_score']:.1f}/100")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Leverage", f"{leverage_info['leverage']}x")
        col_b.metric("Risk/Reward", f"{levels['risk_reward_ratio']:.1f}:1")
        col_c.metric("Expected Return", f"{levels['expected_return_pct']:+.1f}%")
        col_d.metric("ML Direction",
                      ml.get("direction", {}).get("direction", "N/A").capitalize(),
                      f"{ml.get('direction', {}).get('confidence', 0):.0f}% conf.")

        # Score breakdown
        with st.expander("Score Breakdown"):
            components = final.get("components", {})
            score_cols = st.columns(4)
            for i, (name, data) in enumerate(components.items()):
                score_cols[i].metric(
                    name.replace("_", " ").title(),
                    f"{data['raw']:.1f}",
                    f"weight: {data['weight']:.0%}",
                )

        # Reasoning
        with st.expander("Why Selected"):
            st.markdown(reasoning)
            st.caption(levels["methodology"])

        # Leverage warning
        with st.expander("Leverage & Risk Details"):
            st.markdown(f"**{leverage_info['rationale']}**")
            st.error(leverage_info["liquidation_warning"])

        # Sentiment details
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

        # Chart
        with st.expander("Price Chart & Indicators", expanded=True):
            if not ohlcv.empty:
                forecast_prices = ml.get("forecast", {}).get("forecast_prices", [])
                fig = build_price_chart(ohlcv, symbol, forecast_prices)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No chart data available.")

        st.divider()

    # ── Full Rankings Table ──────────────────────────────────────────
    st.header("Full Asset Rankings")
    rows = []
    for cid, data in sorted(all_results.items(),
                             key=lambda x: x[1]["final"]["final_score"],
                             reverse=True):
        rows.append({
            "Rank": len(rows) + 1,
            "Symbol": data["symbol"],
            "Price": f"${data['current_price']:,.4f}",
            "Technical": f"{data['technical']['score']:.0f}",
            "Fundamental": f"{data['fundamental']['score']:.0f}",
            "Sentiment": f"{data['sentiment']['score']:.0f}",
            "ML": f"{data['ml_forecast']['score']:.0f}",
            "Final Score": f"{data['final']['final_score']:.1f}",
            "Trend": data["technical"]["trend"].capitalize(),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── App Entry Point ──────────────────────────────────────────────────────────
st.title("Crypto Investor MVP")
st.markdown(
    "A quant-assisted crypto investment decision-support tool combining "
    "**technical analysis**, **fundamental scoring**, **NLP sentiment analysis**, "
    "and **ML time-series forecasting**."
)

if run_btn:
    with st.spinner("Running full analysis pipeline..."):
        results = run_analysis(risk_level)
    if results:
        st.session_state["results"] = results
        st.session_state["risk_level"] = risk_level

if "results" in st.session_state:
    display_results(st.session_state["results"], st.session_state.get("risk_level", "moderate"))
else:
    st.info("Click **Run Analysis** in the sidebar to begin.", icon="->")

    # Show asset universe
    with st.expander("Asset Universe"):
        for cid, meta in ASSET_UNIVERSE.items():
            st.caption(f"{meta['symbol']} — {cid}")
