"""
Mode-aware walk-forward backtesting engine.

The backtester is intentionally transparent:
- bi-directional for the current production version
- entries require technical + ML agreement
- exits can happen via target, stop, time limit, or signal deterioration

This gives the frontend a repeatable way to compare swing/day/scalp behaviour
without depending on notebook cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.ml_forecast import forecast_asset
from analysis.technical import score_technical
from config import ML_TRAIN_WINDOW, TRADING_MODES
from scoring.engine import derive_trade_setup
from strategy.entry_exit import compute_levels
from utils.helpers import get_logger, safe_div

log = get_logger(__name__)


@dataclass
class TradeState:
    """Mutable state for the currently open position."""

    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    entry_equity: float
    direction: str
    stop_loss: float
    target_price: float
    signal_snapshot: dict


def _bars_per_year(mode: dict) -> int:
    """Translate mode frequency into an annualisation factor."""
    return int(mode.get("bars_per_day", 1) * 365)


def _max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown as a negative percentage."""
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _profit_factor(trade_returns: list[float]) -> float:
    """Sum of winners divided by absolute sum of losers."""
    gains = sum(ret for ret in trade_returns if ret > 0)
    losses = abs(sum(ret for ret in trade_returns if ret < 0))
    return float(gains / losses) if losses else 0.0


def run_mode_backtest(
    df: pd.DataFrame,
    trading_mode: str = "swing",
    risk_level: str = "moderate",
    initial_cash: float = 10_000.0,
) -> dict:
    """
    Walk-forward backtest using the current technical + ML pipeline.

    Returns:
        {
            "metrics": {...},
            "equity_curve": DataFrame,
            "trades": DataFrame,
            "mode": str,
        }
    """
    mode = TRADING_MODES.get(trading_mode, TRADING_MODES["swing"])
    if df.empty or len(df) < mode["warmup_bars"] + 10:
        return {
            "metrics": {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "buy_hold_return_pct": 0.0,
                "benchmark_annualized_return_pct": 0.0,
                "alpha_vs_buy_hold_pct": 0.0,
                "trade_count": 0,
                "win_rate_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return_pct": 0.0,
                "avg_hold_bars": 0.0,
                "exposure_pct": 0.0,
                "annualized_volatility_pct": 0.0,
                "bars_tested": 0,
                "ending_equity": initial_cash,
            },
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "mode": trading_mode,
            "notes": "Insufficient data for backtesting.",
        }

    fee_rate = mode.get("fee_bps", 0) / 10_000
    slippage_rate = mode.get("slippage_bps", 0) / 10_000
    stride = max(int(mode.get("signal_stride_bars", 1)), 1)
    warmup = max(
        int(mode.get("warmup_bars", 120)),
        int(ML_TRAIN_WINDOW + mode.get("forecast_horizon_bars", 1) + 10),
        30,
    )

    df = df.copy().sort_index()
    signal_cache: dict[int, dict] = {}

    def get_signals(index: int) -> dict:
        cached = signal_cache.get(index)
        if cached is not None:
            return cached

        window = df.iloc[: index + 1].copy()
        technical = score_technical(window)
        ml = forecast_asset(
            window,
            horizon=mode["forecast_horizon_bars"],
            return_threshold=mode["classification_threshold"],
        )
        signal_cache[index] = {
            "technical": technical,
            "ml": ml,
            "setup": derive_trade_setup(technical, ml, final_score=ml.get("score", 50)),
        }
        return signal_cache[index]

    cash = float(initial_cash)
    exposure_bars = 0
    state: TradeState | None = None
    trade_log = []
    equity_points = []

    for i in range(warmup, len(df) - 1):
        row = df.iloc[i]
        timestamp = df.index[i]
        close_price = float(row["Close"])
        equity_value = cash

        if state is not None:
            exposure_bars += 1
            high_price = float(row["High"])
            low_price = float(row["Low"])
            hold_bars = i - state.entry_index
            if state.direction == "short":
                equity_value = state.entry_equity * (1 + safe_div(state.entry_price - close_price, state.entry_price, 0))
            else:
                equity_value = state.entry_equity * (close_price / state.entry_price)

            exit_price = None
            exit_reason = None
            if state.direction == "short":
                if high_price >= state.stop_loss:
                    exit_price = state.stop_loss * (1 + slippage_rate)
                    exit_reason = "stop_loss"
                elif low_price <= state.target_price:
                    exit_price = state.target_price * (1 + slippage_rate)
                    exit_reason = "take_profit"
                elif hold_bars >= mode["max_holding_bars"]:
                    exit_price = close_price * (1 + slippage_rate)
                    exit_reason = "time_exit"
                elif hold_bars >= mode["min_holding_bars"] and i % stride == 0:
                    signal_state = get_signals(i)
                    setup = signal_state["setup"]
                    if (
                        setup.get("direction") == "long"
                        or signal_state["ml"].get("direction", {}).get("direction") == "bullish"
                        or setup.get("opportunity_score", 0) < 50
                    ):
                        exit_price = close_price * (1 + slippage_rate)
                        exit_reason = "signal_exit"
            else:
                if low_price <= state.stop_loss:
                    exit_price = state.stop_loss * (1 - slippage_rate)
                    exit_reason = "stop_loss"
                elif high_price >= state.target_price:
                    exit_price = state.target_price * (1 - slippage_rate)
                    exit_reason = "take_profit"
                elif hold_bars >= mode["max_holding_bars"]:
                    exit_price = close_price * (1 - slippage_rate)
                    exit_reason = "time_exit"
                elif hold_bars >= mode["min_holding_bars"] and i % stride == 0:
                    signal_state = get_signals(i)
                    setup = signal_state["setup"]
                    if (
                        setup.get("direction") == "short"
                        or signal_state["ml"].get("direction", {}).get("direction") == "bearish"
                        or setup.get("opportunity_score", 0) < 50
                    ):
                        exit_price = close_price * (1 - slippage_rate)
                        exit_reason = "signal_exit"

            if exit_price is not None:
                if state.direction == "short":
                    gross_return = safe_div(state.entry_price - exit_price, state.entry_price, 0)
                else:
                    gross_return = safe_div(exit_price - state.entry_price, state.entry_price, 0)
                trade_return = gross_return - (2 * fee_rate)
                cash = max(state.entry_equity * (1 + trade_return), 0.0)
                equity_value = cash
                trade_log.append(
                    {
                        "direction": state.direction,
                        "entry_time": state.entry_time,
                        "exit_time": timestamp,
                        "entry_price": round(state.entry_price, 6),
                        "exit_price": round(exit_price, 6),
                        "return_pct": round(trade_return * 100, 2),
                        "bars_held": hold_bars,
                        "exit_reason": exit_reason,
                        "technical_score_at_entry": round(
                            state.signal_snapshot["technical"].get("score", 0), 2
                        ),
                        "ml_score_at_entry": round(state.signal_snapshot["ml"].get("score", 0), 2),
                        "ml_confidence_at_entry": round(
                            state.signal_snapshot["ml"].get("direction", {}).get("confidence", 0), 2
                        ),
                    }
                )
                state = None

        if state is None and i % stride == 0:
            signal_state = get_signals(i)
            technical = signal_state["technical"]
            ml = signal_state["ml"]
            setup = signal_state["setup"]
            setup_direction = setup.get("direction", "neutral")
            min_setup_score = (mode["minimum_ml_score"] + mode["minimum_technical_score"]) / 2

            if setup_direction in {"long", "short"} and setup.get("opportunity_score", 0) >= min_setup_score:
                if setup_direction == "short":
                    next_open = float(df["Open"].iloc[i + 1]) * (1 - slippage_rate)
                else:
                    next_open = float(df["Open"].iloc[i + 1]) * (1 + slippage_rate)
                levels = compute_levels(
                    next_open,
                    technical,
                    ml,
                    risk_level=risk_level,
                    trading_mode=trading_mode,
                    trade_setup=setup,
                )
                state = TradeState(
                    entry_index=i + 1,
                    entry_time=df.index[i + 1],
                    entry_price=next_open,
                    entry_equity=cash,
                    direction=setup_direction,
                    stop_loss=float(levels["stop_loss"]),
                    target_price=float(levels["exit_price"]),
                    signal_snapshot=signal_state,
                )

        if state is not None:
            if state.direction == "short":
                equity_value = state.entry_equity * (1 + safe_div(state.entry_price - close_price, state.entry_price, 0))
            else:
                equity_value = state.entry_equity * (close_price / state.entry_price)
        equity_points.append({"timestamp": timestamp, "equity": equity_value, "close": close_price})

    if state is not None:
        if state.direction == "short":
            final_close = float(df["Close"].iloc[-1]) * (1 + slippage_rate)
            gross_return = safe_div(state.entry_price - final_close, state.entry_price, 0)
        else:
            final_close = float(df["Close"].iloc[-1]) * (1 - slippage_rate)
            gross_return = safe_div(final_close - state.entry_price, state.entry_price, 0)
        trade_return = gross_return - (2 * fee_rate)
        cash = max(state.entry_equity * (1 + trade_return), 0.0)
        trade_log.append(
            {
                "direction": state.direction,
                "entry_time": state.entry_time,
                "exit_time": df.index[-1],
                "entry_price": round(state.entry_price, 6),
                "exit_price": round(final_close, 6),
                "return_pct": round(trade_return * 100, 2),
                "bars_held": len(df) - 1 - state.entry_index,
                "exit_reason": "end_of_test",
                "technical_score_at_entry": round(state.signal_snapshot["technical"].get("score", 0), 2),
                "ml_score_at_entry": round(state.signal_snapshot["ml"].get("score", 0), 2),
                "ml_confidence_at_entry": round(
                    state.signal_snapshot["ml"].get("direction", {}).get("confidence", 0), 2
                ),
            }
        )

    equity_curve = pd.DataFrame(equity_points)
    if equity_curve.empty:
        equity_curve = pd.DataFrame({"timestamp": [], "equity": [], "close": []})
        trade_df = pd.DataFrame(trade_log)
        return {
            "metrics": {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "buy_hold_return_pct": 0.0,
                "benchmark_annualized_return_pct": 0.0,
                "alpha_vs_buy_hold_pct": 0.0,
                "trade_count": 0,
                "win_rate_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return_pct": 0.0,
                "avg_hold_bars": 0.0,
                "exposure_pct": 0.0,
                "annualized_volatility_pct": 0.0,
                "bars_tested": 0,
                "ending_equity": initial_cash,
            },
            "equity_curve": equity_curve,
            "trades": trade_df,
            "mode": trading_mode,
            "notes": "No backtest output generated.",
        }

    equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"])
    equity_curve["asset_normalized"] = initial_cash * (
        equity_curve["close"] / float(equity_curve["close"].iloc[0])
    )
    equity_curve["equity_return"] = equity_curve["equity"].pct_change().fillna(0.0)

    trade_df = pd.DataFrame(trade_log)
    trade_returns = (
        trade_df["return_pct"].div(100).tolist()
        if not trade_df.empty
        else []
    )

    total_return_pct = safe_div(cash - initial_cash, initial_cash, 0) * 100
    buy_hold_return_pct = safe_div(
        float(df["Close"].iloc[-1]) - float(df["Close"].iloc[warmup]),
        float(df["Close"].iloc[warmup]),
        0,
    ) * 100

    bar_returns = equity_curve["equity_return"]
    bars_per_year = _bars_per_year(mode)
    years_tested = max(len(equity_curve) / max(bars_per_year, 1), 1 / max(bars_per_year, 1))
    annualized_return_pct = ((cash / initial_cash) ** (1 / years_tested) - 1) * 100 if initial_cash > 0 else 0.0
    benchmark_ratio = safe_div(float(df["Close"].iloc[-1]), float(df["Close"].iloc[warmup]), 1.0)
    benchmark_annualized_return_pct = ((benchmark_ratio) ** (1 / years_tested) - 1) * 100 if benchmark_ratio > 0 else 0.0
    annualized_volatility_pct = float(bar_returns.std() * np.sqrt(bars_per_year) * 100) if not bar_returns.empty else 0.0
    sharpe_ratio = 0.0
    sortino_ratio = 0.0
    if bar_returns.std() > 0:
        sharpe_ratio = float(bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year))
    downside = bar_returns[bar_returns < 0]
    if not downside.empty and downside.std() > 0:
        sortino_ratio = float(bar_returns.mean() / downside.std() * np.sqrt(bars_per_year))

    metrics = {
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(float(annualized_return_pct), 2),
        "buy_hold_return_pct": round(buy_hold_return_pct, 2),
        "benchmark_annualized_return_pct": round(float(benchmark_annualized_return_pct), 2),
        "alpha_vs_buy_hold_pct": round(total_return_pct - buy_hold_return_pct, 2),
        "trade_count": int(len(trade_df)),
        "win_rate_pct": round(
            float((trade_df["return_pct"] > 0).mean() * 100) if not trade_df.empty else 0.0,
            2,
        ),
        "max_drawdown_pct": round(abs(_max_drawdown(equity_curve["equity"])) * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "sortino_ratio": round(sortino_ratio, 3),
        "profit_factor": round(_profit_factor(trade_returns), 3),
        "avg_trade_return_pct": round(float(trade_df["return_pct"].mean()) if not trade_df.empty else 0.0, 2),
        "avg_hold_bars": round(float(trade_df["bars_held"].mean()) if not trade_df.empty else 0.0, 1),
        "exposure_pct": round(exposure_bars / max(len(equity_curve), 1) * 100, 2),
        "annualized_volatility_pct": round(annualized_volatility_pct, 2),
        "bars_tested": int(len(equity_curve)),
        "ending_equity": round(float(cash), 2),
    }

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trade_df,
        "mode": trading_mode,
        "start": equity_curve["timestamp"].iloc[0],
        "end": equity_curve["timestamp"].iloc[-1],
        "notes": (
            "Backtest uses real historical OHLCV plus the executable strategy logic: "
            "long/short technical signals, ML forecasts, mode-specific holding rules, risk settings, "
            "and transaction-cost assumptions. Historical sentiment is excluded because "
            "reliable archival sentiment data is difficult to source consistently."
        ),
    }
