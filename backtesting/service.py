"""
Historical backtesting service.

This layer turns the engine into a real on-demand module by:
- resolving the requested asset
- fetching real historical OHLCV for that asset and strategy mode
- running the walk-forward engine
- returning metadata that the frontend or API can render directly
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import ASSET_UNIVERSE, TRADING_MODES
from data.market_data import fetch_ohlcv
from utils.helpers import get_logger

from .engine import run_mode_backtest

log = get_logger(__name__)


@dataclass(frozen=True)
class BacktestRequest:
    """Canonical backtest request for one asset and one strategy mode."""

    coin_id: str
    symbol: str
    trading_mode: str
    risk_level: str
    initial_cash: float
    period: str


def list_backtest_assets() -> list[dict]:
    """Return all supported backtest assets in a UI-friendly format."""
    return [
        {
            "coin_id": coin_id,
            "symbol": meta["symbol"],
            "label": f"{meta['symbol']} ({coin_id})",
            "yf": meta["yf"],
        }
        for coin_id, meta in ASSET_UNIVERSE.items()
    ]


def resolve_asset(asset: str) -> tuple[str, dict]:
    """Resolve a user-facing asset identifier to the internal universe key."""
    asset_normalized = asset.strip().lower()

    if asset_normalized in ASSET_UNIVERSE:
        return asset_normalized, ASSET_UNIVERSE[asset_normalized]

    for coin_id, meta in ASSET_UNIVERSE.items():
        if meta["symbol"].lower() == asset_normalized:
            return coin_id, meta

    raise KeyError(f"Unsupported asset: {asset}")


def fetch_backtest_history(
    asset: str,
    trading_mode: str,
    period: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[str, dict, pd.DataFrame]:
    """Fetch real historical OHLCV aligned to the requested trading mode."""
    coin_id, meta = resolve_asset(asset)
    mode = TRADING_MODES.get(trading_mode, TRADING_MODES["swing"])
    history = fetch_ohlcv(
        meta["yf"],
        interval=mode["yfinance_interval"],
        period=period or mode.get("backtest_period") or mode["yfinance_period"],
        start=start,
        end=end,
    )
    return coin_id, meta, history


def run_historical_backtest(
    asset: str,
    trading_mode: str = "swing",
    risk_level: str = "moderate",
    initial_cash: float = 10_000.0,
    period: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict:
    """Run a real historical backtest on demand for one asset and one mode."""
    mode = TRADING_MODES.get(trading_mode, TRADING_MODES["swing"])
    coin_id, meta, history = fetch_backtest_history(
        asset,
        trading_mode=trading_mode,
        period=period,
        start=start,
        end=end,
    )

    if history.empty:
        return {
            "request": BacktestRequest(
                coin_id=coin_id,
                symbol=meta["symbol"],
                trading_mode=trading_mode,
                risk_level=risk_level,
                initial_cash=initial_cash,
                period=period or mode.get("backtest_period") or mode["yfinance_period"],
            ).__dict__,
            "history": pd.DataFrame(),
            "result": {
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
                "notes": "No historical data was returned for the selected asset/mode.",
            },
        }

    result = run_mode_backtest(
        history,
        trading_mode=trading_mode,
        risk_level=risk_level,
        initial_cash=initial_cash,
    )

    payload = {
        "request": BacktestRequest(
            coin_id=coin_id,
            symbol=meta["symbol"],
            trading_mode=trading_mode,
            risk_level=risk_level,
            initial_cash=initial_cash,
            period=period or mode.get("backtest_period") or mode["yfinance_period"],
        ).__dict__,
        "history": history,
        "result": result,
        "history_meta": {
            "symbol": meta["symbol"],
            "coin_id": coin_id,
            "interval": mode["yfinance_interval"],
            "period": period or mode.get("backtest_period") or mode["yfinance_period"],
            "rows": int(len(history)),
            "start": history.index.min(),
            "end": history.index.max(),
        },
    }
    log.info(
        "Completed historical backtest for %s in %s mode with %d bars",
        meta["symbol"],
        trading_mode,
        len(history),
    )
    return payload
