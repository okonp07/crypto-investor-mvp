"""
Entry / Exit / Stop-Loss Price Logic
Transparent methodology combining technical levels, ATR, and ML forecasts.
"""
import numpy as np
from config import RISK_PROFILES, TRADING_MODES
from scoring.engine import derive_trade_setup
from utils.helpers import get_logger, safe_div

log = get_logger(__name__)


def compute_levels(
    current_price: float,
    technical: dict,
    ml_forecast: dict,
    risk_level: str = "moderate",
    trading_mode: str = "swing",
    trade_setup: dict | None = None,
) -> dict:
    """
    Compute entry, exit, and stop-loss prices for a single asset.

    Methodology:
        Entry:  pull back toward support / pivot / EMA, weighted by risk tolerance
        Exit:   resistance / ATR target / forecast target, blended
        Stop:   below support / ATR-based, scaled by risk tolerance

    Args:
        current_price: latest close price
        technical:      output of score_technical() — includes support_resistance
        ml_forecast:    output of forecast_asset()  — includes forecast dict
        risk_level:     "conservative" | "moderate" | "aggressive"

    Returns:
        {
            "trade_direction": str,
            "trade_label": str,
            "opportunity_score": float,
            "entry_price": float,
            "exit_price": float,
            "stop_loss": float,
            "risk_reward_ratio": float,
            "expected_return_pct": float,
            "methodology": str,
        }
    """
    profile = RISK_PROFILES.get(risk_level, RISK_PROFILES["moderate"])
    mode_profile = TRADING_MODES.get(trading_mode, TRADING_MODES["swing"])
    trade_setup = trade_setup or derive_trade_setup(technical, ml_forecast)
    trade_direction = trade_setup.get("direction", "neutral")
    if trade_direction == "neutral":
        trade_direction = "short" if trade_setup.get("short_edge", 0) > trade_setup.get("long_edge", 0) else "long"
    sr = technical.get("support_resistance", {})
    forecast = ml_forecast.get("forecast", {})

    support    = sr.get("support", current_price * 0.95)
    resistance = sr.get("resistance", current_price * 1.10)
    pivot      = sr.get("pivot", current_price)

    forecast_mean = forecast.get("forecast_mean", current_price)
    forecast_high = forecast.get("forecast_high", current_price * 1.08)
    forecast_low = forecast.get("forecast_low", current_price * 0.92)

    atr_mult_stop = profile["stop_loss_atr_mult"] * mode_profile.get("stop_loss_atr_scale", 1.0)
    atr_mult_target = profile["target_atr_mult"] * mode_profile.get("target_atr_scale", 1.0)

    # ── ATR estimate from support/resistance range as proxy ──────────
    # If we had raw ATR we'd use it; approximate from S/R range
    price_range = abs(resistance - support)
    atr_proxy = price_range / 4 if price_range > 0 else current_price * 0.03

    # ── Entry Price ──────────────────────────────────────────────────
    pullback_factors = {"conservative": 0.5, "moderate": 0.3, "aggressive": 0.15}
    pb = pullback_factors.get(risk_level, 0.3)

    if trade_direction == "short":
        entry_from_resistance = current_price + pb * (resistance - current_price)
        entry_from_pivot = (current_price + pivot) / 2
        entry_price = (entry_from_resistance * 0.6 + entry_from_pivot * 0.4)
        entry_price = max(entry_price, current_price * 0.995)
        entry_price = min(entry_price, resistance * 1.02)

        exit_from_support = support
        exit_from_atr = entry_price - atr_proxy * atr_mult_target
        exit_from_forecast = min(forecast_mean, forecast_low * 1.05)
        exit_price = (
            exit_from_support * 0.35
            + exit_from_atr * 0.30
            + exit_from_forecast * 0.35
        )
        exit_price = min(exit_price, entry_price * 0.98)

        stop_from_resistance = resistance + atr_proxy * 0.5
        stop_from_atr = entry_price + atr_proxy * atr_mult_stop
        stop_loss = max(stop_from_resistance, stop_from_atr)
        stop_loss = max(stop_loss, entry_price * 1.03)

        potential_gain = entry_price - exit_price
        potential_loss = stop_loss - entry_price
        expected_return = safe_div(entry_price - exit_price, entry_price, 0) * 100
        methodology = (
            f"Mode: {trading_mode}. Short setup. "
            f"Entry: {pb*100:.0f}% rebound toward resistance ({resistance:.4f}) and pivot ({pivot:.4f}). "
            f"Exit: blend of support ({support:.4f}), "
            f"ATR target ({exit_from_atr:.4f}), "
            f"and ML downside forecast ({exit_from_forecast:.4f}). "
            f"Stop: {atr_mult_stop:.1f}x ATR above entry, anchored beyond resistance."
        )
    else:
        entry_from_support = current_price - pb * (current_price - support)
        entry_from_pivot = (current_price + pivot) / 2
        entry_price = (entry_from_support * 0.6 + entry_from_pivot * 0.4)
        entry_price = min(entry_price, current_price * 0.995)
        entry_price = max(entry_price, support * 0.98)

        exit_from_resistance = resistance
        exit_from_atr = entry_price + atr_proxy * atr_mult_target
        exit_from_forecast = max(forecast_mean, forecast_high * 0.85)
        exit_price = (
            exit_from_resistance * 0.35
            + exit_from_atr * 0.30
            + exit_from_forecast * 0.35
        )
        exit_price = max(exit_price, entry_price * 1.02)

        stop_from_support = support - atr_proxy * 0.5
        stop_from_atr = entry_price - atr_proxy * atr_mult_stop
        stop_loss = max(stop_from_support, stop_from_atr)
        stop_loss = min(stop_loss, entry_price * 0.97)
        stop_loss = max(stop_loss, 0.0)

        potential_gain = exit_price - entry_price
        potential_loss = entry_price - stop_loss
        expected_return = safe_div(exit_price - entry_price, entry_price, 0) * 100
        methodology = (
            f"Mode: {trading_mode}. Long setup. "
            f"Entry: {pb*100:.0f}% pullback toward support ({support:.4f}) and pivot ({pivot:.4f}). "
            f"Exit: blend of resistance ({resistance:.4f}), "
            f"ATR target ({exit_from_atr:.4f}), "
            f"and ML forecast ({exit_from_forecast:.4f}). "
            f"Stop: {atr_mult_stop:.1f}x ATR below entry, floored at support."
        )

    risk_reward = safe_div(potential_gain, potential_loss, 0)

    # Format based on price magnitude
    def fmt(p):
        if current_price > 1000:
            return round(p, 2)
        elif current_price > 1:
            return round(p, 4)
        else:
            return round(p, 6)

    return {
        "trade_direction": trade_direction,
        "trade_label": "Short Setup" if trade_direction == "short" else "Long Setup",
        "opportunity_score": trade_setup.get("opportunity_score", 50),
        "entry_price": fmt(entry_price),
        "exit_price": fmt(exit_price),
        "stop_loss": fmt(stop_loss),
        "risk_reward_ratio": round(float(risk_reward), 2),
        "expected_return_pct": round(float(expected_return), 2),
        "methodology": methodology,
    }
