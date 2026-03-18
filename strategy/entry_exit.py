"""
Entry / Exit / Stop-Loss Price Logic
Transparent methodology combining technical levels, ATR, and ML forecasts.
"""
import numpy as np
from config import RISK_PROFILES, TRADING_MODES
from utils.helpers import get_logger, safe_div

log = get_logger(__name__)


def compute_levels(
    current_price: float,
    technical: dict,
    ml_forecast: dict,
    risk_level: str = "moderate",
    trading_mode: str = "swing",
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
    sr = technical.get("support_resistance", {})
    forecast = ml_forecast.get("forecast", {})

    support    = sr.get("support", current_price * 0.95)
    resistance = sr.get("resistance", current_price * 1.10)
    pivot      = sr.get("pivot", current_price)

    forecast_mean = forecast.get("forecast_mean", current_price)
    forecast_high = forecast.get("forecast_high", current_price * 1.08)

    atr_mult_stop = profile["stop_loss_atr_mult"] * mode_profile.get("stop_loss_atr_scale", 1.0)
    atr_mult_target = profile["target_atr_mult"] * mode_profile.get("target_atr_scale", 1.0)

    # ── ATR estimate from support/resistance range as proxy ──────────
    # If we had raw ATR we'd use it; approximate from S/R range
    price_range = abs(resistance - support)
    atr_proxy = price_range / 4 if price_range > 0 else current_price * 0.03

    # ── Entry Price ──────────────────────────────────────────────────
    # Blend: pullback toward pivot and support
    # Conservative → deeper pullback, Aggressive → closer to market
    pullback_factors = {"conservative": 0.5, "moderate": 0.3, "aggressive": 0.15}
    pb = pullback_factors.get(risk_level, 0.3)

    # Entry = current_price - pullback toward support
    entry_from_support = current_price - pb * (current_price - support)
    entry_from_pivot   = (current_price + pivot) / 2

    entry_price = (entry_from_support * 0.6 + entry_from_pivot * 0.4)
    # Entry should not be above current price
    entry_price = min(entry_price, current_price * 0.995)
    # Entry should not be below support
    entry_price = max(entry_price, support * 0.98)

    # ── Exit Price (Take Profit) ─────────────────────────────────────
    # Blend resistance level, ATR target, and ML forecast
    exit_from_resistance = resistance
    exit_from_atr = entry_price + atr_proxy * atr_mult_target
    exit_from_forecast = max(forecast_mean, forecast_high * 0.85)

    exit_price = (
        exit_from_resistance * 0.35 +
        exit_from_atr * 0.30 +
        exit_from_forecast * 0.35
    )
    # Exit should be above entry
    exit_price = max(exit_price, entry_price * 1.02)

    # ── Stop Loss ────────────────────────────────────────────────────
    stop_from_support = support - atr_proxy * 0.5
    stop_from_atr = entry_price - atr_proxy * atr_mult_stop

    stop_loss = max(stop_from_support, stop_from_atr)
    # Stop should be below entry
    stop_loss = min(stop_loss, entry_price * 0.97)
    # Stop should not go negative
    stop_loss = max(stop_loss, 0.0)

    # ── Risk / Reward ────────────────────────────────────────────────
    potential_gain = exit_price - entry_price
    potential_loss = entry_price - stop_loss
    risk_reward = safe_div(potential_gain, potential_loss, 0)
    expected_return = safe_div(exit_price - entry_price, entry_price, 0) * 100

    # Format based on price magnitude
    def fmt(p):
        if current_price > 1000:
            return round(p, 2)
        elif current_price > 1:
            return round(p, 4)
        else:
            return round(p, 6)

    methodology = (
        f"Mode: {trading_mode}. "
        f"Entry: {pb*100:.0f}% pullback toward support ({fmt(support)}) and pivot ({fmt(pivot)}). "
        f"Exit: blend of resistance ({fmt(resistance)}), "
        f"ATR target ({fmt(exit_from_atr)}), "
        f"and ML forecast ({fmt(exit_from_forecast)}). "
        f"Stop: {atr_mult_stop:.1f}x ATR below entry, floored at support."
    )

    return {
        "entry_price": fmt(entry_price),
        "exit_price": fmt(exit_price),
        "stop_loss": fmt(stop_loss),
        "risk_reward_ratio": round(float(risk_reward), 2),
        "expected_return_pct": round(float(expected_return), 2),
        "methodology": methodology,
    }
