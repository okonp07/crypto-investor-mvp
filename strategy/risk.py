"""
Risk Tolerance & Leverage Logic
Maps user risk preference to concrete trading parameters.
"""
import numpy as np
from config import RISK_PROFILES
from utils.helpers import get_logger

log = get_logger(__name__)


def get_risk_profile(level: str) -> dict:
    """Return the full risk profile dict for the given level."""
    if level not in RISK_PROFILES:
        log.warning("Unknown risk level '%s', defaulting to moderate", level)
        level = "moderate"
    return RISK_PROFILES[level]


def compute_leverage(
    risk_level: str,
    volatility_pct: float,
    confidence: float,
) -> dict:
    """
    Compute suggested leverage, adjusted for asset volatility and signal confidence.

    Args:
        risk_level:      "conservative" | "moderate" | "aggressive"
        volatility_pct:  recent daily volatility as percentage (e.g. 3.5 for 3.5%)
        confidence:      model confidence score (0-100)

    Returns:
        {
            "leverage": float,
            "max_leverage": float,
            "rationale": str,
            "liquidation_warning": str,
        }
    """
    profile = get_risk_profile(risk_level)
    base_leverage = profile["leverage"]
    max_lev = profile["max_leverage"]

    # Adjust down for high volatility
    # If vol > 5%, reduce leverage proportionally
    vol_factor = 1.0
    if volatility_pct > 3.0:
        vol_factor = max(3.0 / volatility_pct, 0.3)

    # Adjust down for low confidence
    conf_factor = 1.0
    if confidence < 50:
        conf_factor = max(confidence / 50, 0.5)

    adjusted = base_leverage * vol_factor * conf_factor
    leverage = round(float(np.clip(adjusted, 1.0, max_lev)), 1)

    rationale_parts = [f"Base leverage for {risk_level}: {base_leverage}x."]
    if vol_factor < 1.0:
        rationale_parts.append(
            f"Reduced by {(1-vol_factor)*100:.0f}% due to {volatility_pct:.1f}% daily volatility."
        )
    if conf_factor < 1.0:
        rationale_parts.append(
            f"Reduced by {(1-conf_factor)*100:.0f}% due to {confidence:.0f}% model confidence."
        )
    rationale_parts.append(f"Final suggested leverage: {leverage}x.")

    # Liquidation distance estimate
    liq_move = 100 / leverage if leverage > 0 else float("inf")

    warning = (
        f"At {leverage}x leverage, a {liq_move:.1f}% adverse price move "
        f"would approach liquidation. "
        f"Leverage amplifies both gains and losses. "
        f"Never risk more than you can afford to lose."
    )

    return {
        "leverage": leverage,
        "max_leverage": max_lev,
        "rationale": " ".join(rationale_parts),
        "liquidation_warning": warning,
    }
