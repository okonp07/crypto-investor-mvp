"""
Combined Scoring & Ranking Engine
Merges technical, fundamental, sentiment, and ML scores into a final
ranking with explanations for each pick.
"""
import numpy as np
from config import SCORING_WEIGHTS, RISK_PROFILES
from utils.helpers import get_logger

log = get_logger(__name__)


def compute_final_score(
    technical: dict,
    fundamental: dict,
    sentiment: dict,
    ml_forecast: dict,
    risk_level: str = "moderate",
) -> dict:
    """
    Combine sub-scores into a single final score, adjusted by risk profile.

    Each input dict must contain a "score" key (0-100).

    Returns:
        {
            "final_score": float,
            "components": {name: {"raw": float, "weighted": float}},
            "risk_adjusted": bool,
        }
    """
    risk_profile = RISK_PROFILES.get(risk_level, RISK_PROFILES["moderate"])
    adjustments = risk_profile["weight_adjustments"]

    raw_scores = {
        "technical":   technical.get("score", 50),
        "fundamental": fundamental.get("score", 50),
        "sentiment":   sentiment.get("score", 50),
        "ml_forecast": ml_forecast.get("score", 50),
    }

    # Apply base weights * risk adjustments, then re-normalise
    adjusted_weights = {}
    for key in SCORING_WEIGHTS:
        adjusted_weights[key] = SCORING_WEIGHTS[key] * adjustments.get(key, 1.0)

    # Normalise so they sum to 1
    total_w = sum(adjusted_weights.values())
    for key in adjusted_weights:
        adjusted_weights[key] /= total_w

    # Weighted sum
    final = sum(raw_scores[k] * adjusted_weights[k] for k in raw_scores)

    components = {}
    for k in raw_scores:
        components[k] = {
            "raw": round(raw_scores[k], 2),
            "weight": round(adjusted_weights[k], 3),
            "weighted": round(raw_scores[k] * adjusted_weights[k], 2),
        }

    return {
        "final_score": round(float(np.clip(final, 0, 100)), 2),
        "components": components,
        "risk_adjusted": risk_level != "moderate",
    }


def rank_assets(all_results: dict, top_n: int = 3) -> list[dict]:
    """
    Rank all analysed assets by final_score and return top N picks.

    Args:
        all_results: {coin_id: {"final": {...}, "technical": {...}, ...}}
        top_n: number of picks to return

    Returns:
        Sorted list of dicts, each with the full analysis for that asset.
    """
    ranked = sorted(
        all_results.items(),
        key=lambda item: item[1]["final"]["final_score"],
        reverse=True,
    )
    return [{"coin_id": cid, **data} for cid, data in ranked[:top_n]]


def generate_reasoning(pick: dict) -> str:
    """
    Generate a human-readable explanation for why an asset was selected.
    """
    coin_id = pick.get("coin_id", "unknown")
    symbol = pick.get("symbol", coin_id.upper())
    final = pick.get("final", {})
    tech = pick.get("technical", {})
    fund = pick.get("fundamental", {})
    sent = pick.get("sentiment", {})
    ml = pick.get("ml_forecast", {})

    lines = []
    score = final.get("final_score", 0)
    lines.append(f"{symbol} scored {score:.1f}/100 overall.")

    # Technical
    t_score = tech.get("score", 50)
    t_trend = tech.get("trend", "neutral")
    lines.append(f"Technical analysis ({t_score:.0f}/100): trend is {t_trend}.")

    # Highlight strongest signals
    signals = tech.get("signals", {})
    bullish_sigs = [k for k, v in signals.items() if v.get("signal") == "bullish"]
    if bullish_sigs:
        lines.append(f"  Bullish signals: {', '.join(bullish_sigs)}.")

    # Fundamental
    f_score = fund.get("score", 50)
    factors = fund.get("factors", {})
    lines.append(f"Fundamentals ({f_score:.0f}/100).")
    # Highlight top factor
    top_factor = max(factors.items(), key=lambda x: x[1].get("score", 0), default=None)
    if top_factor:
        lines.append(f"  Strongest factor: {top_factor[0]} ({top_factor[1]['score']:.0f}/100).")

    # Sentiment
    s_score = sent.get("score", 50)
    s_trend = sent.get("trend", "stable")
    s_count = sent.get("article_count", 0)
    lines.append(f"Sentiment ({s_score:.0f}/100): {s_trend} across {s_count} articles.")

    # ML
    ml_score = ml.get("score", 50)
    direction_info = ml.get("direction", {})
    forecast_info = ml.get("forecast", {})
    direction = direction_info.get("direction", "neutral")
    confidence = direction_info.get("confidence", 0)
    exp_ret = forecast_info.get("expected_return_pct", 0)
    lines.append(
        f"ML forecast ({ml_score:.0f}/100): {direction} direction "
        f"({confidence:.0f}% confidence), expected return {exp_ret:+.1f}%."
    )

    return "\n".join(lines)


def determine_market_regime(all_results: dict) -> str:
    """
    Simple market regime detection based on aggregate scores.
    """
    if not all_results:
        return "unknown"

    avg_tech = np.mean([r.get("technical", {}).get("score", 50) for r in all_results.values()])
    avg_sent = np.mean([r.get("sentiment", {}).get("score", 50) for r in all_results.values()])
    avg_ml = np.mean([r.get("ml_forecast", {}).get("score", 50) for r in all_results.values()])

    composite = (avg_tech + avg_sent + avg_ml) / 3

    if composite >= 58:
        return "bullish"
    elif composite <= 42:
        return "bearish"
    else:
        return "mixed / transitional"
