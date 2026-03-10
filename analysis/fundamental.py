"""
Fundamental / Market Structure Analysis Module
Scores assets on market quality, supply dynamics, relative strength,
developer activity, and liquidity metrics.

All inputs come from CoinGecko market data and GitHub activity.
"""
import numpy as np
import pandas as pd
from config import FUNDAMENTAL_WEIGHTS
from utils.helpers import get_logger, normalise_score, safe_div

log = get_logger(__name__)


def score_fundamental(market_row: pd.Series,
                      all_market: pd.DataFrame,
                      details: dict | None = None,
                      github: dict | None = None) -> dict:
    """
    Compute fundamental score (0-100) for a single asset.

    Args:
        market_row:  one row from the CoinGecko /coins/markets DataFrame
        all_market:  the full market DataFrame (used for relative comparisons)
        details:     optional CoinGecko /coins/{id} response
        github:      optional GitHub activity dict

    Returns:
        {"score": float, "factors": {name: {"value": ..., "score": float}}}
    """
    details = details or {}
    github = github or {}
    factors = {}

    # ── 1. Market cap rank (lower rank = better) ─────────────────────
    rank = market_row.get("market_cap_rank", 999)
    # Top-10 → 100, Top-50 → ~60, Top-100 → ~40
    factors["market_cap_rank"] = {
        "value": rank,
        "score": normalise_score(100 - rank, 0, 100) if rank else 50.0,
    }

    # ── 2. Volume-to-market-cap ratio (higher = more liquid / active) ─
    vol = market_row.get("total_volume", 0) or 0
    mcap = market_row.get("market_cap", 1) or 1
    vol_mcap = safe_div(vol, mcap, 0)
    # Typical healthy range: 0.02-0.15
    factors["volume_mcap_ratio"] = {
        "value": round(vol_mcap, 4),
        "score": normalise_score(vol_mcap, 0, 0.15) * 0.8 + 10,
    }

    # ── 3. Supply ratio (circulating / max) ──────────────────────────
    circ = market_row.get("circulating_supply", 0) or 0
    max_supply = market_row.get("max_supply")
    if max_supply and max_supply > 0:
        supply_ratio = safe_div(circ, max_supply, 0)
        # Higher ratio = less future dilution = better
        factors["supply_ratio"] = {
            "value": round(supply_ratio, 3),
            "score": normalise_score(supply_ratio, 0, 1),
        }
    else:
        factors["supply_ratio"] = {"value": None, "score": 50.0}

    # ── 4. Price-change momentum (7d + 30d) ──────────────────────────
    pct_7d = market_row.get("price_change_percentage_7d_in_currency", 0) or 0
    pct_30d = market_row.get("price_change_percentage_30d_in_currency", 0) or 0
    # Blend: 60% recent, 40% medium-term
    blended_pct = pct_7d * 0.6 + pct_30d * 0.4
    factors["price_change_momentum"] = {
        "value": round(blended_pct, 2),
        "score": float(np.clip(50 + blended_pct * 1.5, 0, 100)),
    }

    # ── 5. Volatility quality ────────────────────────────────────────
    # Low volatility relative to peers is quality in conservative framing.
    # We compute it here; risk module will reweight.
    pct_24h = abs(market_row.get("price_change_percentage_24h", 0) or 0)
    all_vol = all_market.get("price_change_percentage_24h", pd.Series([0])).abs()
    vol_percentile = float((all_vol < pct_24h).mean() * 100) if len(all_vol) > 1 else 50
    factors["volatility_quality"] = {
        "value": round(pct_24h, 2),
        "score": normalise_score(100 - vol_percentile, 0, 100),
    }

    # ── 6. Relative strength vs BTC ─────────────────────────────────
    btc_row = all_market.loc[all_market["id"] == "bitcoin"]
    if not btc_row.empty:
        btc_7d = btc_row.iloc[0].get("price_change_percentage_7d_in_currency", 0) or 0
        rel_strength = pct_7d - btc_7d
    else:
        rel_strength = 0
    factors["relative_strength"] = {
        "value": round(rel_strength, 2),
        "score": float(np.clip(50 + rel_strength * 2, 0, 100)),
    }

    # ── 7. Developer activity (GitHub proxy) ─────────────────────────
    stars = github.get("stars", 0)
    commits = github.get("recent_commits", 0)
    forks = github.get("forks", 0)
    # Composite: stars/1000 + commits/100 + forks/500
    dev_raw = stars / 1000 + commits / 100 + forks / 500
    factors["developer_activity"] = {
        "value": {"stars": stars, "commits_4w": commits, "forks": forks},
        "score": float(np.clip(dev_raw * 10, 0, 100)) if any([stars, commits, forks]) else 50.0,
    }

    # ── 8. Liquidity proxy ───────────────────────────────────────────
    # Simple proxy: volume relative to median volume across universe
    median_vol = all_market["total_volume"].median() if "total_volume" in all_market else 1
    liq_ratio = safe_div(vol, median_vol, 0) if median_vol else 0
    factors["liquidity"] = {
        "value": round(liq_ratio, 2),
        "score": float(np.clip(normalise_score(liq_ratio, 0, 3), 0, 100)),
    }

    # ── Weighted final score ─────────────────────────────────────────
    total = sum(
        factors[k]["score"] * FUNDAMENTAL_WEIGHTS.get(k, 0)
        for k in factors
    )

    return {
        "score": round(float(np.clip(total, 0, 100)), 2),
        "factors": factors,
    }
