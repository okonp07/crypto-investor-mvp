"""
Technical Analysis Module
Computes a comprehensive set of indicators and derives a weighted technical score (0-100).

Indicators: RSI, MACD, EMA/SMA crossovers, Bollinger Bands, ATR, ADX,
            Stochastic Oscillator, OBV, support/resistance approximation,
            momentum composite.
"""
import numpy as np
import pandas as pd
import ta
from config import TECHNICAL_INDICATOR_PARAMS as TIP, TECHNICAL_WEIGHTS
from utils.helpers import get_logger, normalise_score, safe_div

log = get_logger(__name__)


# ── Compute all indicators ───────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators as new columns to an OHLCV DataFrame.
    Expects columns: Open, High, Low, Close, Volume.
    Returns the augmented DataFrame (modifies in place for efficiency).
    """
    if df.empty or len(df) < 30:
        log.warning("Insufficient data for indicators (%d bars)", len(df))
        return df

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── Trend indicators ──────────────────────────────────────────────
    df["ema_short"]  = ta.trend.ema_indicator(c, window=TIP["ema_short"])
    df["ema_long"]   = ta.trend.ema_indicator(c, window=TIP["ema_long"])
    df["sma_short"]  = ta.trend.sma_indicator(c, window=TIP["sma_short"])
    df["sma_long"]   = ta.trend.sma_indicator(c, window=TIP["sma_long"])

    macd_obj = ta.trend.MACD(c, window_slow=TIP["macd_slow"],
                              window_fast=TIP["macd_fast"],
                              window_sign=TIP["macd_signal"])
    df["macd"]       = macd_obj.macd()
    df["macd_signal"]= macd_obj.macd_signal()
    df["macd_diff"]  = macd_obj.macd_diff()

    adx_obj = ta.trend.ADXIndicator(h, l, c, window=TIP["adx_period"])
    df["adx"]        = adx_obj.adx()
    df["adx_pos"]    = adx_obj.adx_pos()
    df["adx_neg"]    = adx_obj.adx_neg()

    # ── Momentum indicators ───────────────────────────────────────────
    df["rsi"] = ta.momentum.rsi(c, window=TIP["rsi_period"])

    stoch = ta.momentum.StochasticOscillator(
        h, l, c, window=TIP["stoch_period"], smooth_window=TIP["stoch_smooth"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── Volatility indicators ─────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, window=TIP["bb_period"], window_dev=TIP["bb_std"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pband"] = bb.bollinger_pband()   # % position within bands

    df["atr"] = ta.volatility.average_true_range(h, l, c, window=TIP["atr_period"])

    # ── Volume indicators ─────────────────────────────────────────────
    df["obv"] = ta.volume.on_balance_volume(c, v)
    df["obv_ema"] = ta.trend.ema_indicator(df["obv"], window=TIP["obv_period"])

    # ── Derived columns ───────────────────────────────────────────────
    df["returns"] = c.pct_change()
    df["momentum_5"]  = c.pct_change(5)
    df["momentum_10"] = c.pct_change(10)
    df["vol_change"]  = v.pct_change(5)

    return df


# ── Support / resistance approximation ───────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Simple support/resistance via rolling min/max and pivot points.
    Returns dict with support and resistance levels.
    """
    recent = df.tail(lookback)
    if recent.empty:
        return {"support": None, "resistance": None, "pivot": None}

    high = recent["High"].max()
    low  = recent["Low"].min()
    close = recent["Close"].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high

    return {
        "support": float(s1),
        "resistance": float(r1),
        "pivot": float(pivot),
        "range_high": float(high),
        "range_low": float(low),
    }


# ── Individual signal scorers (each returns 0-100) ──────────────────────────

def _score_rsi(rsi: float) -> float:
    """RSI: oversold (<30)=bullish, overbought (>70)=bearish, mid=neutral."""
    if np.isnan(rsi):
        return 50.0
    if rsi < 30:
        return normalise_score(30 - rsi, 0, 30)  # lower RSI → higher score
    if rsi > 70:
        return normalise_score(70 - rsi, -30, 0)  # 70→50, 100→0
    return 50.0 + (50 - rsi)  # slight tilt: 50→50, 30→70


def _score_macd(macd_diff: float, prev_diff: float) -> float:
    """MACD histogram: positive & rising = bullish."""
    if np.isnan(macd_diff) or np.isnan(prev_diff):
        return 50.0
    base = 60.0 if macd_diff > 0 else 40.0
    if macd_diff > prev_diff:
        base += 15.0  # rising histogram
    elif macd_diff < prev_diff:
        base -= 15.0
    return float(np.clip(base, 0, 100))


def _score_ema_cross(close: float, ema_s: float, ema_l: float) -> float:
    """EMA crossover: short above long = bullish, price above both = extra."""
    if any(np.isnan(x) for x in (close, ema_s, ema_l)):
        return 50.0
    score = 50.0
    if ema_s > ema_l:
        score += 20.0
    else:
        score -= 20.0
    if close > ema_s:
        score += 10.0
    if close > ema_l:
        score += 10.0
    return float(np.clip(score, 0, 100))


def _score_sma_cross(close: float, sma_s: float, sma_l: float) -> float:
    """SMA crossover: analogous to EMA cross."""
    return _score_ema_cross(close, sma_s, sma_l)


def _score_bollinger(bb_pband: float) -> float:
    """Bollinger %B: <0 oversold (bullish), >1 overbought (bearish)."""
    if np.isnan(bb_pband):
        return 50.0
    if bb_pband < 0.2:
        return 80.0
    if bb_pband > 0.8:
        return 20.0
    return normalise_score(1 - bb_pband, 0, 1)


def _score_adx(adx: float, adx_pos: float, adx_neg: float) -> float:
    """ADX: strong trend (>25) with +DI > -DI = bullish."""
    if any(np.isnan(x) for x in (adx, adx_pos, adx_neg)):
        return 50.0
    if adx < 20:
        return 50.0  # no clear trend
    direction = 70.0 if adx_pos > adx_neg else 30.0
    strength_bonus = min((adx - 20) * 0.5, 15.0)
    return float(np.clip(direction + strength_bonus * (1 if adx_pos > adx_neg else -1), 0, 100))


def _score_stochastic(k: float, d: float) -> float:
    """Stochastic: oversold K<20=bullish, overbought K>80=bearish, K crossing D."""
    if any(np.isnan(x) for x in (k, d)):
        return 50.0
    score = 50.0
    if k < 20:
        score += 25.0
    elif k > 80:
        score -= 25.0
    if k > d:
        score += 10.0
    else:
        score -= 10.0
    return float(np.clip(score, 0, 100))


def _score_obv(obv: float, obv_ema: float) -> float:
    """OBV: above its EMA = accumulation (bullish)."""
    if any(np.isnan(x) for x in (obv, obv_ema)):
        return 50.0
    diff_pct = safe_div(obv - obv_ema, abs(obv_ema), 0) * 100
    return float(np.clip(50 + diff_pct * 2, 0, 100))


def _score_atr_trend(atr: float, atr_prev: float) -> float:
    """ATR trend: decreasing volatility can signal consolidation before breakout."""
    if any(np.isnan(x) for x in (atr, atr_prev)):
        return 50.0
    change = safe_div(atr - atr_prev, atr_prev, 0)
    return float(np.clip(50 - change * 100, 20, 80))


def _score_momentum(mom5: float, mom10: float) -> float:
    """Combined 5d + 10d momentum: positive = bullish."""
    vals = [x for x in (mom5, mom10) if not np.isnan(x)]
    if not vals:
        return 50.0
    avg = np.mean(vals)
    return float(np.clip(50 + avg * 300, 0, 100))


def _score_support_resist(close: float, sr: dict) -> float:
    """Proximity to support (bullish) vs resistance (bearish)."""
    if sr["support"] is None or sr["resistance"] is None:
        return 50.0
    total_range = sr["resistance"] - sr["support"]
    if total_range <= 0:
        return 50.0
    position = (close - sr["support"]) / total_range
    # Closer to support = higher score (buying opportunity)
    return float(np.clip((1 - position) * 80 + 10, 0, 100))


# ── Master technical scorer ──────────────────────────────────────────────────

def score_technical(df: pd.DataFrame) -> dict:
    """
    Compute weighted technical score (0-100) from the latest bar's indicators.

    Returns:
        {
            "score": float,
            "signals": {indicator_name: {"value": float, "score": float, "signal": str}},
            "trend": "bullish" | "bearish" | "neutral",
            "support_resistance": dict,
        }
    """
    if df.empty or len(df) < 30:
        return {"score": 50.0, "signals": {}, "trend": "neutral", "support_resistance": {}}

    df = compute_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    sr = find_support_resistance(df)

    # Score each signal
    signal_scores = {
        "rsi":            _score_rsi(last.get("rsi", 50)),
        "macd":           _score_macd(last.get("macd_diff", 0), prev.get("macd_diff", 0)),
        "ema_cross":      _score_ema_cross(last["Close"], last.get("ema_short", last["Close"]),
                                            last.get("ema_long", last["Close"])),
        "sma_cross":      _score_sma_cross(last["Close"], last.get("sma_short", last["Close"]),
                                            last.get("sma_long", last["Close"])),
        "bollinger":      _score_bollinger(last.get("bb_pband", 0.5)),
        "adx":            _score_adx(last.get("adx", 20), last.get("adx_pos", 0), last.get("adx_neg", 0)),
        "stochastic":     _score_stochastic(last.get("stoch_k", 50), last.get("stoch_d", 50)),
        "obv":            _score_obv(last.get("obv", 0), last.get("obv_ema", 0)),
        "atr_trend":      _score_atr_trend(last.get("atr", 0), prev.get("atr", 0)),
        "momentum":       _score_momentum(last.get("momentum_5", 0), last.get("momentum_10", 0)),
        "support_resist": _score_support_resist(last["Close"], sr),
    }

    # Weighted combination
    total_score = sum(signal_scores[k] * TECHNICAL_WEIGHTS.get(k, 0)
                      for k in signal_scores)

    # Determine trend label
    if total_score >= 60:
        trend = "bullish"
    elif total_score <= 40:
        trend = "bearish"
    else:
        trend = "neutral"

    # Build detailed signal output
    trend_labels = {True: "bullish", False: "bearish"}
    signals = {}
    for name, sc in signal_scores.items():
        signals[name] = {
            "score": round(sc, 1),
            "signal": "bullish" if sc >= 55 else ("bearish" if sc <= 45 else "neutral"),
        }

    return {
        "score": round(total_score, 2),
        "signals": signals,
        "trend": trend,
        "support_resistance": sr,
    }
