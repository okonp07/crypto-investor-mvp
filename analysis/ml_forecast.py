"""
Machine Learning / Time-Series Forecasting Module

Two models:
1. XGBoost classifier → bullish / bearish / neutral direction (7-day horizon)
2. Exponential smoothing (statsmodels) → price forecast band

Produces an ML score (0-100), predicted direction, confidence, and price range.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from config import ML_FORECAST_HORIZON, ML_TRAIN_WINDOW, ML_FEATURES_LAGS
from utils.helpers import get_logger, safe_div

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Feature Engineering ──────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features from OHLCV data:
    - Lagged returns (1d, 2d, 3d, 5d, 7d, 14d, 21d)
    - Rolling volatility
    - Rolling volume ratio
    - RSI proxy (simple)
    - Day-of-week
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Lagged returns
    for lag in ML_FEATURES_LAGS:
        feat[f"ret_{lag}d"] = close.pct_change(lag)

    # Rolling volatility
    feat["vol_7d"]  = close.pct_change().rolling(7).std()
    feat["vol_14d"] = close.pct_change().rolling(14).std()
    feat["vol_21d"] = close.pct_change().rolling(21).std()

    # Rolling volume ratio (current vs 20d avg)
    if "Volume" in df.columns:
        feat["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    else:
        feat["vol_ratio"] = 1.0

    # Simple RSI proxy
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_proxy"] = 100 - (100 / (1 + rs))

    # Price relative to moving averages
    feat["close_sma20_ratio"] = close / close.rolling(20).mean()
    feat["close_sma50_ratio"] = close / close.rolling(50).mean()

    # Range (high-low) / close
    if "High" in df.columns and "Low" in df.columns:
        feat["range_pct"] = (df["High"] - df["Low"]) / close

    # Day of week (cyclical)
    if hasattr(df.index, "dayofweek"):
        feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return feat


def _build_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Target: forward return over *horizon* days → class label.
    0 = bearish (< -2%), 1 = neutral (-2% to +2%), 2 = bullish (> +2%)
    """
    fwd_ret = df["Close"].pct_change(horizon).shift(-horizon)
    target = pd.Series(1, index=df.index, dtype=int)  # neutral default
    target[fwd_ret > 0.02] = 2   # bullish
    target[fwd_ret < -0.02] = 0  # bearish
    return target


# ── Direction Classifier ─────────────────────────────────────────────────────

def classify_direction(df: pd.DataFrame) -> dict:
    """
    Train XGBoost on historical features and predict direction for the next period.

    Returns:
        {
            "direction": "bullish" | "bearish" | "neutral",
            "probabilities": {"bullish": float, "bearish": float, "neutral": float},
            "confidence": float (0-100),
            "cv_accuracy": float,
        }
    """
    if len(df) < ML_TRAIN_WINDOW + ML_FORECAST_HORIZON + 10:
        log.warning("Insufficient data for direction classifier (%d bars)", len(df))
        return {
            "direction": "neutral", "probabilities": {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34},
            "confidence": 33.0, "cv_accuracy": 0.0,
        }

    features = _build_features(df)
    target = _build_target(df, ML_FORECAST_HORIZON)

    # Align and drop NaN
    combined = features.join(target.rename("target")).dropna()
    if len(combined) < 50:
        return {
            "direction": "neutral", "probabilities": {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34},
            "confidence": 33.0, "cv_accuracy": 0.0,
        }

    X = combined.drop(columns=["target"])
    y = combined["target"]

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []

    scaler = StandardScaler()

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        cv_scores.append(accuracy_score(y_val, preds))

    cv_acc = float(np.mean(cv_scores))

    # Final model on all data (except last *horizon* rows with no target)
    X_all = scaler.fit_transform(X)
    model.fit(X_all, y)

    # Predict on the most recent features
    X_last = scaler.transform(X.iloc[[-1]])
    proba = model.predict_proba(X_last)[0]

    # Map class indices to labels
    class_labels = {0: "bearish", 1: "neutral", 2: "bullish"}
    classes = model.classes_
    probabilities = {}
    for idx, cls in enumerate(classes):
        probabilities[class_labels.get(cls, str(cls))] = round(float(proba[idx]), 4)

    # Fill missing classes
    for label in ["bullish", "bearish", "neutral"]:
        probabilities.setdefault(label, 0.0)

    pred_class = classes[np.argmax(proba)]
    direction = class_labels.get(pred_class, "neutral")
    confidence = float(np.max(proba)) * 100

    return {
        "direction": direction,
        "probabilities": probabilities,
        "confidence": round(confidence, 1),
        "cv_accuracy": round(cv_acc, 4),
    }


# ── Price Forecast (Exponential Smoothing) ───────────────────────────────────

def forecast_price(df: pd.DataFrame, horizon: int = ML_FORECAST_HORIZON) -> dict:
    """
    Forecast close prices using Holt-Winters exponential smoothing.

    Returns:
        {
            "forecast_prices": list[float],
            "forecast_mean": float,
            "forecast_low": float,
            "forecast_high": float,
            "current_price": float,
            "expected_return_pct": float,
        }
    """
    close = df["Close"].dropna()
    if len(close) < 30:
        current = float(close.iloc[-1]) if len(close) > 0 else 0
        return {
            "forecast_prices": [], "forecast_mean": current,
            "forecast_low": current, "forecast_high": current,
            "current_price": current, "expected_return_pct": 0.0,
        }

    current = float(close.iloc[-1])

    try:
        # Holt-Winters with additive trend (no seasonality — crypto trades 24/7)
        model = ExponentialSmoothing(
            close.values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = model.forecast(horizon)
    except Exception as e:
        log.warning("Exponential smoothing failed: %s — falling back to simple EMA", e)
        # Fallback: extrapolate from EMA trend
        ema = close.ewm(span=14).mean()
        daily_drift = float((ema.iloc[-1] - ema.iloc[-5]) / 5)
        forecast = np.array([current + daily_drift * (i + 1) for i in range(horizon)])

    forecast_prices = [round(float(p), 6) for p in forecast]
    forecast_mean = float(np.mean(forecast))

    # Simple confidence band: use recent volatility
    daily_vol = float(close.pct_change().tail(30).std())
    band_width = daily_vol * np.sqrt(horizon) * current
    forecast_low = forecast_mean - band_width
    forecast_high = forecast_mean + band_width

    expected_return = safe_div(forecast_mean - current, current, 0) * 100

    return {
        "forecast_prices": forecast_prices,
        "forecast_mean": round(forecast_mean, 6),
        "forecast_low": round(float(forecast_low), 6),
        "forecast_high": round(float(forecast_high), 6),
        "current_price": round(current, 6),
        "expected_return_pct": round(float(expected_return), 2),
    }


# ── Combined ML Score ────────────────────────────────────────────────────────

def forecast_asset(df: pd.DataFrame) -> dict:
    """
    Run the full ML forecasting pipeline for one asset.

    Returns:
        {
            "score": float (0-100),
            "direction": dict (from classify_direction),
            "forecast": dict (from forecast_price),
        }
    """
    direction = classify_direction(df)
    forecast = forecast_price(df)

    # Composite ML score:
    # - Direction probability contributes 60%
    # - Expected return direction contributes 40%
    dir_score = direction["probabilities"].get("bullish", 0.33) * 100
    ret_score = float(np.clip(50 + forecast["expected_return_pct"] * 3, 0, 100))

    score = dir_score * 0.6 + ret_score * 0.4

    # Boost score if direction and forecast agree
    if direction["direction"] == "bullish" and forecast["expected_return_pct"] > 0:
        score = min(score + 5, 100)
    elif direction["direction"] == "bearish" and forecast["expected_return_pct"] < 0:
        score = max(score - 5, 0)

    return {
        "score": round(float(np.clip(score, 0, 100)), 2),
        "direction": direction,
        "forecast": forecast,
    }
