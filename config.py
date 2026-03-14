"""
Crypto Investor MVP — Central Configuration
All tunable parameters, asset universe, API endpoints, and scoring weights.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Asset Universe ────────────────────────────────────────────────────────────
# Stable internal asset ID → metadata used across modules
ASSET_UNIVERSE = {
    "bitcoin":        {"symbol": "BTC",   "yf": "BTC-USD",   "paprika": "btc-bitcoin",       "github": "bitcoin/bitcoin"},
    "ethereum":       {"symbol": "ETH",   "yf": "ETH-USD",   "paprika": "eth-ethereum",      "github": "ethereum/go-ethereum"},
    "solana":         {"symbol": "SOL",   "yf": "SOL-USD",   "paprika": "sol-solana",        "github": "solana-labs/solana"},
    "binancecoin":    {"symbol": "BNB",   "yf": "BNB-USD",   "paprika": "bnb-binance-coin",  "github": None},
    "ripple":         {"symbol": "XRP",   "yf": "XRP-USD",   "paprika": "xrp-xrp",           "github": "ripple/rippled"},
    "cardano":        {"symbol": "ADA",   "yf": "ADA-USD",   "paprika": "ada-cardano",       "github": "input-output-hk/cardano-node"},
    "avalanche-2":    {"symbol": "AVAX",  "yf": "AVAX-USD",  "paprika": "avax-avalanche",    "github": "ava-labs/avalanchego"},
    "polkadot":       {"symbol": "DOT",   "yf": "DOT-USD",   "paprika": "dot-polkadot",      "github": "paritytech/polkadot-sdk"},
    "chainlink":      {"symbol": "LINK",  "yf": "LINK-USD",  "paprika": "link-chainlink",    "github": "smartcontractkit/chainlink"},
    "matic-network":  {"symbol": "MATIC", "yf": "POL28321-USD", "paprika": "matic-polygon",     "github": "maticnetwork/bor"},
    "litecoin":       {"symbol": "LTC",   "yf": "LTC-USD",   "paprika": "ltc-litecoin",      "github": "litecoin-project/litecoin"},
    "uniswap":        {"symbol": "UNI",   "yf": "UNI7083-USD",   "paprika": "uni-uniswap",       "github": "Uniswap/v3-core"},
    "near":           {"symbol": "NEAR",  "yf": "NEAR-USD",  "paprika": "near-near-protocol","github": "near/nearcore"},
    "aptos":          {"symbol": "APT",   "yf": "APT21794-USD",   "paprika": "apt-aptos",         "github": "aptos-labs/aptos-core"},
    "sui":            {"symbol": "SUI",   "yf": "SUI20947-USD",   "paprika": "sui-sui",           "github": None},
}

# ── API Configuration ─────────────────────────────────────────────────────────
COINPAPRIKA_BASE = "https://api.coinpaprika.com/v1"
COINPAPRIKA_RATE_LIMIT = float(os.getenv("COINPAPRIKA_RATE_LIMIT", "0.2"))

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# RSS news feeds for sentiment
NEWS_RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/feed",
]

# ── Technical Analysis ────────────────────────────────────────────────────────
OHLC_HISTORY_DAYS = 180          # days of historical data to fetch
TECHNICAL_INDICATOR_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ema_short": 12,
    "ema_long": 26,
    "sma_short": 20,
    "sma_long": 50,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "adx_period": 14,
    "stoch_period": 14,
    "stoch_smooth": 3,
    "obv_period": 20,  # for OBV EMA smoothing
}

# Weights within the technical score (must sum to 1.0)
TECHNICAL_WEIGHTS = {
    "rsi":           0.12,
    "macd":          0.14,
    "ema_cross":     0.10,
    "bollinger":     0.10,
    "adx":           0.10,
    "stochastic":    0.08,
    "obv":           0.08,
    "atr_trend":     0.06,
    "sma_cross":     0.08,
    "momentum":      0.07,
    "support_resist": 0.07,
}

# ── Fundamental Analysis ──────────────────────────────────────────────────────
FUNDAMENTAL_WEIGHTS = {
    "market_cap_rank":       0.15,
    "volume_mcap_ratio":     0.15,
    "supply_ratio":          0.10,
    "price_change_momentum": 0.15,
    "volatility_quality":    0.10,
    "relative_strength":     0.15,
    "developer_activity":    0.10,
    "liquidity":             0.10,
}

# ── Sentiment Analysis ────────────────────────────────────────────────────────
SENTIMENT_MAX_ARTICLES = 30      # per asset
SENTIMENT_LOOKBACK_DAYS = 7

# ── ML / Forecasting ─────────────────────────────────────────────────────────
ML_FORECAST_HORIZON = 7          # days ahead
ML_TRAIN_WINDOW = 120            # days for training
ML_FEATURES_LAGS = [1, 2, 3, 5, 7, 14, 21]  # lagged return periods

# ── Combined Scoring Weights ──────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "technical":    0.30,
    "fundamental":  0.20,
    "sentiment":    0.20,
    "ml_forecast":  0.30,
}

# ── Risk Profiles ─────────────────────────────────────────────────────────────
RISK_PROFILES = {
    "conservative": {
        "leverage": 1.0,
        "max_leverage": 1.0,
        "max_volatility_percentile": 50,
        "stop_loss_atr_mult": 2.5,
        "target_atr_mult": 3.0,
        "min_confidence": 60,
        "description": "No leverage. Prioritises stability and strong fundamentals.",
        "weight_adjustments": {
            "technical": 1.0,
            "fundamental": 1.3,
            "sentiment": 0.8,
            "ml_forecast": 0.9,
        },
    },
    "moderate": {
        "leverage": 2.5,
        "max_leverage": 3.0,
        "max_volatility_percentile": 75,
        "stop_loss_atr_mult": 2.0,
        "target_atr_mult": 4.5,
        "min_confidence": 50,
        "description": "Moderate leverage (2-3x). Balanced across all signals.",
        "weight_adjustments": {
            "technical": 1.0,
            "fundamental": 1.0,
            "sentiment": 1.0,
            "ml_forecast": 1.0,
        },
    },
    "aggressive": {
        "leverage": 5.0,
        "max_leverage": 5.0,
        "max_volatility_percentile": 100,
        "stop_loss_atr_mult": 1.5,
        "target_atr_mult": 6.0,
        "min_confidence": 35,
        "description": "High leverage (up to 5x). Favours momentum and sentiment.",
        "weight_adjustments": {
            "technical": 1.0,
            "fundamental": 0.7,
            "sentiment": 1.3,
            "ml_forecast": 1.2,
        },
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
