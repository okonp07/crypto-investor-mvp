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

# Free, high-signal sentiment/news sources.
# The pipeline treats professional editorial sources more heavily than
# community feeds, then blends both into the final sentiment view.
NEWS_RSS_FEEDS = [
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "source_type": "news",
        "weight": 1.0,
    },
    {
        "name": "Cointelegraph",
        "url": "https://cointelegraph.com/rss",
        "source_type": "news",
        "weight": 0.95,
    },
    {
        "name": "Decrypt",
        "url": "https://decrypt.co/feed",
        "source_type": "news",
        "weight": 0.9,
    },
    {
        "name": "Bitcoin Magazine",
        "url": "https://bitcoinmagazine.com/feed",
        "source_type": "news",
        "weight": 0.85,
    },
    {
        "name": "Blockworks",
        "url": "https://blockworks.co/feed",
        "source_type": "news",
        "weight": 0.95,
    },
    {
        "name": "The Defiant",
        "url": "https://thedefiant.io/feed",
        "source_type": "news",
        "weight": 0.9,
    },
]

REDDIT_COMMUNITY_FEEDS = {
    "*": [
        {
            "name": "r/CryptoCurrency",
            "url": "https://www.reddit.com/r/CryptoCurrency/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "BTC": [
        {
            "name": "r/Bitcoin",
            "url": "https://www.reddit.com/r/Bitcoin/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "ETH": [
        {
            "name": "r/ethereum",
            "url": "https://www.reddit.com/r/ethereum/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "SOL": [
        {
            "name": "r/solana",
            "url": "https://www.reddit.com/r/solana/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "XRP": [
        {
            "name": "r/Ripple",
            "url": "https://www.reddit.com/r/Ripple/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "ADA": [
        {
            "name": "r/cardano",
            "url": "https://www.reddit.com/r/cardano/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "AVAX": [
        {
            "name": "r/Avax",
            "url": "https://www.reddit.com/r/Avax/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "DOT": [
        {
            "name": "r/dot",
            "url": "https://www.reddit.com/r/dot/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "LINK": [
        {
            "name": "r/Chainlink",
            "url": "https://www.reddit.com/r/Chainlink/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "MATIC": [
        {
            "name": "r/0xPolygon",
            "url": "https://www.reddit.com/r/0xPolygon/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "LTC": [
        {
            "name": "r/litecoin",
            "url": "https://www.reddit.com/r/litecoin/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "UNI": [
        {
            "name": "r/UniSwap",
            "url": "https://www.reddit.com/r/UniSwap/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "NEAR": [
        {
            "name": "r/nearprotocol",
            "url": "https://www.reddit.com/r/nearprotocol/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "APT": [
        {
            "name": "r/Aptos",
            "url": "https://www.reddit.com/r/Aptos/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
    "SUI": [
        {
            "name": "r/Sui",
            "url": "https://www.reddit.com/r/Sui/.rss",
            "source_type": "community",
            "weight": 0.55,
        },
    ],
}

SENTIMENT_SOURCE_WEIGHTS = {
    "news": 1.0,
    "community": 0.6,
    "aggregator": 0.9,
    "market_mood": 0.7,
}

ALTERNATIVE_ME_FNG_URL = "https://api.alternative.me/fng/"

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
ML_FORECAST_HORIZON = 7          # bars ahead; overridden by trading mode
ML_TRAIN_WINDOW = 120            # days for training
ML_FEATURES_LAGS = [1, 2, 3, 5, 7, 14, 21]  # lagged return periods

# ── Trading Modes ────────────────────────────────────────────────────────────
TRADING_MODES = {
    "swing": {
        "label": "Swing",
        "description": "Daily-bar positioning for 20-60 day trades.",
        "holding_period_label": "20-60 days",
        "yfinance_period": "2y",
        "backtest_period": "5y",
        "yfinance_interval": "1d",
        "forecast_horizon_bars": 30,
        "classification_threshold": 0.03,
        "max_holding_bars": 60,
        "min_holding_bars": 5,
        "warmup_bars": 120,
        "signal_stride_bars": 1,
        "bars_per_day": 1,
        "minimum_ml_score": 57,
        "minimum_technical_score": 54,
        "exit_ml_score": 44,
        "exit_technical_score": 45,
        "stop_loss_atr_scale": 1.2,
        "target_atr_scale": 1.4,
        "fee_bps": 10,
        "slippage_bps": 5,
    },
    "day": {
        "label": "Day",
        "description": "Hourly-bar setups designed for 1-3 day holds.",
        "holding_period_label": "1-3 days",
        "yfinance_period": "60d",
        "backtest_period": "730d",
        "yfinance_interval": "1h",
        "forecast_horizon_bars": 24,
        "classification_threshold": 0.012,
        "max_holding_bars": 72,
        "min_holding_bars": 4,
        "warmup_bars": 120,
        "signal_stride_bars": 2,
        "bars_per_day": 24,
        "minimum_ml_score": 55,
        "minimum_technical_score": 52,
        "exit_ml_score": 46,
        "exit_technical_score": 46,
        "stop_loss_atr_scale": 0.75,
        "target_atr_scale": 0.9,
        "fee_bps": 12,
        "slippage_bps": 6,
    },
    "scalp": {
        "label": "Scalp",
        "description": "15-minute intraday tactics for sub-day trading.",
        "holding_period_label": "Less than 1 day",
        "yfinance_period": "30d",
        "backtest_period": "60d",
        "yfinance_interval": "15m",
        "forecast_horizon_bars": 16,
        "classification_threshold": 0.004,
        "max_holding_bars": 32,
        "min_holding_bars": 2,
        "warmup_bars": 160,
        "signal_stride_bars": 4,
        "bars_per_day": 96,
        "minimum_ml_score": 53,
        "minimum_technical_score": 51,
        "exit_ml_score": 47,
        "exit_technical_score": 47,
        "stop_loss_atr_scale": 0.45,
        "target_atr_scale": 0.6,
        "fee_bps": 14,
        "slippage_bps": 8,
    },
}

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
