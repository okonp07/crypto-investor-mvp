# Crypto Investor MVP

A quant-assisted crypto investment decision-support tool that combines technical analysis, fundamental scoring, NLP sentiment analysis, and ML time-series forecasting to identify and rank promising crypto assets.

**This is NOT financial advice. Use at your own risk.**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure API keys for enhanced data
cp .env.example .env
# Edit .env with your CryptoPanic / GitHub tokens

# 3. Run the app
streamlit run app.py
```

## Architecture

```
crypto_investor/
├── app.py                    # Streamlit frontend
├── config.py                 # All configuration & tunable parameters
├── data/
│   ├── market_data.py        # CoinGecko + Yahoo Finance data ingestion
│   └── news_data.py          # RSS feeds + CryptoPanic news collection
├── analysis/
│   ├── technical.py          # 11 technical indicators → weighted score
│   ├── fundamental.py        # 8 fundamental factors → weighted score
│   ├── sentiment.py          # VADER + TextBlob NLP → sentiment score
│   └── ml_forecast.py        # XGBoost direction + Exp. Smoothing forecast
├── scoring/
│   └── engine.py             # Combined ranking, reasoning, market regime
├── strategy/
│   ├── entry_exit.py         # Entry / exit / stop-loss price computation
│   └── risk.py               # Risk tolerance → leverage mapping
├── utils/
│   └── helpers.py            # Logging, retry, normalisation utilities
├── requirements.txt
├── .env.example
└── README.md
```

## How Picks Are Generated

### 1. Data Collection
- **OHLCV prices** from Yahoo Finance (180 days daily)
- **Market metadata** from CoinGecko (market cap, volume, supply)
- **News articles** from RSS feeds (CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine)
- **Developer activity** from GitHub API (stars, commits, forks)

### 2. Four Analysis Pillars

| Pillar | Method | Score Range |
|--------|--------|-------------|
| **Technical** | RSI, MACD, EMA/SMA cross, Bollinger, ADX, Stochastic, OBV, ATR, momentum, support/resistance | 0-100 |
| **Fundamental** | Market cap rank, volume/mcap ratio, supply dynamics, price momentum, relative strength vs BTC, developer activity, liquidity | 0-100 |
| **Sentiment** | VADER + TextBlob on recent news headlines, trend detection, positive/negative classification | 0-100 |
| **ML Forecast** | XGBoost direction classifier (bullish/bearish/neutral) + Holt-Winters exponential smoothing price forecast | 0-100 |

### 3. Combined Scoring
```
Final Score = 0.30 * Technical + 0.20 * Fundamental + 0.20 * Sentiment + 0.30 * ML
```
Weights are adjusted by risk tolerance (e.g., conservative boosts fundamentals, aggressive boosts sentiment/ML).

### 4. Entry / Exit / Stop-Loss
- **Entry**: Pullback toward support and pivot, depth varies by risk level
- **Exit**: Blend of resistance level, ATR-based target, and ML forecast
- **Stop-loss**: ATR-multiple below entry, floored at support

### 5. Leverage
- Conservative: 1x (no leverage)
- Moderate: 2-3x
- Aggressive: up to 5x
- Adjusted down for high volatility and low model confidence

## Asset Universe

15 major crypto assets: BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, LINK, MATIC, LTC, UNI, NEAR, APT, SUI

## Configuration

All parameters are in `config.py`:
- Asset universe
- Technical indicator periods
- Scoring weights
- Risk profiles (leverage, stop-loss, target multipliers)
- API endpoints and rate limits

## Limitations

- **Not real-time**: Runs on-demand, not streaming
- **Sentiment coverage**: RSS feeds may not capture all relevant news; CryptoPanic key improves coverage
- **ML model**: XGBoost with simple features; no deep learning (intentional for MVP transparency)
- **No backtesting**: Forward-looking only (backtesting module is a future extension)
- **API rate limits**: CoinGecko free tier limits ~10-30 calls/min; full analysis takes 2-5 minutes
- **No order execution**: Analysis only, no trading integration

## Future Improvements

- Backtesting engine with historical signal evaluation
- Portfolio allocation optimisation (Markowitz / risk parity)
- Signal history tracking and performance monitoring
- FinBERT or domain-specific transformer for sentiment
- LSTM / Temporal Fusion Transformer for forecasting
- WebSocket streaming for real-time updates
- Cron-scheduled periodic refresh
- Downloadable PDF report
- On-chain data integration (Dune Analytics, Glassnode free tier)

## Tech Stack

All open-source:
- **Frontend**: Streamlit + Plotly
- **Data**: yfinance, CoinGecko API, feedparser
- **Technical Analysis**: ta (Python)
- **NLP**: VADER, TextBlob
- **ML**: XGBoost, scikit-learn, statsmodels
- **Language**: Python 3.10+
