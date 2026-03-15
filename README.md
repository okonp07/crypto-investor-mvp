# TalentPoint

## Live App

Use the live Streamlit app here:

[Open TalentPoint](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)

A Streamlit-based crypto research dashboard that ranks a curated asset universe using technical analysis, market-structure fundamentals, news sentiment, and machine-learning forecasts.

The app is designed as a decision-support tool, not an execution bot. It produces a ranked market view plus actionable trade-style levels such as suggested entry, stop-loss, take-profit, leverage guidance, and score breakdowns.

This is not financial advice. Use at your own risk.

## What The App Does

For each tracked asset, the app:

- pulls price history from Yahoo Finance
- pulls market metadata from CoinPaprika
- scans crypto news sources for sentiment
- measures developer activity from GitHub when available
- computes four pillar scores: technical, fundamental, sentiment, and ML
- combines those scores into a final ranking
- suggests entry price, stop-loss, take-profit, leverage, and risk/reward
- checkpoints progress after each completed asset so interrupted runs can resume
- shows partial results during the scan instead of waiting for all 15 assets to finish
- generates per-asset transparency reports explaining score construction, data sources, and trade-level logic

## Current Product State

The current frontend is a polished Streamlit dashboard with:

- a live deployment on Streamlit Community Cloud: [crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)
- a hero-style landing section and upgraded visual theme
- `Overview`, `Picks`, and `Universe` tabs to reduce scroll length
- a visible resume-status card that shows checkpoint progress, next asset, and save time
- a live runboard that surfaces rolling rankings and latest-completed asset details during the scan
- an interactive rankings table with progress bars, mini price-action sparklines, and direct report links for each asset
- persistent asset selection across tabs
- a one-click watchlist in the sidebar
- a score-weighted suggested allocation panel for the current top picks
- a health-check panel that shows whether the full pipeline and ML layer are active
- downloadable transparency reports for selected assets, top picks, and non-top-ranked assets opened from the rankings or universe views
- detailed top-pick cards with chart, sentiment, rationale, and risk controls

## Data Sources

- `Yahoo Finance` via `yfinance`: OHLCV history
- `CoinPaprika`: market metadata such as price, market cap, rank, supply, and volume
- `RSS feeds` + optional `CryptoPanic`: sentiment/news input
- `GitHub API`: developer activity proxy

## Scoring Model

The app scores each asset across four pillars:

| Pillar | What It Uses | Score Range |
| --- | --- | --- |
| Technical | RSI, MACD, EMA/SMA cross, Bollinger Bands, ADX, Stochastic, OBV, ATR, momentum, support/resistance | 0-100 |
| Fundamental | Market cap rank, volume/market-cap ratio, supply dynamics, price momentum, relative strength, developer activity, liquidity | 0-100 |
| Sentiment | VADER + TextBlob over recent crypto news | 0-100 |
| ML Forecast | XGBoost direction classifier + exponential smoothing forecast | 0-100 |

Default combined score:

```text
Final Score = 0.30 * Technical
            + 0.20 * Fundamental
            + 0.20 * Sentiment
            + 0.30 * ML
```

Risk profiles reweight those components and also affect leverage, stop-loss width, and target sizing.

## Trade-Level Outputs

For each strong candidate, the app surfaces:

- `Entry price`
- `Stop-loss`
- `Take-profit / exit price`
- `Risk/reward ratio`
- `Expected return`
- `Suggested leverage`

These are computed from support/resistance, ATR-style logic, forecast levels, confidence, and user-selected risk tolerance.

## Transparency Reports

TalentPoint includes a built-in transparency report for every analysed asset in a run.

Each report explains:

- the raw and weighted contribution of each score pillar
- the underlying technical and fundamental factor scores
- sentiment article counts and headline summaries
- ML direction, confidence, forecast range, and model status
- data-source provenance for the asset analysis
- why the suggested entry, stop-loss, and take-profit levels were chosen
- leverage rationale and risk/reward context

Reports can be viewed in-app and downloaded as markdown files for review or sharing.

You can also open a report directly from the rolling rankings table via the `Open report` link beside each asset, which deep-links into that asset's review panel even if the asset is not in the top 3.

## Run Experience

TalentPoint now supports a more resilient long-running scan flow:

- each asset is checkpointed after completion
- interrupted runs can resume from the next pending asset instead of restarting from asset 1
- the app shows partial results while the 15-asset loop is still in progress
- users can explicitly clear the checkpoint and start a fresh run
- the UI exposes checkpoint state before a run starts

This keeps the app useful even when live APIs are slow or a session is interrupted.

## Asset Universe

The default universe includes:

- BTC
- ETH
- SOL
- BNB
- XRP
- ADA
- AVAX
- DOT
- LINK
- MATIC
- LTC
- UNI
- NEAR
- APT
- SUI

## Project Structure

```text
talentpoint/
├── app.py                    # Streamlit UI and app flow
├── config.py                 # Asset universe, weights, risk profiles, API settings
├── data/
│   ├── market_data.py        # CoinPaprika + Yahoo Finance ingestion, caching
│   └── news_data.py          # RSS + CryptoPanic ingestion, dedupe, caching
├── analysis/
│   ├── technical.py          # Technical indicators and technical score
│   ├── fundamental.py        # Fundamental / market structure score
│   ├── sentiment.py          # News sentiment scoring
│   └── ml_forecast.py        # XGBoost direction + price forecast
├── scoring/
│   └── engine.py             # Final score combination, ranking, reasoning
├── strategy/
│   ├── entry_exit.py         # Entry, exit, stop-loss computation
│   └── risk.py               # Leverage and risk profile logic
├── utils/
│   └── helpers.py            # Logging, retries, normalization helpers
├── requirements.txt
├── ARTICLE.md
├── PRD_UI_AND_RELIABILITY_UPDATE.md
└── README.md
```

## Running Locally

Live app:

[https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional environment variables:

- `CRYPTOPANIC_API_KEY`
- `GITHUB_TOKEN`
- `COINPAPRIKA_RATE_LIMIT`
- `LOG_LEVEL`

## Streamlit Cloud Deployment

This project is suitable for Streamlit Community Cloud deployment.

Current deployed app:

[Open the deployed Streamlit app](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)

Typical setup:

1. Push the repo to GitHub.
2. In Streamlit Community Cloud, create a new app from this repo.
3. Set the main file path to `app.py`.
4. Add any needed secrets such as `CRYPTOPANIC_API_KEY` and `GITHUB_TOKEN`.
5. Deploy.

Notes:

- The app is designed to run from Streamlit, not as a static site.
- First load may take longer because it fetches live data.
- In-process caching helps repeated reruns within the same session.
- interrupted runs can resume from a saved checkpoint for the same risk profile

## Limitations

- Live external APIs can still slow down or fail transiently
- Sentiment coverage is limited to configured news sources unless expanded
- ML is intentionally lightweight and transparent, not institutional-grade forecasting
- There is no brokerage integration or automatic trade execution
- There is no historical backtesting module yet
- resume state is stored locally per app instance, so deployment environments should be tested for persistence expectations

## Recent Improvements

- migrated market metadata from CoinGecko to CoinPaprika
- fixed broken Yahoo Finance tickers for several assets
- added market-data and RSS caching to reduce repeated network cost
- repaired the local XGBoost runtime and graceful fallback path
- redesigned the Streamlit UI with tabs, sparklines, watchlist, and allocation guidance
- added resumable scan checkpoints and live partial-results rendering during long runs
- added a visible pipeline health-check section and resume-status card
- added per-asset transparency reports and downloadable explanation files
- added direct transparency-report links inside the rolling rankings table for all analysed assets
- deployed the app to Streamlit Community Cloud: [crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)

## Tech Stack

- `Streamlit`
- `Plotly`
- `pandas`
- `numpy`
- `yfinance`
- `CoinPaprika API`
- `feedparser`
- `BeautifulSoup`
- `VADER`
- `TextBlob`
- `XGBoost`
- `scikit-learn`
- `statsmodels`

## Quick Access

- Live app: [https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)
- PRD for the latest upgrade: [https://github.com/okonp07/crypto-investor-mvp/blob/main/PRD_UI_AND_RELIABILITY_UPDATE.md](https://github.com/okonp07/crypto-investor-mvp/blob/main/PRD_UI_AND_RELIABILITY_UPDATE.md)
