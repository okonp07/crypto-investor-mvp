# Building a Quant-Assisted Crypto Investment Tool — From Data to Decision

*A transparent, multi-signal approach to cutting through crypto market noise*

---

Crypto markets are loud. Scroll through any trading forum and you'll find a firehose of contradictory signals — one analyst screaming "breakout," another calling for a crash, and a chorus of influencers shilling whatever they bought yesterday. The data is there — price action, on-chain metrics, news sentiment, volume patterns — but turning that data into a coherent investment thesis is genuinely hard. Emotions creep in. Recency bias takes over. And most retail tools are either black boxes or glorified chart viewers.

We asked a simple question: *what if we combined four distinct analytical lenses — technical, fundamental, sentiment, and machine learning — into a single transparent score?* Not a magic number that tells you what to buy, but an explainable framework that shows you *why* it's suggesting what it's suggesting.

So we built it. An open-source MVP that analyses 15 major cryptocurrencies, scores each one across four pillars, and returns its top 3 picks with concrete entry prices, exit targets, stop-losses, and leverage recommendations — all adjusted to your risk tolerance.

---

## What the Tool Does

At its core, the tool runs a full analytical pipeline across 15 liquid crypto assets: Bitcoin, Ethereum, Solana, BNB, XRP, Cardano, Avalanche, Polkadot, Chainlink, Polygon, Litecoin, Uniswap, NEAR, Aptos, and Sui.

For each asset, it computes scores across four pillars — technical analysis, fundamental/market structure, news sentiment, and a machine learning forecast — then blends them into a single composite score from 0 to 100. The top 3 assets by score become the tool's investment picks.

But a score alone isn't actionable. For each pick, the tool also calculates:

- **Entry price** — a specific pullback level toward support, so you're not chasing the current price
- **Exit price** — a take-profit target blending resistance levels, ATR projections, and the ML forecast
- **Stop-loss** — an ATR-based floor, anchored at support
- **Leverage recommendation** — dynamically adjusted for volatility and model confidence
- **Risk/reward ratio and expected return** — so you can size the trade appropriately

Every recommendation comes with a plain-English explanation of *why* it was chosen — which signals fired, what the sentiment looks like, where the ML model sees price heading. No black boxes.

> [Screenshot: The main dashboard showing the top 3 picks with score breakdowns, entry/exit levels, and market regime indicator]

---

## The Four Analytical Pillars

This is the engine room. Each pillar contributes a score from 0 to 100, and each approaches the market from a fundamentally different angle. The diversity is the point — technical momentum can confirm what sentiment is hinting at, while fundamentals can flag when hype has outrun reality.

### A. Technical Analysis — 11 Indicators, One Composite

The technical module runs 11 indicator signals across 180 days of daily OHLCV data, each scored independently and then weighted into a composite.

The indicators and their weights:

| Indicator | Weight | What It Tells You |
|-----------|--------|-------------------|
| MACD (12/26/9) | 14% | Momentum direction and acceleration |
| RSI (14-period) | 12% | Overbought/oversold conditions |
| EMA Cross (12/26) | 10% | Short-term trend direction |
| Bollinger Bands (20/2) | 10% | Volatility squeeze and mean reversion |
| ADX (14-period) | 10% | Trend strength (not direction) |
| SMA Cross (20/50) | 8% | Medium-term trend confirmation |
| OBV (20-period EMA) | 8% | Volume-confirmed accumulation or distribution |
| Stochastic (14/3) | 8% | Momentum within a range |
| Momentum (5d/10d) | 7% | Raw price velocity |
| Support/Resistance | 7% | Proximity to key price levels |
| ATR Trend | 6% | Volatility expansion or contraction |

Each indicator produces a score from 0 (strongly bearish) to 100 (strongly bullish). RSI below 30 scores bullish; above 70 scores bearish. A positive and rising MACD histogram scores high. Price sitting near support (calculated via pivot points) scores higher than price near resistance, since it implies a better entry.

The key insight is *weighting*. MACD and RSI carry the most weight because they're the most reliable momentum signals in crypto's noisy, trend-driven markets. ATR trend gets the least because volatility direction alone is ambiguous — it's more useful for position sizing than for direction.

### B. Fundamental / Market Structure — 8 Factors

Traditional fundamental analysis doesn't map cleanly to crypto — there are no earnings reports or P/E ratios. But there *are* structural signals that separate solid assets from fragile ones.

The fundamental module scores 8 factors:

| Factor | Weight | What It Captures |
|--------|--------|------------------|
| Market Cap Rank | 15% | Size and institutional credibility |
| Volume/Market Cap Ratio | 15% | Liquidity health (sweet spot: 2-15%) |
| Price Change Momentum | 15% | Blend of 7d (60%) and 30d (40%) returns |
| Relative Strength vs BTC | 15% | Is it outperforming the benchmark? |
| Developer Activity | 10% | GitHub stars, commits, forks — is anyone building? |
| Supply Dynamics | 10% | Circulating vs max supply — dilution risk |
| Volatility Quality | 10% | Lower relative volatility = higher score |
| Liquidity Proxy | 10% | Volume vs universe median |

Developer activity is worth calling out. The tool hits the GitHub API (with an optional token for higher rate limits) and computes a composite from stars, recent commits, and forks. It's a rough but real signal — projects with active development tend to have stronger long-term fundamentals than abandoned codebases with a token still trading.

Relative strength vs BTC is another important one. In crypto, BTC is the benchmark. An altcoin returning 5% in a week where BTC returned 8% is actually *underperforming*. This factor captures that relative dynamic.

### C. Sentiment Analysis — NLP on Crypto News

Markets move on narratives, and crypto markets especially so. The sentiment module scrapes recent headlines from four major outlets — CoinDesk, CoinTelegraph, Decrypt, and Bitcoin Magazine — via RSS feeds, plus an optional CryptoPanic API integration for broader coverage.

For each asset, it pulls up to 30 articles from the last 7 days, filters for relevance by matching the asset's symbol in titles and summaries, then runs each headline through two NLP engines:

- **VADER** (Valence Aware Dictionary and Sentiment Reasoner) — a lexicon-based analyser tuned for social media and news text, contributing 60% of the blend
- **TextBlob** — a simpler polarity scorer, contributing the remaining 40%

The blended score classifies each article as positive (>0.05), negative (<-0.05), or neutral. The module also detects *sentiment trend* by comparing the newer half of articles against the older half — if the shift exceeds 0.1, sentiment is "improving"; below -0.1, it's "declining."

Here's a concrete example. A headline like *"Ethereum TVL hits all-time high as institutional inflows surge"* would score positively on both VADER and TextBlob. Meanwhile, *"SEC files emergency motion to halt Ethereum ETF approvals"* would score negatively. The tool doesn't just count positive vs negative — it tracks the *direction* of sentiment, which is often more predictive than the level.

The final sentiment score maps the mean blended sentiment from [-1, 1] to [0, 100]. The output also surfaces the top 3 most positive and top 3 most negative headlines, so you can see exactly what's driving the score.

### D. Machine Learning Forecast — XGBoost + Holt-Winters

The ML module combines a classification model for direction with a statistical model for price levels.

**Direction forecast (XGBoost):** A gradient-boosted classifier trained on 120 days of data predicts whether the next 7 days will be bullish (>+2% return), bearish (<-2%), or neutral. Features include:

- Lagged returns at 1, 2, 3, 5, 7, 14, and 21 days
- Rolling volatility windows (7d, 14d, 21d)
- Volume ratio vs 20-day average
- RSI proxy and price-to-SMA ratios (20d, 50d)
- Day-of-week cyclical encoding

The model uses time-series cross-validation with 3 splits — no random shuffling, no future data leaking into training. Hyperparameters are conservative: 100 estimators, max depth 4, learning rate 0.05. This isn't trying to be a high-frequency alpha model; it's a directional signal.

**Price forecast (Holt-Winters):** Exponential smoothing with an additive trend produces a 7-day price trajectory with confidence bands scaled by daily volatility and the square root of the horizon. If the statistical model fails to converge (which happens with very noisy series), it falls back to EMA trend extrapolation.

**Combined ML score:** 60% from the XGBoost bullish probability, 40% from an expected return score. A +5 bonus applies when both models agree on direction; a -5 penalty when they disagree.

---

## How Scores Combine

Each pillar produces a score from 0 to 100. The default blending weights are:

```
Final Score = 0.30 × Technical + 0.20 × Fundamental + 0.20 × Sentiment + 0.30 × ML
```

Technical and ML get the heaviest weights because they're the most directly predictive of short-term price action. Fundamentals and sentiment provide context and confirmation.

But these weights shift with your risk profile. A **conservative** profile boosts fundamentals by 1.3x and dials down sentiment to 0.8x and ML to 0.9x — favouring stability over momentum. An **aggressive** profile does the opposite: it boosts sentiment by 1.3x and ML by 1.2x while reducing fundamentals to 0.7x — leaning into momentum signals. The weights are re-normalised to sum to 1.0 after adjustment.

The tool also determines an overall **market regime** — bullish (composite ≥ 58), bearish (≤ 42), or mixed — by averaging the technical, sentiment, and ML scores across all 15 assets. This provides a macro backdrop for interpreting individual scores.

---

## Entry, Exit & Risk Management

A score without a trade plan is just a number. The strategy module converts each pick into actionable levels.

**Entry price** is calculated as a pullback toward support. The depth varies by risk tolerance: conservative pulls back 50% of the distance from current price to support (patient entry), moderate pulls back 30%, and aggressive only 15% (near-market entry). The entry is blended 60/40 between the support-based pullback and the midpoint between current price and pivot, then bounded so it never exceeds 99.5% of the current price or drops below 98% of support.

**Exit price (take-profit)** blends three components with roughly equal weight: the calculated resistance level (35%), an ATR-based target (30%), and the ML price forecast (35%). The ATR multiple scales with risk appetite — 3.0x for conservative, 4.5x for moderate, 6.0x for aggressive. A hard floor ensures the exit is at least 2% above entry.

**Stop-loss** takes the higher of a support-based stop (support minus half the ATR proxy) and an ATR-based stop (entry minus 1.5x to 2.5x ATR, depending on risk level). It's capped at 3% below entry to prevent unreasonably tight stops in volatile conditions.

**Leverage** starts at a base level — 1x for conservative, 2.5x for moderate, 5x for aggressive — then gets dynamically reduced. If daily volatility exceeds 3%, leverage scales down proportionally (minimum factor: 0.3x). If model confidence is below 50%, it scales down further (minimum factor: 0.5x). The tool also calculates the **liquidation distance**: at 5x leverage, a 20% adverse move approaches liquidation; at 2.5x, it takes 40%. This math is displayed prominently so you understand the risk.

---

## Using the Tool — A Walkthrough

Getting started takes about two minutes:

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set up API keys for broader news coverage and GitHub data
cp .env.example .env
# Edit .env with your CryptoPanic API key and/or GitHub token

# Launch the app
streamlit run app.py
```

The app opens in your browser with a sidebar on the left. Select your risk tolerance — Conservative, Moderate, or Aggressive — and review the effective scoring weights displayed below the selector. Hit **Run Analysis**.

> [Screenshot: Sidebar showing risk tolerance selector with weight breakdown and the Run Analysis button]

The pipeline takes a couple of minutes as it fetches data from CoinPaprika, Yahoo Finance, RSS feeds, and optionally CryptoPanic and GitHub. A progress indicator tracks each asset.

When it finishes, the main panel shows the market regime (bullish/bearish/mixed), the analysis timestamp, and three cards for the top picks. Each card displays the current price, entry/exit/stop-loss levels, final score, leverage, risk/reward ratio, expected return, and ML direction with confidence.

> [Screenshot: Top pick card showing all metrics — score breakdown, entry/exit levels, leverage recommendation]

Expand any card to drill into the score breakdown (raw scores and weights for each pillar), the reasoning narrative, leverage rationale with liquidation warning, sentiment details with top headlines, and an interactive Plotly chart showing candlesticks, EMAs, Bollinger Bands, RSI, volume, and the 7-day forecast overlay.

> [Screenshot: Expanded chart view with candlestick, indicators, and forecast overlay]

Below the top picks, a full rankings table shows all 15 assets sorted by final score, with individual pillar scores and sentiment trend.

> [Screenshot: Full rankings table with all 15 assets]

---

## Under the Hood — Architecture

The codebase follows a clean pipeline architecture:

```
data/          → Fetch market data (CoinPaprika, Yahoo Finance) and news (RSS, CryptoPanic)
analysis/      → Score each pillar (technical, fundamental, sentiment, ML)
scoring/       → Blend pillar scores, determine market regime, rank assets
strategy/      → Calculate entry/exit/stop-loss, estimate leverage
app.py         → Streamlit frontend — orchestrates the pipeline and renders results
config.py      → Single source of truth for all weights, thresholds, and parameters
```

Everything is config-driven. Want to change the RSI period from 14 to 21? Edit one line in `config.py`. Want to add a 16th asset? Add its internal asset ID, CoinPaprika ID, Yahoo ticker, and symbol to the asset list. Want to swap XGBoost for a different classifier? The ML module has a clean interface — plug in your model and return a direction, probabilities, and confidence.

The stack is entirely open-source: **Streamlit** for the UI, **Plotly** for interactive charts, **yfinance** for price data, the **ta** library for technical indicators, **VADER** and **TextBlob** for sentiment, **XGBoost** for classification, and **statsmodels** for exponential smoothing. No paid APIs are required for core functionality — CryptoPanic and GitHub tokens are optional enhancements.

---

## Limitations & What's Next

We built this as an MVP, and it's important to be honest about its boundaries.

**Not real-time.** The tool fetches data on-demand. It's designed for daily or weekly analysis, not intraday trading. Even with CoinPaprika and caching, the full pipeline still takes a few minutes because it relies on multiple live APIs.

**Sentiment coverage is narrow.** Four RSS feeds and an optional API capture major headlines but miss crypto Twitter, Reddit, Discord, and on-chain social signals — which often move markets faster than news articles.

**The ML model is simple.** A 100-estimator XGBoost trained on 120 days of data is a useful directional signal, not a production-grade forecasting system. It doesn't incorporate alternative data, cross-asset correlations, or order book features.

**No backtesting.** The current version doesn't backtest its recommendations against historical data. You're trusting the framework's logic, not a verified track record.

**What's on the roadmap:** A backtesting engine to validate the scoring system against history. FinBERT or a crypto-tuned transformer to replace VADER/TextBlob. Portfolio-level optimisation (correlation-aware allocation, not just top-3 picking). On-chain data integration (whale flows, exchange balances, staking dynamics). And eventually, real-time streaming for intraday use.

---

## Closing

This tool isn't trying to predict the future. It's trying to make the present *legible* — to take the noise of 15 assets, dozens of indicators, hundreds of headlines, and a statistical forecast, and distil it into a transparent, explainable recommendation that you can interrogate and override.

The entire codebase is open source. Fork it, extend it, swap out the models, add your own data sources. If you disagree with the weights, change them — they're all in one config file.

The best investment tools aren't the ones that tell you what to do. They're the ones that help you think clearly about what you're doing and why.

---

> **Disclaimer:** This tool is for educational and research purposes only. It is not financial advice. Cryptocurrency trading involves substantial risk of loss, and leveraged trading can result in losses exceeding your initial investment. Always do your own research and consult a qualified financial advisor before making investment decisions. Past performance of any model or scoring system does not guarantee future results.

---

*Built with Python, Streamlit, and a healthy respect for how much we don't know about markets.*
