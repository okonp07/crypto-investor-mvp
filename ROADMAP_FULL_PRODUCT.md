# TalentPoint Product Roadmap

## Goal

Turn the capstone-style crypto strategy project into a production-oriented decision platform with:

- walk-forward backtesting
- three trading modes: swing, day, scalp
- richer free sentiment coverage
- a frontend that exposes rankings, trade setups, and historical validation

## Current Foundation

The app already has:

- live market ingestion
- technical analysis
- fundamentals
- sentiment scoring
- ML forecasting
- a Streamlit frontend

The first production expansion slice now adds:

- explicit trading-mode configuration in `config.py`
- richer sentiment ingestion in `data/news_data.py`
- weighted sentiment scoring in `analysis/sentiment.py`
- mode-aware ML horizons in `analysis/ml_forecast.py`
- mode-aware trade level logic in `strategy/entry_exit.py`
- a walk-forward backtester in `backtesting/engine.py`
- frontend controls for trading mode plus a `Backtests` tab in `app.py`

## Product Architecture

### 1. Research Layer

- Historical OHLCV by mode-specific interval
- Technical features per mode
- Optional macro overlays
- Point-in-time sentiment archive

### 2. Signal Layer

- Technical score
- Fundamental score
- Sentiment score
- ML score
- Final score engine

### 3. Strategy Layer

- Entry / stop / target logic
- Position sizing and leverage logic
- Mode-specific holding constraints
- Walk-forward simulation engine

### 4. Product Layer

- Ranked market dashboard
- Asset detail views
- Backtest inspection
- Transparency reports
- Deployment, logging, and caching

## Recommended Next Implementation Phases

### Phase 1. Historical Data Correctness

- Persist OHLCV snapshots locally for reproducible backtests
- Build a point-in-time sentiment store instead of scoring only live articles
- Add transaction-cost assumptions by venue and mode
- Add benchmark strategies: buy-and-hold, RSI/MACD baseline, naive momentum baseline

### Phase 2. Model Separation By Mode

- Train separate feature sets for swing, day, and scalp
- Tune thresholds and horizons per mode
- Add mode-specific target labels instead of one generic label scheme
- Evaluate per-asset and per-mode performance independently

### Phase 3. Better Backtesting

- Add rolling retraining / walk-forward validation windows
- Add equity curve, trade distribution, monthly returns, drawdown charts
- Add benchmark comparison and regime segmentation
- Add portfolio-level backtesting instead of asset-by-asset only

### Phase 4. Sentiment Pipeline Upgrade

- Store article and community-post history with timestamps
- Add deduplication at content level, not title only
- Add source reliability scores and asset-entity recognition
- Add richer community sources where terms and access rules allow

### Phase 5. Productionization

- Split frontend, service layer, and research jobs
- Add scheduled refresh jobs and cached feature stores
- Add API endpoints for assets, rankings, and backtests
- Add observability: logs, health checks, error monitoring
- Add user auth and saved watchlists if needed

## Suggested Tech Direction

- Frontend now: Streamlit is acceptable for MVP validation
- Backend next: FastAPI for APIs and background jobs
- Storage: Postgres for assets/runs, object storage or parquet for historical bars/articles
- Scheduling: cron, GitHub Actions, or Prefect depending on deployment preference
- Model tracking: MLflow or a lightweight experiment table

## Important Constraint

Backtests are only as trustworthy as the data replay.

The current backtester is a strong first foundation, but true institutional-grade validation will require:

- point-in-time sentiment history
- point-in-time fundamentals
- realistic fees/slippage
- exchange-specific execution assumptions
