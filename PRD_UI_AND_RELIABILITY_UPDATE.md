# PRD: UI And Reliability Update

## Document Status

- Status: Implemented
- Product: Crypto Investor MVP
- Surface: Streamlit application
- Scope: Reliability, data-source migration, ML runtime health, and frontend usability

## Summary

This update improves the Crypto Investor MVP across three areas:

1. reliability of the live analysis pipeline
2. usability and aesthetics of the Streamlit frontend
3. operator confidence through visible health and status indicators

The result is a more production-ready dashboard that can be deployed on Streamlit Community Cloud and used as a real decision-support interface rather than a rough MVP prototype.

## Problem Statement

Before this update, the app had several product and operational issues:

- CoinGecko detail calls could trigger aggressive rate-limit failures mid-run
- several Yahoo Finance symbols no longer returned valid price history
- the XGBoost runtime could fail to load locally due to an OpenMP mismatch
- the frontend felt like a default Streamlit report instead of a polished dashboard
- rankings were harder to scan because the UI lacked dense visual summaries
- users had no explicit health indicator confirming whether the ML layer was active
- selected assets did not persist naturally across views

## Goals

- make the full 15-asset pipeline run more reliably
- preserve full ML functionality where possible and expose fallback/health clearly
- improve frontend aesthetics without changing the core analysis concept
- make the app easier to scan, compare, and act on
- support a cleaner deployment path to Streamlit Community Cloud

## Non-Goals

- building brokerage or exchange execution
- introducing portfolio backtesting
- redesigning the scoring methodology from scratch
- replacing Streamlit with another frontend framework

## User Stories

- As a user, I want the full asset universe to analyse without the app appearing to freeze midway through the run.
- As a user, I want a clear ranking table with visual score bars and sparklines so I can scan the market quickly.
- As a user, I want to click into an asset and keep that context while switching tabs.
- As a user, I want a watchlist so I can promote interesting assets during exploration.
- As a user, I want suggested entry, stop-loss, and take-profit levels surfaced prominently.
- As a user, I want to know whether the ML model is active or in fallback mode.

## Functional Requirements

### Reliability

- Replace CoinGecko market metadata dependency with CoinPaprika.
- Fix broken Yahoo Finance symbols in the configured asset universe.
- Add in-process caching for market overview and per-asset metadata calls.
- Add RSS caching so repeated sentiment runs do not refetch the same feeds on every asset.

### ML Health

- Preserve XGBoost forecasting when the runtime is available.
- Fall back gracefully when XGBoost cannot load.
- Expose model status in the analysis output.
- Display a pipeline health section in the frontend showing ML status coverage.

### Frontend

- Introduce a more polished visual theme and branded dashboard shell.
- Add `Overview`, `Picks`, and `Universe` tabs.
- Provide an interactive rankings table with:
  - progress-bar score columns
  - sparkline price previews
  - quick trend and sentiment labels
- Add a persistent selected-asset detail panel.
- Add one-click watchlist behavior.
- Add a suggested allocation panel in the sidebar for top picks.

### Risk And Output Presentation

- Keep recommended `entry price`, `stop-loss`, and `take-profit` as first-class outputs.
- Present risk/reward and leverage more prominently.
- Update the disclaimer language to use a more formal investment-risk and liability notice.

## UX Requirements

- The app should feel closer to a financial dashboard than a notebook output.
- Core market signals should be understandable within a few seconds of opening the app.
- The selected asset should persist across tabs in the same session.
- Watchlist actions should require one click.
- The health-check area should clearly communicate whether the ML layer is functioning.

## Technical Changes Implemented

- migrated fundamentals market data from CoinGecko to CoinPaprika
- added CoinPaprika IDs and corrected Yahoo tickers in config
- added process-local cache for market overview and per-coin details
- added process-local RSS cache
- repaired local libomp / XGBoost runtime mismatch
- added ML status reporting
- redesigned Streamlit layout and styling
- added tabbed navigation
- added interactive ranking table with sparklines
- added persistent selected asset state via Streamlit session state
- added watchlist and suggested allocation sidebar modules
- added pipeline health-check section

## Success Criteria

- full configured asset universe completes analysis successfully under normal upstream API conditions
- rankings table is visually scannable and interactive
- selected asset context persists across tabs
- ML health is visible to the user
- app is deployable on Streamlit Community Cloud

## Risks

- live APIs can still degrade or fail transiently
- Yahoo Finance and RSS sources are unofficial or semi-structured and may change over time
- Streamlit Community Cloud environment may differ from local runtime and should be smoke-tested after deploy

## Deployment Notes

- Main app entrypoint: `app.py`
- Live app target: Streamlit Community Cloud
- Live URL: [https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/](https://crypto-investor-mvp-gxv6u8tvd7btcjyr3fx26n.streamlit.app/)

