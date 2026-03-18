# PR: Long And Short Trade Predictions

## Document Status

- Status: Implemented and pushed
- Product: TalentPoint
- Surface: Streamlit application and historical backtesting engine
- Scope: Bi-directional trade setup generation, ranking, UI presentation, and backtest execution

## Summary

This update expands TalentPoint from a long-only decision-support flow into a bi-directional trading assistant that can identify both `long` and `short` opportunities.

The change does not add exchange execution. Instead, it upgrades the app's analysis, ranking, trade-level generation, and historical backtesting logic so bearish setups can be surfaced and evaluated alongside bullish ones.

## Problem Statement

Before this update, the product was structurally biased toward bullish trades:

- the app primarily framed outputs as long opportunities
- ranked picks could only be interpreted naturally as buy-side ideas
- trade levels were designed around long entries, long stops, and long targets
- the backtester did not behave like a fully bi-directional strategy module
- UI language such as "allocation" implied only long exposure

That meant the system underused bearish signals and could not represent two-sided market conditions cleanly.

## Goals

- support both `long` and `short` trade setup generation
- keep trade direction consistent across scoring, rankings, trade levels, and backtests
- present bearish setups clearly in the frontend without forcing long-only language
- ensure historical backtests can open, manage, and close short positions
- preserve the existing app structure rather than rewriting the product

## Non-Goals

- direct broker or exchange execution
- margin, borrow-fee, funding-rate, or liquidation-engine simulation
- portfolio optimization across mixed long/short baskets
- point-in-time historical sentiment replay

## User Stories

- As a user, I want the app to tell me when the best current idea is a short, not only when it is a long.
- As a user, I want entry, stop-loss, and exit levels to reflect the actual trade side.
- As a user, I want ranked assets to prioritize actionable long and short setups instead of only bullish names.
- As a user, I want backtests to evaluate bearish trades on real historical price data.
- As a user, I want the UI language to make sense whether the idea is bullish or bearish.

## Functional Requirements

### Trade Setup Logic

- derive a single normalized trade setup for each asset with:
  - `direction`
  - `bias`
  - `opportunity_score`
  - `long_edge`
  - `short_edge`
- classify each asset as `long`, `short`, or `neutral`
- rank actionable long/short setups ahead of neutral ones

### Trade Levels

- generate side-aware trade levels
- for `long` setups:
  - compute pullback-style entries
  - compute upside targets
  - compute downside stops
- for `short` setups:
  - compute rebound-style entries
  - compute downside targets
  - compute upside stops

### Frontend

- show `Long Candidate` and `Short Candidate` states
- display trade side in detailed panels and top-pick cards
- present an opportunity score alongside the recommended side
- replace long-only sidebar framing with direction-aware positioning language

### Backtesting

- support short entries and exits in the historical engine
- mark trade direction in the backtest trade log
- value open short positions correctly
- exit short positions via stop, target, time limit, or signal deterioration

## Technical Changes Implemented

- added `derive_trade_setup(...)` in `scoring/engine.py`
- updated ranking logic to prioritize actionable setup direction plus opportunity strength
- updated reasoning generation so selected assets are described as `long`, `short`, or mixed
- upgraded `compute_levels(...)` in `strategy/entry_exit.py` to support side-aware trade construction
- added consistent `trade_setup` passing so UI and level generation do not disagree on direction
- upgraded `backtesting/engine.py` to support bi-directional historical position handling
- updated the Streamlit UI in `app.py` to render long/short recommendations, banners, metrics, and table columns
- replaced the sidebar's long-only "Suggested Allocation" framing with direction-aware "Suggested Positioning"

## Files Changed

- `app.py`
- `backtesting/engine.py`
- `scoring/engine.py`
- `strategy/entry_exit.py`

## Product Impact

After this update, TalentPoint can:

- rank bearish opportunities as top trade setups
- show short-biased trade plans with side-correct entries, targets, and stops
- present top picks as trade setups rather than implied buy-only ideas
- run historical backtests that include both long and short positions

This makes the application better aligned with real crypto market conditions, where strong opportunities can exist in either direction.

## Validation

The update was validated with:

- syntax compilation:
  - `python3 -m py_compile app.py backtesting/engine.py scoring/engine.py strategy/entry_exit.py`
- targeted smoke testing of:
  - one bullish sample setup
  - one bearish sample setup
- ranking validation to confirm actionable `long` and `short` setups sort ahead of neutral ones

Observed smoke-test outcome:

- bullish sample produced a valid `long` setup with long-style trade levels
- bearish sample produced a valid `short` setup with short-style trade levels

## Risks And Limitations

- short backtests still use a simplified cash/equity model rather than a full margin account simulation
- borrow fees, perpetual funding, liquidation mechanics, and venue-specific execution constraints are not modeled
- historical sentiment remains excluded from backtests because reliable archival sentiment data is difficult to source consistently
- live third-party data quality can still affect both signals and backtest realism

## Deployment Notes

- Repository branch: `main`
- Implementation commit: `7ab8bc8`
- Commit message: `Add long and short trade predictions`

## Suggested Next Steps

- add explicit side filtering in the UI so users can view only long or only short setups
- extend backtesting to model short-side financing and venue frictions
- add portfolio-level analytics for mixed long/short books
- document the directional scoring model in the main README
