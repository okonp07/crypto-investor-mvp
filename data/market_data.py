"""
Data ingestion layer.
- CoinPaprika: market overview, coin details
- Yahoo Finance (via yfinance): OHLCV historical data
"""
import time
import copy
import functools
import requests
import pandas as pd
import yfinance as yf
from config import (
    ASSET_UNIVERSE, COINPAPRIKA_BASE, COINPAPRIKA_RATE_LIMIT,
    OHLC_HISTORY_DAYS, GITHUB_TOKEN,
)
from utils.helpers import get_logger, retry

log = get_logger(__name__)


# ── CoinPaprika helpers ───────────────────────────────────────────────────────

def _cp_get(endpoint: str, params: dict | None = None) -> dict | list:
    """Rate-limited GET against CoinPaprika with retry on 429 / 5xx."""
    url = f"{COINPAPRIKA_BASE}{endpoint}"
    time.sleep(COINPAPRIKA_RATE_LIMIT)
    for attempt in range(3):
        resp = requests.get(url, params=params or {}, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = COINPAPRIKA_RATE_LIMIT * (attempt + 2)
            log.warning("CoinPaprika %s retry, waiting %.1fs...", resp.status_code, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


@retry(max_attempts=3, delay=2.0)
def _fetch_market_overview_cached() -> pd.DataFrame:
    """
    Fetch current market snapshot for every asset in the universe.
    Returns one row per asset with the same columns the scoring layer expects.
    """
    rows = []
    for coin_id, meta in ASSET_UNIVERSE.items():
        paprika_id = meta["paprika"]
        ticker = _cp_get(f"/tickers/{paprika_id}")
        usd = ticker.get("quotes", {}).get("USD", {})
        rows.append({
            "id": coin_id,
            "symbol": meta["symbol"],
            "name": ticker.get("name", coin_id),
            "current_price": usd.get("price", 0.0),
            "market_cap": usd.get("market_cap", 0.0),
            "market_cap_rank": ticker.get("rank"),
            "total_volume": usd.get("volume_24h", 0.0),
            "circulating_supply": ticker.get("circulating_supply") or ticker.get("total_supply"),
            "max_supply": ticker.get("max_supply"),
            "price_change_percentage_24h": usd.get("percent_change_24h", 0.0),
            "price_change_percentage_7d_in_currency": usd.get("percent_change_7d", 0.0),
            "price_change_percentage_30d_in_currency": usd.get("percent_change_30d", 0.0),
        })

    df = pd.DataFrame(rows)
    log.info("Fetched market overview for %d assets", len(df))
    return df


@functools.lru_cache(maxsize=1)
def _market_overview_cache() -> pd.DataFrame:
    """Process-local cache to avoid refetching overview data on Streamlit reruns."""
    return _fetch_market_overview_cached()


def fetch_market_overview() -> pd.DataFrame:
    """Return a copy of the cached market overview snapshot."""
    return _market_overview_cache().copy(deep=True)


@retry(max_attempts=1, delay=2.0)
def _fetch_coin_details_cached(coin_id: str) -> dict:
    """Fetch extended coin data from CoinPaprika."""
    paprika_id = ASSET_UNIVERSE[coin_id]["paprika"]
    return _cp_get(f"/coins/{paprika_id}")


@functools.lru_cache(maxsize=64)
def _coin_details_cache(coin_id: str) -> dict:
    """Process-local cache for per-asset metadata."""
    return _fetch_coin_details_cached(coin_id)


def fetch_coin_details(coin_id: str) -> dict:
    """Return a copy of cached per-asset details."""
    return copy.deepcopy(_coin_details_cache(coin_id))


# ── Yahoo Finance OHLCV ──────────────────────────────────────────────────────

@retry(max_attempts=2, delay=1.0)
def fetch_ohlcv(
    yf_ticker: str,
    days: int = OHLC_HISTORY_DAYS,
    interval: str = "1d",
    period: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance for the requested interval.
    Returns DataFrame with columns: Open, High, Low, Close, Volume (DatetimeIndex).
    """
    if period is None and start is None and end is None:
        period_map = {d: p for d, p in [(30, "1mo"), (90, "3mo"), (180, "6mo"), (365, "1y"), (730, "2y")]}
        period = "6mo"
        for threshold, label in sorted(period_map.items()):
            if days <= threshold:
                period = label
                break

    ticker = yf.Ticker(yf_ticker)
    history_kwargs = {"interval": interval}
    if start is not None or end is not None:
        history_kwargs["start"] = start
        history_kwargs["end"] = end
    else:
        history_kwargs["period"] = period

    df = ticker.history(**history_kwargs)
    if df.empty:
        label = f"{start} -> {end}" if start is not None or end is not None else period
        log.warning("No OHLCV data for %s (%s, %s)", yf_ticker, label, interval)
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    label = f"{start} -> {end}" if start is not None or end is not None else period
    log.info("Fetched %d OHLCV bars for %s (%s, %s)", len(df), yf_ticker, label, interval)
    return df


# ── GitHub developer activity proxy ──────────────────────────────────────────

def fetch_github_activity(repo_slug: str) -> dict:
    """
    Fetch commit activity and repo stats as a developer-activity proxy.
    Returns dict with stars, forks, open_issues, recent_commits.
    """
    if not repo_slug:
        return {"stars": 0, "forks": 0, "open_issues": 0, "recent_commits": 0}

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    base = f"https://api.github.com/repos/{repo_slug}"
    result = {"stars": 0, "forks": 0, "open_issues": 0, "recent_commits": 0}

    try:
        repo = requests.get(base, headers=headers, timeout=10).json()
        result["stars"] = repo.get("stargazers_count", 0)
        result["forks"] = repo.get("forks_count", 0)
        result["open_issues"] = repo.get("open_issues_count", 0)

        # Commit activity (last 52 weeks)
        activity = requests.get(f"{base}/stats/commit_activity",
                                headers=headers, timeout=10).json()
        if isinstance(activity, list) and len(activity) >= 4:
            result["recent_commits"] = sum(w.get("total", 0) for w in activity[-4:])
    except Exception as e:
        log.warning("GitHub fetch failed for %s: %s", repo_slug, e)

    return result


# ── Convenience: fetch everything for one asset ──────────────────────────────

def fetch_all_asset_data(coin_id: str) -> dict:
    """
    Fetch OHLCV + market row + coin details + GitHub for a single asset.
    Returns a dict with keys: ohlcv, market, details, github.
    """
    meta = ASSET_UNIVERSE[coin_id]
    ohlcv = fetch_ohlcv(meta["yf"])
    details = {}
    github = {}

    try:
        details = fetch_coin_details(coin_id)
    except Exception as e:
        log.warning("CoinPaprika details failed for %s: %s", coin_id, e)

    try:
        github = fetch_github_activity(meta.get("github"))
    except Exception as e:
        log.warning("GitHub fetch failed for %s: %s", coin_id, e)

    return {"ohlcv": ohlcv, "details": details, "github": github}
