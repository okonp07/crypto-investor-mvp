"""
Data ingestion layer.
- CoinGecko: market overview, coin details
- Yahoo Finance (via yfinance): OHLCV historical data
"""
import time
import requests
import pandas as pd
import yfinance as yf
from config import (
    ASSET_UNIVERSE, COINGECKO_BASE, COINGECKO_RATE_LIMIT,
    OHLC_HISTORY_DAYS, GITHUB_TOKEN,
)
from utils.helpers import get_logger, retry

log = get_logger(__name__)


# ── CoinGecko helpers ─────────────────────────────────────────────────────────

def _cg_get(endpoint: str, params: dict | None = None) -> dict | list:
    """Rate-limited GET against CoinGecko free API."""
    url = f"{COINGECKO_BASE}{endpoint}"
    time.sleep(COINGECKO_RATE_LIMIT)
    resp = requests.get(url, params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


@retry(max_attempts=3, delay=2.0)
def fetch_market_overview() -> pd.DataFrame:
    """
    Fetch current market snapshot for every asset in the universe.
    Returns one row per asset with price, market cap, volume, supply, etc.
    """
    ids = ",".join(ASSET_UNIVERSE.keys())
    data = _cg_get("/coins/markets", {
        "vs_currency": "usd",
        "ids": ids,
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d,30d",
    })
    df = pd.DataFrame(data)
    log.info("Fetched market overview for %d assets", len(df))
    return df


@retry(max_attempts=2, delay=3.0)
def fetch_coin_details(coin_id: str) -> dict:
    """Fetch extended coin data (developer stats, community, etc.)."""
    data = _cg_get(f"/coins/{coin_id}", {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "true",
        "developer_data": "true",
    })
    return data


# ── Yahoo Finance OHLCV ──────────────────────────────────────────────────────

@retry(max_attempts=2, delay=1.0)
def fetch_ohlcv(yf_ticker: str, days: int = OHLC_HISTORY_DAYS) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance.
    Returns DataFrame with columns: Open, High, Low, Close, Volume (DatetimeIndex).
    """
    period_map = {d: p for d, p in [(30, "1mo"), (90, "3mo"), (180, "6mo"), (365, "1y")]}
    period = "6mo"
    for threshold, label in sorted(period_map.items()):
        if days <= threshold:
            period = label
            break

    ticker = yf.Ticker(yf_ticker)
    df = ticker.history(period=period, interval="1d")
    if df.empty:
        log.warning("No OHLCV data for %s", yf_ticker)
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    log.info("Fetched %d OHLCV bars for %s", len(df), yf_ticker)
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
        log.warning("CoinGecko details failed for %s: %s", coin_id, e)

    try:
        github = fetch_github_activity(meta.get("github"))
    except Exception as e:
        log.warning("GitHub fetch failed for %s: %s", coin_id, e)

    return {"ohlcv": ohlcv, "details": details, "github": github}
