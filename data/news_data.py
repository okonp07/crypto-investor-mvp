"""
News / text data collection for sentiment analysis.
Sources (in priority order):
1. RSS feeds from major crypto outlets (no key needed)
2. CryptoPanic API (optional, free tier)
"""
import re
import time
import socket
from datetime import datetime, timedelta
import requests
import feedparser
from bs4 import BeautifulSoup
from config import ASSET_UNIVERSE, NEWS_RSS_FEEDS, CRYPTOPANIC_API_KEY, SENTIMENT_LOOKBACK_DAYS
from utils.helpers import get_logger

log = get_logger(__name__)
_RSS_CACHE: dict = {"articles": [], "fetched_at": None, "loaded": False}


def _clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _query_terms_for_symbol(symbol: str) -> set[str]:
    """Expand a ticker into a small keyword set for feed matching."""
    terms = {symbol.lower()}
    for coin_id, meta in ASSET_UNIVERSE.items():
        if meta["symbol"].lower() != symbol.lower():
            continue
        terms.add(coin_id.lower())
        terms.add(coin_id.replace("-", " ").lower())
        if "-" in coin_id:
            terms.add(coin_id.split("-")[0].lower())
        break
    return {term for term in terms if term}


def _fetch_rss_feed_entries(force_refresh: bool = False) -> list[dict]:
    """
    Fetch RSS feeds once per process and reuse them across asset analyses.
    This removes dozens of repeated network calls during a single app run.
    """
    if _RSS_CACHE["loaded"] and not force_refresh:
        return _RSS_CACHE["articles"]

    cutoff = datetime.utcnow() - timedelta(days=SENTIMENT_LOOKBACK_DAYS)
    entries = []

    for feed_url in NEWS_RSS_FEEDS:
        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(15)
            try:
                feed = feedparser.parse(
                    feed_url,
                    request_headers={"User-Agent": "CryptoInvestorMVP/1.0"},
                )
            finally:
                socket.setdefaulttimeout(old_timeout)

            source = feed.feed.get("title", feed_url)
            for entry in feed.entries[:60]:
                title = entry.get("title", "")
                summary = _clean_html(entry.get("summary", entry.get("description", "")))

                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6])

                if published and published < cutoff:
                    continue

                entries.append({
                    "title": title,
                    "summary": summary[:500],
                    "source": source,
                    "published": published,
                    "url": entry.get("link", ""),
                })
        except Exception as e:
            log.warning("RSS feed %s failed: %s", feed_url, e)

    _RSS_CACHE["articles"] = entries
    _RSS_CACHE["fetched_at"] = datetime.utcnow()
    _RSS_CACHE["loaded"] = True
    log.info("RSS cache refreshed with %d articles", len(entries))
    return entries


# ── RSS Feeds ─────────────────────────────────────────────────────────────────

def fetch_rss_news(symbol: str, max_items: int = 30) -> list[dict]:
    """
    Fetch recent articles from RSS feeds and filter by asset symbol/name.
    Returns list of {"title", "summary", "source", "published", "url"}.
    """
    results = []
    query_terms = _query_terms_for_symbol(symbol)
    for article in _fetch_rss_feed_entries():
        combined = f"{article['title']} {article.get('summary', '')}".lower()
        if not any(term in combined for term in query_terms):
            continue
        results.append(article)

    log.info("RSS: found %d articles for %s", len(results), symbol)
    return results


# ── CryptoPanic (optional) ───────────────────────────────────────────────────

def fetch_cryptopanic_news(symbol: str, max_items: int = 20) -> list[dict]:
    """
    Fetch news from CryptoPanic free API.  Requires CRYPTOPANIC_API_KEY in .env.
    Returns same format as fetch_rss_news.
    """
    if not CRYPTOPANIC_API_KEY:
        return []

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": symbol,
        "kind": "news",
        "filter": "important",
        "public": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("CryptoPanic failed for %s: %s", symbol, e)
        return []

    results = []
    for item in data.get("results", [])[:max_items]:
        published = None
        if item.get("published_at"):
            try:
                published = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass
        results.append({
            "title": item.get("title", ""),
            "summary": "",
            "source": item.get("source", {}).get("title", "CryptoPanic"),
            "published": published,
            "url": item.get("url", ""),
        })

    log.info("CryptoPanic: found %d articles for %s", len(results), symbol)
    return results


# ── Unified entry point ──────────────────────────────────────────────────────

def fetch_news(symbol: str, max_items: int = 30) -> list[dict]:
    """
    Collect news from all available sources, deduplicate, and sort by date.
    """
    articles = fetch_rss_news(symbol, max_items)
    articles.extend(fetch_cryptopanic_news(symbol, max_items))

    # Deduplicate by title similarity
    seen_titles = set()
    unique = []
    for a in articles:
        key = a["title"].lower().strip()[:80]
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    # Sort newest first
    unique.sort(key=lambda x: x.get("published") or datetime.min, reverse=True)
    return unique[:max_items]
