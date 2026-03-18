"""
News / text data collection for sentiment analysis.

Source families:
1. Editorial RSS feeds from professional crypto publishers
2. Reddit community feeds for market/community tone
3. CryptoPanic (optional free API key)
4. Alternative.me Fear & Greed index for broad market mood
"""
import re
import socket
from datetime import datetime, timedelta

import feedparser
import requests
from bs4 import BeautifulSoup

from config import (
    ALTERNATIVE_ME_FNG_URL,
    ASSET_UNIVERSE,
    CRYPTOPANIC_API_KEY,
    NEWS_RSS_FEEDS,
    REDDIT_COMMUNITY_FEEDS,
    SENTIMENT_LOOKBACK_DAYS,
)
from utils.helpers import get_logger

log = get_logger(__name__)
_FEED_CACHE: dict[str, dict] = {}


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


def _parse_published(entry) -> datetime | None:
    """Convert parsed feed timestamps into naive UTC datetimes."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6])
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime(*entry.updated_parsed[:6])
    return None


def _fetch_feed_entries(feed_config: dict, force_refresh: bool = False) -> list[dict]:
    """Fetch one RSS feed and cache the parsed entries across reruns."""
    url = feed_config["url"]
    cached = _FEED_CACHE.get(url)
    if cached and not force_refresh:
        return cached["articles"]

    cutoff = datetime.utcnow() - timedelta(days=SENTIMENT_LOOKBACK_DAYS)
    entries = []

    try:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            feed = feedparser.parse(
                url,
                request_headers={"User-Agent": "TalentPoint/1.0"},
            )
        finally:
            socket.setdefaulttimeout(old_timeout)

        source = feed_config.get("name") or feed.feed.get("title", url)
        for entry in feed.entries[:80]:
            published = _parse_published(entry)
            if published and published < cutoff:
                continue

            entries.append(
                {
                    "title": entry.get("title", ""),
                    "summary": _clean_html(entry.get("summary", entry.get("description", "")))[:500],
                    "source": source,
                    "published": published,
                    "url": entry.get("link", ""),
                    "source_type": feed_config.get("source_type", "news"),
                    "source_weight": float(feed_config.get("weight", 1.0)),
                }
            )
    except Exception as exc:
        log.warning("RSS feed %s failed: %s", url, exc)

    _FEED_CACHE[url] = {"articles": entries, "fetched_at": datetime.utcnow()}
    log.info("Feed %s cached with %d articles", url, len(entries))
    return entries


def fetch_rss_news(symbol: str, max_items: int = 30) -> list[dict]:
    """Fetch editorial crypto news and filter it to the asset's query terms."""
    query_terms = _query_terms_for_symbol(symbol)
    results = []

    for feed in NEWS_RSS_FEEDS:
        for article in _fetch_feed_entries(feed):
            combined = f"{article['title']} {article.get('summary', '')}".lower()
            if any(term in combined for term in query_terms):
                results.append(article)

    log.info("Editorial RSS: found %d articles for %s", len(results), symbol)
    return results[:max_items]


def fetch_reddit_news(symbol: str, max_items: int = 20) -> list[dict]:
    """Fetch community sentiment via Reddit RSS for broad and asset-specific channels."""
    query_terms = _query_terms_for_symbol(symbol)
    feed_configs = list(REDDIT_COMMUNITY_FEEDS.get("*", []))
    feed_configs.extend(REDDIT_COMMUNITY_FEEDS.get(symbol.upper(), []))

    results = []
    for feed in feed_configs:
        for article in _fetch_feed_entries(feed):
            combined = f"{article['title']} {article.get('summary', '')}".lower()
            if feed in REDDIT_COMMUNITY_FEEDS.get(symbol.upper(), []):
                results.append(article)
                continue
            if any(term in combined for term in query_terms):
                results.append(article)

    log.info("Community feeds: found %d posts for %s", len(results), symbol)
    return results[:max_items]


def fetch_cryptopanic_news(symbol: str, max_items: int = 20) -> list[dict]:
    """Fetch optional CryptoPanic coverage using its free API tier."""
    if not CRYPTOPANIC_API_KEY:
        return []

    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": symbol,
        "kind": "news",
        "filter": "important",
        "public": "true",
    }

    try:
        resp = requests.get("https://cryptopanic.com/api/v1/posts/", params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("CryptoPanic failed for %s: %s", symbol, exc)
        return []

    results = []
    for item in payload.get("results", [])[:max_items]:
        published = None
        if item.get("published_at"):
            try:
                published = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except ValueError:
                published = None
        results.append(
            {
                "title": item.get("title", ""),
                "summary": "",
                "source": item.get("source", {}).get("title", "CryptoPanic"),
                "published": published,
                "url": item.get("url", ""),
                "source_type": "aggregator",
                "source_weight": 0.9,
            }
        )

    log.info("CryptoPanic: found %d articles for %s", len(results), symbol)
    return results


def fetch_market_mood_context() -> list[dict]:
    """Fetch a lightweight market-wide sentiment context from Fear & Greed."""
    try:
        resp = requests.get(ALTERNATIVE_ME_FNG_URL, params={"limit": 1, "format": "json"}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        row = payload.get("data", [{}])[0]
    except Exception as exc:
        log.warning("Alternative.me Fear & Greed fetch failed: %s", exc)
        return []

    score = row.get("value")
    classification = row.get("value_classification", "Unknown")
    timestamp = row.get("timestamp")
    published = datetime.utcnow()
    if timestamp:
        try:
            published = datetime.utcfromtimestamp(int(timestamp))
        except (TypeError, ValueError):
            published = datetime.utcnow()

    title = f"Crypto Fear & Greed Index: {classification} ({score})"
    summary = (
        "Market-wide sentiment proxy from Alternative.me. "
        "Useful as a broad crypto risk appetite overlay."
    )
    return [
        {
            "title": title,
            "summary": summary,
            "source": "Alternative.me",
            "published": published,
            "url": "https://alternative.me/crypto/fear-and-greed-index/",
            "source_type": "market_mood",
            "source_weight": 0.7,
        }
    ]


def fetch_news(symbol: str, max_items: int = 30) -> list[dict]:
    """Collect, deduplicate, and sort sentiment inputs across all free sources."""
    articles = []
    articles.extend(fetch_rss_news(symbol, max_items=max_items))
    articles.extend(fetch_reddit_news(symbol, max_items=max_items // 2))
    articles.extend(fetch_cryptopanic_news(symbol, max_items=max_items // 2))
    articles.extend(fetch_market_mood_context())

    seen_titles = set()
    unique = []
    for article in articles:
        title_key = article.get("title", "").lower().strip()[:120]
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        unique.append(article)

    unique.sort(key=lambda item: item.get("published") or datetime.min, reverse=True)
    return unique[:max_items]
