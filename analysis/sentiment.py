"""
Sentiment Analysis Module
Analyses crypto news headlines/summaries using VADER and TextBlob.
Produces a sentiment score (0-100) with trend and narrative summary.
"""
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from utils.helpers import get_logger

log = get_logger(__name__)

_vader = SentimentIntensityAnalyzer()


def _analyse_single(text: str) -> dict:
    """Run VADER + TextBlob on a single text string. Returns blended scores."""
    vader_scores = _vader.polarity_scores(text)
    blob = TextBlob(text)

    # VADER compound: [-1, 1], TextBlob polarity: [-1, 1]
    # Blend: 60% VADER (better for social/informal), 40% TextBlob
    blended = vader_scores["compound"] * 0.6 + blob.sentiment.polarity * 0.4

    return {
        "vader_compound": vader_scores["compound"],
        "textblob_polarity": blob.sentiment.polarity,
        "blended": blended,
        "positive": vader_scores["pos"],
        "negative": vader_scores["neg"],
        "neutral": vader_scores["neu"],
    }


def analyse_sentiment(news_items: list[dict]) -> dict:
    """
    Analyse sentiment of a list of news items for one asset.

    Args:
        news_items: list of {"title", "summary", "published", "source", "url"}

    Returns:
        {
            "score": float (0-100),
            "trend": "improving" | "declining" | "stable",
            "blended_mean": float (-1 to 1),
            "positive_pct": float,
            "negative_pct": float,
            "neutral_pct": float,
            "article_count": int,
            "top_positive": [{"title", "score"}],
            "top_negative": [{"title", "score"}],
            "recent_vs_older": float,  # sentiment shift
        }
    """
    if not news_items:
        log.warning("No news items for sentiment analysis")
        return {
            "score": 50.0, "trend": "stable", "blended_mean": 0.0,
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0,
            "article_count": 0, "top_positive": [], "top_negative": [],
            "recent_vs_older": 0.0,
        }

    results = []
    for item in news_items:
        text = f"{item['title']}. {item.get('summary', '')}"
        scores = _analyse_single(text)
        scores["title"] = item["title"]
        scores["published"] = item.get("published")
        results.append(scores)

    blended_values = [r["blended"] for r in results]
    mean_blend = np.mean(blended_values)

    # Classify each article
    pos_count = sum(1 for v in blended_values if v > 0.05)
    neg_count = sum(1 for v in blended_values if v < -0.05)
    neu_count = len(blended_values) - pos_count - neg_count
    total = len(blended_values)

    # Score: map blended mean from [-1, 1] to [0, 100]
    score = float(np.clip((mean_blend + 1) / 2 * 100, 0, 100))

    # Sentiment trend: compare recent half vs older half
    mid = len(results) // 2
    if mid > 0:
        # results are sorted newest-first
        recent_mean = np.mean([r["blended"] for r in results[:mid]])
        older_mean = np.mean([r["blended"] for r in results[mid:]])
        shift = recent_mean - older_mean
    else:
        shift = 0.0

    if shift > 0.1:
        trend = "improving"
    elif shift < -0.1:
        trend = "declining"
    else:
        trend = "stable"

    # Top headlines
    sorted_pos = sorted(results, key=lambda r: r["blended"], reverse=True)
    sorted_neg = sorted(results, key=lambda r: r["blended"])

    top_positive = [{"title": r["title"], "score": round(r["blended"], 3)}
                    for r in sorted_pos[:3] if r["blended"] > 0.05]
    top_negative = [{"title": r["title"], "score": round(r["blended"], 3)}
                    for r in sorted_neg[:3] if r["blended"] < -0.05]

    return {
        "score": round(score, 2),
        "trend": trend,
        "blended_mean": round(float(mean_blend), 4),
        "positive_pct": round(pos_count / total * 100, 1),
        "negative_pct": round(neg_count / total * 100, 1),
        "neutral_pct": round(neu_count / total * 100, 1),
        "article_count": total,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "recent_vs_older": round(float(shift), 4),
    }
