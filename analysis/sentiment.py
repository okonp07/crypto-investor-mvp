"""
Sentiment Analysis Module

Analyses crypto text inputs using VADER and TextBlob, then weights them by
source quality so professional editorial sources count more than broad
community chatter.
"""
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import SENTIMENT_SOURCE_WEIGHTS
from utils.helpers import get_logger

log = get_logger(__name__)
_vader = SentimentIntensityAnalyzer()


def _analyse_single(text: str) -> dict:
    """Run VADER + TextBlob on a single text string and blend the outputs."""
    vader_scores = _vader.polarity_scores(text)
    blob = TextBlob(text)
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
    """Analyse a list of sentiment inputs for one asset."""
    if not news_items:
        log.warning("No news items for sentiment analysis")
        return {
            "score": 50.0,
            "trend": "stable",
            "blended_mean": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "article_count": 0,
            "weighted_article_count": 0.0,
            "top_positive": [],
            "top_negative": [],
            "recent_vs_older": 0.0,
            "source_breakdown": {},
        }

    results = []
    for item in news_items:
        text = f"{item.get('title', '')}. {item.get('summary', '')}".strip()
        scores = _analyse_single(text)
        source_type = item.get("source_type", "news")
        type_weight = float(SENTIMENT_SOURCE_WEIGHTS.get(source_type, 1.0))
        item_weight = float(item.get("source_weight", 1.0))
        total_weight = type_weight * item_weight

        scores.update(
            {
                "title": item.get("title", ""),
                "source": item.get("source", "Unknown"),
                "source_type": source_type,
                "published": item.get("published"),
                "weight": total_weight,
            }
        )
        results.append(scores)

    weights = np.array([max(r["weight"], 0.05) for r in results], dtype=float)
    blended_values = np.array([r["blended"] for r in results], dtype=float)
    mean_blend = np.average(blended_values, weights=weights)

    pos_weight = float(sum(r["weight"] for r in results if r["blended"] > 0.05))
    neg_weight = float(sum(r["weight"] for r in results if r["blended"] < -0.05))
    total_weight = float(weights.sum())
    neu_weight = max(total_weight - pos_weight - neg_weight, 0.0)

    score = float(np.clip((mean_blend + 1) / 2 * 100, 0, 100))

    mid = len(results) // 2
    shift = 0.0
    if mid > 0:
        recent = results[:mid]
        older = results[mid:]
        recent_mean = np.average(
            [r["blended"] for r in recent],
            weights=[max(r["weight"], 0.05) for r in recent],
        )
        older_mean = np.average(
            [r["blended"] for r in older],
            weights=[max(r["weight"], 0.05) for r in older],
        )
        shift = float(recent_mean - older_mean)

    if shift > 0.1:
        trend = "improving"
    elif shift < -0.1:
        trend = "declining"
    else:
        trend = "stable"

    sorted_pos = sorted(results, key=lambda row: row["blended"], reverse=True)
    sorted_neg = sorted(results, key=lambda row: row["blended"])

    top_positive = [
        {
            "title": row["title"],
            "score": round(float(row["blended"]), 3),
            "source": row["source"],
        }
        for row in sorted_pos[:4]
        if row["blended"] > 0.05
    ]
    top_negative = [
        {
            "title": row["title"],
            "score": round(float(row["blended"]), 3),
            "source": row["source"],
        }
        for row in sorted_neg[:4]
        if row["blended"] < -0.05
    ]

    source_breakdown = {}
    for row in results:
        source_type = row["source_type"]
        entry = source_breakdown.setdefault(
            source_type,
            {"count": 0, "weight": 0.0},
        )
        entry["count"] += 1
        entry["weight"] += float(row["weight"])

    return {
        "score": round(score, 2),
        "trend": trend,
        "blended_mean": round(float(mean_blend), 4),
        "positive_pct": round(pos_weight / total_weight * 100, 1),
        "negative_pct": round(neg_weight / total_weight * 100, 1),
        "neutral_pct": round(neu_weight / total_weight * 100, 1),
        "article_count": len(results),
        "weighted_article_count": round(total_weight, 2),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "recent_vs_older": round(shift, 4),
        "source_breakdown": source_breakdown,
    }
