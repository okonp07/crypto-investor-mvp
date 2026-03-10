"""
Shared utilities: logging, retry logic, normalisation helpers.
"""
import logging
import time
import functools
import numpy as np
from config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator: retry a function on exception with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    logging.getLogger(func.__module__).warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        func.__name__, attempt, max_attempts, e, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator


def normalise_score(value: float, low: float, high: float) -> float:
    """Linearly normalise *value* from [low, high] → [0, 100], clamped."""
    if high == low:
        return 50.0
    return float(np.clip((value - low) / (high - low) * 100, 0, 100))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns *default* when denominator is zero / NaN."""
    try:
        if b == 0 or np.isnan(b):
            return default
        return a / b
    except (TypeError, ValueError):
        return default
