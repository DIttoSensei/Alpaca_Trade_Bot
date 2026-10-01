"""Trend regime classification via multi-feature scoring (spec sections 10-11).

Do NOT rely on a single indicator. Score several features and classify. All
thresholds are configurable and are NOT assumed optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.strategy_config import RegimeConfig
from app.core.enums import TrendRegime


@dataclass(slots=True)
class TrendScore:
    score: int
    regime: TrendRegime
    contributions: dict[str, int] = field(default_factory=dict)


def score_trend(features: dict[str, float], cfg: RegimeConfig) -> TrendScore:
    """Compute the trend score from a feature snapshot.

    +2 price > SMA200
    +1 EMA20 > EMA50
    +1 EMA50 > EMA200
    +1 ADX > threshold
    +1 positive 4h return (or slope proxy)
    """
    c: dict[str, int] = {}
    close = features.get("close")
    if close is None:
        # fall back: use last close embedded via ema? require close feature.
        close = features.get("_close")

    def cmp(key_a: str, key_b: str) -> bool | None:
        a = features.get(key_a)
        b = features.get(key_b)
        if a is None or b is None:
            return None
        return a > b

    price_above_sma = None
    if close is not None and features.get("sma200") is not None:
        price_above_sma = close > features["sma200"]
    if price_above_sma:
        c["price>sma200"] = 2
    elif price_above_sma is False:
        c["price>sma200"] = -2

    ema20_50 = cmp("ema20", "ema50")
    if ema20_50:
        c["ema20>ema50"] = 1
    elif ema20_50 is False:
        c["ema20>ema50"] = -1

    ema50_200 = cmp("ema50", "ema200")
    if ema50_200:
        c["ema50>ema200"] = 1
    elif ema50_200 is False:
        c["ema50>ema200"] = -1

    # ADX strength (signed by DI)
    adx = features.get("adx")
    if adx is not None and adx > cfg.adx_trend_threshold:
        di_plus = features.get("di_plus")
        di_minus = features.get("di_minus")
        direction = 1
        if di_plus is not None and di_minus is not None:
            direction = 1 if di_plus >= di_minus else -1
        c["adx"] = direction

    # return / slope
    ret = features.get("return_6")
    if ret is None:
        ret = features.get("slope20")
    if ret is not None:
        c["return"] = 1 if ret > 0 else -1

    total = sum(c.values())
    regime = classify(total, cfg)
    return TrendScore(score=total, regime=regime, contributions=c)


def classify(score: int, cfg: RegimeConfig) -> TrendRegime:
    if score >= cfg.strong_up_threshold:
        return TrendRegime.STRONG_UP
    if score >= cfg.weak_up_threshold:
        return TrendRegime.WEAK_UP
    if score <= cfg.strong_down_threshold:
        return TrendRegime.STRONG_DOWN
    if score <= cfg.weak_down_threshold:
        return TrendRegime.WEAK_DOWN
    return TrendRegime.RANGE
