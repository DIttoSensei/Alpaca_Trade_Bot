"""Volatility indicators: ATR, ATR%, Bollinger Bands, stddev, percentile."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import core


def compute(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    out: dict[str, list] = {}
    atr_period = params.get("atr_period", 14)
    atr_series = core.atr(highs, lows, closes, period=atr_period)
    out["atr"] = atr_series
    out["atr_pct"] = [
        (a / c) if (a is not None and c) else None for a, c in zip(atr_series, closes)
    ]

    upper, mid, lower = core.bollinger(closes, period=params.get("bb_period", 20))
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["bb_width"] = [
        ((u - l) / m) if (u is not None and l is not None and m) else None
        for u, l, m in zip(upper, lower, mid)
    ]

    out["stddev20"] = core.rolling_std(closes, 20)
    out["vol_percentile"] = core.percentile_rank(out["atr_pct"], lookback=params.get("vol_lookback", 100))
    return out
