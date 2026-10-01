"""Trend indicators: SMAs, EMAs, ADX/DI, slope."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import core


def compute(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    out: dict[str, list] = {}
    for period in (20, 50, 100, 200):
        out[f"sma{period}"] = core.sma(closes, period)
    for period in (9, 20, 50, 200):
        out[f"ema{period}"] = core.ema(closes, period)

    adx, di_plus, di_minus = core.adx_di(highs, lows, closes, period=params.get("adx_period", 14))
    out["adx"] = adx
    out["di_plus"] = di_plus
    out["di_minus"] = di_minus
    out["slope20"] = core.slope(closes, 20)
    return out
