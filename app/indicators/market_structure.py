"""Market structure: recent/swing highs & lows, range, breakout distance."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import core


def compute(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    n = len(candles)

    out: dict[str, list] = {}
    recent = params.get("recent_period", 20)
    swing = params.get("swing_period", 10)

    recent_high = core.rolling_max(highs, recent)
    recent_low = core.rolling_min(lows, recent)
    out["recent_high"] = recent_high
    out["recent_low"] = recent_low

    range_width: list = [None] * n
    for i in range(n):
        if recent_high[i] is not None and recent_low[i] is not None and recent_low[i] > 0:
            range_width[i] = (recent_high[i] - recent_low[i]) / recent_low[i]
    out["range_width"] = range_width

    # distance from recent high (negative-ish when below)
    breakout_distance: list = [None] * n
    for i in range(n):
        if recent_high[i] and recent_high[i] > 0:
            breakout_distance[i] = (closes[i] - recent_high[i]) / recent_high[i]
    out["breakout_distance"] = breakout_distance

    out["swing_high"] = core.rolling_max(highs, swing)
    out["swing_low"] = core.rolling_min(lows, swing)

    # range efficiency: net move / sum of absolute moves over window
    window = params.get("efficiency_period", 20)
    eff: list = [None] * n
    for i in range(window, n):
        net = abs(closes[i] - closes[i - window])
        path = sum(abs(closes[j] - closes[j - 1]) for j in range(i - window + 1, i + 1))
        eff[i] = (net / path) if path > 0 else 0.0
    out["range_efficiency"] = eff

    # VWAP over the window
    vwap: list = [None] * n
    for i in range(window - 1, n):
        pv = 0.0
        vv = 0.0
        for j in range(i - window + 1, i + 1):
            typical = (highs[j] + lows[j] + closes[j]) / 3.0
            pv += typical * candles[j].volume
            vv += candles[j].volume
        vwap[i] = (pv / vv) if vv > 0 else closes[i]
    out["vwap"] = vwap
    return out
