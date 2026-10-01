"""Mean-reversion indicators: z-score, distance from EMA/BB mean."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import core


def compute(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    closes = [c.close for c in candles]

    out: dict[str, list] = {}
    out["zscore"] = core.zscore(closes, period=params.get("zscore_period", 20))

    ema20 = core.ema(closes, 20)
    out["dist_from_ema20"] = [
        ((c - e) / e) if (e and e > 0) else None for c, e in zip(closes, ema20)
    ]

    upper, mid, lower = core.bollinger(closes, period=20)
    out["dist_from_bb_mid"] = [
        ((c - m) / m) if (m and m > 0) else None for c, m in zip(closes, mid)
    ]

    # short-term return deviation: return_1 minus its 20-bar mean
    r1 = core.roc(closes, 1)
    r1_clean = [x if x is not None else 0.0 for x in r1]
    r1_mean = core.sma(r1_clean, 20)
    out["return_deviation"] = [
        (r - m) if (r is not None and m is not None) else None for r, m in zip(r1, r1_mean)
    ]
    return out
