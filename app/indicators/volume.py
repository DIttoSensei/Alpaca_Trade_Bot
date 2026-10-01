"""Volume indicators: volume SMA, ratio, OBV, relative volume."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import core


def compute(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    out: dict[str, list] = {}
    for period in (5, 20, 50):
        out[f"volume_sma{period}"] = core.sma(volumes, period)

    vol_sma20 = core.sma(volumes, 20)
    out["volume_ratio"] = [
        (v / s) if (s and s > 0) else None for v, s in zip(volumes, vol_sma20)
    ]

    obv_series = core.obv(closes, volumes)
    out["obv"] = obv_series
    out["obv_slope"] = core.slope(obv_series, 20)

    # volume momentum: current vs n bars ago
    n = params.get("volume_momentum_period", 10)
    vmom: list = [None] * len(volumes)
    for i in range(n, len(volumes)):
        base = vol_sma20[i - n] if vol_sma20[i - n] else None
        if base and base > 0:
            vmom[i] = (volumes[i] - base) / base
    out["volume_momentum"] = vmom
    return out
