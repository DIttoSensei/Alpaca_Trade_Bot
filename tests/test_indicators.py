"""Indicator correctness and feature-frame integration tests."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app.core.enums import Timeframe
from app.core.types import Candle
from app.indicators import compute_frame
from app.indicators.core import atr, ema, rsi, sma
from app.data.feature_store import CORE_FEATURES


def test_sma_values_and_warmup():
    out = sma([1, 2, 3, 4, 5], 3)
    assert out[0] is None and out[1] is None
    assert out[2] == 2 and out[3] == 3 and out[4] == 4


def test_ema_tracks_trend_and_length():
    # On a flat-then-jump series the EMA (seeded from the SMA) weights the most
    # recent bar more heavily, so it must sit above the arithmetic mean.
    series = [1, 2, 3, 4, 5, 5, 5, 10]
    out = ema(series, 3)
    assert len(out) == len(series)
    assert out[0] is None and out[1] is None  # warm-up window
    assert out[-1] is not None and out[-1] > sma(series, 3)[-1]


def test_rsi_bounds_and_extremes():
    rising = rsi(list(range(1, 40)), 14)
    assert all(v is None or 0.0 <= v <= 100.0 for v in rising)
    assert rising[-1] > 90  # monotonic rise -> saturated


def test_atr_requires_warmup_then_positive():
    highs = [10 + i for i in range(30)]
    lows = [9 + i for i in range(30)]
    closes = [9.5 + i for i in range(30)]
    out = atr(highs, lows, closes, 14)
    assert out[0] is None
    assert out[-1] is not None and out[-1] > 0


def _series(n: int) -> list[Candle]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    out = []
    price = 100.0
    for i in range(n):
        price *= 1 + 0.001 * math.sin(i / 5.0) + 0.0005
        high = price * 1.002
        low = price * 0.998
        out.append(Candle(timestamp=base + timedelta(minutes=15 * i), symbol="BTC/USD",
                          timeframe=Timeframe.M15, open=price, high=high, low=low,
                          close=price, volume=100 + i))
    return out


def test_compute_frame_produces_core_features():
    frame = compute_frame(_series(260))
    assert frame  # non-empty
    missing = [name for name in CORE_FEATURES if name not in frame]
    # Some features are optional; the bulk must be present for the ML vector.
    assert len(missing) <= 2, f"unexpected missing features: {missing}"
