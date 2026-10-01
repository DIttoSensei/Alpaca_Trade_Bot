"""Momentum indicators: RSI, MACD, ROC, returns."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import core


def compute(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    closes = [c.close for c in candles]

    out: dict[str, list] = {}
    out["rsi"] = core.rsi(closes, period=params.get("rsi_period", 14))
    macd_line, signal_line, hist = core.macd(closes)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["roc"] = core.roc(closes, period=params.get("roc_period", 10))

    # simple momentum returns over a few horizons
    for h in (1, 3, 6, 12):
        seq: list = [None] * len(closes)
        for i in range(h, len(closes)):
            if closes[i - h] != 0:
                seq[i] = (closes[i] - closes[i - h]) / closes[i - h]
        out[f"return_{h}"] = seq
    return out
