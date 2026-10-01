"""Technical indicators (implemented in-house for portability).

All indicator functions are pure and lookahead-free. Use ``compute_frame`` to
calculate the full feature set for a candle sequence.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import Candle
from app.indicators import (
    market_structure,
    mean_reversion,
    momentum,
    trend,
    volatility,
    volume,
)

MODULES = {
    "trend": trend,
    "momentum": momentum,
    "volatility": volatility,
    "volume": volume,
    "mean_reversion": mean_reversion,
    "market_structure": market_structure,
}


def compute_frame(candles: Sequence[Candle], **params: Any) -> dict[str, list]:
    """Compute every indicator series for a candle sequence.

    Returns a dict of ``name -> list`` aligned to candle indices (None during
    warm-up). This is the single source of truth for feature values used by
    both live and backtest.
    """
    frame: dict[str, list] = {}
    for module in MODULES.values():
        frame.update(module.compute(candles, **params))
    return frame


__all__ = ["MODULES", "compute_frame"]
