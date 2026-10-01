"""Volatility regime classification using ATR percentile bands."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.strategy_config import RegimeConfig
from app.core.enums import VolatilityRegime


@dataclass(slots=True)
class VolatilityScore:
    percentile: float
    regime: VolatilityRegime
    details: dict[str, float] = field(default_factory=dict)


def classify_volatility(features: dict[str, float], cfg: RegimeConfig) -> VolatilityScore:
    pct = features.get("vol_percentile")
    atr_pct = features.get("atr_pct", 0.0)
    if pct is None:
        return VolatilityScore(percentile=0.5, regime=VolatilityRegime.UNKNOWN, details={"atr_pct": atr_pct})

    if pct >= cfg.extreme_vol_percentile:
        regime = VolatilityRegime.EXTREME
    elif pct >= cfg.high_vol_percentile:
        regime = VolatilityRegime.HIGH
    elif pct <= cfg.low_vol_percentile:
        regime = VolatilityRegime.LOW
    else:
        regime = VolatilityRegime.NORMAL
    return VolatilityScore(percentile=pct, regime=regime, details={"atr_pct": atr_pct})
