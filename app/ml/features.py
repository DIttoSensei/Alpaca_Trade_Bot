"""Feature vector assembly for the ML meta-filter.

The meta-filter is a *veto* layer: it learns from the exact ``FeatureSnapshot``
that the rule-based stack produced at entry time, so there is no train/serve
skew — the very same numbers seen live are the numbers the model scored. Because
snapshots are stored verbatim with every trade, the training set can be rebuilt
from history without re-simulating (spec section 85).

The vector is a FIXED, ordered set of names plus a regime one-hot block, so the
dimensionality is stable across symbols and time. Missing values default to 0.0
and non-finite values are sanitised.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from app.core.enums import Regime
from app.core.types import FeatureSnapshot

# A curated subset of the feature store's vector. Kept explicit (not "all keys")
# so the model's input contract never silently changes when new indicators land.
ML_FEATURES: list[str] = [
    "close",
    "adx",
    "di_plus",
    "di_minus",
    "slope20",
    "rsi",
    "macd_hist",
    "roc",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "atr_pct",
    "bb_width",
    "stddev20",
    "vol_percentile",
    "volume_ratio",
    "obv_slope",
    "volume_momentum",
    "zscore",
    "dist_from_ema20",
    "dist_from_bb_mid",
    "return_deviation",
    "range_width",
    "breakout_distance",
    "range_efficiency",
]

# Stable regime ordering for the one-hot block.
REGIME_ORDER: list[Regime] = [
    Regime.STRONG_UPTREND,
    Regime.WEAK_UPTREND,
    Regime.RANGE,
    Regime.WEAK_DOWNTREND,
    Regime.STRONG_DOWNTREND,
    Regime.HIGH_VOLATILITY,
    Regime.LOW_VOLATILITY,
    Regime.PANIC,
    Regime.RECOVERY,
    Regime.UNKNOWN,
]


def feature_names(include_regime: bool = True) -> list[str]:
    names = list(ML_FEATURES)
    if include_regime:
        names += [f"regime_{r}" for r in REGIME_ORDER]
    return names


def _finite(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def build_vector(
    snapshot: Optional[FeatureSnapshot],
    regime: Optional[Regime],
    include_regime: bool = True,
) -> list[float]:
    """Return the fixed-length numeric vector for one observation."""
    values = snapshot.values if snapshot is not None else {}
    vector = [_finite(values.get(name, 0.0)) for name in ML_FEATURES]
    if include_regime:
        vector += [1.0 if regime == r else 0.0 for r in REGIME_ORDER]
    return vector


def build_matrix(
    snapshots: Sequence[Optional[FeatureSnapshot]],
    regimes: Sequence[Optional[Regime]],
    include_regime: bool = True,
) -> list[list[float]]:
    return [build_vector(s, r, include_regime) for s, r in zip(snapshots, regimes)]
