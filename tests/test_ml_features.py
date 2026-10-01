"""Tests for the ML feature vector (stability, sanitisation, no train/serve skew)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.enums import Regime, Timeframe
from app.core.types import FeatureSnapshot
from app.ml.features import (
    ML_FEATURES,
    REGIME_ORDER,
    build_matrix,
    build_vector,
    feature_names,
)


def _snap(values: dict) -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="BTC/USD",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.M15,
        values=values,
    )


def test_vector_length_is_stable_and_matches_names():
    names = feature_names(include_regime=True)
    vec = build_vector(_snap({"close": 100.0, "rsi": 55.0}), Regime.RANGE)
    assert len(vec) == len(names) == len(ML_FEATURES) + len(REGIME_ORDER)


def test_missing_and_non_finite_values_are_sanitised():
    vec = build_vector(_snap({"close": float("nan"), "rsi": float("inf")}), Regime.RANGE)
    # Missing, NaN and inf must all collapse to 0.0 — never poison the model.
    assert all(v == 0.0 or v == 1.0 for v in vec)
    assert vec[ML_FEATURES.index("close")] == 0.0
    assert vec[ML_FEATURES.index("rsi")] == 0.0


def test_regime_one_hot_is_exactly_one_hot():
    vec = build_vector(_snap({}), Regime.PANIC)
    block = vec[len(ML_FEATURES):]
    assert sum(block) == 1.0
    assert block[REGIME_ORDER.index(Regime.PANIC)] == 1.0


def test_none_snapshot_yields_zero_features():
    vec = build_vector(None, Regime.UNKNOWN)
    assert vec[: len(ML_FEATURES)] == [0.0] * len(ML_FEATURES)


def test_matrix_matches_row_count():
    snaps = [_snap({"close": 1.0}), None, _snap({"rsi": 60.0})]
    regimes = [Regime.RANGE, Regime.PANIC, Regime.STRONG_UPTREND]
    matrix = build_matrix(snaps, regimes)
    assert len(matrix) == 3
    assert all(len(row) == len(feature_names()) for row in matrix)
