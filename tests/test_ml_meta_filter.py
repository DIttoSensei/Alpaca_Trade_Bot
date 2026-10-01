"""Tests for the veto-only meta-filter wrapper and its fail-open guarantees."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.enums import Regime, Timeframe
from app.core.types import FeatureSnapshot
from app.ml.meta_filter import MetaFilterModel

pytest.importorskip("numpy")


class _FakeModel:
    """A minimal classifier stub so these tests do not require scikit-learn."""

    classes_ = [0, 1]

    def __init__(self, p_win: float) -> None:
        self._p = p_win

    def predict_proba(self, X):  # noqa: N803 - sklearn signature
        return [[1.0 - self._p, self._p] for _ in X]


class _BrokenModel:
    classes_ = [0, 1]

    def predict_proba(self, X):  # noqa: N803
        raise RuntimeError("boom")


def _snap() -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="BTC/USD",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.M15,
        values={"close": 100.0, "rsi": 55.0},
    )


def test_disabled_filter_never_vetoes():
    model = MetaFilterModel(model=None)
    assert model.enabled is False
    decision = model.evaluate(_snap(), Regime.RANGE)
    assert decision.veto is False


def test_high_probability_is_allowed():
    model = MetaFilterModel(model=_FakeModel(0.9), threshold=0.5)
    assert model.vetoes(_snap(), Regime.RANGE) is False


def test_low_probability_is_vetoed():
    model = MetaFilterModel(model=_FakeModel(0.2), threshold=0.5)
    decision = model.evaluate(_snap(), Regime.RANGE)
    assert decision.veto is True
    assert decision.probability == pytest.approx(0.2)


def test_threshold_is_respected():
    model = MetaFilterModel(model=_FakeModel(0.6), threshold=0.65)
    assert model.vetoes(_snap(), Regime.RANGE) is True


def test_unscoreable_candidate_fails_open():
    """A model error must NEVER block a trade (fail-open)."""
    model = MetaFilterModel(model=_BrokenModel(), threshold=0.99)
    decision = model.evaluate(_snap(), Regime.RANGE)
    assert decision.veto is False


def test_save_and_load_roundtrip(tmp_path):
    pytest.importorskip("joblib")
    model = MetaFilterModel(model=_FakeModel(0.7), threshold=0.4, model_name="unit")
    path = model.save(tmp_path / "meta_filter.joblib")
    assert path.exists()
    assert (tmp_path / "meta_filter.joblib.meta.json").exists()

    loaded = MetaFilterModel.load(path)
    assert loaded.enabled is True
    assert loaded.threshold == pytest.approx(0.4)
    assert loaded.model_name == "unit"


def test_load_missing_file_is_disabled(tmp_path):
    loaded = MetaFilterModel.load(tmp_path / "nope.joblib")
    assert loaded.enabled is False
