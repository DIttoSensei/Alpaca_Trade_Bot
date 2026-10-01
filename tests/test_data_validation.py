"""Tests for candle validation, dedupe and aggregation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.enums import Timeframe
from app.core.types import Candle
from app.data.candle_store import aggregate
from app.data.data_validator import DataIssue, dedupe, validate_candle, validate_series


def _c(ts: datetime, o=100.0, h=101.0, low=99.0, c=100.5, v=10.0) -> Candle:
    return Candle(timestamp=ts, symbol="BTC/USD", timeframe=Timeframe.M15,
                  open=o, high=h, low=low, close=c, volume=v)


def test_valid_candle_passes():
    res = validate_candle(_c(datetime(2024, 1, 1, tzinfo=timezone.utc)), Timeframe.M15)
    assert res.ok is True
    assert res.has_fatal() is False


def test_non_positive_price_is_fatal():
    res = validate_candle(_c(datetime(2024, 1, 1, tzinfo=timezone.utc), o=0.0, h=1.0, low=0.0, c=0.0),
                          Timeframe.M15)
    assert DataIssue.NON_POSITIVE_PRICE in res.issues
    assert res.has_fatal() is True
    assert res.ok is False


def test_negative_volume_is_fatal_but_zero_volume_is_not():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    neg = validate_candle(_c(ts, v=-5.0), Timeframe.M15)
    assert DataIssue.NEGATIVE_VOLUME in neg.issues and neg.ok is False
    zero = validate_candle(_c(ts, v=0.0), Timeframe.M15)
    assert DataIssue.ZERO_VOLUME in zero.issues and zero.ok is True


def test_inconsistent_ohlc_is_fatal():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    res = validate_candle(_c(ts, o=100, h=99, low=98, c=100), Timeframe.M15)
    assert DataIssue.OHLC_INCONSISTENT in res.issues
    assert res.ok is False


def test_series_detects_duplicates_and_gaps():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [_c(base), _c(base), _c(base + timedelta(hours=5))]
    res = validate_series(bars, Timeframe.M15, max_gap_multiplier=3.0)
    assert DataIssue.DUPLICATE_TIMESTAMP in res.issues
    assert DataIssue.GAP in res.issues


def test_large_move_is_flagged_but_not_fatal():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [_c(base, c=100.0), _c(base + timedelta(minutes=15), o=100, h=180, low=100, c=170.0)]
    res = validate_series(bars, Timeframe.M15, spike_threshold=0.35)
    assert DataIssue.ABNORMAL_SPIKE in res.issues
    # A genuine crash/spike must NOT be deleted as bad data.
    assert DataIssue.ABNORMAL_SPIKE not in __import__("app.data.data_validator", fromlist=["FATAL_ISSUES"]).FATAL_ISSUES


def test_dedupe_keeps_latest_and_sorts():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    a = _c(base, c=1.0)
    b = _c(base + timedelta(minutes=15), c=2.0)
    a2 = _c(base, c=9.0)  # fresher duplicate
    out = dedupe([b, a, a2])
    assert len(out) == 2
    assert out[0].close == 9.0 and out[1].close == 2.0


def test_aggregate_m15_to_h1():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [
        _c(base, o=100, h=101, low=99, c=100, v=1),
        _c(base + timedelta(minutes=15), o=100, h=102, low=98, c=101, v=2),
        _c(base + timedelta(minutes=30), o=101, h=103, low=100, c=102, v=3),
        _c(base + timedelta(minutes=45), o=102, h=104, low=101, c=103, v=4),
    ]
    out = aggregate(bars, Timeframe.H1)
    assert len(out) == 1
    h1 = out[0]
    assert h1.open == 100 and h1.close == 103
    assert h1.high == 104 and h1.low == 98
    assert h1.volume == 10
