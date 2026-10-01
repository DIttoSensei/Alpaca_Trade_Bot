"""Tests for meta-filter training: chronological split, lift metrics, guards."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from app.core.enums import Regime, Timeframe
from app.core.types import FeatureSnapshot, TradeRecord
from app.ml.training import dataset_from_trades, train_meta_filter

pytest.importorskip("sklearn")


def _trade(i: int, good: bool, base: datetime) -> TradeRecord:
    vals = {
        "close": 100 + i * 0.1,
        "adx": 60.0 if good else 12.0,
        "rsi": 68.0 if good else 35.0,
        "atr_pct": 0.01 if good else 0.05,
        "volume_ratio": 1.5 if good else 0.4,
        "macd_hist": 0.02 if good else -0.02,
        "return_6": 0.03 if good else -0.02,
    }
    snap = FeatureSnapshot(
        symbol="BTC/USD",
        timestamp=base + timedelta(hours=i),
        timeframe=Timeframe.M15,
        values=vals,
    )
    pnl = random.uniform(5, 50) if good else -random.uniform(5, 40)
    return TradeRecord(
        trade_id=str(i),
        symbol="BTC/USD",
        strategy="s",
        entry_price=100.0,
        exit_price=100.0,
        qty=1.0,
        fees=0.0,
        slippage=0.0,
        gross_pnl=pnl,
        net_pnl=pnl,
        entry_time=base + timedelta(hours=i),
        exit_time=base + timedelta(hours=i + 1),
        holding_time_seconds=3600,
        exit_reason="TAKE_PROFIT",
        regime=Regime.STRONG_UPTREND if good else Regime.RANGE,
        signal_score=1.0,
        expected_edge=0.01,
        estimated_cost=0.0,
        actual_cost=0.0,
        feature_snapshot=snap,
    )


def _trades(n: int, seed: int = 0) -> list[TradeRecord]:
    random.seed(seed)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [_trade(i, good=random.random() < 0.5, base=base) for i in range(n)]


def test_dataset_drops_trades_without_snapshot_and_orders_chronologically():
    trades = _trades(10)
    trades[3].feature_snapshot = None
    ds = dataset_from_trades(trades)
    assert len(ds) == 9
    assert ds.exit_times == sorted(ds.exit_times)


def test_label_uses_net_pnl_not_gross():
    trades = _trades(6)
    # Force a losing net trade that is "gross positive" to prove costs matter.
    trades[0].gross_pnl = 10.0
    trades[0].net_pnl = -1.0
    ds = dataset_from_trades(trades)
    # The first chronological trade is trades[0] (same ordering by exit_time).
    assert ds.y[0] == 0


def test_training_reports_lift_and_chronological_split():
    trades = _trades(400, seed=1)
    report = train_meta_filter(trades, model_type="gradient_boosting", min_trades=50)
    assert report.train_size + report.test_size == 400
    assert report.train_size == int(400 * 0.7)
    metrics = report.metrics
    # The model should separate obviously-good from obviously-bad features.
    assert metrics.get("kept_win_rate") is not None
    assert metrics.get("win_rate_lift", 0) >= 0
    assert metrics.get("roc_auc", 0) > 0.6


def test_training_refuses_when_too_few_trades():
    with pytest.raises(RuntimeError):
        train_meta_filter(_trades(10), min_trades=50)


def test_training_refuses_single_class_training_split():
    # All-winning trades -> the training split has only one class.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trades = [_trade(i, good=True, base=base) for i in range(100)]
    for t in trades:
        t.net_pnl = abs(t.net_pnl)
    with pytest.raises(RuntimeError):
        train_meta_filter(trades, min_trades=50)
