"""Tests for the append-only experiment registry."""

from __future__ import annotations

from app.ml.registry import ExperimentRecord, ExperimentRegistry


def test_log_and_read_all(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    reg.log(ExperimentRecord(name="a", kind="backtest", metrics={"net_pnl": 1.0}))
    reg.log(ExperimentRecord(name="b", kind="backtest", metrics={"net_pnl": 3.0}))
    records = reg.read_all()
    assert len(records) == 2
    assert records[0]["name"] == "a"


def test_latest_and_find(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    reg.log(ExperimentRecord(name="a", kind="backtest"))
    reg.log(ExperimentRecord(name="a", kind="walk_forward"))
    reg.log(ExperimentRecord(name="a", kind="backtest"))
    assert reg.latest(kind="backtest")["kind"] == "backtest"
    assert len(reg.find("a")) == 3
    assert reg.latest(kind="missing") is None


def test_best_by_metric(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    reg.log(ExperimentRecord(name="a", kind="backtest", metrics={"sharpe": 0.5}))
    reg.log(ExperimentRecord(name="a", kind="backtest", metrics={"sharpe": 1.5}))
    reg.log(ExperimentRecord(name="a", kind="backtest", metrics={"sharpe": -1.0}))
    best = reg.best_by("sharpe", kind="backtest")
    assert best["metrics"]["sharpe"] == 1.5
    worst = reg.best_by("sharpe", kind="backtest", higher_is_better=False)
    assert worst["metrics"]["sharpe"] == -1.0


def test_read_all_on_empty_registry(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    assert reg.read_all() == []
    assert reg.latest() is None


def test_malformed_lines_are_skipped(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    reg.log(ExperimentRecord(name="good", kind="backtest"))
    with reg.path.open("a", encoding="utf-8") as fh:
        fh.write("this is not json\n")
    records = reg.read_all()
    assert len(records) == 1
    assert records[0]["name"] == "good"
