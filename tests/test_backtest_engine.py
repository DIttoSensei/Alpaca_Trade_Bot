"""Regression tests for the backtest engine's indexing and exit labelling.

These lock in the fixes for the bug where the O(n) rewrite addressed candles by
ABSOLUTE index while ``CandleStore`` truncated to ``max_bars`` (default 5000).
For histories longer than the cap the store dropped the oldest bars, so
``snapshot_at(index)`` read the wrong (future) bar — producing phantom one-bar
price moves and inverted exit labels.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from app.assembly import build_stack
from app.backtest import BacktestConfig, BacktestEngine
from app.backtest.simulator import OpenPosition, SimConfig, Simulator
from app.config import load_config
from app.config.risk_config import RiskConfig
from app.config.settings import FeeModel
from app.core.enums import ExitReason, Regime, Timeframe
from app.core.types import Candle, FeatureSnapshot
from app.data.candle_store import CandleStore


# -- helpers --------------------------------------------------------------
def _synthetic(bars: int, seed: int = 11) -> tuple[dict, Timeframe]:
    rng = random.Random(seed)
    exec_tf = Timeframe.M15
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    universe = [("BTC/USD", 60_000.0, 0.0040), ("ETH/USD", 3_000.0, 0.0050)]
    out: dict = {}
    for symbol, base, vol in universe:
        price = base
        candles: list[Candle] = []
        for i in range(bars):
            phase = math.sin(i / (96 * 8))
            mu = 0.00002 + 0.0006 * phase
            price = max(1.0, price * (1.0 + mu + rng.gauss(0.0, vol)))
            op = price * (1.0 + rng.gauss(0.0, vol / 4.0))
            candles.append(
                Candle(
                    timestamp=start + timedelta(minutes=15 * i),
                    symbol=symbol,
                    timeframe=exec_tf,
                    open=op,
                    high=max(op, price) * (1.0 + abs(rng.gauss(0.0, vol / 2.0))),
                    low=max(0.01, min(op, price) * (1.0 - abs(rng.gauss(0.0, vol / 2.0)))),
                    close=price,
                    volume=100.0 + rng.random() * 50.0,
                )
            )
        out[(symbol, exec_tf)] = candles
    return out, exec_tf


def _engine(by_key, exec_tf, *, preload: bool = True, uncapped: bool = True) -> BacktestEngine:
    cfg = load_config()
    store = CandleStore(max_bars=10**9) if uncapped else CandleStore()
    stack = build_stack(cfg, candles=store)
    bt = BacktestConfig(warmup_bars=210, ruin_fraction=0.0, preload_features=preload)
    return BacktestEngine(by_key, stack=stack, backtest=bt, collection_timeframe=exec_tf)


# -- the regression: absolute indices must align with the store -----------
def test_store_retains_full_history_and_snapshot_indices_align():
    """> max_bars history must not be truncated, and snapshot_at(i) must read bar i.

    Before the fix the store kept 5000 of 6000 bars and ``snapshot_at(100)``
    silently returned bar 5100 — the future.
    """
    n = 6000
    assert n > 5000  # must exceed the historical default cap to be meaningful
    by_key, exec_tf = _synthetic(n)
    engine = _engine(by_key, exec_tf)
    stack = engine.stack

    for symbol, bars in engine._exec.items():
        assert stack.candles.count(symbol, exec_tf) == len(bars), "store truncated the series"

    sample = [0, 1, 100, 2500, 4999, 5000, 5555, n - 1]
    for symbol, bars in engine._exec.items():
        for i in sample:
            snap = stack.features.snapshot_at(symbol, exec_tf, i)
            assert snap.values["close"] == bars[i].close
            assert snap.timestamp == bars[i].timestamp


def test_preload_matches_incremental_full_load():
    """Preloading the frame is value-identical to loading the series in run()."""
    by_key, exec_tf = _synthetic(900)
    a = _engine(by_key, exec_tf, preload=True).run()
    b = _engine(by_key, exec_tf, preload=False).run()

    def key(t):
        return (t.symbol, t.entry_time, t.exit_time, round(t.entry_price, 6),
                round(t.exit_price, 6), str(t.exit_reason), round(t.net_pnl, 6))

    assert [key(t) for t in a.trades] == [key(t) for t in b.trades]


def test_backtest_prices_are_physical_and_labels_not_inverted():
    """Exit prices must be reachable from the entry bar; no obvious label inversion."""
    by_key, exec_tf = _synthetic(6000)
    result = _engine(by_key, exec_tf).run()

    # Timestamp -> index lookup built from the raw source data.
    ts_index = {sym: {c.timestamp: i for i, c in enumerate(bars)}
                for (sym, _tf), bars in by_key.items()}

    for t in result.trades:
        i = ts_index[t.symbol][t.entry_time]
        j = ts_index[t.symbol][t.exit_time]
        span = by_key[(t.symbol, exec_tf)][i:j + 1]
        lo = min(b.low for b in span)
        hi = max(b.high for b in span)
        # The exit fill must lie within the traded range (allowing cost slippage).
        assert lo * 0.995 <= t.exit_price <= hi * 1.005, (t.symbol, t.entry_time)
        if t.exit_reason is ExitReason.TAKE_PROFIT:
            assert t.net_pnl > 0, "TAKE_PROFIT booked a loss"
        if t.exit_reason is ExitReason.STOP_LOSS:
            assert t.exit_price < t.entry_price, "STOP_LOSS exited above entry"


# -- exit labelling -------------------------------------------------------
def _sim() -> Simulator:
    return Simulator(SimConfig(), RiskConfig(), FeeModel())


def _open(stop: float, target: float) -> OpenPosition:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return OpenPosition(
        symbol="BTC/USD",
        qty=1.0,
        entry_price=100.0,
        entry_index=0,
        entry_time=ts,
        stop_price=stop,
        target_price=target,
        initial_risk_per_unit=max(1e-12, 100.0 - stop),
        strategy="trend_pullback",
        regime=Regime.STRONG_UPTREND,
        feature_snapshot=FeatureSnapshot(
            symbol="BTC/USD", timestamp=ts, timeframe=Timeframe.M15, values={}
        ),
        expected_edge=0.02,
        estimated_cost=0.001,
        risk_amount=10.0,
        entry_fee=0.0,
    )


def _bar(low: float, high: float) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
        symbol="BTC/USD",
        timeframe=Timeframe.M15,
        open=100.0,
        high=high,
        low=low,
        close=100.0,
        volume=1.0,
    )


def test_trailing_stop_above_entry_is_labelled_trailing():
    sim = _sim()
    pos = _open(stop=105.0, target=200.0)  # trailed above entry
    sim.positions["BTC/USD"] = pos
    trades = sim.check_bar_exits("BTC/USD", _bar(low=104.0, high=106.0))
    assert len(trades) == 1
    assert trades[0].exit_reason is ExitReason.TRAILING_STOP
    assert trades[0].exit_price > trades[0].entry_price


def test_initial_stop_below_entry_is_labelled_stop_loss():
    sim = _sim()
    sim.positions["BTC/USD"] = _open(stop=90.0, target=200.0)
    trades = sim.check_bar_exits("BTC/USD", _bar(low=89.0, high=101.0))
    assert len(trades) == 1
    assert trades[0].exit_reason is ExitReason.STOP_LOSS


def test_breakeven_stop_is_labelled_trailing():
    sim = _sim()
    sim.positions["BTC/USD"] = _open(stop=100.0, target=200.0)
    trades = sim.check_bar_exits("BTC/USD", _bar(low=99.0, high=101.0))
    assert len(trades) == 1
    assert trades[0].exit_reason is ExitReason.TRAILING_STOP
