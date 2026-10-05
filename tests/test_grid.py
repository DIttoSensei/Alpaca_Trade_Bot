"""Tests for the grid trading strategy and its backtest engine.

These lock in the behaviours that actually determine grid P&L and safety, so a
regression is caught before it can flatter or hide results:

* range geometry and the fee-viability gate (safety #2),
* the core cycle rule: a buy at level k is sold ONE level up (k+1), then
  re-armed at level k — booking a positive gross.
* max simultaneously-filled buys (safety #3) and the symbol-exposure cap
  (safety #4),
* the documented worst-case intrabar ordering (low first, then high),
* regime-exit flatten + hard-bound trigger (safety #1),
* durable ladder state round-trip,
* the end-to-end portfolio runner over a synthetic, regime-varying series.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from app.config import load_config
from app.config.risk_config import RiskConfig
from app.config.settings import FeeModel
from app.config.strategy_config import GridParams
from app.core.enums import Action, ExitReason, Regime, Timeframe
from app.core.types import Candle, FeatureSnapshot, MarketState
from app.strategies.grid import (
    FILL_ORDER_LOW_FIRST,
    GridBacktester,
    GridState,
    GridStrategy,
    build_range,
    dump_state,
    fee_safety_ok,
    load_state,
    run_grid_backtest,
)


# -- helpers ---------------------------------------------------------------
def _params(**overrides) -> GridParams:
    base = dict(
        enabled=True,
        grid_levels=6,
        range_multiplier=2.0,
        recenter_interval=10_000,
        capital_per_grid=0.5,
        max_filled_levels=5,
        fee_safety_multiple=3.0,
        atr_period=14,
        reference_sma_period=100,
        hard_bound_multiplier=1.5,
    )
    base.update(overrides)
    return GridParams(**base)


def _bar(ts: datetime, low: float, high: float, close: float, open: float | None = None) -> Candle:
    return Candle(
        timestamp=ts,
        symbol="BTC/USD",
        timeframe=Timeframe.M15,
        open=open if open is not None else close,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


def _fresh_engine(params: GridParams | None = None, equity: float = 10_000.0) -> GridBacktester:
    params = params or _params()
    return GridBacktester(
        "BTC/USD",
        params,
        FeeModel(),
        RiskConfig(),
        starting_equity=equity,
        apply_slippage=False,
        apply_spread=False,
    )


# -- geometry --------------------------------------------------------------
def test_build_range_levels_are_equal_and_span_bounds():
    rng = build_range(100.0, 2.0, levels=5, range_multiplier=2.0)
    assert rng.upper == 104.0
    assert rng.lower == 96.0
    assert len(rng.levels) == 5
    assert rng.levels[0] == 96.0
    assert rng.levels[-1] == 104.0
    diffs = [round(rng.levels[i + 1] - rng.levels[i], 9) for i in range(4)]
    assert len(set(diffs)) == 1  # equal spacing
    assert rng.in_range(100.0) and not rng.in_range(105.0)


def test_fee_safety_gate_rejects_tight_range():
    fees = FeeModel()
    cost = fees.round_trip_cost()
    # A gap of exactly the round-trip cost cannot clear a 3x multiple.
    assert not fee_safety_ok(100.0 * cost, 100.0, fees, 3.0)
    # 3x the cost (plus a hair) clears it.
    assert fee_safety_ok(100.0 * cost * 3.0 + 1e-9, 100.0, fees, 3.0)


def test_recenter_skips_grid_when_fees_dominate():
    # Deliberately absurd geometry: 100 rungs inside a fraction of a basis point.
    engine = _fresh_engine(_params(grid_levels=100, range_multiplier=0.000001))
    ok = engine._recenter(100.0, 1.0, 100.0, 0)  # real fee check, NOT skipped
    assert ok is False
    assert engine.result.levels_placed_skipped == 1


# -- strategy shell --------------------------------------------------------
def test_grid_strategy_is_inert_and_regime_gated():
    strat = GridStrategy(_params())
    assert strat.name == "grid"
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    feats = FeatureSnapshot(symbol="BTC/USD", timestamp=ts, timeframe=Timeframe.M15, values={})
    in_range = MarketState(symbol="BTC/USD", timestamp=ts, regime=Regime.RANGE, features=feats)
    sig = strat.evaluate(in_range)
    assert sig.action == Action.NO_SIGNAL  # grid never emits a directional signal

    trending = MarketState(symbol="BTC/USD", timestamp=ts, regime=Regime.STRONG_UPTREND, features=feats)
    assert strat.evaluate(trending).action == Action.NO_SIGNAL


# -- the core cycle: buy at k, sell at k+1, re-arm at k --------------------
def test_buy_then_sell_one_level_up_then_rebuy():
    engine = _fresh_engine()
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # Build a ladder explicitly: range 96..104, 6 rungs, price 100.
    assert engine._recenter(100.0, 2.0, 100.0, 0, skip_fee_check=True)
    levels = engine.state.levels
    assert len(levels) == 6

    # Bar 1: descend from 97.0 to the lowest rung (96) only -> exactly one buy.
    engine.process_bar(_bar(t0, low=96.0, high=97.0, close=96.5, open=97.0), 1, Regime.RANGE)
    assert levels[0].inventory > 0, "buy at the lowest rung should have filled"
    buy_fills = [f for f in engine.result.fills if f.side == "buy"]
    assert len(buy_fills) == 1

    # Bar 2: rally to 98 -> that inventory (bought at 96) sells at the NEXT
    # rung up (97.6). This is the profit leg; the same-level bug would sell at 96.
    engine.process_bar(_bar(t0 + timedelta(minutes=15), low=97.0, high=98.0, close=97.5, open=97.0), 2, Regime.RANGE)
    sells = [f for f in engine.result.fills if f.side == "sell"]
    assert len(sells) == 1
    assert sells[0].level_price == levels[1].price, "inventory must sell one level UP"
    assert levels[0].inventory == 0.0, "re-buy: the original rung is re-armed once sold"

    # Exactly one completed round-trip, and it made a POSITIVE gross spread.
    assert len(engine.result.trades) == 1
    trade = engine.result.trades[0]
    assert trade.entry_price < trade.exit_price
    assert trade.gross_pnl > 0
    # Entry is the buy rung, exit is one rung above it.
    assert round(trade.entry_price, 6) == round(levels[0].price, 6)
    assert round(trade.exit_price, 6) == round(levels[1].price, 6)


def test_multi_level_cycle_books_profitable_round_trips():
    engine = _fresh_engine()
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert engine._recenter(100.0, 2.0, 100.0, 0, skip_fee_check=True)
    # One wide bar sweeping the whole range: buys on the low leg, sells up on
    # the high leg. Open at the centre bounds the down-leg to the lower rungs.
    engine.process_bar(_bar(t0, low=95.0, high=105.0, close=100.0, open=100.0), 1, Regime.RANGE)
    assert engine.result.trades, "a full sweep must complete at least one cycle"
    assert all(t.gross_pnl > 0 for t in engine.result.trades)
    assert engine.state.inventory_qty >= 0.0


# -- safety #3: max simultaneously-filled buys -----------------------------
def test_max_filled_levels_caps_concurrent_buys_per_bar():
    params = _params(max_filled_levels=2, grid_levels=8)
    engine = _fresh_engine(params)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert engine._recenter(100.0, 3.0, 100.0, 0, skip_fee_check=True)
    # A bar that undercuts every rung: far more than max_filled_levels are touched.
    engine.process_bar(_bar(t0, low=90.0, high=100.0, close=95.0, open=100.0), 1, Regime.RANGE)
    buys = [f for f in engine.result.fills if f.side == "buy"]
    assert len(buys) == 2, "concurrent buys must be capped by max_filled_levels"


# -- safety #4: exposure cap ----------------------------------------------
def test_symbol_exposure_cap_limits_committed_capital():
    risk = RiskConfig()
    # 20 small (~100 notional) rungs, all reachable, so the symbol cap (2000)
    # binds before capital_per_grid would.
    params = _params(
        capital_per_grid=0.2,
        grid_levels=20,
        max_filled_levels=20,
        fee_safety_multiple=0.1,
    )
    engine = GridBacktester(
        "BTC/USD",
        params,
        FeeModel(),
        risk,
        starting_equity=10_000.0,
        apply_slippage=False,
        apply_spread=False,
    )
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert engine._recenter(100.0, 10.0, 100.0, 0, skip_fee_check=True)
    engine.process_bar(_bar(t0, low=50.0, high=120.0, close=110.0, open=120.0), 1, Regime.RANGE)
    cap = risk.max_symbol_exposure * 10_000.0
    assert engine.result.peak_committed <= cap + 1e-6, "exposure cap must never be breached"
    assert engine.result.peak_committed > 0.0


# -- safety #1: hard-bound trigger ----------------------------------------
def test_hard_bound_trigger_flattens_and_deactivates():
    engine = _fresh_engine()
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert engine._recenter(100.0, 2.0, 100.0, 0, skip_fee_check=True)
    engine.process_bar(_bar(t0, low=96.0, high=97.0, close=96.5, open=97.0), 1, Regime.RANGE)
    assert engine.state.inventory_qty > 0

    # Range width is 8; a move beyond the bound by > 1.5x width (>= 12) trips it.
    breakout = _bar(t0 + timedelta(minutes=15), low=80.0, high=100.0, close=82.0, open=85.0)
    engine.process_bar(breakout, 2, Regime.RANGE)
    assert engine.state.hard_bound_triggers == 1
    assert engine.result.hard_bound_triggers == 1
    assert engine.state.active is False
    assert engine.state.inventory_qty == 0.0
    assert engine.result.regime_exit_flattens == 1


# -- ordering assumption is explicit and conservative ----------------------
def test_fill_order_is_documented_low_first():
    engine = _fresh_engine()
    assert engine.fill_order == FILL_ORDER_LOW_FIRST
    result = run_grid_backtest({}, load_config())
    assert result.fill_order == FILL_ORDER_LOW_FIRST


# -- durable state round-trip ---------------------------------------------
def test_grid_state_round_trip():
    st = GridState(symbol="BTC/USD", ref=100.0, upper=104.0, lower=96.0, width=8.0, spacing=2.0)
    st.recenter_count = 3
    st.regime_exit_flattens = 2
    st.hard_bound_triggers = 1
    st.active = False
    payload = dump_state(st)
    restored = load_state(payload)
    assert restored.symbol == st.symbol
    assert restored.recenter_count == 3
    assert restored.regime_exit_flattens == 2
    assert restored.hard_bound_triggers == 1
    assert restored.active is False


# -- end-to-end runner -----------------------------------------------------
def _synthetic(bars: int = 1400, seed: int = 5) -> tuple[dict, Timeframe]:
    """Mean-reverting series with alternating low- and high-volatility blocks."""
    rng = random.Random(seed)
    tf = Timeframe.M15
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    base = 30_000.0
    candles: list[Candle] = []
    price = base
    for i in range(bars):
        # ~48-bar blocks alternating between tight (range) and wide (trend) vol.
        block = (i // 48) % 2
        vol = 0.0009 if block == 0 else 0.0045
        drift = 0.0 if block == 0 else 0.0016
        mean_pull = (base - price) / base * 0.05
        price = max(1.0, price * (1.0 + drift + mean_pull + rng.gauss(0.0, vol)))
        op = price
        high = max(op, price) * (1.0 + abs(rng.gauss(0.0, vol / 2.0)))
        low = max(0.01, min(op, price) * (1.0 - abs(rng.gauss(0.0, vol / 2.0))))
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=15 * i),
                symbol="BTC/USD",
                timeframe=tf,
                open=op,
                high=high,
                low=low,
                close=price,
                volume=100.0,
            )
        )
    return {("BTC/USD", tf): candles}, tf


def test_run_grid_backtest_end_to_end():
    by_key, exec_tf = _synthetic()
    cfg = load_config()
    cfg.strategies.grid.enabled = True
    result = run_grid_backtest(by_key, cfg, collection_timeframe=exec_tf)
    assert "BTC/USD" in result.per_symbol
    res = result.per_symbol["BTC/USD"]
    assert res.final_equity == res.final_equity  # not NaN
    assert math.isfinite(res.final_equity)
    summary = result.summary()
    assert summary["fill_order"] == FILL_ORDER_LOW_FIRST
    assert summary["trades"] >= 0
    # Trades, if any, must be labelled as grid and structurally valid.
    for t in result.trades:
        assert t.strategy == "grid"
        assert t.exit_time >= t.entry_time
        assert t.qty > 0
        assert t.entry_price > 0 and t.exit_price > 0


def test_grid_disabled_is_a_noop():
    by_key, exec_tf = _synthetic(400)
    cfg = load_config()
    cfg.strategies.grid.enabled = False
    result = run_grid_backtest(by_key, cfg, collection_timeframe=exec_tf)
    assert result.per_symbol == {}
    assert result.trades == []


def test_backtest_grid_report_renders_sanity_checks():
    from scripts.backtest_grid import render_report

    by_key, exec_tf = _synthetic()
    cfg = load_config()
    result = run_grid_backtest(by_key, cfg, collection_timeframe=exec_tf)
    report = render_report(result, cfg, days=200, exec_tf=exec_tf, equity=10_000.0)
    for needle in (
        "GRID BACKTEST",
        FILL_ORDER_LOW_FIRST,
        "HOLD TIME",
        "MAX SINGLE-TRADE PRICE MOVE",
        "NON-EXACT-LEVEL FILLS",
        "UNIQUE TRADING DAYS",
        "SAFETY #1",
        "SAFETY #4",
    ):
        assert needle in report, needle
    # Every generated fill must land exactly on a configured level.
    assert " 0  (must be 0)" in report
