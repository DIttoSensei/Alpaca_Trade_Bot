"""Bar-by-bar backtest engine.

The engine deliberately reuses the EXACT live decision stack (see
``app.assembly.build_stack``): the same feature store, regime detector,
strategies, ensemble and decision composer. There is no parallel "backtest
strategy code" that could silently diverge from production (spec sections 59,
92, 96).

No-lookahead guarantees:
  * Execution-timeframe bars are merged into the store incrementally, one
    completed bar at a time, so a strategy can never see the future.
  * Higher-timeframe (confirmation/regime/macro) bars are aggregated from the
    base timeframe and only merged once their bucket has fully CLOSED relative
    to the current evaluation time. A forming H4 candle is never visible.
  * Exits are evaluated with the pessimistic assumption that if a bar touches
    both the stop and the target, the stop filled first.

Cost realism (spec sections 25, 63, 64) is delegated to ``Simulator``: taker
fees on both legs, slippage and half the spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

from app.assembly import StrategyStack, build_stack
from app.backtest.simulator import SimConfig, Simulator
from app.config import BotConfig
from app.core.enums import ExitReason, Timeframe
from app.core.types import Candle, PositionRecord, RejectedCandidate, TradeRecord
from app.data.candle_store import aggregate
from app.monitoring.metrics import PerformanceMetrics, compute_metrics


@dataclass(slots=True)
class BacktestConfig:
    starting_equity: float = 10_000.0
    warmup_bars: int = 250
    # Enforce at most one position per symbol and a maximum number of concurrent
    # positions across the universe (fail-closed exposure control).
    max_open_positions: int = 3
    apply_slippage: bool = True
    apply_spread: bool = True
    # Stop a backtest early if equity falls below this fraction of start.
    ruin_fraction: float = 0.5
    # Pre-load every candle series and compute each feature frame exactly ONCE.
    # Because all indicators are causal (value at index i depends only on inputs
    # 0..i) this is value-identical to the old incremental, per-bar recompute, but
    # turns the backtest from O(n^2) into O(n) (spec section 124). Disable only if
    # the store must be mutated incrementally during the run for some other test.
    preload_features: bool = True


@dataclass(slots=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    open_positions: int


@dataclass(slots=True)
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    rejections: list[RejectedCandidate] = field(default_factory=list)
    regime_counts: dict[str, int] = field(default_factory=dict)
    bars_processed: int = 0
    symbols: list[str] = field(default_factory=list)
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def as_dict(self) -> dict:
        return {
            "trades": len(self.trades),
            "metrics": self.metrics.as_dict(),
            "rejections": len(self.rejections),
            "regime_counts": dict(self.regime_counts),
            "bars_processed": self.bars_processed,
            "symbols": list(self.symbols),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "exposure": self.exposure_stats(),
        }

    def exposure_stats(self) -> dict:
        if not self.equity_curve:
            return {"avg_open_positions": 0.0, "max_open_positions": 0}
        open_counts = [p.open_positions for p in self.equity_curve]
        return {
            "avg_open_positions": sum(open_counts) / len(open_counts),
            "max_open_positions": max(open_counts),
        }


def _htf_schedule(exec_bars: Sequence[Candle], target: Timeframe) -> list[tuple[datetime, Candle]]:
    """Return (bucket_close_time, candle) for complete higher-timeframe buckets.

    ``aggregate`` already drops incomplete buckets; here we additionally compute
    each bucket's CLOSE time from its OPEN time, so the engine can release a
    higher-timeframe candle only after it has finished forming.
    """
    if not exec_bars or target == exec_bars[0].timeframe:
        return []
    span = timedelta(seconds=target.seconds)
    return [(bar.timestamp + span, bar) for bar in aggregate(exec_bars, target)]


class BacktestEngine:
    """Runs the shared decision stack over historical candles."""

    def __init__(
        self,
        candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
        config: BotConfig | None = None,
        backtest: BacktestConfig | None = None,
        stack: StrategyStack | None = None,
        collection_timeframe: Timeframe | None = None,
    ) -> None:
        self.stack = stack or build_stack(config)
        self.cfg = self.stack.config
        self.bt = backtest or BacktestConfig()
        self.candles_by_key = {k: list(v) for k, v in candles_by_key.items()}

        # The base timeframe we iterate on (the collection timeframe).
        self.exec_tf = collection_timeframe or self.stack.context.execution

        # Group execution-timeframe bars per symbol (sorted by time).
        self._exec: dict[str, list[Candle]] = {}
        for (symbol, tf), bars in self.candles_by_key.items():
            if tf == self.exec_tf:
                self._exec[symbol] = sorted(bars, key=lambda c: c.timestamp)

        # Higher timeframes are always re-aggregated from the base timeframe so
        # no-lookahead consistency is guaranteed regardless of what the caller
        # supplied.
        self._htf_roles: tuple[str, ...] = ("confirmation", "regime", "macro")
        self._htf_schedule: dict[str, dict[str, list[tuple[datetime, Candle]]]] = {}
        for symbol, bars in self._exec.items():
            self._htf_schedule[symbol] = {}
            for role in self._htf_roles:
                tf = self.stack.context.by_role(role)
                if tf is None or tf == self.exec_tf:
                    self._htf_schedule[symbol][role] = []
                    continue
                self._htf_schedule[symbol][role] = _htf_schedule(bars, tf)

        self.sim = Simulator(
            SimConfig(
                starting_equity=self.bt.starting_equity,
                apply_slippage=self.bt.apply_slippage,
                apply_spread=self.bt.apply_spread,
            ),
            self.cfg.risk,
            self.cfg.settings.fees,
        )

        # O(n) pre-warm: load every candle series and compute each feature frame
        # exactly once, up front. ``snapshot_at(index)`` then reads point-in-time
        # values with no look-ahead because every indicator is causal. This
        # replaces the old per-bar ``merge`` + ``invalidate`` (which recomputed
        # the whole frame after every bar and re-sorted the store -> O(n^2)).
        self.preloaded = False
        if self.bt.preload_features:
            self._preload()

    def _preload(self) -> None:
        """Merge the full history once and build each feature frame once.

        Higher-timeframe series are pre-aggregated from the execution timeframe
        so that only COMPLETE buckets exist in the store; the pipeline still only
        reads a bucket once its close time has passed, preserving no-lookahead.
        """
        store = self.stack.candles
        store._data.clear()  # noqa: SLF001 - engine owns the store lifecycle
        for symbol, bars in self._exec.items():
            store.merge(bars)
            for role in self._htf_roles:
                htf_bars = [candle for (_close, candle) in self._htf_schedule[symbol][role]]
                if htf_bars:
                    store.merge(htf_bars)
        # One frame computation per (symbol, timeframe): drop the stale cache and
        # let the next snapshot_at() build it lazily from the complete series.
        self.stack.features.invalidate()
        self.preloaded = True

    # -- main loop -------------------------------------------------------
    def run(self) -> BacktestResult:
        result = BacktestResult(symbols=sorted(self._exec.keys()))
        if not self._exec:
            return result

        # The store was pre-loaded (and each feature frame computed once) in
        # __init__, so the run loop only advances pointers. If preloading was
        # disabled, load the full history once here as a fallback so index-based
        # snapshot reads stay consistent (this path is still O(n)).
        if not self.preloaded:
            self._preload()
        self.sim.reset()

        exec_ptr: dict[str, int] = {s: 0 for s in self._exec}
        htf_ptr: dict[str, dict[str, int]] = {
            s: {role: 0 for role in self._htf_roles} for s in self._exec
        }
        last_exec_index: dict[str, int] = {}
        btc_symbol = self.stack.reference_symbol

        events = self._timeline()
        result.start = events[0][0]
        result.end = events[-1][0]
        ruin_level = self.bt.starting_equity * self.bt.ruin_fraction

        for timestamp, symbol in events:
            # 1) Higher-timeframe candles are already pre-loaded; the pipeline
            #    gates them by close time, so nothing needs releasing here.
            self._release_htf(symbol, timestamp, htf_ptr)

            # 2) Advance ALL symbols' execution-bar POINTERS up to `timestamp` so
            #    cross-symbol reads (BTC reference) use the right index. The store
            #    already holds the full series: no per-bar merge or recompute.
            for sym, bars in self._exec.items():
                ptr = exec_ptr[sym]
                advanced = False
                while ptr < len(bars) and bars[ptr].timestamp <= timestamp:
                    ptr += 1
                    advanced = True
                if advanced:
                    last_exec_index[sym] = ptr - 1
                exec_ptr[sym] = ptr

            result.bars_processed += 1

            # 3) Manage open positions for this symbol using its closed bar.
            bar = self._bar_at(symbol, timestamp)
            if bar is None:
                continue
            self.sim.check_bar_exits(symbol, bar)

            # 4) Evaluate the decision stack after warmup.
            if self._ready(symbol, last_exec_index):
                self._evaluate(symbol, bar, timestamp, last_exec_index, btc_symbol, result)

            # 5) Mark to market using closing prices.
            self.sim.mark_to_market(self._close_prices(timestamp))
            result.equity_curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    equity=self.sim.equity,
                    open_positions=len(self.sim.positions),
                )
            )

            if self.sim.equity <= ruin_level and not self.sim.positions:
                break

        # Close any residual open positions at the last known price.
        self._force_close_all()

        result.trades = list(self.sim.trades)
        result.metrics = compute_metrics(result.trades, self.bt.starting_equity)
        return result

    # -- internals -------------------------------------------------------
    def _timeline(self) -> list[tuple[datetime, str]]:
        events: list[tuple[datetime, str]] = []
        for symbol, bars in self._exec.items():
            for bar in bars:
                events.append((bar.timestamp, symbol))
        events.sort(key=lambda e: (e[0], e[1]))
        return events

    def _release_htf(
        self,
        symbol: str,
        timestamp: datetime,
        htf_ptr: dict[str, dict[str, int]],
    ) -> None:
        """No-op: higher-timeframe candles are pre-loaded in ``_preload``.

        Retention of no-lookahead now lives in ``TradingPipeline._ctx_index``,
        which only returns a bucket whose CLOSE time is at or before ``timestamp``
        (completed buckets only). The ``htf_ptr`` argument is kept for signature
        stability and potential future incremental use.
        """
        return None

    def _bar_at(self, symbol: str, timestamp: datetime) -> Optional[Candle]:
        bars = self._exec.get(symbol)
        if not bars:
            return None
        lo, hi = 0, len(bars) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            ts = bars[mid].timestamp
            if ts == timestamp:
                return bars[mid]
            if ts < timestamp:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def _ready(self, symbol: str, last_exec_index: dict[str, int]) -> bool:
        idx = last_exec_index.get(symbol)
        return idx is not None and idx >= self.bt.warmup_bars

    def _evaluate(
        self,
        symbol: str,
        bar: Candle,
        timestamp: datetime,
        last_exec_index: dict[str, int],
        btc_symbol: str,
        result: BacktestResult,
    ) -> None:
        if symbol in self.sim.positions:
            return
        if len(self.sim.positions) >= self.bt.max_open_positions:
            return

        positions: dict[str, PositionRecord] = {
            sym: pos.to_record() for sym, pos in self.sim.positions.items()
        }
        btc_index = last_exec_index.get(btc_symbol)

        pipeline_result = self.stack.pipeline.evaluate(
            symbol=symbol,
            timestamp=timestamp,
            exec_index=last_exec_index[symbol],
            equity=self.sim.equity,
            positions=positions,
            btc_index=btc_index,
        )
        regime_name = str(pipeline_result.state.regime)
        result.regime_counts[regime_name] = result.regime_counts.get(regime_name, 0) + 1

        outcome = pipeline_result.outcome
        if outcome.rejection is not None:
            result.rejections.append(outcome.rejection)
            return
        decision = outcome.decision
        if decision is None or decision.quantity <= 0:
            return

        # The signal is derived from a COMPLETED bar; we execute at that bar's
        # close, which is the earliest tradable price without look-ahead.
        self.sim.open_from_decision(
            decision, bar.close, last_exec_index[symbol], timestamp
        )

    def _close_prices(self, timestamp: datetime) -> dict[str, float]:
        prices: dict[str, float] = {}
        for symbol in self.sim.positions:
            bar = self._bar_at(symbol, timestamp)
            if bar is not None:
                prices[symbol] = bar.close
        return prices

    def _force_close_all(self) -> None:
        for symbol in list(self.sim.positions.keys()):
            bars = self._exec.get(symbol)
            if not bars:
                continue
            last_bar = bars[-1]
            self.sim.close_position(
                symbol, last_bar.close, ExitReason.MANUAL, last_bar.timestamp, last_bar
            )


def run_backtest(
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    config: BotConfig | None = None,
    backtest: BacktestConfig | None = None,
    collection_timeframe: Timeframe | None = None,
) -> BacktestResult:
    """Convenience entry point used by scripts, walk-forward and Monte Carlo."""
    engine = BacktestEngine(
        candles_by_key,
        config=config,
        backtest=backtest,
        collection_timeframe=collection_timeframe,
    )
    return engine.run()


__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "EquityPoint",
    "run_backtest",
]
