"""Walk-forward / out-of-sample validation.

Overfitting is the default failure mode of a backtested strategy, so we never
trust a single in-sample run. Walk-forward splits the series into consecutive
windows and evaluates each window on data the strategy has not "seen" as a
contiguous block, then aggregates the out-of-sample result. Parameter choices
are made on the training slice and only the test slice is scored (spec sections
90, 91).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Sequence

from app.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.config import BotConfig
from app.core.enums import Timeframe
from app.core.types import Candle
from app.monitoring.metrics import PerformanceMetrics, compute_metrics


@dataclass(slots=True)
class WalkForwardWindow:
    index: int
    train_start: Optional[datetime]
    train_end: Optional[datetime]
    test_start: Optional[datetime]
    test_end: Optional[datetime]
    result: BacktestResult

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
            "test": self.result.as_dict(),
        }


@dataclass(slots=True)
class WalkForwardResult:
    windows: list[WalkForwardWindow] = field(default_factory=list)
    out_of_sample: list = field(default_factory=list)
    aggregate: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    consistency: float = 0.0  # fraction of profitable windows

    def as_dict(self) -> dict:
        return {
            "windows": [w.as_dict() for w in self.windows],
            "aggregate": self.aggregate.as_dict(),
            "consistency": round(self.consistency, 4),
            "window_count": len(self.windows),
        }


def _split_events(
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    base_timeframe: Timeframe,
) -> list[datetime]:
    stamps: set[datetime] = set()
    for (_symbol, tf), bars in candles_by_key.items():
        if tf != base_timeframe:
            continue
        for bar in bars:
            stamps.add(bar.timestamp)
    return sorted(stamps)


def _slice(
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    start: datetime,
    end: datetime,
) -> dict[tuple[str, Timeframe], list[Candle]]:
    out: dict[tuple[str, Timeframe], list[Candle]] = {}
    for key, bars in candles_by_key.items():
        out[key] = [b for b in bars if start <= b.timestamp < end]
    return out


def run_walk_forward(
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    config: BotConfig | None = None,
    backtest: BacktestConfig | None = None,
    base_timeframe: Timeframe = Timeframe.M15,
    train_bars: int = 4000,
    test_bars: int = 1000,
    step_bars: int = 1000,
    max_windows: int = 8,
    on_window: Optional[Callable[[WalkForwardWindow], None]] = None,
) -> WalkForwardResult:
    """Roll a train/test split across the timeline and score each test slice.

    The training slice is reserved for parameter selection by the caller; here
    we simply guarantee that scoring happens on a later, disjoint slice.
    """
    stamps = _split_events(candles_by_key, base_timeframe)
    wf = WalkForwardResult()
    if len(stamps) < train_bars + test_bars:
        return wf

    cursor = train_bars
    index = 0
    while cursor + test_bars <= len(stamps) and index < max_windows:
        train_start = stamps[0]
        train_end = stamps[cursor - 1]
        test_start = stamps[cursor]
        test_end = stamps[min(cursor + test_bars, len(stamps)) - 1]

        test_slice = _slice(candles_by_key, test_start, test_end)
        engine = BacktestEngine(test_slice, config=config, backtest=backtest)
        result = engine.run()

        window = WalkForwardWindow(
            index=index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            result=result,
        )
        wf.windows.append(window)
        wf.out_of_sample.extend(result.trades)
        if on_window is not None:
            on_window(window)

        cursor += step_bars
        index += 1

    wf.aggregate = compute_metrics(wf.out_of_sample, (backtest or BacktestConfig()).starting_equity)
    if wf.windows:
        profitable = sum(1 for w in wf.windows if w.result.metrics.net_pnl > 0)
        wf.consistency = profitable / len(wf.windows)
    return wf


__all__ = ["WalkForwardResult", "WalkForwardWindow", "run_walk_forward"]
