"""Performance and operational metrics.

Performance metrics are computed from TradeRecords (net of costs!) so we never
confuse gross P&L with what actually landed in the account (spec sections 69,
110). Operational counters are simple in-process tallies for dashboards/alerts.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.core.types import TradeRecord


@dataclass(slots=True)
class PerformanceMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    payoff_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    average_hold_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "net_pnl": round(self.net_pnl, 4),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 6),
            "average_win": round(self.average_win, 4),
            "average_loss": round(self.average_loss, 4),
            "payoff_ratio": round(self.payoff_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe": round(self.sharpe, 4),
            "total_fees": round(self.total_fees, 4),
            "total_slippage": round(self.total_slippage, 4),
            "average_hold_seconds": round(self.average_hold_seconds, 2),
        }


def compute_metrics(trades: Sequence[TradeRecord], starting_equity: float | None = None) -> PerformanceMetrics:
    m = PerformanceMetrics()
    if not trades:
        return m

    ordered = sorted(trades, key=lambda t: t.exit_time)
    wins = [t for t in ordered if t.net_pnl > 0]
    losses = [t for t in ordered if t.net_pnl < 0]

    m.trades = len(ordered)
    m.wins = len(wins)
    m.losses = len(losses)
    m.win_rate = len(wins) / m.trades
    m.gross_profit = sum(t.net_pnl for t in wins)
    m.gross_loss = abs(sum(t.net_pnl for t in losses))
    m.net_pnl = sum(t.net_pnl for t in ordered)
    m.total_fees = sum(t.fees for t in ordered)
    m.total_slippage = sum(t.slippage for t in ordered)
    m.average_hold_seconds = sum(t.holding_time_seconds for t in ordered) / m.trades

    if m.gross_loss > 0:
        m.profit_factor = m.gross_profit / m.gross_loss
    elif m.gross_profit > 0:
        m.profit_factor = math.inf
    m.expectancy = m.net_pnl / m.trades
    m.average_win = (m.gross_profit / len(wins)) if wins else 0.0
    m.average_loss = (-m.gross_loss / len(losses)) if losses else 0.0
    if m.average_loss != 0:
        m.payoff_ratio = abs(m.average_win / m.average_loss)

    m.max_drawdown, m.max_drawdown_pct = _max_drawdown(ordered, starting_equity)
    m.sharpe = _trade_sharpe([t.net_pnl for t in ordered])
    return m


def _max_drawdown(trades: Sequence[TradeRecord], starting_equity: float | None) -> tuple[float, float]:
    equity = starting_equity if starting_equity and starting_equity > 0 else 0.0
    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0
    for t in trades:
        equity += t.net_pnl
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak) if peak > 0 else 0.0
    if starting_equity is None and max_dd_pct == 0.0:
        # Without an equity base, report drawdown in absolute terms only.
        return max_dd, 0.0
    return max_dd, max_dd_pct


def _trade_sharpe(pnls: Sequence[float]) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    # Per-trade Sharpe scaled by sqrt(assumed trades/year) is deliberately left
    # to the reporting layer; here we return the raw risk-adjusted ratio.
    return mean / sd


@dataclass(slots=True)
class RollingsStats:
    """Rolling window stats used for adaptive strategy weighting."""

    window: int = 50
    _results: deque = field(default_factory=lambda: deque(maxlen=50))

    def add(self, net_pnl: float) -> None:
        self._results.append(net_pnl)

    @property
    def trade_count(self) -> int:
        return len(self._results)

    @property
    def recent_performance(self) -> float:
        if not self._results:
            return 0.0
        return sum(self._results) / len(self._results)

    @property
    def win_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r > 0) / len(self._results)


class MetricsRegistry:
    """Cheap in-process operational counters."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}

    def incr(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def snapshot(self) -> dict:
        return {"counters": dict(self._counters), "gauges": dict(self._gauges)}


def per_strategy_metrics(trades: Iterable[TradeRecord]) -> dict[str, PerformanceMetrics]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        grouped[t.strategy].append(t)
    return {name: compute_metrics(items) for name, items in grouped.items()}
