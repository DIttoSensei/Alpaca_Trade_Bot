"""Monte Carlo and bootstrap robustness analysis.

A single backtest produces a single path. The honest question is: "given the
distribution of per-trade outcomes we actually observed, how likely is this
result to be luck?" We answer with a bootstrap over the realised trade returns
plus a randomised trade-order Monte Carlo to estimate drawdown risk (spec
sections 93, 94).

Nothing here changes the strategy; it only quantifies confidence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.core.types import TradeRecord


@dataclass(slots=True)
class MonteCarloResult:
    iterations: int
    starting_equity: float
    # Distribution of final equity / total return.
    mean_return: float = 0.0
    median_return: float = 0.0
    p05_return: float = 0.0
    p95_return: float = 0.0
    # Probability the simulated result is profitable, and risk of ruin.
    prob_profit: float = 0.0
    prob_ruin: float = 0.0
    ruin_fraction: float = 0.5
    # Worst-case drawdown distribution (as a fraction of peak equity).
    median_max_drawdown_pct: float = 0.0
    worst_max_drawdown_pct: float = 0.0
    samples: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "starting_equity": self.starting_equity,
            "mean_return": round(self.mean_return, 6),
            "median_return": round(self.median_return, 6),
            "p05_return": round(self.p05_return, 6),
            "p95_return": round(self.p95_return, 6),
            "prob_profit": round(self.prob_profit, 4),
            "prob_ruin": round(self.prob_ruin, 4),
            "ruin_fraction": self.ruin_fraction,
            "median_max_drawdown_pct": round(self.median_max_drawdown_pct, 4),
            "worst_max_drawdown_pct": round(self.worst_max_drawdown_pct, 4),
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _max_drawdown_pct(pnls: Sequence[float], starting_equity: float) -> float:
    equity = starting_equity
    peak = starting_equity
    worst = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def bootstrap_returns(
    trades: Sequence[TradeRecord],
    starting_equity: float = 10_000.0,
    iterations: int = 2000,
    ruin_fraction: float = 0.5,
    seed: Optional[int] = 42,
    risk_per_trade_fraction: Optional[float] = None,
) -> MonteCarloResult:
    """Resample the realised trades WITH replacement to build return confidence.

    If ``risk_per_trade_fraction`` is provided, per-trade P&L is normalised to a
    fraction of equity (fixed-fractional) so the bootstrap is not dominated by a
    single large position size.
    """
    result = MonteCarloResult(
        iterations=iterations,
        starting_equity=starting_equity,
        ruin_fraction=ruin_fraction,
    )
    if not trades:
        return result

    rng = random.Random(seed)
    if risk_per_trade_fraction is not None and risk_per_trade_fraction > 0:
        # Express each trade as a return on the equity that was risked.
        unit = starting_equity * risk_per_trade_fraction
        pool = [t.net_pnl / unit for t in trades]
    else:
        pool = [t.net_pnl for t in trades]

    n = len(pool)
    returns: list[float] = []
    drawdowns: list[float] = []
    ruin_count = 0
    ruin_level = starting_equity * (1.0 - ruin_fraction)

    for _ in range(iterations):
        equity = starting_equity
        peak = starting_equity
        worst_dd = 0.0
        ruined = False
        for _ in range(n):
            step = pool[rng.randrange(n)]
            if risk_per_trade_fraction is not None and risk_per_trade_fraction > 0:
                equity += step * equity * risk_per_trade_fraction
            else:
                equity += step
            peak = max(peak, equity)
            if peak > 0:
                worst_dd = max(worst_dd, (peak - equity) / peak)
            if equity <= ruin_level:
                ruined = True
                break
        if ruined:
            ruin_count += 1
        returns.append((equity - starting_equity) / starting_equity)
        drawdowns.append(worst_dd)

    ordered = sorted(returns)
    ordered_dd = sorted(drawdowns)
    result.samples = sorted(returns)
    result.mean_return = sum(returns) / len(returns)
    result.median_return = _percentile(ordered, 0.5)
    result.p05_return = _percentile(ordered, 0.05)
    result.p95_return = _percentile(ordered, 0.95)
    result.prob_profit = sum(1 for r in returns if r > 0) / len(returns)
    result.prob_ruin = ruin_count / iterations
    result.median_max_drawdown_pct = _percentile(ordered_dd, 0.5)
    result.worst_max_drawdown_pct = ordered_dd[-1]
    return result


def shuffle_trades(
    trades: Sequence[TradeRecord],
    starting_equity: float = 10_000.0,
    iterations: int = 2000,
    ruin_fraction: float = 0.5,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """Randomise trade ORDER to stress-test path-dependent drawdown."""
    result = MonteCarloResult(
        iterations=iterations,
        starting_equity=starting_equity,
        ruin_fraction=ruin_fraction,
    )
    if not trades:
        return result

    rng = random.Random(seed)
    pnls = [t.net_pnl for t in trades]
    returns: list[float] = []
    drawdowns: list[float] = []
    ruin_count = 0
    ruin_level = starting_equity * (1.0 - ruin_fraction)

    for _ in range(iterations):
        order = pnls[:]
        rng.shuffle(order)
        equity = starting_equity
        peak = starting_equity
        worst_dd = 0.0
        ruined = False
        for pnl in order:
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                worst_dd = max(worst_dd, (peak - equity) / peak)
            if equity <= ruin_level:
                ruined = True
                break
        if ruined:
            ruin_count += 1
        returns.append((equity - starting_equity) / starting_equity)
        drawdowns.append(worst_dd)

    ordered = sorted(returns)
    ordered_dd = sorted(drawdowns)
    result.samples = ordered
    result.mean_return = sum(returns) / len(returns)
    result.median_return = _percentile(ordered, 0.5)
    result.p05_return = _percentile(ordered, 0.05)
    result.p95_return = _percentile(ordered, 0.95)
    result.prob_profit = sum(1 for r in returns if r > 0) / len(returns)
    result.prob_ruin = ruin_count / iterations
    result.median_max_drawdown_pct = _percentile(ordered_dd, 0.5)
    result.worst_max_drawdown_pct = ordered_dd[-1]
    return result


__all__ = ["MonteCarloResult", "bootstrap_returns", "shuffle_trades"]
