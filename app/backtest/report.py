"""Backtest reporting.

Turns a ``BacktestResult`` (plus optional walk-forward and Monte Carlo output)
into something a human can actually evaluate: a plain-text summary, machine
readable JSON, a CSV of trades, and — when matplotlib is available — equity and
drawdown charts. No charting dependency is mandatory (spec section 105).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.backtest.engine import BacktestResult
from app.backtest.monte_carlo import MonteCarloResult
from app.backtest.walk_forward import WalkForwardResult
from app.core.types import TradeRecord
from app.monitoring.metrics import PerformanceMetrics, compute_metrics


@dataclass(slots=True)
class ReportPaths:
    json_path: Optional[Path] = None
    text_path: Optional[Path] = None
    trades_csv: Optional[Path] = None
    equity_png: Optional[Path] = None


def _regime_breakdown(trades: Sequence[TradeRecord]) -> dict[str, dict]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        grouped[str(t.regime)].append(t)
    return {name: compute_metrics(items).as_dict() for name, items in grouped.items()}


def _strategy_breakdown(trades: Sequence[TradeRecord]) -> dict[str, dict]:
    grouped: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        grouped[t.strategy].append(t)
    return {name: compute_metrics(items).as_dict() for name, items in grouped.items()}


def _exit_reason_breakdown(trades: Sequence[TradeRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for t in trades:
        counts[str(t.exit_reason)] += 1
    return dict(counts)


def build_report(
    result: BacktestResult,
    walk_forward: Optional[WalkForwardResult] = None,
    monte_carlo: Optional[MonteCarloResult] = None,
    label: str = "backtest",
) -> dict:
    payload: dict = {
        "label": label,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": result.as_dict(),
        "per_strategy": _strategy_breakdown(result.trades),
        "per_regime": _regime_breakdown(result.trades),
        "exit_reasons": _exit_reason_breakdown(result.trades),
    }
    if walk_forward is not None:
        payload["walk_forward"] = walk_forward.as_dict()
    if monte_carlo is not None:
        payload["monte_carlo"] = monte_carlo.as_dict()
    return payload


def _fmt_metrics(m: PerformanceMetrics) -> list[str]:
    d = m.as_dict()
    pf = d["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return [
        f"  trades:            {d['trades']}  (win rate {d['win_rate'] * 100:.1f}%)",
        f"  net P&L:           {d['net_pnl']:,.2f}",
        f"  expectancy/trade:  {d['expectancy']:,.4f}",
        f"  profit factor:     {pf_s}",
        f"  payoff ratio:      {d['payoff_ratio']:.2f}",
        f"  max drawdown:      {d['max_drawdown']:,.2f} ({d['max_drawdown_pct'] * 100:.2f}%)",
        f"  trade sharpe:      {d['sharpe']:.3f}",
        f"  total fees:        {d['total_fees']:,.2f}",
        f"  total slippage:    {d['total_slippage']:,.2f}",
        f"  avg hold (s):      {d['average_hold_seconds']:,.0f}",
    ]


def render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"BACKTEST REPORT: {report.get('label', 'backtest')}")
    lines.append(f"generated: {report.get('generated_at', '')}")
    lines.append("=" * 68)

    summary = report.get("summary", {})
    period = f"{summary.get('start')} -> {summary.get('end')}"
    lines.append(f"period: {period}")
    lines.append(f"symbols: {', '.join(summary.get('symbols', []))}")
    exposure = summary.get("exposure", {}) or {}
    lines.append(
        f"bars processed: {summary.get('bars_processed', 0)} | "
        f"avg open: {exposure.get('avg_open_positions', 0):.2f} | "
        f"max open: {exposure.get('max_open_positions', 0)}"
    )
    lines.append("")
    lines.append("OVERALL")
    metrics = PerformanceMetrics(**{k: v for k, v in summary.get("metrics", {}).items()
                                    if k in PerformanceMetrics.__dataclass_fields__})
    lines.extend(_fmt_metrics(metrics))

    lines.append("")
    lines.append("PER STRATEGY")
    for name, m in sorted(report.get("per_strategy", {}).items()):
        m_metric = PerformanceMetrics(**{k: v for k, v in m.items()
                                         if k in PerformanceMetrics.__dataclass_fields__})
        lines.append(f"  [{name}] net {m_metric.net_pnl:,.2f} over {m_metric.trades} trades "
                     f"(PF {'inf' if m_metric.profit_factor == float('inf') else f'{m_metric.profit_factor:.2f}'})")

    lines.append("")
    lines.append("PER REGIME")
    for name, m in sorted(report.get("per_regime", {}).items()):
        lines.append(f"  [{name}] net {m.get('net_pnl', 0):,.2f} over {m.get('trades', 0)} trades")

    lines.append("")
    lines.append("EXIT REASONS")
    for reason, count in sorted(report.get("exit_reasons", {}).items()):
        lines.append(f"  {reason}: {count}")

    if "walk_forward" in report:
        wf = report["walk_forward"]
        lines.append("")
        lines.append("WALK-FORWARD (out-of-sample)")
        lines.append(f"  windows: {wf.get('window_count', 0)} | consistency: {wf.get('consistency', 0) * 100:.1f}%")
        agg = wf.get("aggregate", {})
        lines.append(f"  OOS net: {agg.get('net_pnl', 0):,.2f} | OOS trades: {agg.get('trades', 0)} | "
                     f"OOS PF: {agg.get('profit_factor', 0)}")

    if "monte_carlo" in report:
        mc = report["monte_carlo"]
        lines.append("")
        lines.append("MONTE CARLO")
        lines.append(f"  iterations: {mc.get('iterations')} | prob(profit): {mc.get('prob_profit', 0) * 100:.1f}% | "
                     f"prob(ruin): {mc.get('prob_ruin', 0) * 100:.2f}%")
        lines.append(f"  return p05/median/p95: {mc.get('p05_return', 0) * 100:.1f}% / "
                     f"{mc.get('median_return', 0) * 100:.1f}% / {mc.get('p95_return', 0) * 100:.1f}%")
        lines.append(f"  median max DD: {mc.get('median_max_drawdown_pct', 0) * 100:.1f}% | "
                     f"worst: {mc.get('worst_max_drawdown_pct', 0) * 100:.1f}%")

    lines.append("=" * 68)
    return "\n".join(lines)


def write_trades_csv(trades: Iterable[TradeRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [t.as_dict() for t in trades]
    fieldnames = [
        "trade_id", "symbol", "strategy", "entry_price", "exit_price", "qty",
        "fees", "slippage", "gross_pnl", "net_pnl", "entry_time", "exit_time",
        "holding_time_seconds", "exit_reason", "regime", "expected_edge",
        "estimated_cost", "actual_cost",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def save_report(
    result: BacktestResult,
    out_dir: Path,
    label: str = "backtest",
    walk_forward: Optional[WalkForwardResult] = None,
    monte_carlo: Optional[MonteCarloResult] = None,
    charts: bool = True,
) -> ReportPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    paths = ReportPaths()

    report = build_report(result, walk_forward=walk_forward, monte_carlo=monte_carlo, label=label)

    paths.json_path = out_dir / f"{label}-{stamp}.json"
    paths.json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    paths.text_path = out_dir / f"{label}-{stamp}.txt"
    paths.text_path.write_text(render_text(report), encoding="utf-8")

    if result.trades:
        paths.trades_csv = write_trades_csv(result.trades, out_dir / f"{label}-{stamp}-trades.csv")

    if charts:
        try:
            paths.equity_png = _plot_equity(result, out_dir / f"{label}-{stamp}-equity.png")
        except Exception:
            # Charting is best-effort; never fail a report because of matplotlib.
            paths.equity_png = None

    return paths


def _plot_equity(result: BacktestResult, path: Path) -> Optional[Path]:
    if not result.equity_curve:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    times = [p.timestamp for p in result.equity_curve]
    equity = [p.equity for p in result.equity_curve]
    peak = []
    running = float("-inf")
    for e in equity:
        running = max(running, e)
        peak.append(running)
    drawdown = [((p - e) / p * 100.0) if p > 0 else 0.0 for p, e in zip(peak, equity)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(times, equity, color="#1f77b4", linewidth=1.2)
    ax1.set_title("Equity curve")
    ax1.set_ylabel("Equity")
    ax1.grid(alpha=0.3)
    ax2.fill_between(times, [-d for d in drawdown], color="#d62728", alpha=0.4)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


__all__ = [
    "ReportPaths",
    "build_report",
    "render_text",
    "save_report",
    "write_trades_csv",
]
