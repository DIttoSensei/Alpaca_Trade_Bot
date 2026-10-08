"""Scenario sweep: is ANY realistic configuration net-profitable on real data?

Tests the trend_pullback strategy (the only one that fires often) across a grid
of:
  * entry score threshold (6 / 7 / 8)
  * target R multiple (2 / 3 / 4)
  * fee regime (baseline 0.25%/side, low 0.05%/side, ZERO cost)

The ZERO-cost row is the key diagnostic: if it is still negative, the signal has
no gross edge and no fee/parameter choice can rescue it. If ZERO is positive but
baseline is negative, costs are the killer and only dramatic fee reduction helps.

Runs on the configured execution timeframe (M15). Outputs diagnostics/scenario_sweep.txt.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest import BacktestConfig  # noqa: E402
from app.config import load_config  # noqa: E402
from app.config.settings import FeeModel  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402
from app.monitoring.metrics import compute_metrics  # noqa: E402
from scripts.compare import ARM_TREND, run_arm  # noqa: E402

logging.basicConfig(level="ERROR")

# (label, threshold, target_r, taker, slip, spread)
SCENARIOS = [
    ("baseline thr6 R2", 6.0, 2.0, 0.0025, 0.0010, 0.0005),
    ("thr7 R2", 7.0, 2.0, 0.0025, 0.0010, 0.0005),
    ("thr8 R2", 8.0, 2.0, 0.0025, 0.0010, 0.0005),
    ("thr6 R3", 6.0, 3.0, 0.0025, 0.0010, 0.0005),
    ("thr6 R4", 6.0, 4.0, 0.0025, 0.0010, 0.0005),
    ("thr7 R3", 7.0, 3.0, 0.0025, 0.0010, 0.0005),
    ("LOWFEE thr7 R3", 7.0, 3.0, 0.0005, 0.0005, 0.0002),
    ("ZEROCOST thr6 R2", 6.0, 2.0, 0.0, 0.0, 0.0),
    ("ZEROCOST thr7 R3", 7.0, 3.0, 0.0, 0.0, 0.0),
    ("ZEROCOST thr8 R4", 8.0, 4.0, 0.0, 0.0, 0.0),
]


def _only_trend(cfg) -> None:
    for name in ("breakout", "mean_reversion", "momentum", "recovery"):
        p = cfg.strategies.get(name)
        if p is not None:
            p.enabled = False
    cfg.strategies.trend_pullback.enabled = True
    cfg.strategies.trend = True
    cfg.strategies.grid.enabled = False


def main() -> int:
    base = load_config()
    tf = base.symbols.timeframes.get("execution", Timeframe.M15)
    cache = ParquetCache(base.settings.paths.data)
    by_key = {}
    for sym in base.symbols.enabled():
        try:
            bars = cache.load(sym, tf)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sym}: {exc}")
            continue
        if bars:
            by_key[(sym, tf)] = sorted(bars, key=lambda c: c.timestamp)

    bt = BacktestConfig(starting_equity=10_000.0, warmup_bars=210)
    rows = []
    for label, thr, tgt, taker, slip, spread in SCENARIOS:
        cfg = copy.deepcopy(base)
        _only_trend(cfg)
        cfg.strategies.trend_pullback.signal_threshold = thr
        cfg.risk.target_r_multiple = tgt
        cfg.settings.fees = FeeModel(
            maker_fee=taker, taker_fee=taker, estimated_slippage=slip, spread_cost=spread
        )
        try:
            res = run_arm(ARM_TREND, by_key, cfg, tf, bt)
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: ERROR {exc}")
            continue
        m = compute_metrics(res.trades, 10_000.0)
        gross = sum(t.gross_pnl for t in res.trades)
        rows.append((label, len(res.trades), m.win_rate, gross, m.net_pnl,
                     m.profit_factor, m.max_drawdown_pct))
        print(f"{label:<20} trades={len(res.trades):>4} win={m.win_rate*100:5.1f}% "
              f"gross={gross:>9.0f} net={m.net_pnl:>9.0f} PF={m.profit_factor:.2f}")

    lines = [
        "=" * 92,
        "SCENARIO SWEEP -- trend_pullback on real M15 history (730d, 3 symbols)",
        "=" * 92,
        f"{'scenario':<20}{'trades':>7}{'win%':>8}{'gross':>11}{'net':>11}{'PF':>8}{'maxDD%':>9}",
        "-" * 92,
    ]
    for label, n, wr, gross, net, pf, dd in rows:
        lines.append(f"{label:<20}{n:>7}{wr*100:>7.1f}%{gross:>11,.0f}{net:>11,.0f}"
                     f"{pf:>8.2f}{dd*100:>8.1f}%")
    lines.append("-" * 92)
    lines.append("If ZEROCOST rows are negative -> no gross edge (unfixable by fees).")
    lines.append("If ZEROCOST positive but baseline negative -> costs dominate.")
    lines.append("=" * 92)
    text = "\n".join(lines)
    (Path(__file__).resolve().parent / "scenario_sweep.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
