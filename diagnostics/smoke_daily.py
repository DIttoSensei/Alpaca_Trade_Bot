"""Smoke test: does the default (daily-trend) config actually trade on real data?

Not a verdict on profitability -- just a wiring/health check that the daily-trend
edge is registered, produces signals, and that exits/trend-mode behave.

    python diagnostics/smoke_daily.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest import BacktestConfig  # noqa: E402
from app.config import load_config  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402
from app.strategies import STRATEGY_CLASSES, build_strategies  # noqa: E402
from scripts.compare import ARM_MULTI, run_arm  # noqa: E402


def main() -> int:
    cfg = load_config()
    print(f"daily_trend switch   : {cfg.strategies.daily_trend}")
    print(f"trend_daily params   : enabled={cfg.strategies.trend_daily.enabled} "
          f"min_data={cfg.strategies.trend_daily.minimum_data}")
    print(f"'trend_daily' in reg : {'trend_daily' in STRATEGY_CLASSES}")
    built = build_strategies(cfg.strategies)
    print(f"build_strategies()   : {sorted(built)}")
    print(f"grid enabled default : {cfg.strategies.grid.enabled}")

    tf = cfg.symbols.timeframes.get("execution", Timeframe.M15)
    cache = ParquetCache(cfg.settings.paths.data)
    by_key = {}
    for sym in cfg.symbols.enabled():
        try:
            bars = cache.load(sym, tf)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sym}: {exc}")
            continue
        if bars:
            by_key[(sym, tf)] = sorted(bars, key=lambda c: c.timestamp)
    print(f"bars loaded          : {{sym: len for {len(by_key)} series}} -> "
          f"{ {s: len(b) for (s, _t), b in by_key.items()} }")
    if not by_key:
        print("NO DATA")
        return 1

    bt = BacktestConfig(starting_equity=10_000.0, warmup_bars=210)
    res = run_arm(ARM_MULTI, by_key, cfg, tf, bt)
    m = res.metrics
    print("\n--- default (daily-trend) arm on real M15 ---")
    print(f"trades               : {len(res.trades)}")
    print(f"net_pnl              : {m.net_pnl:.2f}")
    print(f"win_rate             : {m.win_rate:.3f}")
    print(f"profit_factor        : {m.profit_factor}")
    per_strat = defaultdict(int)
    exits = defaultdict(int)
    holds = []
    for t in res.trades:
        per_strat[t.strategy] += 1
        exits[str(t.exit_reason)] += 1
        holds.append(t.holding_time_seconds / 86400.0)
    print(f"per strategy         : {dict(per_strat)}")
    print(f"exit reasons         : {dict(exits)}")
    if holds:
        print(f"avg hold (days)      : {sum(holds) / len(holds):.2f}  max={max(holds):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
