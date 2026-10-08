"""End-to-end profitability & robustness validation over REAL cached candles.

Honest diagnostics. It answers the user's question directly: does this bot make
money, under which scenarios, and why? It tests:

  * the SHIPPED default configuration (grid only -- the five strategies are
    disabled by default) via the real grid backtester;
  * each of the five strategies in ISOLATION (full composer: BTC + relative
    strength + expected-edge/cost filters + risk), and all five as an ensemble;
  * the ungated legacy benchmark (no regime/BTC/cost filters) for reference;

on real BTC/USD, ETH/USD, SOL/USD history, using the SAME cost model and risk
engine the live loop uses. It records gross vs net P&L (cost drag), per-year
robustness, Monte Carlo bootstrap, rejection reasons and per-symbol grid detail.

Outputs diagnostics/validation_results.json and .txt. Read-only w.r.t. the bot.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest import BacktestConfig, bootstrap_returns, build_report  # noqa: E402
from app.config import BotConfig, load_config  # noqa: E402
from app.config.strategy_config import StrategyConfig  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402
from app.monitoring.metrics import compute_metrics  # noqa: E402
from app.strategies.grid import run_grid_backtest  # noqa: E402
from scripts.compare import (  # noqa: E402
    ARM_CURRENT,
    ARM_MULTI,
    ARM_TREND,
    run_arm,
)

logger = logging.getLogger("diagnostics.validate")

SINGLES = ["trend_pullback", "breakout", "mean_reversion", "momentum", "recovery"]


def load_cached(cfg: BotConfig, tf: Timeframe) -> dict:
    """Load the exact-timeframe cached series for the enabled universe."""
    cache = ParquetCache(cfg.settings.paths.data)
    by_key: dict = {}
    for sym in cfg.symbols.enabled():
        try:
            bars = cache.load(sym, tf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache read failed %s %s: %s", sym, tf, exc)
            bars = []
        if bars:
            by_key[(sym, tf)] = bars
    return by_key


def enabled_config(cfg: BotConfig, names: list[str]) -> BotConfig:
    """A BotConfig with exactly ``names`` of the five strategies enabled.

    IMPORTANT: ``StrategyConfig`` reuses the same attribute name for the master
    switch (bool) of the first strategy and for the ``StrategyParams`` object of
    the others, so the later dataclass fields SHADOW the boolean switches. The
    effective per-strategy gate used by the engine is therefore ``params.enabled``
    (see ``app.strategies.build_strategies`` / ``eligible_strategies``). We flip
    that flag for every named strategy and disable all others.
    """
    sc = StrategyConfig()
    # NOTE: the `grid` field is a GridParams object (not a bool switch); set its
    # `.enabled` flag. `grid_enabled()` reads `config.grid and config.grid.enabled`.
    sc.grid.enabled = False  # grid is not part of the streaming ensemble
    for n in SINGLES:
        p = sc.get(n)
        if p is not None:
            p.enabled = n in names
    # `trend` is the one master switch that is NOT shadowed; keep it consistent.
    sc.trend = "trend_pullback" in names
    return BotConfig(
        settings=cfg.settings,
        symbols=cfg.symbols,
        strategies=sc,
        risk=cfg.risk,
        regime=cfg.regime,
    )


def _rej_tally(rejections) -> dict:
    t: dict[str, int] = defaultdict(int)
    for r in rejections:
        t[str(r.reason)] += 1
    return dict(sorted(t.items(), key=lambda kv: -kv[1]))


def _year_breakdown(trades) -> dict:
    g: dict[int, list] = defaultdict(list)
    for t in trades:
        g[t.entry_time.year].append(t)
    out: dict = {}
    for y, ts in sorted(g.items()):
        m = compute_metrics(ts)
        pf = None if m.profit_factor == float("inf") else round(m.profit_factor, 3)
        out[str(y)] = {
            "trades": m.trades,
            "net_pnl": round(m.net_pnl, 2),
            "win_rate": round(m.win_rate, 4),
            "profit_factor": pf,
        }
    return out


def summarize_arm(arm: str, result, equity: float, mc_iter: int) -> dict:
    m = result.metrics
    trades = result.trades
    gross = sum(t.gross_pnl for t in trades)
    net = sum(t.net_pnl for t in trades)
    mc = bootstrap_returns(trades, starting_equity=equity, iterations=mc_iter)
    report = build_report(result, label=arm)
    return {
        "arm": arm,
        "trades": len(trades),
        "gross_pnl": round(gross, 2),
        "net_pnl": round(net, 2),
        "return_pct": round(net / equity * 100, 4) if equity else 0.0,
        "cost_drag": round(m.total_fees + m.total_slippage, 2),
        "win_rate": round(m.win_rate, 4),
        "profit_factor": (None if m.profit_factor == float("inf") else round(m.profit_factor, 3)),
        "expectancy": round(m.expectancy, 4),
        "average_win": round(m.average_win, 2),
        "average_loss": round(m.average_loss, 2),
        "max_dd_pct": round(m.max_drawdown_pct, 4),
        "sharpe": round(m.sharpe, 3),
        "fees": round(m.total_fees, 2),
        "slippage": round(m.total_slippage, 2),
        "avg_hold_s": round(m.average_hold_seconds, 0),
        "regime_counts": dict(result.regime_counts),
        "exit_reasons": report.get("exit_reasons", {}),
        "per_strategy": {
            k: {"trades": v["trades"], "net_pnl": v["net_pnl"]}
            for k, v in report.get("per_strategy", {}).items()
        },
        "n_rejections": len(result.rejections),
        "top_rejections": dict(list(_rej_tally(result.rejections).items())[:8]),
        "monte_carlo": mc.as_dict(),
        "by_year": _year_breakdown(trades),
    }


def summarize_grid(tfname: str, result, equity: float) -> dict:
    s = result.summary()
    return {
        "arm": f"grid[{tfname}]",
        "trades": s["trades"],
        "net_pnl": round(s["net_pnl"], 2),
        "return_pct": round(s["return_pct"], 4),
        "base_equity": s["base_equity"],
        "peak_committed": round(s["peak_committed"], 2),
        "regime_exit_flattens": s["regime_exit_flattens"],
        "hard_bound_triggers": s["hard_bound_triggers"],
        "per_symbol": {
            sym: {
                "trades": len(r.trades),
                "fills": len(r.fills),
                "final_equity": round(r.final_equity, 2),
                "skipped": r.levels_placed_skipped,
                "recenters": r.recenter_count,
            }
            for sym, r in result.per_symbol.items()
        },
    }


def _parse_tf(name: str) -> Timeframe:
    token = name.strip().upper()
    if token in Timeframe.__members__:
        return Timeframe[token]
    return Timeframe(name.strip().lower())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profitability & robustness validation")
    ap.add_argument(
        "--tfs", nargs="*", default=None,
        help="execution timeframes (default: the configured execution timeframe)",
    )
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm keys")
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load_config()
    # The strategy stack resolves features on ``context.execution`` (M15 by
    # default), so the loaded series MUST be on that timeframe -- otherwise the
    # feature snapshot is empty and every regime degrades to RANGE.
    exec_tf = cfg.symbols.timeframes.get("execution", Timeframe.M15)
    if not args.tfs:
        args.tfs = [exec_tf.name]
    bt = BacktestConfig(starting_equity=args.equity, warmup_bars=210)
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity": args.equity,
        "execution_timeframe": exec_tf.name,
        "fees": {
            "taker_fee": cfg.settings.fees.taker_fee,
            "slippage": cfg.settings.fees.estimated_slippage,
            "spread": cfg.settings.fees.spread_cost,
            "round_trip_cost": cfg.settings.fees.round_trip_cost(),
        },
        "scenarios": {},
    }

    for tfname in args.tfs:
        try:
            tf = _parse_tf(tfname)
        except Exception as exc:  # noqa: BLE001
            logger.error("unknown timeframe %r: %s", tfname, exc)
            continue
        by_key = load_cached(cfg, tf)
        if not by_key:
            logger.warning("no cached data for %s; skipping", tfname)
            continue
        scenario: dict = {
            "tf": tf.name,
            "bars": {sym: len(b) for (sym, _tf), b in by_key.items()},
            "arms": {},
        }

        arm_specs: list[tuple[str, str, list[str] | None]] = []
        want = args.arms
        if not want or "current_bot" in want:
            arm_specs.append(("current_bot", ARM_CURRENT, None))
        if not want or "v2_trend_only" in want:
            arm_specs.append(("v2_trend_only", ARM_TREND, ["trend_pullback"]))
        if not want or "v2_multi" in want:
            arm_specs.append(("v2_multi", ARM_MULTI, SINGLES))
        for n in SINGLES:
            key = f"only_{n}"
            if not want or key in want:
                arm_specs.append((key, ARM_MULTI, [n]))

        for key, display, names in arm_specs:
            cfg_arm = cfg if names is None else enabled_config(cfg, names)
            try:
                res = run_arm(display, by_key, cfg_arm, tf, bt)
                scenario["arms"][key] = summarize_arm(key, res, args.equity, args.mc)
                logger.info("%s %s: %d trades net=%.2f", tf.name, key, len(res.trades), res.metrics.net_pnl)
            except Exception as exc:  # noqa: BLE001
                logger.exception("arm %s failed: %s", key, exc)
                scenario["arms"][key] = {"error": str(exc)}

        # Grid (the only strategy enabled by default) via its dedicated runner.
        if not want or "grid" in want:
            try:
                gres = run_grid_backtest(
                    by_key, cfg, starting_equity=args.equity, collection_timeframe=tf
                )
                scenario["arms"]["grid"] = summarize_grid(tf.name, gres, args.equity)
                logger.info("%s grid: %d trades net=%.2f", tf.name,
                            gres.summary()["trades"], gres.summary()["net_pnl"])
            except Exception as exc:  # noqa: BLE001
                logger.exception("grid failed: %s", exc)
                scenario["arms"]["grid"] = {"error": str(exc)}

        out["scenarios"][tf.name] = scenario

    # -- write artefacts -------------------------------------------------
    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "validation_results.json"
    txt_path = out_dir / "validation_results.txt"
    json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(render(out), encoding="utf-8")
    print(render(out))
    print(f"\nJSON: {json_path}")
    print(f"TXT : {txt_path}")
    return 0


def render(out: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("PROFITABILITY & ROBUSTNESS VALIDATION (real cached candles)")
    lines.append(f"generated: {out.get('generated_at')}   equity: {out.get('equity'):,.0f}")
    f = out.get("fees", {})
    lines.append(
        f"cost model: taker {f.get('taker_fee')} + slip {f.get('slippage')} + spread "
        f"{f.get('spread')}  => round-trip {f.get('round_trip_cost'):.4f} "
        f"({f.get('round_trip_cost', 0) * 100:.2f}%)"
    )
    lines.append("=" * 96)
    for tfname, sc in out.get("scenarios", {}).items():
        lines.append("")
        lines.append(f"### EXECUTION TIMEFRAME {tfname}   bars: {sc.get('bars')}")
        lines.append(
            f"{'arm':<18}{'trades':>7}{'win%':>8}{'gross':>11}{'net':>11}"
            f"{'ret%':>8}{'PF':>7}{'maxDD%':>8}{'P(prof)':>9}{'P(ruin)':>8}"
        )
        lines.append("-" * 96)
        for arm, d in sc.get("arms", {}).items():
            if "error" in d:
                lines.append(f"{arm:<18} ERROR: {d['error'][:60]}")
                continue
            pf = d.get("profit_factor")
            pf_s = "inf" if pf is None else f"{pf:.2f}"
            mc = d.get("monte_carlo", {})
            lines.append(
                f"{arm:<18}{d.get('trades', 0):>7}{d.get('win_rate', 0) * 100:>7.1f}%"
                f"{d.get('gross_pnl', 0):>11,.0f}{d.get('net_pnl', 0):>11,.0f}"
                f"{d.get('return_pct', 0):>8.2f}{pf_s:>7}"
                f"{d.get('max_dd_pct', 0) * 100:>7.2f}%"
                f"{mc.get('prob_profit', 0) * 100:>8.1f}%{mc.get('prob_ruin', 0) * 100:>7.2f}%"
            )
        lines.append("-" * 96)
        # per-year robustness for arms with trades
        for arm, d in sc.get("arms", {}).items():
            if "error" in d or not d.get("by_year"):
                continue
            by = d["by_year"]
            parts = [f"{y}: net={v['net_pnl']:+.0f} ({v['trades']}t)" for y, v in by.items()]
            lines.append(f"  by-year {arm:<16} " + " | ".join(parts))
        # grid detail
        g = sc.get("arms", {}).get("grid")
        if g and "error" not in g:
            lines.append(
                f"  grid detail: trades={g['trades']} net={g['net_pnl']} "
                f"flattens={g['regime_exit_flattens']} hardbound={g['hard_bound_triggers']} "
                f"peak_committed={g['peak_committed']}"
            )
            for sym, v in g.get("per_symbol", {}).items():
                lines.append(
                    f"    {sym:<9} trades={v['trades']:<4} fills={v['fills']:<5} "
                    f"final={v['final_equity']:>10,.2f} skipped_recenters={v['skipped']}"
                )
        # rejection + exit reason detail for the multi arm
        mv = sc.get("arms", {}).get("v2_multi")
        if mv and "error" not in mv:
            lines.append(f"  v2_multi rejections (total {mv.get('n_rejections')}): {mv.get('top_rejections')}")
            lines.append(f"  v2_multi exit reasons: {mv.get('exit_reasons')}")
            lines.append(f"  v2_multi per-strategy: {mv.get('per_strategy')}")
    lines.append("=" * 96)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
