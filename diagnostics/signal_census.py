"""Signal census: WHY do the five streaming strategies never trade on real data?

For every execution bar (after warmup) on the real cached H1 series, this tool
mirrors the exact pipeline decision path and records, per strategy:

  * how often the strategy is regime-eligible at all;
  * how often its RAW logic would fire a BUY (bypassing the regime gate);
  * how often that raw BUY clears the configured ``signal_threshold``;
  * the distribution (mean/max) of the raw score.

It also tallies the regime mix and how often the *ensemble* is actionable.

Read-only w.r.t. the bot. Outputs diagnostics/signal_census.txt.
"""

from __future__ import annotations

import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.assembly import build_stack  # noqa: E402
from app.config import load_config  # noqa: E402
from app.core.enums import Action, Timeframe  # noqa: E402
from app.data.candle_store import CandleStore, aggregate  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402
from app.decision.signal_engine import SignalEngine  # noqa: E402
from app.config.strategy_config import StrategyConfig  # noqa: E402
from app.strategies import STRATEGY_CLASSES  # noqa: E402
from app.strategies.base import ScoreBuilder  # noqa: E402

logging.basicConfig(level="WARNING")

STRATS = ["trend_pullback", "breakout", "mean_reversion", "momentum", "recovery"]


def main() -> int:
    cfg = load_config()
    # Use the CONFIGURED execution timeframe. Feeding a series on a different
    # timeframe than context.execution leaves the feature store empty (the bug
    # that made every earlier arm look like 100% RANGE / zero trades).
    tf = cfg.symbols.timeframes.get("execution", Timeframe.M15)
    cache = ParquetCache(cfg.settings.paths.data)
    print(f"[census] execution timeframe = {tf}")
    # Higher-timeframe context roles, re-aggregated from the execution series
    # exactly as app.backtest.engine._preload does (no-lookahead consistency).
    ctx_roles = [stack_tf for stack_tf in ("confirmation", "regime", "macro")
                 if cfg.symbols.timeframes.get(stack_tf) not in (None, tf)]

    sc = StrategyConfig()
    for n in STRATS:
        p = sc.get(n)
        if p is not None:
            p.enabled = True

    regime_counts: Counter = Counter()
    per = {n: {"eligible": 0, "raw_buy": 0, "raw_sell": 0, "passed": 0, "scores": []} for n in STRATS}
    ens_any_signal = 0
    ens_actionable = 0
    total_bars = 0

    for sym in cfg.symbols.enabled():
        try:
            bars = cache.load(sym, tf)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sym}: {exc}")
            continue
        bars = sorted(bars, key=lambda c: c.timestamp)
        if not bars:
            continue

        store = CandleStore(max_bars=10**9)
        stack = build_stack(cfg, candles=store)
        store.merge(bars)
        for role in ctx_roles:
            htf = cfg.symbols.timeframes.get(role)
            if htf is None or htf == tf:
                continue
            agg = aggregate(bars, htf)
            if agg:
                store.merge(agg)
        stack.features.invalidate()

        strategies = {n: STRATEGY_CLASSES[n](params=sc.get(n)) for n in STRATS}
        engine = SignalEngine(strategies, sc)
        ref = stack.reference_symbol
        warmup = 220

        for i in range(warmup, len(bars)):
            ts = bars[i].timestamp
            btc_feats = None
            rs = None
            if sym != ref:
                try:
                    btc_feats = stack.features.snapshot_at(ref, tf, i, ts)
                    rs = stack.features.relative_strength(
                        sym, ref, stack.context.regime, lookback_bars=4, as_of=ts
                    )
                except Exception:  # noqa: BLE001
                    pass
            state, _rr = stack.pipeline.build_state(
                sym, ts, i, btc_features=btc_feats, relative_strength=rs
            )
            total_bars += 1
            regime_counts[str(state.regime)] += 1

            for n in STRATS:
                st = strategies[n]
                if not (st.capability.allows_regime(state.regime) and st.capability.supports_symbol(sym)):
                    continue
                per[n]["eligible"] += 1
                builder = ScoreBuilder()
                try:
                    act = st._evaluate(state, builder)
                except Exception:  # noqa: BLE001
                    continue
                score, _ = builder.capped_scores()
                if act == Action.BUY:
                    per[n]["raw_buy"] += 1
                    per[n]["scores"].append(score)
                    thr = getattr(st.params, "signal_threshold", 6.0)
                    if score >= thr:
                        per[n]["passed"] += 1
                elif act == Action.SELL:
                    per[n]["raw_sell"] += 1

            es = engine.evaluate(state)
            actionable = es.actionable()
            if es.signals and any(s.action != Action.NO_SIGNAL for s in es.signals):
                ens_any_signal += 1
            if actionable:
                ens_actionable += 1

    # -- render -----------------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("SIGNAL CENSUS (real H1 candles) -- why the streaming strategies don't trade")
    lines.append(f"execution bars evaluated (post-warmup): {total_bars:,}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("regime mix over evaluated bars:")
    for reg, cnt in regime_counts.most_common():
        pct = cnt / total_bars * 100 if total_bars else 0.0
        lines.append(f"  {reg:<18} {cnt:>8,}  {pct:6.1f}%")
    lines.append("")
    lines.append(f"ensemble non-NO_SIGNAL bars : {ens_any_signal:,}")
    lines.append(f"ensemble ACTIONABLE bars    : {ens_actionable:,}")
    lines.append("")
    lines.append(f"{'strategy':<18}{'eligible':>10}{'raw_buy':>10}{'raw_sell':>10}{'passed':>10}"
                 f"{'mean_score':>12}{'max_score':>11}")
    lines.append("-" * 88)
    for n in STRATS:
        d = per[n]
        scores = d["scores"]
        mean_s = f"{statistics.mean(scores):.2f}" if scores else "-"
        max_s = f"{max(scores):.2f}" if scores else "-"
        lines.append(
            f"{n:<18}{d['eligible']:>10,}{d['raw_buy']:>10,}{d['raw_sell']:>10,}"
            f"{d['passed']:>10,}{mean_s:>12}{max_s:>11}"
        )
    lines.append("-" * 88)
    lines.append("")
    lines.append("Reading: 'raw_buy' = strategy logic fired BUY when its regime was allowed;")
    lines.append("'passed' = raw BUY also cleared signal_threshold (6.0). If raw_buy is ~0 the")
    lines.append("logic never fires on real data; if raw_buy>0 but passed=0 the score gate is")
    lines.append("unreachable. Eligible=0 means the regime never permits the strategy at all.")
    lines.append("=" * 88)

    text = "\n".join(lines)
    out_path = Path(__file__).resolve().parent / "signal_census.txt"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nTXT: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
