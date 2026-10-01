"""Head-to-head comparison: CURRENT BOT vs V2 TREND-ONLY vs V2 MULTI-STRATEGY.

This is the final deliverable harness (spec section 111). It runs every arm over
the SAME candles, the SAME cost model and the SAME risk engine so the only thing
that changes is the decision logic:

  * ``current_bot``   - the original behaviour: only the legacy trend-pullback
    logic, no regime awareness, no BTC/relative-strength filters and no
    cost/edge filter. ATR-based sizing and stops are retained, because a
    benchmark that never sizes or stops would not be a fair comparison.
  * ``v2_trend_only`` - the V2 stack with a single ``trend_pullback`` strategy,
    the full composer (BTC + relative-strength + cost/edge filters) and regime
    gating.
  * ``v2_multi``      - the full V2 ensemble (trend_pullback, breakout,
    mean_reversion, momentum, recovery) with the same composer and gating.
  * ``v2_multi_ml``   - ``v2_multi`` plus the optional veto-only ML meta-filter
    (only when ``--with-ml`` and a trained model exists).

Results are written to ``reports/comparison.json`` and ``reports/comparison.txt``.

Usage::

    # Real data (requires Alpaca credentials in .env)
    python scripts/compare.py --days 730
    python scripts/compare.py --days 730 --with-ml --walk-forward

    # Offline harness validation (deterministic synthetic candles, no network)
    python scripts/compare.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.assembly import StrategyStack, build_context, build_meta_filter  # noqa: E402
from app.backtest import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    bootstrap_returns,
    build_report,
)
from app.backtest.walk_forward import _slice, _split_events  # noqa: E402
from app.config import BotConfig, load_config  # noqa: E402
from app.config.strategy_config import StrategyConfig, StrategyParams  # noqa: E402
from app.core.enums import Action, Regime, RejectReason, Timeframe  # noqa: E402
from app.core.types import Candle, PositionRecord, Signal, TradeDecision, utcnow  # noqa: E402
from app.data.candle_store import CandleStore  # noqa: E402
from app.data.feature_store import FeatureStore  # noqa: E402
from app.decision.ensemble import Ensemble  # noqa: E402
from app.decision.pipeline import TradingPipeline  # noqa: E402
from app.decision.signal_engine import SignalEngine, SignalSet  # noqa: E402
from app.decision.trade_decision import DecisionOutcome, TradeDecisionComposer  # noqa: E402
from app.monitoring.metrics import compute_metrics  # noqa: E402
from app.regimes.detector import RegimeDetector  # noqa: E402
from app.risk.engine import RiskEngine  # noqa: E402
from app.strategies import build_strategies  # noqa: E402
from app.strategies.base import Capability  # noqa: E402
from app.strategies.legacy import LegacyTrendPullbackStrategy  # noqa: E402
from app.strategies.trend_pullback import TrendPullbackStrategy  # noqa: E402

logger = logging.getLogger("scripts.compare")

# -- arm identifiers ------------------------------------------------------
ARM_CURRENT = "current_bot"
ARM_TREND = "v2_trend_only"
ARM_MULTI = "v2_multi"
ARM_MULTI_ML = "v2_multi_ml"

ARM_DESCRIPTIONS = {
    ARM_CURRENT: "Original bot: legacy trend-pullback only, no regime/cost filters",
    ARM_TREND: "V2 trend-only: trend_pullback + full composer + regime gating",
    ARM_MULTI: "V2 multi-strategy: 5-strategy ensemble + full composer",
    ARM_MULTI_ML: "V2 multi-strategy + veto-only ML meta-filter",
}


class _LegacyStrategyConfig(StrategyConfig):
    """Strategy config used by the CURRENT-BOT arm.

    The shipped config gives ``legacy_trend_pullback`` an ensemble weight of 0.0
    (it is a benchmark that normally never trades). To benchmark it we must give
    it a real, single-strategy weight so the ensemble can act on it.
    """

    def all_params(self) -> dict[str, StrategyParams]:
        params = super().all_params()
        params["legacy_trend_pullback"] = StrategyParams(
            signal_threshold=6.0, risk_multiplier=1.0, ensemble_weight=1.0
        )
        return params


class _SingleStrategyEngine:
    """Runs exactly one strategy, ignoring the regime->strategy eligibility map.

    Used for the CURRENT-BOT arm so the benchmark is not silently granted V2's
    regime-awareness machinery.
    """

    def __init__(self, strategy) -> None:
        self.strategy = strategy

    def evaluate(self, state) -> SignalSet:
        try:
            signal = self.strategy.evaluate(state)
        except Exception as exc:  # noqa: BLE001 - a broken strategy must not kill the run
            signal = Signal(
                action=Action.NO_SIGNAL,
                confidence=0.0,
                score=0.0,
                strategy_name=getattr(self.strategy, "name", "legacy"),
                reason=f"strategy error: {exc}",
            )
        return SignalSet(symbol=state.symbol, regime=str(state.regime), signals=[signal])


class _LegacyComposer(TradeDecisionComposer):
    """Composer that mirrors the ORIGINAL bot.

    It keeps risk sizing and the ATR stop/target (so trades are still bounded),
    but deliberately omits the V2 improvements: the panic gate, the BTC filter,
    the relative-strength filter and the expected-edge/cost filter.
    """

    def compose(  # type: ignore[override]
        self,
        state,
        ensemble,
        equity: float,
        positions: dict[str, PositionRecord],
        return_series: Optional[dict[str, list[float]]] = None,
        buying_power: Optional[float] = None,
        recent_volume: Optional[float] = None,
        ml_veto: bool = False,
    ) -> DecisionOutcome:
        if ensemble.action is not Action.BUY:
            return self._reject(state, ensemble, RejectReason.NO_SIGNAL, f"ensemble action {ensemble.action}")

        existing = positions.get(state.symbol)
        if existing is not None and existing.qty != 0:
            return self._reject(state, ensemble, RejectReason.DUPLICATE_POSITION, "position already open")

        price = state.features.values.get("close")
        atr = state.features.values.get("atr")
        if price is None or atr is None or price <= 0:
            return self._reject(state, ensemble, RejectReason.NO_SIGNAL, "missing price/ATR")

        risk_plan = self.risk_engine.plan(
            equity=equity,
            symbol=state.symbol,
            entry_price=price,
            atr=atr,
            strategy_name="legacy_trend_pullback",
            positions=positions,
            return_series=return_series,
            buying_power=buying_power,
            recent_volume=recent_volume,
        )
        if not risk_plan.allowed or risk_plan.stop_plan is None or risk_plan.sizing is None:
            return self._reject(
                state,
                ensemble,
                risk_plan.reject_reason or RejectReason.NO_SIGNAL,
                risk_plan.detail or "risk rejected",
            )

        stop_plan = risk_plan.stop_plan
        decision = TradeDecision(
            symbol=state.symbol,
            side="buy",
            quantity=risk_plan.sizing.quantity,
            entry_type="market",
            limit_price=None,
            stop_price=stop_plan.stop_price,
            target_price=stop_plan.target_price,
            strategy="legacy_trend_pullback",
            score=ensemble.weighted_score,
            expected_edge=0.0,
            regime=state.regime,
            reason=ensemble.reason,
            risk_amount=risk_plan.risk_amount,
            risk_multiplier=risk_plan.effective_risk_multiplier,
            feature_snapshot=state.features,
            estimated_cost=0.0,
            metadata={"cost": None, "legacy": True},
        )
        return DecisionOutcome(decision=decision, rejection=None)


# -- stack construction per arm ------------------------------------------
def _arm_config(cfg: BotConfig, arm: str) -> BotConfig:
    """Return a BotConfig whose strategy config matches the arm's intent."""
    strategies = _LegacyStrategyConfig() if arm == ARM_CURRENT else cfg.strategies
    return BotConfig(
        settings=cfg.settings,
        symbols=cfg.symbols,
        strategies=strategies,
        risk=cfg.risk,
        regime=cfg.regime,
    )


def _full_composer(cfg: BotConfig, risk_engine: RiskEngine, reference_symbol: str) -> TradeDecisionComposer:
    return TradeDecisionComposer(
        fees=cfg.settings.fees,
        risk=cfg.risk,
        risk_engine=risk_engine,
        reference_symbol=reference_symbol,
        require_btc_confirmation=cfg.strategies.require_btc_confirmation,
        min_relative_strength=cfg.strategies.min_relative_strength,
    )


def build_arm_stack(cfg: BotConfig, arm: str, with_ml: bool = False) -> StrategyStack:
    """Construct a fresh strategy stack configured for a single comparison arm."""
    arm_cfg = _arm_config(cfg, arm)
    candles = CandleStore()
    features = FeatureStore(candles)
    context = build_context(cfg.symbols)
    reference_symbol = cfg.symbols.reference_symbol() or "BTC/USD"
    regime_detector = RegimeDetector(cfg.regime)
    risk_engine = RiskEngine(cfg.risk, cfg.symbols)

    if arm == ARM_CURRENT:
        legacy = LegacyTrendPullbackStrategy(
            params=StrategyParams(signal_threshold=6.0, risk_multiplier=1.0, ensemble_weight=1.0)
        )
        # Regime-unaware: the original bot had no regime filter, so widen the
        # capability to every regime instead of inheriting V2's gating.
        legacy.capability = Capability(
            allowed_regimes=tuple(Regime), minimum_data=200, risk_multiplier=1.0
        )
        strategies = {legacy.name: legacy}
        signal_engine = _SingleStrategyEngine(legacy)
        composer = _LegacyComposer(
            fees=cfg.settings.fees,
            risk=cfg.risk,
            risk_engine=risk_engine,
            reference_symbol=reference_symbol,
        )
    elif arm == ARM_TREND:
        strategies = {"trend_pullback": TrendPullbackStrategy(params=cfg.strategies.trend_pullback)}
        signal_engine = SignalEngine(strategies, arm_cfg.strategies)
        composer = _full_composer(cfg, risk_engine, reference_symbol)
    else:  # ARM_MULTI / ARM_MULTI_ML
        strategies = build_strategies(cfg.strategies)
        signal_engine = SignalEngine(strategies, arm_cfg.strategies)
        composer = _full_composer(cfg, risk_engine, reference_symbol)

    ensemble = Ensemble(strategies, arm_cfg.strategies)
    meta_filter = build_meta_filter(cfg) if (with_ml and arm == ARM_MULTI) else None

    pipeline = TradingPipeline(
        candles=candles,
        features=features,
        regime_detector=regime_detector,
        signal_engine=signal_engine,
        ensemble=ensemble,
        composer=composer,
        context=context,
        reference_symbol=reference_symbol,
        meta_filter=meta_filter,
    )

    return StrategyStack(
        config=arm_cfg,
        candles=candles,
        features=features,
        strategies=strategies,
        signal_engine=signal_engine,
        ensemble=ensemble,
        regime_detector=regime_detector,
        risk_engine=risk_engine,
        composer=composer,
        pipeline=pipeline,
        context=context,
        reference_symbol=reference_symbol,
    )


# -- running a single arm -------------------------------------------------
def _base_arm(display_arm: str) -> str:
    return ARM_MULTI if display_arm == ARM_MULTI_ML else display_arm


def run_arm(
    display_arm: str,
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    cfg: BotConfig,
    exec_tf: Timeframe,
    backtest: BacktestConfig,
    with_ml: bool = False,
) -> BacktestResult:
    stack = build_arm_stack(cfg, _base_arm(display_arm), with_ml=with_ml)
    engine = BacktestEngine(
        candles_by_key, stack=stack, backtest=backtest, collection_timeframe=exec_tf
    )
    result = engine.run()
    logger.info("%s: %d trades, net %.2f", display_arm, len(result.trades), result.metrics.net_pnl)
    return result


def _walk_forward_arm(
    display_arm: str,
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    cfg: BotConfig,
    exec_tf: Timeframe,
    backtest: BacktestConfig,
    with_ml: bool = False,
    train_bars: int = 900,
    test_bars: int = 400,
    step_bars: int = 400,
    max_windows: int = 6,
) -> dict:
    """Out-of-sample walk-forward for one arm using the arm's own stack."""
    stamps = _split_events(candles_by_key, exec_tf)
    if len(stamps) < train_bars + test_bars:
        return {"windows": [], "aggregate": {}, "consistency": 0.0, "window_count": 0}

    all_trades = []
    windows: list[dict] = []
    cursor = train_bars
    index = 0
    while cursor + test_bars <= len(stamps) and index < max_windows:
        test_start = stamps[cursor]
        test_end = stamps[min(cursor + test_bars, len(stamps)) - 1]
        test_slice = _slice(candles_by_key, test_start, test_end)

        stack = build_arm_stack(cfg, _base_arm(display_arm), with_ml=with_ml)
        engine = BacktestEngine(
            test_slice, stack=stack, backtest=backtest, collection_timeframe=exec_tf
        )
        res = engine.run()
        all_trades.extend(res.trades)
        windows.append(
            {
                "index": index,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "trades": len(res.trades),
                "net_pnl": round(res.metrics.net_pnl, 4),
            }
        )
        cursor += step_bars
        index += 1

    aggregate = compute_metrics(all_trades, backtest.starting_equity)
    profitable = sum(1 for w in windows if w["net_pnl"] > 0)
    consistency = (profitable / len(windows)) if windows else 0.0
    return {
        "windows": windows,
        "aggregate": aggregate.as_dict(),
        "consistency": round(consistency, 4),
        "window_count": len(windows),
    }


# -- synthetic history (offline harness validation) ----------------------
def _synthetic_history(bars: int = 700, seed: int = 7) -> tuple[dict, Timeframe]:
    """Deterministic synthetic M15 candles for offline harness validation.

    These are NOT market data and no trading conclusion should be drawn from
    them. They exist solely so the comparison harness can be exercised end to
    end without credentials or connectivity.

    Note: the backtest engine recomputes indicator frames per bar, so runtime
    grows roughly quadratically with ``bars``. Keep this modest for harness
    checks (a few hundred bars run in seconds; several thousand take minutes).
    """
    import random

    rng = random.Random(seed)
    exec_tf = Timeframe.M15
    bars_per_day = 96
    n = max(400, bars)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    universe = [
        ("BTC/USD", 60_000.0, 0.0040, 0.00002),
        ("ETH/USD", 3_000.0, 0.0050, 0.000015),
        ("SOL/USD", 150.0, 0.0070, 0.00001),
    ]
    out: dict[tuple[str, Timeframe], list[Candle]] = {}
    for symbol, base, vol, drift in universe:
        price = base
        candles: list[Candle] = []
        for i in range(n):
            # A slow cycle so both trending and ranging phases appear.
            phase = math.sin(i / (bars_per_day * 8))
            mu = drift + 0.0006 * phase
            shock = rng.gauss(0.0, vol)
            price = max(1.0, price * (1.0 + mu + shock))
            op = price * (1.0 + rng.gauss(0.0, vol / 4.0))
            hi = max(op, price) * (1.0 + abs(rng.gauss(0.0, vol / 2.0)))
            lo = min(op, price) * (1.0 - abs(rng.gauss(0.0, vol / 2.0)))
            candles.append(
                Candle(
                    timestamp=start + timedelta(minutes=15 * i),
                    symbol=symbol,
                    timeframe=exec_tf,
                    open=op,
                    high=max(hi, op, price),
                    low=max(0.01, min(lo, op, price)),
                    close=price,
                    volume=100.0 + rng.random() * 50.0,
                )
            )
        out[(symbol, exec_tf)] = candles
    return out, exec_tf


# -- reporting ------------------------------------------------------------
def _fmt_row(name: str, m: dict) -> str:
    pf = m.get("profit_factor", 0.0)
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (
        f"{name:<16} {m.get('trades', 0):>7} {m.get('win_rate', 0) * 100:>8.1f}% "
        f"{m.get('net_pnl', 0):>12,.2f} {pf_s:>7} {m.get('expectancy', 0):>11,.4f} "
        f"{m.get('max_drawdown_pct', 0) * 100:>9.2f}% {m.get('sharpe', 0):>8.3f} "
        f"{m.get('total_fees', 0):>10,.2f}"
    )


def render_comparison(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 104)
    lines.append("V2 STRATEGY COMPARISON: CURRENT BOT vs V2 TREND-ONLY vs V2 MULTI-STRATEGY")
    lines.append(f"generated: {payload.get('generated_at', '')}")
    period = payload.get("period", {})
    lines.append(f"period: {period.get('start')} -> {period.get('end')}   source: {payload.get('source')}")
    lines.append(f"symbols: {', '.join(payload.get('symbols', []))}   equity: {payload.get('starting_equity', 0):,.0f}")
    lines.append("=" * 104)
    lines.append(
        f"{'arm':<16} {'trades':>7} {'win%':>9} {'net P&L':>12} {'PF':>7} "
        f"{'expectancy':>11} {'maxDD%':>10} {'sharpe':>8} {'fees':>10}"
    )
    lines.append("-" * 104)
    for arm, data in payload.get("results", {}).items():
        lines.append(_fmt_row(arm, data.get("metrics", {})))
    lines.append("-" * 104)
    lines.append("")
    lines.append("ARM DESCRIPTIONS")
    for arm, desc in ARM_DESCRIPTIONS.items():
        if arm in payload.get("results", {}):
            lines.append(f"  {arm:<16} {desc}")
    lines.append("")
    lines.append("MONTE CARLO (bootstrap of realised trades)")
    for arm, data in payload.get("results", {}).items():
        mc = data.get("monte_carlo", {})
        lines.append(
            f"  {arm:<16} prob(profit)={mc.get('prob_profit', 0) * 100:5.1f}%  "
            f"prob(ruin)={mc.get('prob_ruin', 0) * 100:5.2f}%  "
            f"median_return={mc.get('median_return', 0) * 100:+.2f}%  "
            f"p05={mc.get('p05_return', 0) * 100:+.2f}%  p95={mc.get('p95_return', 0) * 100:+.2f}%"
        )
    wf_arms = [a for a, d in payload.get("results", {}).items() if d.get("walk_forward")]
    if wf_arms:
        lines.append("")
        lines.append("WALK-FORWARD (out-of-sample)")
        for arm in wf_arms:
            wf = payload["results"][arm]["walk_forward"]
            agg = wf.get("aggregate", {})
            lines.append(
                f"  {arm:<16} windows={wf.get('window_count', 0)}  "
                f"OOS trades={agg.get('trades', 0)}  OOS net={agg.get('net_pnl', 0):,.2f}  "
                f"consistency={wf.get('consistency', 0) * 100:.1f}%"
            )
    lines.append("")
    per_strategy_sections = payload.get("per_strategy", {})
    if per_strategy_sections:
        lines.append("PER-STRATEGY CONTRIBUTION")
        for arm, breakdown in per_strategy_sections.items():
            lines.append(f"  [{arm}]")
            for strat, m in sorted(breakdown.items()):
                lines.append(
                    f"    {strat:<22} trades={m.get('trades', 0):<5} "
                    f"net={m.get('net_pnl', 0):>11,.2f}  win={m.get('win_rate', 0) * 100:5.1f}%"
                )
    lines.append("=" * 104)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CURRENT BOT vs V2 strategies")
    parser.add_argument("--days", type=int, default=365, help="history window (ignored with --synthetic)")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--synthetic", action="store_true", help="use offline synthetic candles")
    parser.add_argument("--synthetic-bars", type=int, default=700, help="bars per symbol with --synthetic")
    parser.add_argument("--with-ml", action="store_true", help="add the ML meta-filter arm")
    parser.add_argument("--walk-forward", action="store_true", help="add out-of-sample walk-forward")
    parser.add_argument("--mc-iterations", type=int, default=2000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load_config()

    if args.synthetic:
        candles_by_key, exec_tf = _synthetic_history(bars=args.synthetic_bars)
        source = "synthetic (harness validation only)"
    else:
        from scripts._common import load_history

        candles_by_key, exec_tf = load_history(cfg, days=args.days)
        source = f"alpaca {args.days}d"

    if not candles_by_key:
        logger.error("No candles available; nothing to compare.")
        return 1

    # A lower warmup than the engine default keeps the harness tractable on
    # short windows while still allowing the 200-period indicators to warm up.
    backtest = BacktestConfig(starting_equity=args.equity, warmup_bars=210)
    arms = [ARM_CURRENT, ARM_TREND, ARM_MULTI]
    if args.with_ml:
        arms.append(ARM_MULTI_ML)

    payload: dict = {
        "generated_at": utcnow().isoformat(),
        "source": source,
        "starting_equity": args.equity,
        "exec_tf": str(exec_tf),
        "symbols": sorted({sym for (sym, _tf) in candles_by_key}),
        "period": {},
        "results": {},
        "per_strategy": {},
    }

    for arm in arms:
        with_ml = arm == ARM_MULTI_ML
        result = run_arm(
            arm, candles_by_key, cfg, exec_tf, backtest, with_ml=with_ml
        )
        mc = bootstrap_returns(
            result.trades, starting_equity=args.equity, iterations=args.mc_iterations
        )
        report = build_report(result, label=arm)
        payload["results"][arm] = {
            "description": ARM_DESCRIPTIONS.get(arm, ""),
            "metrics": result.metrics.as_dict(),
            "trades": len(result.trades),
            "regime_counts": result.regime_counts,
            "exit_reasons": report.get("exit_reasons", {}),
            "monte_carlo": mc.as_dict(),
        }
        payload["per_strategy"][arm] = report.get("per_strategy", {})

        if args.walk_forward:
            payload["results"][arm]["walk_forward"] = _walk_forward_arm(
                arm, candles_by_key, cfg, exec_tf, backtest, with_ml=with_ml
            )

        if not payload["period"]:
            payload["period"] = {
                "start": result.start.isoformat() if result.start else None,
                "end": result.end.isoformat() if result.end else None,
            }

    text = render_comparison(payload)

    reports_dir = cfg.settings.paths.reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "comparison.json"
    text_path = reports_dir / "comparison.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nJSON: {json_path}")
    print(f"Text: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
