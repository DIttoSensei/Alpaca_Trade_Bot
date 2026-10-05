"""Grid-strategy backtest in isolation, with mandatory sanity checks.

Example::

    python scripts/backtest_grid.py --days 730 --equity 10000

Runs ONLY the grid strategy (the other five are disabled by default) against
real Alpaca history via the shared data loader, then prints P&L alongside the
sanity checks that would expose the classic grid backtest lies:

* hold-time distribution (min / median / max) — a grid run reduced to a handful
  of instant round-trips is not really a grid;
* max single-trade price move % — a value near a full range width means a whole
  cycle was booked inside one bar, i.e. the intrabar ordering dominated P&L;
* non-exact-level fills (MUST be 0) — every fill must land on a configured rung;
* unique trading days active, with the first/last dates;
* regime-exit flattens and hard-bound triggers (safety #1) actually fired;
* peak committed capital vs the configured cap (safety #4).

Backtest-only: this script does NOT touch live/paper execution.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.candle_store import aggregate  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402
from app.strategies.grid import FILL_ORDER_LOW_FIRST, run_grid_backtest  # noqa: E402
from scripts._common import load_history  # noqa: E402

logger = logging.getLogger("scripts.backtest_grid")

_EXACT_LEVEL_TOL = 1e-6


def _load_from_cache(cfg, target_tf: Timeframe | None = None) -> tuple[dict, object]:
    """Offline fallback: read the local parquet cache, no network needed.

    Lets the grid backtest be reproduced without alpaca-py / credentials, which
    is exactly when you most want to re-check a suspicious result.

    The cache is keyed by the EXACT timeframe that was downloaded (typically the
    configured execution timeframe, M15). To honour an ``--execution-timeframe``
    override offline, we try the exact file first and otherwise aggregate a
    finer cached series up to ``target_tf`` (e.g. M15 -> H1). This keeps the
    sweep runnable with no credentials; a live fetch (when available) creates
    the exact-timeframe file directly.
    """
    cache = ParquetCache(cfg.settings.paths.data)
    target = target_tf or cfg.symbols.execution_timeframe()
    by_key: dict = {}
    for symbol in cfg.symbols.enabled():
        # 1) Exact cached timeframe.
        try:
            bars = cache.load(symbol, target)
        except Exception as exc:  # noqa: BLE001 - pandas/pyarrow not installed
            logger.warning("cache read failed for %s (%s)", symbol, exc)
            return {}, target
        # 2) Otherwise aggregate a finer cached series (M15 -> H1) if present.
        if not bars:
            for src in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
                if src.seconds >= target.seconds:
                    continue
                try:
                    fine = cache.load(symbol, src)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("cache read failed for %s (%s)", symbol, exc)
                    fine = []
                if fine:
                    bars = aggregate(fine, target)
                    logger.info(
                        "cache: aggregated %s %s -> %s (%d bars)",
                        symbol, src, target, len(bars),
                    )
                    break
        if bars:
            by_key[(symbol, target)] = bars
    return by_key, target


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def render_report(result, cfg, *, days: int, exec_tf, equity: float) -> str:
    """Build the full human-readable report string (pure; no I/O)."""
    summary = result.summary()
    trades = result.trades
    fills = [f for r in result.per_symbol.values() for f in r.fills]
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"GRID BACKTEST (isolated)  fill-order assumption: {result.fill_order}")
    if result.fill_order != FILL_ORDER_LOW_FIRST:
        lines.append(f"  !! unexpected fill ordering: {result.fill_order}")
    lines.append("=" * 72)
    lines.append(f"days                : {days}")
    lines.append(f"execution timeframe : {exec_tf}")
    lines.append(f"fee-safety multiple : {cfg.strategies.grid.fee_safety_multiple:g}x")
    lines.append(f"symbols run         : {sorted(result.per_symbol)}")
    lines.append(f"trades              : {summary['trades']}")
    lines.append(f"net P&L             : {summary['net_pnl']:.2f}")
    lines.append(f"return %            : {summary['return_pct']:.3f}% (base {summary['base_equity']:.0f})")
    lines.append(f"total fills         : {len(fills)}")
    for sym, res in sorted(result.per_symbol.items()):
        lines.append(
            f"  {sym:<10} trades={len(res.trades):<5} fills={len(res.fills):<5} "
            f"equity {res.starting_equity:.0f} -> {res.final_equity:.2f} "
            f"skipped={res.levels_placed_skipped}"
        )

    holds = [t.holding_time_seconds / 60.0 for t in trades if t.holding_time_seconds is not None]
    lines.append("-" * 72)
    lines.append("HOLD TIME (minutes)")
    lines.append(
        f"  min={min(holds):.2f}  median={_median(holds):.2f}  max={max(holds):.2f}"
        if holds
        else "  (no completed trades)"
    )

    moves = [abs(t.exit_price - t.entry_price) / t.entry_price * 100.0 for t in trades if t.entry_price > 0]
    lines.append("MAX SINGLE-TRADE PRICE MOVE")
    lines.append(f"  max={max(moves):.3f}%" if moves else "  (no completed trades)")

    non_exact = 0
    for f in fills:
        if f.spacing <= 0:
            continue
        ratio = (f.level_price - f.lower) / f.spacing
        if abs(ratio - round(ratio)) > _EXACT_LEVEL_TOL:
            non_exact += 1
    lines.append("NON-EXACT-LEVEL FILLS")
    lines.append(f"  {non_exact}  (must be 0)")

    days_active = sorted({t.entry_time.date() for t in trades})
    lines.append("UNIQUE TRADING DAYS")
    lines.append(
        f"  count={len(days_active)}  first={days_active[0]}  last={days_active[-1]}"
        if days_active
        else "  count=0"
    )

    lines.append("SAFETY #1 (regime exit / hard bound)")
    lines.append(f"  regime-exit flattens : {summary['regime_exit_flattens']}")
    lines.append(f"  hard-bound triggers  : {summary['hard_bound_triggers']}")

    cap = cfg.risk.max_symbol_exposure * equity
    lines.append("SAFETY #4 (peak committed vs cap)")
    lines.append(
        f"  peak_committed={summary['peak_committed']:.2f}  cap={cap:.2f}  "
        f"breached={'YES' if summary['peak_committed'] > cap + 1e-6 else 'no'}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def _parse_timeframe(value: str) -> Timeframe:
    """Parse ``--execution-timeframe`` (accepts ``M15``/``m15``/``15m`` or ``H1``/``1h``)."""
    token = value.strip()
    by_name = {m.name.upper(): m for m in Timeframe}
    if token.upper() in by_name:
        return by_name[token.upper()]
    try:
        return Timeframe(token.lower())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown timeframe {value!r}; use one of {[t.name for t in Timeframe]} "
            f"or {[t.value for t in Timeframe]}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the grid strategy in isolation")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--symbols", nargs="*", default=None, help="override the universe")
    parser.add_argument(
        "--fee-safety-multiple",
        type=float,
        default=None,
        metavar="X",
        help="override GridParams.fee_safety_multiple: the adjacent-level gap must clear "
        "X times round-trip cost or the whole grid is skipped (safety #2). Default 3.0.",
    )
    parser.add_argument(
        "--execution-timeframe",
        type=_parse_timeframe,
        default=None,
        metavar="TF",
        help="override the execution timeframe (e.g. M15 or H1). The grid is primarily "
        "gated by this timeframe; a coarser one widens the ATR-scaled level spacing.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load_config()

    # -- Programmatic overrides (no code change needed) --------------------
    # Both are applied to the shared BotConfig BEFORE the stack/loader is
    # built, so the execution context, feature store and the fee gate all
    # observe the same values.
    if args.fee_safety_multiple is not None:
        cfg.strategies.grid.fee_safety_multiple = float(args.fee_safety_multiple)
        logger.info("override: fee_safety_multiple=%.3gx", args.fee_safety_multiple)
    if args.execution_timeframe is not None:
        cfg.symbols.timeframes["execution"] = args.execution_timeframe
        logger.info("override: execution_timeframe=%s", args.execution_timeframe.value)

    try:
        candles_by_key, exec_tf = load_history(cfg, days=args.days, symbols=args.symbols)
    except Exception as exc:  # noqa: BLE001 - missing alpaca-py / offline
        logger.warning("live data fetch unavailable (%s); falling back to local parquet cache", exc)
        candles_by_key, exec_tf = _load_from_cache(cfg, args.execution_timeframe)
    if not candles_by_key:
        logger.error("No historical data available (live or cached); nothing to backtest.")
        return 1

    result = run_grid_backtest(
        candles_by_key,
        cfg,
        starting_equity=args.equity,
        collection_timeframe=exec_tf,
    )

    print(render_report(result, cfg, days=args.days, exec_tf=exec_tf, equity=args.equity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
