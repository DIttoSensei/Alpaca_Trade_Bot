"""Run a full backtest with optional walk-forward and Monte Carlo validation.

Example::

    python scripts/backtest.py --days 730 --walk-forward --monte-carlo

The heavy lifting is the shared engine used by live trading; this script only
supplies history and formatting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest import BacktestConfig, run_backtest, save_report  # noqa: E402
from app.backtest.monte_carlo import bootstrap_returns  # noqa: E402
from app.backtest.walk_forward import run_walk_forward  # noqa: E402
from app.config import load_config  # noqa: E402
from scripts._common import load_history  # noqa: E402

logger = logging.getLogger("scripts.backtest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the trading strategies")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--monte-carlo", action="store_true")
    parser.add_argument("--mc-iterations", type=int, default=2000)
    parser.add_argument("--label", default="backtest")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    cfg = load_config()
    candles_by_key, exec_tf = load_history(cfg, days=args.days)
    if not candles_by_key:
        logger.error("No historical data loaded; check credentials/connectivity.")
        return 1

    backtest = BacktestConfig(starting_equity=args.equity)
    result = run_backtest(candles_by_key, config=cfg, backtest=backtest,
                          collection_timeframe=exec_tf)
    logger.info("in-sample: %d trades, net %.2f", len(result.trades), result.equity_curve[-1].equity
                - args.equity if result.equity_curve else 0.0)

    wf_result = None
    if args.walk_forward:
        wf_result = run_walk_forward(
            candles_by_key, config=cfg, backtest=backtest,
            base_timeframe=exec_tf,
            on_window=lambda w: logger.info(
                "walk-forward window %d: %d trades", w.index, len(w.result.trades)
            ),
        )
        logger.info("walk-forward windows: %d consistent %.1f%%",
                    len(wf_result.windows), wf_result.consistency * 100)

    mc_result = None
    if args.monte_carlo:
        mc_result = bootstrap_returns(
            result.trades, starting_equity=args.equity, iterations=args.mc_iterations
        )
        logger.info("monte carlo: prob(profit)=%.1f%% prob(ruin)=%.2f%%",
                    mc_result.prob_profit * 100, mc_result.prob_ruin * 100)

    paths = save_report(result, cfg.settings.paths.reports, label=args.label,
                        walk_forward=wf_result, monte_carlo=mc_result,
                        charts=not args.no_charts)
    print(f"Report: {paths.text_path}")
    print(f"JSON:   {paths.json_path}")
    if paths.trades_csv:
        print(f"Trades: {paths.trades_csv}")
    if paths.equity_png:
        print(f"Equity: {paths.equity_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
