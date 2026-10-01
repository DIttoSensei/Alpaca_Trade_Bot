"""Train the optional ML meta-filter from a backtest's realised trades.

The meta-filter is a veto-only layer. This script:
  1. runs a backtest with the rule-based stack (as-is),
  2. builds a chronological dataset from the trades' stored feature snapshots,
  3. trains a classifier to predict P(win after costs),
  4. saves the model to the models directory and logs an experiment record.

Example::

    python scripts/train.py --days 730 --model logistic
    # then enable it: ML_ENABLED=true python -m app.main --mode backtest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest import BacktestConfig, run_backtest  # noqa: E402
from app.config import load_config  # noqa: E402
from app.ml.meta_filter import DEFAULT_MODEL_FILENAME  # noqa: E402
from app.ml.registry import ExperimentRecord, ExperimentRegistry  # noqa: E402
from app.ml.training import train_meta_filter  # noqa: E402
from scripts._common import load_history  # noqa: E402

logger = logging.getLogger("scripts.train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the ML meta-filter")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument(
        "--model",
        choices=["gradient_boosting", "random_forest", "logistic"],
        default="gradient_boosting",
    )
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--no-regime", action="store_true", help="omit regime one-hot features")
    parser.add_argument("--label", default="meta-filter")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    cfg = load_config()
    candles_by_key, exec_tf = load_history(cfg, days=args.days)
    if not candles_by_key:
        logger.error("No historical data loaded; cannot generate a training set.")
        return 1

    logger.info("generating trades via backtest for the training set...")
    result = run_backtest(
        candles_by_key,
        config=cfg,
        backtest=BacktestConfig(starting_equity=args.equity),
        collection_timeframe=exec_tf,
    )
    logger.info("backtest produced %d trades", len(result.trades))

    try:
        report = train_meta_filter(
            result.trades,
            model_type=args.model,
            test_fraction=args.test_fraction,
            threshold=args.threshold,
            include_regime=not args.no_regime,
            min_trades=args.min_trades,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    out_path = cfg.settings.paths.models / DEFAULT_MODEL_FILENAME
    report.model.save(out_path)
    logger.info("saved meta-filter model to %s", out_path)

    metrics = report.as_dict()["metrics"]
    registry = ExperimentRegistry(cfg.settings.paths.reports)
    registry.log(
        ExperimentRecord(
            name="meta-filter",
            kind="ml_train",
            label=args.label,
            params={
                "days": args.days,
                "model": args.model,
                "threshold": args.threshold,
                "test_fraction": args.test_fraction,
                "include_regime": not args.no_regime,
            },
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            notes="veto-only meta-filter trained on backtest feature snapshots",
            artifacts={"model": str(out_path)},
        )
    )

    print(f"Model:      {out_path}")
    print(f"Train/Test: {report.train_size}/{report.test_size}")
    print(f"Base win:   {report.base_win_rate:.3f}")
    if metrics.get("kept_win_rate") is not None:
        print(f"Kept win:   {metrics['kept_win_rate']:.3f} "
              f"(lift {metrics.get('win_rate_lift', 0):+.3f}, veto {metrics.get('veto_rate', 0):.1%})")
    if metrics.get("roc_auc") is not None:
        print(f"ROC AUC:    {metrics['roc_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
