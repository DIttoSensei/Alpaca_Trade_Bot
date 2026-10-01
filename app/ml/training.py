"""Training pipeline for the ML meta-filter.

Design principles (spec sections 85, 137):
  * The label is realised trade success (net P&L > 0 after costs) — we never
    train on gross P&L, so the model learns to avoid trades that only LOOK good
    before fees and slippage.
  * Splitting is CHRONOLOGICAL, never random: a random split leaks the future
    into the training set and manufactures a fantasy score.
  * Success is measured by the LIFT the filter provides over taking every trade
    (win-rate of kept trades vs. all trades), not by raw accuracy — with an
    imbalanced label an "always take" model would look accurate and be useless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from app.core.enums import Regime
from app.core.types import TradeRecord
from app.ml.features import build_vector, feature_names
from app.ml.meta_filter import MetaFilterModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Dataset:
    X: list[list[float]]
    y: list[int]
    feature_names: list[str]
    exit_times: list[datetime] = field(default_factory=list)
    regimes: list[Optional[Regime]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def base_win_rate(self) -> float:
        return (sum(self.y) / len(self.y)) if self.y else 0.0


@dataclass(slots=True)
class TrainingReport:
    model: MetaFilterModel
    train_size: int
    test_size: int
    base_win_rate: float
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "base_win_rate": round(self.base_win_rate, 4),
            "threshold": self.model.threshold,
            "metrics": self.metrics,
        }


def dataset_from_trades(
    trades: Sequence[TradeRecord],
    include_regime: bool = True,
) -> Dataset:
    """Build a chronologically ordered dataset from completed trades.

    Trades without a stored feature snapshot are dropped: we refuse to
    reconstruct features from current data (that would leak the future).
    """
    ordered = sorted(
        (t for t in trades if t.feature_snapshot is not None),
        key=lambda t: t.exit_time,
    )
    X: list[list[float]] = []
    y: list[int] = []
    times: list[datetime] = []
    regimes: list[Optional[Regime]] = []
    for t in ordered:
        X.append(build_vector(t.feature_snapshot, t.regime, include_regime))
        y.append(1 if t.net_pnl > 0 else 0)
        times.append(t.exit_time)
        regimes.append(t.regime)
    return Dataset(X=X, y=y, feature_names=feature_names(include_regime),
                   exit_times=times, regimes=regimes)


def _require_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        return {
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "metrics": (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score),
        }
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "scikit-learn is required to train the meta-filter. "
            "Install it with: pip install 'scikit-learn>=1.4.0'"
        ) from exc


def _make_estimator(model_type: str, random_state: int):
    sk = _require_sklearn()
    if model_type == "logistic":
        return sk["LogisticRegression"](max_iter=1000, class_weight="balanced")
    if model_type == "random_forest":
        return sk["RandomForestClassifier"](
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    # Default: gradient boosting — strong on tabular data with modest data volume.
    return sk["GradientBoostingClassifier"](
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, random_state=random_state,
    )


def train_meta_filter(
    trades: Sequence[TradeRecord],
    model_type: str = "gradient_boosting",
    test_fraction: float = 0.3,
    threshold: float = 0.5,
    include_regime: bool = True,
    random_state: int = 42,
    min_trades: int = 50,
) -> TrainingReport:
    """Train and evaluate the veto model. Raises if there is too little data."""
    dataset = dataset_from_trades(trades, include_regime=include_regime)
    n = len(dataset)
    if n < min_trades:
        raise RuntimeError(
            f"not enough trades with feature snapshots to train ({n} < {min_trades}); "
            "run a longer backtest or wait for more live history"
        )

    split = max(1, int(n * (1.0 - test_fraction)))
    if split >= n:
        split = n - 1
    X_train, y_train = dataset.X[:split], dataset.y[:split]
    X_test, y_test = dataset.X[split:], dataset.y[split:]

    # Guard: a single-class training set cannot fit a classifier.
    if len(set(y_train)) < 2:
        raise RuntimeError("training split contains only one class; cannot fit a classifier")

    estimator = _make_estimator(model_type, random_state)
    estimator.fit(X_train, y_train)

    metrics: dict = {
        "model_type": model_type,
        "feature_count": len(dataset.feature_names),
        "class_balance_train": {
            "positive": sum(y_train), "negative": len(y_train) - sum(y_train)
        },
    }

    if X_test and len(set(y_test)) >= 1:
        metrics.update(_evaluate(estimator, X_test, y_test, threshold))
    else:
        metrics["note"] = "no usable test split"

    model = MetaFilterModel(
        model=estimator,
        threshold=threshold,
        model_name=f"meta_filter:{model_type}",
        feature_columns=dataset.feature_names,
        include_regime=include_regime,
    )
    report = TrainingReport(
        model=model,
        train_size=len(y_train),
        test_size=len(y_test),
        base_win_rate=dataset.base_win_rate,
        metrics=metrics,
    )
    return report


def _evaluate(estimator, X_test, y_test, threshold: float) -> dict:
    sk = _require_sklearn()
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score = sk["metrics"]

    proba = [p[1] for p in estimator.predict_proba(X_test)]
    preds = [1 if p >= threshold else 0 for p in proba]

    out: dict = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision_win": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall_win": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1_win": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "base_win_rate_test": round(sum(y_test) / len(y_test), 4),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
    except Exception:  # noqa: BLE001
        out["roc_auc"] = None

    # The number that actually matters: does keeping only non-vetoed trades
    # raise the win rate relative to taking every trade?
    kept = [y for y, p in zip(y_test, proba) if p >= threshold]
    out["veto_rate"] = round(1 - (len(kept) / len(y_test)), 4) if y_test else 0.0
    if kept:
        kept_wr = sum(kept) / len(kept)
        out["kept_win_rate"] = round(kept_wr, 4)
        out["win_rate_lift"] = round(kept_wr - out["base_win_rate_test"], 4)
    else:
        out["kept_win_rate"] = None
        out["win_rate_lift"] = None
    return out
