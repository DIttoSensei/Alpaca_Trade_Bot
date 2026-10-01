"""ML meta-filter (veto-only, fail-open on absent model).

Contract (spec section 85):
  * The rule-based stack ALWAYS runs first and produces a candidate.
  * The meta-filter may only VETO a candidate — it can never create one.
  * When no model is loaded, or sklearn is unavailable, the filter is a no-op
    (``enabled is False``), so the bot behaves exactly like the rule-based core.

The model predicts P(trade is a winner). A candidate is vetoed when that
probability falls below the configured threshold. We deliberately do NOT let the
model scale position size: keeping the filter to a single, auditable decision
("take it / skip it") avoids accidentally compounding risk through an opaque
model output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.enums import Regime
from app.core.types import FeatureSnapshot
from app.ml.features import build_vector, feature_names

logger = logging.getLogger(__name__)

DEFAULT_MODEL_FILENAME = "meta_filter.joblib"


@dataclass(slots=True)
class VetoDecision:
    veto: bool
    probability: float
    threshold: float
    model_name: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "veto": self.veto,
            "probability": round(self.probability, 6),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "reason": self.reason,
        }


def _import_joblib():
    try:
        import joblib  # type: ignore

        return joblib
    except Exception:  # noqa: BLE001 - optional dependency
        return None


class MetaFilterModel:
    """Thin wrapper around a scikit-learn binary classifier.

    Loading is best-effort: a missing/corrupt file leaves the filter disabled
    rather than crashing the trading process. Decision-making must never fail
    because an OPTIONAL enhancement is unavailable.
    """

    def __init__(
        self,
        model=None,
        threshold: float = 0.5,
        model_name: str = "meta_filter",
        feature_columns: Optional[list[str]] = None,
        include_regime: bool = True,
    ) -> None:
        self.model = model
        self.threshold = float(threshold)
        self.model_name = model_name
        self.feature_columns = feature_columns or feature_names(include_regime)
        self.include_regime = include_regime

    @property
    def enabled(self) -> bool:
        return self.model is not None

    # -- prediction ------------------------------------------------------
    def win_probability(
        self,
        snapshot: Optional[FeatureSnapshot],
        regime: Optional[Regime],
    ) -> Optional[float]:
        if not self.enabled:
            return None
        vector = build_vector(snapshot, regime, self.include_regime)
        try:
            import numpy as np  # type: ignore

            x = np.asarray([vector], dtype=float)
            proba = self.model.predict_proba(x)  # type: ignore[union-attr]
            classes = list(getattr(self.model, "classes_", [0, 1]))
            if 1 in classes:
                return float(proba[0][classes.index(1)])
            return float(proba[0][-1])
        except Exception as exc:  # noqa: BLE001 - never let ML break trading
            logger.debug("meta-filter prediction failed (allowing trade): %s", exc)
            return None

    def evaluate(
        self,
        snapshot: Optional[FeatureSnapshot],
        regime: Optional[Regime],
    ) -> VetoDecision:
        if not self.enabled:
            return VetoDecision(False, 1.0, self.threshold, self.model_name, "disabled")

        proba = self.win_probability(snapshot, regime)
        if proba is None:
            # Fail-OPEN: an un-scoreable candidate is left to the rule-based stack.
            return VetoDecision(
                False, float("nan"), self.threshold, self.model_name, "unscoreable; allowed"
            )
        veto = proba < self.threshold
        reason = f"P(win)={proba:.3f} {'<' if veto else '>='} threshold={self.threshold:.3f}"
        return VetoDecision(veto, proba, self.threshold, self.model_name, reason)

    def vetoes(self, snapshot: Optional[FeatureSnapshot], regime: Optional[Regime]) -> bool:
        return self.evaluate(snapshot, regime).veto

    # -- persistence -----------------------------------------------------
    def save(self, path: Path | str) -> Path:
        joblib = _import_joblib()
        if joblib is None:
            raise RuntimeError("joblib is required to persist the meta-filter model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # joblib handles numpy/scikit estimators far more robustly than pickle.
        joblib.dump(self.model, path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "threshold": self.threshold,
                    "include_regime": self.include_regime,
                    "feature_columns": self.feature_columns,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path | str,
        threshold: Optional[float] = None,
        include_regime: bool = True,
    ) -> "MetaFilterModel":
        path = Path(path)
        if not path.exists():
            logger.info("no meta-filter model at %s; filter disabled", path)
            return cls(model=None, threshold=threshold or 0.5, include_regime=include_regime)

        meta: dict = {}
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}

        joblib = _import_joblib()
        if joblib is None:
            logger.warning("joblib unavailable; cannot load meta-filter model")
            return cls(model=None, threshold=threshold or 0.5, include_regime=include_regime)

        try:
            model = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load meta-filter model (%s); filter disabled", exc)
            model = None

        return cls(
            model=model,
            threshold=threshold if threshold is not None else float(meta.get("threshold", 0.5)),
            model_name=str(meta.get("model_name", "meta_filter")),
            feature_columns=meta.get("feature_columns"),
            include_regime=bool(meta.get("include_regime", include_regime)),
        )

    @classmethod
    def load_default(
        cls,
        models_dir: Path | str,
        threshold: Optional[float] = None,
        include_regime: bool = True,
    ) -> "MetaFilterModel":
        return cls.load(
            Path(models_dir) / DEFAULT_MODEL_FILENAME,
            threshold=threshold,
            include_regime=include_regime,
        )
