"""ML meta-filter layer (optional).

A veto-only layer on top of the rule-based stack. Importing this package never
requires scikit-learn: it is only needed when actually training or loading a
persisted model, so the core bot runs without the optional dependency.
"""

from app.ml.features import ML_FEATURES, REGIME_ORDER, build_matrix, build_vector, feature_names
from app.ml.meta_filter import MetaFilterModel, VetoDecision
from app.ml.registry import ExperimentRecord, ExperimentRegistry
from app.ml.training import Dataset, TrainingReport, dataset_from_trades, train_meta_filter

__all__ = [
    "ML_FEATURES",
    "REGIME_ORDER",
    "build_matrix",
    "build_vector",
    "feature_names",
    "MetaFilterModel",
    "VetoDecision",
    "ExperimentRecord",
    "ExperimentRegistry",
    "Dataset",
    "TrainingReport",
    "dataset_from_trades",
    "train_meta_filter",
]
