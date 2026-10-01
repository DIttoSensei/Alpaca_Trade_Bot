"""Experiment registry.

A plain append-only JSONL log of every experiment (backtests, walk-forward,
Monte Carlo, meta-filter training) so results are reproducible and comparable
over time. Deliberately dependency-free: no MLflow, no server — just a file you
can diff, grep and archive (spec sections 90, 137).

Each record captures the intent (what did we change?) separately from the
outcome (what happened?), because a metric without its configuration is noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core.types import iso, utcnow

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = "experiments.jsonl"


@dataclass(slots=True)
class ExperimentRecord:
    name: str
    kind: str  # "backtest" | "walk_forward" | "monte_carlo" | "ml_train"
    timestamp: datetime = field(default_factory=utcnow)
    label: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "timestamp": iso(self.timestamp),
            "label": self.label,
            "params": self.params,
            "metrics": self.metrics,
            "notes": self.notes,
            "artifacts": self.artifacts,
        }


class ExperimentRegistry:
    """Append-only JSONL registry rooted in a directory."""

    def __init__(self, root: Path | str, filename: str = DEFAULT_REGISTRY) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / filename

    def log(self, record: ExperimentRecord) -> ExperimentRecord:
        line = json.dumps(record.as_dict(), default=str)
        # Append with a single write so concurrent runs cannot interleave lines.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info("logged experiment '%s' (%s) -> %s", record.name, record.kind, self.path)
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed registry line")
        return out

    def find(self, name: str) -> list[dict]:
        return [r for r in self.read_all() if r.get("name") == name]

    def latest(self, kind: Optional[str] = None) -> Optional[dict]:
        records = [r for r in self.read_all() if kind is None or r.get("kind") == kind]
        return records[-1] if records else None

    def best_by(self, metric: str, kind: Optional[str] = None, higher_is_better: bool = True) -> Optional[dict]:
        records = [
            r for r in self.read_all()
            if (kind is None or r.get("kind") == kind) and metric in (r.get("metrics") or {})
        ]
        if not records:
            return None
        key = lambda r: r["metrics"][metric]  # noqa: E731
        return max(records, key=key) if higher_is_better else min(records, key=key)
