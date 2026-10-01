"""Checkpoint helpers.

Thin typed wrapper over the ``checkpoints`` table for small key/value state such
as rolling strategy stats, circuit-breaker snapshots and last-known regimes.
"""

from __future__ import annotations

from typing import Any

from app.state.database import Database

# Well-known checkpoint keys.
KEY_TRADING_STATE = "trading_state"
KEY_CIRCUIT_BREAKERS = "circuit_breakers"
KEY_STRATEGY_WEIGHTS = "strategy_weights"
KEY_LAST_SYNC = "last_sync"
KEY_OPEN_POSITION_SNAPSHOT = "open_position_snapshot"


class CheckpointStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, key: str, value: Any) -> None:
        self.db.save_checkpoint(key, value)

    def load(self, key: str, default: Any = None) -> Any:
        return self.db.get_checkpoint(key, default)
