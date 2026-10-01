"""SQLite database access layer.

Thread-safe (per-call connections with WAL mode). Provides typed helpers for
every table. All multi-statement writes that must survive a crash are wrapped in
transactions (spec section 127).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from app.core.types import ensure_utc, iso, utcnow
from app.state.models import SCHEMA_STATEMENTS, SCHEMA_VERSION

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    # -- connection management ------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(str(self.path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self.transaction() as conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- account ---------------------------------------------------------
    def save_account(self, equity: float, cash: float, buying_power: float, last_sync: datetime | None = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO account_state (id, account_equity, cash, buying_power, last_sync)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    account_equity=excluded.account_equity,
                    cash=excluded.cash,
                    buying_power=excluded.buying_power,
                    last_sync=excluded.last_sync
                """,
                (equity, cash, buying_power, iso(last_sync or utcnow())),
            )

    def get_account(self) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM account_state WHERE id = 1").fetchone()
            return dict(row) if row else None

    # -- positions -------------------------------------------------------
    def upsert_position(self, pos: dict[str, Any]) -> None:
        pos = dict(pos)
        pos.setdefault("updated_at", iso(utcnow()))
        cols = [
            "symbol", "qty", "average_entry", "current_price", "unrealized_pnl",
            "strategy", "entry_time", "stop_price", "target_price", "initial_stop_price",
            "initial_risk_per_unit", "risk_amount", "trailing_state", "state", "regime",
            "feature_snapshot", "updated_at",
        ]
        values = [pos.get(c) for c in cols]
        if isinstance(pos.get("trailing_state"), (dict, list)):
            idx = cols.index("trailing_state")
            values[idx] = json.dumps(pos["trailing_state"], default=str)
        if isinstance(pos.get("feature_snapshot"), (dict,)):
            idx = cols.index("feature_snapshot")
            values[idx] = json.dumps(pos["feature_snapshot"], default=str)
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "symbol")
        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO positions ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT(symbol) DO UPDATE SET {updates}",
                values,
            )

    def delete_position(self, symbol: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    def get_positions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM positions").fetchall()]

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()
            return dict(row) if row else None

    # -- orders ----------------------------------------------------------
    def upsert_order(self, order: dict[str, Any]) -> None:
        order = dict(order)
        order.setdefault("updated_at", iso(utcnow()))
        cols = [
            "client_order_id", "alpaca_order_id", "symbol", "side", "order_type", "qty",
            "limit_price", "stop_price", "status", "intent", "filled_qty",
            "average_fill_price", "strategy", "submitted_at", "updated_at",
        ]
        values = [order.get(c) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "client_order_id")
        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO orders ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT(client_order_id) DO UPDATE SET {updates}",
                values,
            )

    def get_order(self, client_order_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_order_by_alpaca_id(self, alpaca_order_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE alpaca_order_id = ?", (alpaca_order_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_open_orders(self) -> list[dict[str, Any]]:
        open_states = ("PLANNED", "VALIDATED", "SUBMITTED", "PARTIALLY_FILLED", "PROTECTED", "UNKNOWN")
        placeholders = ", ".join(["?"] * len(open_states))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM orders WHERE status IN ({placeholders})", open_states
            ).fetchall()
            return [dict(r) for r in rows]

    def get_orders_for_symbol(self, symbol: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM orders WHERE symbol = ? ORDER BY updated_at DESC", (symbol,)
                ).fetchall()
            ]

    # -- trades ----------------------------------------------------------
    def insert_trade(self, trade: dict[str, Any]) -> None:
        cols = [
            "trade_id", "symbol", "strategy", "entry_price", "exit_price", "qty", "fees",
            "slippage", "gross_pnl", "net_pnl", "entry_time", "exit_time",
            "holding_time_seconds", "exit_reason", "regime", "signal_score",
            "expected_edge", "estimated_cost", "actual_cost", "feature_snapshot",
        ]
        values = []
        for c in cols:
            v = trade.get(c)
            if c == "feature_snapshot" and isinstance(v, dict):
                v = json.dumps(v, default=str)
            values.append(v)
        placeholders = ", ".join(["?"] * len(cols))
        with self.transaction() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO trades ({', '.join(cols)}) VALUES ({placeholders})", values
            )

    def get_trades(self, strategy: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM trades"
        clauses = []
        params: list[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY exit_time"
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    # -- signals / decisions / rejections --------------------------------
    def insert_signal(self, signal: dict[str, Any]) -> None:
        cols = [
            "symbol", "timestamp", "regime", "strategy", "decision", "score", "confidence",
            "expected_edge", "estimated_cost", "reject_reason", "detail", "feature_snapshot",
            "created_at",
        ]
        values = []
        for c in cols:
            v = signal.get(c)
            if c == "feature_snapshot" and isinstance(v, dict):
                v = json.dumps(v, default=str)
            if c == "created_at" and v is None:
                v = iso(utcnow())
            values.append(v)
        placeholders = ", ".join(["?"] * len(cols))
        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO signals ({', '.join(cols)}) VALUES ({placeholders})", values
            )

    def get_signals(self, symbol: str | None = None, decision: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM signals"
        clauses = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp"
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    # -- heartbeats ------------------------------------------------------
    def save_heartbeat(self, component: str, detail: str = "") -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO heartbeats (component, last_seen, detail) VALUES (?, ?, ?) "
                "ON CONFLICT(component) DO UPDATE SET last_seen=excluded.last_seen, detail=excluded.detail",
                (component, iso(utcnow()), detail),
            )

    def get_heartbeats(self) -> dict[str, datetime]:
        out: dict[str, datetime] = {}
        with self.connect() as conn:
            for row in conn.execute("SELECT component, last_seen FROM heartbeats").fetchall():
                out[row["component"]] = datetime.fromisoformat(row["last_seen"])
        return out

    # -- checkpoints -----------------------------------------------------
    def save_checkpoint(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO checkpoints (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, payload, iso(utcnow())),
            )

    def get_checkpoint(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM checkpoints WHERE key = ?", (key,)).fetchone()
            if not row:
                return default
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return default

    # -- events ----------------------------------------------------------
    def insert_event(self, level: str, event: str, message: str = "", data: Any = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO events (timestamp, level, event, message, data) VALUES (?, ?, ?, ?, ?)",
                (iso(utcnow()), level, event, message, json.dumps(data, default=str) if data is not None else None),
            )

    # -- strategy health -------------------------------------------------
    def upsert_strategy_health(self, health: dict[str, Any]) -> None:
        cols = [
            "strategy", "trade_count", "win_rate", "profit_factor", "expectancy",
            "average_win", "average_loss", "max_drawdown", "recent_performance",
            "weight", "enabled", "updated_at",
        ]
        values = [health.get(c) for c in cols]
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "strategy")
        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO strategy_health ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))}) "
                f"ON CONFLICT(strategy) DO UPDATE SET {updates}",
                values,
            )

    def get_strategy_health(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM strategy_health").fetchall()]

    # -- experiments -----------------------------------------------------
    def save_experiment(self, experiment: dict[str, Any]) -> None:
        cols = ["experiment_id", "git_commit", "config_hash", "dataset_hash", "mode", "created_at", "summary"]
        values = [experiment.get(c) for c in cols]
        if values[-2] is None:
            values[-2] = iso(utcnow())
        with self.transaction() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO experiments ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
                values,
            )

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            return dict(row) if row else None

    # -- maintenance -----------------------------------------------------
    def healthy(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("database health check failed: %s", exc)
            return False
