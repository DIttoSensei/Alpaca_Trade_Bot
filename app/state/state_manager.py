"""State manager: the single in-memory source of truth, mirrored to SQLite.

Holds account, positions, orders and rolling stats. Every mutation that matters
is persisted immediately (write-through) so a crash cannot lose the fact that we
submitted an order.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from app.core.enums import OrderStatus, PositionState, TradingState
from app.core.types import (
    AccountState,
    OrderRecord,
    PositionRecord,
    TradeRecord,
    ensure_utc,
    iso,
    utcnow,
)
from app.state.checkpoints import CheckpointStore, KEY_TRADING_STATE
from app.state.database import Database

logger = logging.getLogger(__name__)


def _dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    try:
        return ensure_utc(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


class StateManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.checkpoints = CheckpointStore(db)
        self.account: Optional[AccountState] = None
        self.positions: dict[str, PositionRecord] = {}
        self.orders: dict[str, OrderRecord] = {}
        self.trading_state: TradingState = TradingState.STARTING
        self._load()

    # -- load ------------------------------------------------------------
    def _load(self) -> None:
        acct = self.db.get_account()
        if acct:
            self.account = AccountState(
                equity=acct["account_equity"],
                cash=acct["cash"],
                buying_power=acct["buying_power"],
                last_sync=_dt(acct["last_sync"]) or utcnow(),
            )
        for row in self.db.get_positions():
            self.positions[row["symbol"]] = self._row_to_position(row)
        for row in self.db.get_open_orders():
            self.orders[row["client_order_id"]] = self._row_to_order(row)
        saved_state = self.checkpoints.load(KEY_TRADING_STATE)
        if saved_state:
            try:
                self.trading_state = TradingState(saved_state)
            except ValueError:
                self.trading_state = TradingState.STARTING

    @staticmethod
    def _row_to_position(row: dict[str, Any]) -> PositionRecord:
        import json

        trailing = row.get("trailing_state")
        if isinstance(trailing, str) and trailing:
            try:
                trailing = json.loads(trailing)
            except json.JSONDecodeError:
                trailing = {}
        regime = row.get("regime")
        try:
            from app.core.enums import Regime

            regime_val = Regime(regime) if regime else None
        except (ValueError, TypeError):
            regime_val = None
        return PositionRecord(
            symbol=row["symbol"],
            qty=row["qty"],
            average_entry=row["average_entry"],
            current_price=row.get("current_price", 0.0) or 0.0,
            unrealized_pnl=row.get("unrealized_pnl", 0.0) or 0.0,
            strategy=row.get("strategy") or "",
            entry_time=_dt(row.get("entry_time")),
            stop_price=row.get("stop_price"),
            target_price=row.get("target_price"),
            initial_stop_price=row.get("initial_stop_price"),
            initial_risk_per_unit=row.get("initial_risk_per_unit") or 0.0,
            risk_amount=row.get("risk_amount") or 0.0,
            trailing_state=trailing or {},
            state=PositionState(row["state"]) if row.get("state") else PositionState.FLAT,
            regime=regime_val,
        )

    @staticmethod
    def _row_to_order(row: dict[str, Any]) -> OrderRecord:
        return OrderRecord(
            client_order_id=row["client_order_id"],
            alpaca_order_id=row.get("alpaca_order_id"),
            symbol=row["symbol"],
            side=row["side"],
            order_type=row["order_type"],
            qty=row["qty"],
            limit_price=row.get("limit_price"),
            stop_price=row.get("stop_price"),
            status=OrderStatus(row["status"]) if row.get("status") else OrderStatus.PLANNED,
            submitted_at=_dt(row.get("submitted_at")),
            updated_at=_dt(row.get("updated_at")),
            filled_qty=row.get("filled_qty") or 0.0,
            average_fill_price=row.get("average_fill_price"),
            strategy=row.get("strategy") or "",
            intent=row.get("intent") or "entry",
        )

    # -- trading state ---------------------------------------------------
    def set_trading_state(self, state: TradingState) -> None:
        self.trading_state = state
        self.checkpoints.save(KEY_TRADING_STATE, str(state))
        logger.info("trading state -> %s", state)

    # -- account ---------------------------------------------------------
    def update_account(self, equity: float, cash: float, buying_power: float, last_sync: datetime | None = None) -> None:
        last_sync = ensure_utc(last_sync or utcnow())
        self.account = AccountState(equity=equity, cash=cash, buying_power=buying_power, last_sync=last_sync)
        self.db.save_account(equity, cash, buying_power, last_sync)

    @property
    def equity(self) -> float:
        return self.account.equity if self.account else 0.0

    # -- positions -------------------------------------------------------
    def upsert_position(self, pos: PositionRecord) -> None:
        self.positions[pos.symbol] = pos
        payload = pos.as_dict()
        self.db.upsert_position(payload)

    def remove_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        self.db.delete_position(symbol)

    # -- orders ----------------------------------------------------------
    def record_order(self, order: OrderRecord) -> None:
        self.orders[order.client_order_id] = order
        self.db.upsert_order(
            {
                "client_order_id": order.client_order_id,
                "alpaca_order_id": order.alpaca_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "qty": order.qty,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "status": str(order.status),
                "intent": order.intent,
                "filled_qty": order.filled_qty,
                "average_fill_price": order.average_fill_price,
                "strategy": order.strategy,
                "submitted_at": iso(order.submitted_at) if order.submitted_at else None,
                "updated_at": iso(order.updated_at or utcnow()),
            }
        )

    def update_order_status(
        self,
        client_order_id: str,
        status: OrderStatus,
        filled_qty: float | None = None,
        avg_fill_price: float | None = None,
        alpaca_order_id: str | None = None,
    ) -> None:
        order = self.orders.get(client_order_id)
        if order is None:
            existing = self.db.get_order(client_order_id)
            if existing:
                order = self._row_to_order(existing)
                self.orders[client_order_id] = order
        if order is None:
            return
        order.status = status
        order.updated_at = utcnow()
        if filled_qty is not None:
            order.filled_qty = filled_qty
        if avg_fill_price is not None:
            order.average_fill_price = avg_fill_price
        if alpaca_order_id is not None:
            order.alpaca_order_id = alpaca_order_id
        self.record_order(order)

    # -- trades ----------------------------------------------------------
    def record_trade(self, trade: TradeRecord) -> None:
        payload = trade.as_dict()
        if trade.feature_snapshot is not None:
            payload["feature_snapshot"] = trade.feature_snapshot.as_dict()
        self.db.insert_trade(payload)

    # -- signals / decisions ---------------------------------------------
    def record_signal(self, payload: dict[str, Any]) -> None:
        self.db.insert_signal(payload)

    # -- heartbeats ------------------------------------------------------
    def heartbeat(self, component: str, detail: str = "") -> None:
        self.db.save_heartbeat(component, detail)

    def heartbeats(self) -> dict[str, datetime]:
        return self.db.get_heartbeats()
