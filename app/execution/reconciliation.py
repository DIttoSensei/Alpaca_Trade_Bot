"""Reconciliation: the broker is the source of truth (spec section 80).

At startup and periodically we compare what the bot believes against what the
broker actually reports, then repair local state. We never fabricate fills and we
never assume our local view is correct. Unresolvable discrepancies are surfaced
and can halt trading (fail-closed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.core.enums import OrderStatus, PositionState, Regime
from app.core.types import (
    OrderRecord,
    PositionRecord,
    iso,
    utcnow,
)
from app.execution.broker import Broker, BrokerError, BrokerPosition
from app.state.state_manager import StateManager

logger = logging.getLogger(__name__)

_QTY_TOL = 1e-8
_PRICE_TOL = 1e-6


@dataclass(slots=True)
class ReconcileIssue:
    kind: str
    symbol: str
    detail: str
    severity: str = "warning"  # "warning" | "critical"


@dataclass(slots=True)
class ReconcileReport:
    timestamp: datetime = field(default_factory=utcnow)
    positions_broker: int = 0
    positions_local: int = 0
    orders_checked: int = 0
    account_mismatch: bool = False
    issues: list[ReconcileIssue] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    repaired: bool = False

    @property
    def clean(self) -> bool:
        return not self.issues

    @property
    def critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    def as_dict(self) -> dict:
        return {
            "timestamp": iso(self.timestamp),
            "positions_broker": self.positions_broker,
            "positions_local": self.positions_local,
            "orders_checked": self.orders_checked,
            "account_mismatch": self.account_mismatch,
            "clean": self.clean,
            "critical": self.critical,
            "issues": [
                {"kind": i.kind, "symbol": i.symbol, "detail": i.detail, "severity": i.severity}
                for i in self.issues
            ],
            "actions": self.actions,
        }


class Reconciler:
    def __init__(self, qty_tolerance: float = _QTY_TOL) -> None:
        self.qty_tolerance = qty_tolerance

    def reconcile(
        self,
        broker: Broker,
        state: StateManager,
        auto_repair: bool = True,
        reference_regime: Optional[Regime] = None,
    ) -> ReconcileReport:
        report = ReconcileReport()

        # 1) Account truth (broker authoritative).
        try:
            account = broker.get_account()
            if state.account is None or abs(state.account.equity - account.equity) > max(
                1.0, account.equity * 1e-6
            ):
                report.account_mismatch = True
            state.update_account(account.equity, account.cash, account.buying_power, account.last_sync)
        except BrokerError as exc:
            report.issues.append(ReconcileIssue("account_unavailable", "", str(exc), "critical"))
            return report

        # 2) Positions.
        try:
            broker_positions = {p.symbol: p for p in broker.list_positions()}
        except BrokerError as exc:
            report.issues.append(ReconcileIssue("positions_unavailable", "", str(exc), "critical"))
            return report

        report.positions_broker = len(broker_positions)
        report.positions_local = len(state.positions)

        for symbol, bp in broker_positions.items():
            local = state.positions.get(symbol)
            if local is None:
                report.issues.append(
                    ReconcileIssue(
                        "position_missing_local",
                        symbol,
                        f"broker qty={bp.qty} not tracked locally",
                        "critical",
                    )
                )
                if auto_repair:
                    self._adopt_position(state, bp, reference_regime)
                    report.actions.append(f"adopted {symbol} from broker")
            else:
                if abs(local.qty - bp.qty) > self.qty_tolerance:
                    report.issues.append(
                        ReconcileIssue(
                            "position_qty_mismatch",
                            symbol,
                            f"local={local.qty} broker={bp.qty}",
                            "warning",
                        )
                    )
                    if auto_repair:
                        local.qty = bp.qty
                        report.actions.append(f"corrected qty for {symbol} -> broker")

        for symbol, local in list(state.positions.items()):
            if symbol not in broker_positions:
                report.issues.append(
                    ReconcileIssue(
                        "position_missing_broker",
                        symbol,
                        f"local qty={local.qty} absent at broker",
                        "warning",
                    )
                )
                if auto_repair:
                    state.remove_position(symbol)
                    report.actions.append(f"removed stale local position {symbol}")

        # 3) Orders: sync statuses, flag orphans.
        try:
            broker_orders = {o.client_order_id: o for o in broker.list_open_orders()}
        except BrokerError as exc:
            report.issues.append(ReconcileIssue("orders_unavailable", "", str(exc), "critical"))
            return report

        report.orders_checked = len(broker_orders)
        for cid, bo in broker_orders.items():
            local_order = state.orders.get(cid)
            if local_order is None:
                existing = state.db.get_order(cid)
                if existing:
                    continue
                report.issues.append(
                    ReconcileIssue(
                        "orphan_broker_order",
                        bo.symbol,
                        f"broker order {cid} unknown locally",
                        "critical",
                    )
                )
                if auto_repair:
                    state.record_order(bo)
                    report.actions.append(f"adopted broker order {cid}")
            elif local_order.status != bo.status:
                state.record_order(bo)
                report.actions.append(f"synced order {cid} -> {bo.status}")

        # Local orders believed open but unknown to the broker are dangerous.
        for cid, local_order in list(state.orders.items()):
            if local_order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED) and cid not in broker_orders:
                try:
                    maybe = broker.get_order(cid)
                except BrokerError:
                    maybe = None
                if maybe is None:
                    report.issues.append(
                        ReconcileIssue(
                            "orphan_local_order",
                            local_order.symbol,
                            f"local order {cid} not found at broker",
                            "critical",
                        )
                    )
                    if auto_repair:
                        state.update_order_status(cid, OrderStatus.UNKNOWN)
                        report.actions.append(f"marked {cid} UNKNOWN")

        report.repaired = bool(report.actions)
        state.db.insert_event(
            "warning" if report.critical else "info",
            "reconcile",
            "reconciliation complete",
            report.as_dict(),
        )
        return report

    @staticmethod
    def _adopt_position(state: StateManager, bp: BrokerPosition, regime: Optional[Regime]) -> None:
        """Adopt a broker position we did not know about (e.g. after a crash).

        Stop/target are left unset; the execution monitor derives a protective
        stop from entry price and ATR on the next tick.
        """
        entry_time = utcnow()
        pos = PositionRecord(
            symbol=bp.symbol,
            qty=bp.qty,
            average_entry=bp.average_entry,
            current_price=bp.current_price or bp.average_entry,
            unrealized_pnl=bp.unrealized_pnl,
            strategy="reconciled",
            entry_time=entry_time,
            stop_price=None,
            target_price=None,
            state=PositionState.OPEN,
            regime=regime,
        )
        state.upsert_position(pos)

    @staticmethod
    def needs_protection(broker: Broker, state: StateManager) -> list[str]:
        """Return positions that have no stop order at the broker."""
        try:
            open_orders = broker.list_open_orders()
        except BrokerError:
            return list(state.positions.keys())
        protected: set[str] = set()
        for o in open_orders:
            if getattr(o, "order_type", "") in ("stop", "stop_limit"):
                protected.add(o.symbol)
        return [s for s in state.positions if s not in protected]
