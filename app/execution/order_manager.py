"""Order manager: safe, idempotent order submission and lifecycle tracking.

The invariants enforced here are the ones that keep real money safe:

* record the intent locally BEFORE submitting (so a crash cannot lose an order);
* deterministic client order ids (so a retry never double-fills);
* on ambiguity ("did it go through?") we RECONCILE by client order id rather
  than blindly resubmitting (spec sections 73, 79, 80, 127).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from app.core.enums import OrderStatus
from app.core.types import OrderRecord, utcnow
from app.execution.broker import (
    Broker,
    BrokerError,
    BrokerOrderResult,
    OrderRequest,
    RejectedOrderError,
    TransientBrokerError,
)
from app.execution.order_planner import OrderPlan, OrderPlanner
from app.state.state_manager import StateManager

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        state: StateManager,
        planner: OrderPlanner,
        retry_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 16.0,
    ) -> None:
        self.broker = broker
        self.state = state
        self.planner = planner
        self.retry_attempts = max(1, retry_attempts)
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _apply_result(rec: OrderRecord, result: BrokerOrderResult) -> OrderRecord:
        rec.alpaca_order_id = result.alpaca_order_id or rec.alpaca_order_id
        rec.status = result.status
        rec.filled_qty = result.filled_qty or rec.filled_qty
        rec.average_fill_price = result.average_fill_price or rec.average_fill_price
        rec.submitted_at = result.submitted_at or rec.submitted_at or utcnow()
        rec.updated_at = utcnow()
        return rec

    def _backoff(self, attempt: int) -> None:
        delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
        time.sleep(delay)

    def _safe_get_order(self, client_order_id: str) -> Optional[BrokerOrderResult]:
        for _ in range(2):
            try:
                return self.broker.get_order(client_order_id)
            except BrokerError as exc:
                logger.warning("get_order(%s) failed: %s", client_order_id, exc)
                time.sleep(self.retry_base_delay)
        return None

    def _submit_with_retry(self, request: OrderRequest, rec: OrderRecord) -> OrderRecord:
        last_exc: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                result = self.broker.submit_order(request)
                return self._apply_result(rec, result)
            except RejectedOrderError:
                raise
            except TransientBrokerError as exc:
                last_exc = exc
                logger.warning(
                    "transient submit failure for %s (attempt %d/%d): %s",
                    request.client_order_id,
                    attempt + 1,
                    self.retry_attempts,
                    exc,
                )
                if attempt < self.retry_attempts - 1:
                    self._backoff(attempt)
            except BrokerError as exc:
                # Unknown broker error: do NOT resubmit blindly. Reconcile.
                last_exc = exc
                break

        existing = self._safe_get_order(request.client_order_id)
        if existing is not None:
            logger.info("recovered order %s via reconciliation", request.client_order_id)
            return self._apply_result(rec, existing)
        if last_exc is not None:
            raise last_exc
        raise BrokerError("order submission failed and could not be reconciled")

    # -- public API ------------------------------------------------------
    def execute_entry(self, plan: OrderPlan) -> OrderRecord:
        """Submit an entry order and persist its outcome.

        Writes the intent first, then submits. If the submit result is ambiguous
        we reconcile by client order id before returning.
        """
        rec = OrderRecord(
            client_order_id=plan.entry.client_order_id,
            alpaca_order_id=None,
            symbol=plan.entry.symbol,
            side=plan.entry.side,
            order_type=plan.entry.order_type,
            qty=plan.entry.qty,
            limit_price=plan.entry.limit_price,
            stop_price=plan.protection.stop_price,
            status=OrderStatus.PLANNED,
            updated_at=utcnow(),
            strategy=plan.strategy,
            intent="entry",
        )
        self.state.record_order(rec)  # durable intent BEFORE any broker call

        rec.status = OrderStatus.VALIDATED
        self.state.record_order(rec)

        result = self._submit_with_retry(plan.entry, rec)
        self.state.record_order(result)
        logger.info(
            "entry %s %s qty=%.8f status=%s",
            plan.symbol,
            plan.strategy,
            plan.entry.qty,
            result.status,
        )
        return result

    def execute_exit(
        self,
        symbol: str,
        qty: float,
        strategy: str = "",
        intent: str = "exit",
        now: Optional[datetime] = None,
    ) -> Optional[OrderRecord]:
        """Flatten (or reduce) a position with a market sell."""
        request = self.planner.exit_order(symbol, qty, now=now, intent=intent)
        if request is None:
            logger.warning("exit request for %s rounds to zero qty", symbol)
            return None
        rec = OrderRecord(
            client_order_id=request.client_order_id,
            alpaca_order_id=None,
            symbol=symbol,
            side="sell",
            order_type=request.order_type,
            qty=request.qty,
            status=OrderStatus.PLANNED,
            updated_at=utcnow(),
            strategy=strategy,
            intent=intent,
        )
        self.state.record_order(rec)
        rec.status = OrderStatus.VALIDATED
        self.state.record_order(rec)
        try:
            result = self._submit_with_retry(request, rec)
        except BrokerError as exc:
            logger.error("exit submission failed for %s: %s", symbol, exc)
            rec.status = OrderStatus.UNKNOWN
            self.state.record_order(rec)
            raise
        self.state.record_order(result)
        logger.info("exit %s qty=%.8f status=%s", symbol, request.qty, result.status)
        return result

    def cancel(self, client_order_id: str) -> bool:
        try:
            ok = self.broker.cancel_order(client_order_id)
        except BrokerError as exc:
            logger.error("cancel %s failed: %s", client_order_id, exc)
            return False
        if ok:
            self.state.update_order_status(client_order_id, OrderStatus.CANCELLED)
        return ok

    def sync_order(self, client_order_id: str) -> Optional[OrderRecord]:
        """Poll the broker for an order's latest state and persist it."""
        try:
            result = self.broker.get_order(client_order_id)
        except BrokerError as exc:
            logger.warning("sync_order(%s) failed: %s", client_order_id, exc)
            return None
        if result is None:
            return None
        return self.state.record_order(result)

    def sync_open_orders(self) -> list[OrderRecord]:
        updated: list[OrderRecord] = []
        try:
            open_orders = self.broker.list_open_orders()
        except BrokerError as exc:
            logger.warning("list_open_orders failed: %s", exc)
            return updated
        for order in open_orders:
            self.state.record_order(order)
            updated.append(order)
        return updated
