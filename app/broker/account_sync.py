"""Periodic account/position synchronisation.

Keeps the StateManager aligned with the broker between reconciliations and feeds
the circuit breakers with live equity so daily-loss / drawdown limits are
evaluated continuously, not only after a trade closes (spec sections 32, 80).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.enums import PositionState
from app.core.types import PositionRecord, utcnow
from app.execution.broker import Broker, BrokerError
from app.risk.circuit_breaker import CircuitBreakerManager
from app.state.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyncResult:
    ok: bool
    equity: float = 0.0
    synced_at: datetime = None  # type: ignore[assignment]
    error: str = ""


class AccountSync:
    def __init__(
        self,
        broker: Broker,
        state: StateManager,
        breakers: CircuitBreakerManager | None = None,
        interval_seconds: int = 30,
    ) -> None:
        self.broker = broker
        self.state = state
        self.breakers = breakers
        self.interval_seconds = interval_seconds
        self._last_sync: datetime | None = None

    def due(self, now: datetime | None = None) -> bool:
        now = now or utcnow()
        if self._last_sync is None:
            return True
        return now - self._last_sync >= timedelta(seconds=self.interval_seconds)

    def sync(self, update_positions: bool = True) -> SyncResult:
        try:
            account = self.broker.get_account()
        except BrokerError as exc:
            logger.error("account sync failed: %s", exc)
            return SyncResult(ok=False, error=str(exc))

        self.state.update_account(account.equity, account.cash, account.buying_power, account.last_sync)
        self._last_sync = utcnow()

        if update_positions:
            try:
                self._sync_positions()
            except BrokerError as exc:
                logger.warning("position sync failed: %s", exc)

        if self.breakers is not None:
            unrealized = sum(p.unrealized_pnl for p in self.state.positions.values())
            for trip in self.breakers.evaluate(account.equity, unrealized):
                logger.warning("circuit breaker tripped: %s (%s)", trip.kind, trip.detail)

        return SyncResult(ok=True, equity=account.equity, synced_at=self._last_sync)

    def _sync_positions(self) -> None:
        broker_positions = {p.symbol: p for p in self.broker.list_positions()}
        for symbol, bp in broker_positions.items():
            local = self.state.positions.get(symbol)
            if local is None:
                local = PositionRecord(
                    symbol=symbol,
                    qty=bp.qty,
                    average_entry=bp.average_entry,
                    current_price=bp.current_price or bp.average_entry,
                    unrealized_pnl=bp.unrealized_pnl,
                    strategy="synced",
                    entry_time=utcnow(),
                    state=PositionState.OPEN,
                )
            else:
                local.qty = bp.qty
                local.average_entry = bp.average_entry
                local.current_price = bp.current_price
                local.unrealized_pnl = bp.unrealized_pnl
            self.state.upsert_position(local)

        for symbol in list(self.state.positions.keys()):
            if symbol not in broker_positions:
                # Broker closed it (stop/target filled). Remove local mirror;
                # trade recording is handled by reconciliation/trade tracker.
                logger.info("position %s closed at broker; removing local mirror", symbol)
                self.state.remove_position(symbol)
