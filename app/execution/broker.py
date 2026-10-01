"""Broker abstraction for the execution layer.

The execution layer must not know about Alpaca. It talks to this Protocol, and
the concrete adapter (``app.broker.trading_client``) translates Alpaca responses
into these broker-neutral objects. This keeps execution testable and prevents
broker JSON from leaking into strategy code (spec section 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable

from app.core.enums import OrderStatus
from app.core.types import Quote, utcnow


class BrokerError(Exception):
    """Base class for broker interaction failures."""


class TransientBrokerError(BrokerError):
    """A failure that is safe to retry (timeout, 5xx, connection reset)."""


class RejectedOrderError(BrokerError):
    """The broker explicitly rejected the order; retrying will not help."""


@dataclass(slots=True)
class OrderRequest:
    """Broker-neutral order submission request."""

    client_order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    order_type: str  # "market" | "limit" | "stop" | "stop_limit"
    qty: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "gtc"
    notional: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "qty": self.qty,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "notional": self.notional,
        }


@dataclass(slots=True)
class BrokerOrderResult:
    """Normalised order state returned by the broker."""

    client_order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    status: OrderStatus
    alpaca_order_id: Optional[str] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: float = 0.0
    average_fill_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(slots=True)
class BrokerPosition:
    """Normalised broker position."""

    symbol: str
    qty: float
    average_entry: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(slots=True)
class BrokerAccount:
    """Normalised broker account snapshot."""

    equity: float
    cash: float
    buying_power: float
    last_sync: datetime = field(default_factory=utcnow)


@runtime_checkable
class Broker(Protocol):
    """Everything the execution layer needs from a broker."""

    def submit_order(self, request: OrderRequest) -> BrokerOrderResult:
        """Submit an order. Raises BrokerError subclasses on failure."""

    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an order by client order id. Returns True if cancelled."""

    def get_order(self, client_order_id: str) -> Optional[BrokerOrderResult]:
        """Fetch a single order, or None if the broker does not know it."""

    def list_open_orders(self) -> list[BrokerOrderResult]:
        """List currently open orders."""

    def list_positions(self) -> list[BrokerPosition]:
        """List currently open positions."""

    def get_account(self) -> BrokerAccount:
        """Fetch the account snapshot."""

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Latest top-of-book quote, or None if unavailable."""

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Latest trade price, or None if unavailable."""
