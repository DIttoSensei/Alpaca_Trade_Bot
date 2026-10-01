"""Concrete Alpaca implementation of the execution layer's ``Broker`` protocol.

All conversion from Alpaca's response objects to our canonical types happens
here. Nothing above this layer ever sees an alpaca-py object (spec section 2).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from app.broker.alpaca_client import AlpacaClient
from app.core.enums import OrderStatus
from app.core.types import Quote, ensure_utc
from app.execution.broker import (
    Broker,
    BrokerAccount,
    BrokerError,
    BrokerOrderResult,
    BrokerPosition,
    OrderRequest,
    RejectedOrderError,
)

logger = logging.getLogger(__name__)


class QuoteProvider(Protocol):
    def get_quote(self, symbol: str) -> Optional[Quote]: ...

    def get_last_price(self, symbol: str) -> Optional[float]: ...


_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CLOSED,
    "stopped": OrderStatus.CLOSED,
    "done_for_day": OrderStatus.CLOSED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "calculated": OrderStatus.CLOSED,
    "held": OrderStatus.SUBMITTED,
}


def _map_status(raw: Any) -> OrderStatus:
    value = getattr(raw, "value", raw)
    return _STATUS_MAP.get(str(value).lower(), OrderStatus.UNKNOWN)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    try:
        return ensure_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


class AlpacaTradingClient(Broker):
    def __init__(self, client: AlpacaClient, quotes: QuoteProvider | None = None) -> None:
        self.client = client
        self.quotes = quotes

    # -- request construction -------------------------------------------
    @staticmethod
    def _build_request(request: OrderRequest) -> Any:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side = OrderSide.BUY if request.side == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if request.time_in_force.lower() == "gtc" else TimeInForce.IOC
        common = {
            "symbol": request.symbol,
            "qty": request.qty,
            "side": side,
            "time_in_force": tif,
            "client_order_id": request.client_order_id,
        }
        otype = request.order_type.lower()
        if otype == "market":
            return MarketOrderRequest(**common)
        if otype == "limit":
            if request.limit_price is None:
                raise RejectedOrderError("limit order requires limit_price")
            return LimitOrderRequest(limit_price=request.limit_price, **common)
        if otype == "stop":
            if request.stop_price is None:
                raise RejectedOrderError("stop order requires stop_price")
            return StopOrderRequest(stop_price=request.stop_price, **common)
        if otype == "stop_limit":
            if request.stop_price is None or request.limit_price is None:
                raise RejectedOrderError("stop_limit requires stop_price and limit_price")
            return StopLimitOrderRequest(
                stop_price=request.stop_price, limit_price=request.limit_price, **common
            )
        raise RejectedOrderError(f"unsupported order_type: {request.order_type}")

    @staticmethod
    def _normalize_order(order: Any) -> BrokerOrderResult:
        return BrokerOrderResult(
            client_order_id=str(getattr(order, "client_order_id", "")),
            symbol=str(getattr(order, "symbol", "")),
            side=str(getattr(getattr(order, "side", ""), "value", getattr(order, "side", ""))),
            order_type=str(
                getattr(getattr(order, "order_type", ""), "value", getattr(order, "order_type", ""))
            ),
            qty=_as_float(getattr(order, "qty", 0.0)),
            status=_map_status(getattr(order, "status", None)),
            alpaca_order_id=str(getattr(order, "id", "") or "") or None,
            limit_price=_as_float(getattr(order, "limit_price", None), 0.0) or None,
            stop_price=_as_float(getattr(order, "stop_price", None), 0.0) or None,
            filled_qty=_as_float(getattr(order, "filled_qty", 0.0)),
            average_fill_price=_as_float(getattr(order, "filled_avg_price", None), 0.0) or None,
            submitted_at=_as_dt(getattr(order, "submitted_at", None)),
            updated_at=_as_dt(getattr(order, "updated_at", None)),
        )

    # -- Broker protocol -------------------------------------------------
    def submit_order(self, request: OrderRequest) -> BrokerOrderResult:
        order_data = self._build_request(request)
        order = self.client.call(self.client.trading.submit_order, order_data=order_data)
        return self._normalize_order(order)

    def cancel_order(self, client_order_id: str) -> bool:
        try:
            order = self.client.call(
                self.client.trading.get_order_by_client_id, client_order_id
            )
        except BrokerError:
            return False
        if order is None:
            return False
        order_id = getattr(order, "id", None)
        if order_id is None:
            return False
        self.client.call(self.client.trading.cancel_order_by_id, order_id)
        return True

    def get_order(self, client_order_id: str) -> Optional[BrokerOrderResult]:
        try:
            order = self.client.call(
                self.client.trading.get_order_by_client_id, client_order_id
            )
        except RejectedOrderError:
            return None
        if order is None:
            return None
        return self._normalize_order(order)

    def list_open_orders(self) -> list[BrokerOrderResult]:
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(status="open")
        orders = self.client.call(self.client.trading.get_orders, filter=request)
        return [self._normalize_order(o) for o in (orders or [])]

    def list_positions(self) -> list[BrokerPosition]:
        positions = self.client.call(self.client.trading.get_all_positions)
        out: list[BrokerPosition] = []
        for p in positions or []:
            out.append(
                BrokerPosition(
                    symbol=str(getattr(p, "symbol", "")),
                    qty=_as_float(getattr(p, "qty", 0.0)),
                    average_entry=_as_float(getattr(p, "avg_entry_price", 0.0)),
                    current_price=_as_float(getattr(p, "current_price", 0.0)),
                    unrealized_pnl=_as_float(getattr(p, "unrealized_pl", 0.0)),
                )
            )
        return out

    def get_account(self) -> BrokerAccount:
        account = self.client.call(self.client.trading.get_account)
        return BrokerAccount(
            equity=_as_float(getattr(account, "equity", 0.0)),
            cash=_as_float(getattr(account, "cash", 0.0)),
            buying_power=_as_float(getattr(account, "buying_power", 0.0)),
            last_sync=datetime.now(timezone.utc),
        )

    # -- quotes (delegated) ---------------------------------------------
    def get_quote(self, symbol: str) -> Optional[Quote]:
        if self.quotes is None:
            return None
        return self.quotes.get_quote(symbol)

    def get_last_price(self, symbol: str) -> Optional[float]:
        if self.quotes is None:
            return None
        return self.quotes.get_last_price(symbol)
