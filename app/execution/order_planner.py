"""Order planning: turn a TradeDecision into concrete, quantised orders.

The planner is deterministic. Given the same decision at the same decision time
it produces the same ``client_order_id``. That idempotency is what stops a
restart mid-submission from creating a duplicate order (spec sections 73, 79).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.config.symbols import SymbolRegistry
from app.core.enums import Action, RejectReason
from app.core.types import Quote, TradeDecision, iso, utcnow
from app.execution.broker import OrderRequest

logger = logging.getLogger(__name__)

_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]")

# Alpaca crypto client order ids: <=48 chars, letters/digits/dash/underscore.
_MAX_CLIENT_ID_LEN = 48


def sanitize_symbol(symbol: str) -> str:
    return _ID_SAFE.sub("", symbol.replace("/", "").replace("-", "")).upper()


def make_client_order_id(
    symbol: str,
    strategy: str,
    intent: str,
    decision_time: datetime,
    fingerprint: str = "",
) -> str:
    """Deterministic, restart-safe client order id.

    The ``fingerprint`` should encode the trade essentials (qty, stop, target) so
    a re-run of the exact same decision before submission yields the same id.
    """
    stamp = decision_time.strftime("%Y%m%d%H%M%S")
    strat = _ID_SAFE.sub("", strategy)[:12] or "strat"
    body = f"{sanitize_symbol(symbol)}-{strat}-{intent}-{stamp}"
    if fingerprint:
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:6]
        body = f"{body}-{digest}"
    return body[:_MAX_CLIENT_ID_LEN]


def quantize_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    return (int(value / step)) * step


def quantize_price(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round(round(value / tick) * tick, 10)


@dataclass(slots=True)
class ProtectionPlan:
    stop_price: float
    target_price: float
    stop_distance_pct: float


@dataclass(slots=True)
class OrderPlan:
    symbol: str
    entry: OrderRequest
    protection: ProtectionPlan
    strategy: str
    decision: TradeDecision
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stop_price(self) -> float:
        return self.protection.stop_price

    @property
    def target_price(self) -> float:
        return self.protection.target_price


@dataclass(slots=True)
class PlanRejection:
    symbol: str
    reason: RejectReason
    detail: str


class OrderPlanner:
    def __init__(
        self,
        symbols: SymbolRegistry,
        min_notional: float = 1.0,
        default_qty_step: float = 1e-8,
    ) -> None:
        self.symbols = symbols
        self.min_notional = min_notional
        self.default_qty_step = default_qty_step

    def _qty_step(self, symbol: str) -> float:
        cfg = self.symbols.get(symbol)
        if cfg and cfg.tick_size > 0:
            return cfg.tick_size
        return self.default_qty_step

    def plan(
        self,
        decision: TradeDecision,
        now: Optional[datetime] = None,
    ) -> OrderPlan | PlanRejection:
        now = now or utcnow()
        cfg = self.symbols.get(decision.symbol)
        if cfg is None:
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "unknown symbol")
        if not cfg.enabled:
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "symbol disabled")
        if decision.side != "buy":
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "only long entries supported")

        step = self._qty_step(decision.symbol)
        qty = quantize_down(decision.quantity, step)
        if qty <= 0:
            return PlanRejection(decision.symbol, RejectReason.INSUFFICIENT_BUYING_POWER, "qty rounds to zero")

        entry_price = decision.limit_price
        if entry_price is None and decision.feature_snapshot is not None:
            entry_price = decision.feature_snapshot.values.get("close")
        if entry_price is None or entry_price <= 0:
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "missing reference price")

        notional = qty * entry_price
        if notional < self.min_notional:
            return PlanRejection(
                decision.symbol,
                RejectReason.INSUFFICIENT_BUYING_POWER,
                f"notional {notional:.4f} below minimum {self.min_notional}",
            )

        stop_price = decision.stop_price
        target_price = decision.target_price
        if stop_price is None or target_price is None:
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "missing stop/target")
        if stop_price >= entry_price:
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "stop not below entry")
        if target_price <= entry_price:
            return PlanRejection(decision.symbol, RejectReason.NO_SIGNAL, "target not above entry")

        tick = cfg.tick_size if cfg.tick_size > 0 else 1e-8
        stop_price = quantize_price(stop_price, tick)
        target_price = quantize_price(target_price, tick)
        if decision.entry_type == "limit" and decision.limit_price is not None:
            limit_price = quantize_price(decision.limit_price, tick)
        else:
            limit_price = None

        fingerprint = f"{decision.symbol}|{qty:.10f}|{stop_price:.10f}|{target_price:.10f}|{decision.strategy}"
        client_id = make_client_order_id(
            decision.symbol, decision.strategy, "entry", now, fingerprint
        )

        entry = OrderRequest(
            client_order_id=client_id,
            symbol=decision.symbol,
            side="buy",
            order_type=decision.entry_type or "market",
            qty=qty,
            limit_price=limit_price,
        )
        protection = ProtectionPlan(
            stop_price=stop_price,
            target_price=target_price,
            stop_distance_pct=(entry_price - stop_price) / entry_price,
        )
        return OrderPlan(
            symbol=decision.symbol,
            entry=entry,
            protection=protection,
            strategy=decision.strategy,
            decision=decision,
            metadata={
                "decision_time": iso(now),
                "reference_price": entry_price,
                "notional": notional,
                "per_unit_risk": entry_price - stop_price,
            },
        )

    def exit_order(
        self,
        symbol: str,
        qty: float,
        now: Optional[datetime] = None,
        intent: str = "exit",
    ) -> Optional[OrderRequest]:
        """Build a market sell to flatten a position (or part of it)."""
        cfg = self.symbols.get(symbol)
        step = self._qty_step(symbol) if cfg else self.default_qty_step
        q = quantize_down(abs(qty), step)
        if q <= 0:
            return None
        now = now or utcnow()
        client_id = make_client_order_id(symbol, "exit", intent, now, f"{q:.10f}")
        return OrderRequest(
            client_order_id=client_id,
            symbol=symbol,
            side="sell",
            order_type="market",
            qty=q,
        )
