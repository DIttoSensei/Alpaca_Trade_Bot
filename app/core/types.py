"""Canonical internal data structures.

Strategy code must NEVER depend on Alpaca response JSON. Broker adapters convert
Alpaca responses into these objects, and the rest of the system only ever sees
these.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.enums import (
    Action,
    ExitReason,
    OrderStatus,
    PositionState,
    Regime,
    RejectReason,
    SignalCategory,
    Timeframe,
)


def utcnow() -> datetime:
    """Timezone-aware UTC now. All internal timestamps are UTC."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Normalise any datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt: datetime) -> str:
    """Serialise a datetime to an ISO-8601 UTC string."""
    return ensure_utc(dt).isoformat()


@dataclass(slots=True)
class Candle:
    """Canonical OHLCV candle."""

    timestamp: datetime
    symbol: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": iso(self.timestamp),
            "symbol": self.symbol,
            "timeframe": str(self.timeframe),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(slots=True)
class Quote:
    """Latest top-of-book quote for a symbol."""

    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp: datetime = field(default_factory=utcnow)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        return (self.spread / mid) if mid > 0 else 0.0


@dataclass(slots=True)
class Signal:
    """A single strategy output. Strategies never place orders."""

    action: Action
    confidence: float
    score: float
    strategy_name: str
    reason: str = ""
    stop_model: Optional[dict[str, Any]] = None
    target_model: Optional[dict[str, Any]] = None
    category_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = str(self.action)
        return d


@dataclass(slots=True)
class FeatureSnapshot:
    """Immutable set of feature values at a point in time.

    Stored verbatim with every trade so we can answer "what did the bot know?"
    months later. Never reconstruct from current data.
    """

    symbol: str
    timestamp: datetime
    timeframe: Timeframe
    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": iso(self.timestamp),
            "timeframe": str(self.timeframe),
            **self.values,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), default=str)


@dataclass(slots=True)
class MarketState:
    """Everything a strategy is allowed to see when evaluating.

    Both live and backtest construct this object, then call strategy.evaluate().
    This is the key to preventing backtest/live divergence.
    """

    symbol: str
    timestamp: datetime
    regime: Regime
    features: FeatureSnapshot
    context: dict[str, FeatureSnapshot] = field(default_factory=dict)
    # context holds higher-timeframe snapshots, e.g. {"1h": ..., "4h": ...}
    btc_features: Optional[FeatureSnapshot] = None
    relative_strength: Optional[float] = None
    quote: Optional[Quote] = None
    is_panic: bool = False
    is_recovery: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TradeDecision:
    """Output of the decision layer; input to the order planner."""

    symbol: str
    side: str  # "buy" (long spot only initially)
    quantity: float
    entry_type: str  # "market" | "limit"
    limit_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    strategy: str
    score: float
    expected_edge: float
    regime: Regime
    reason: str = ""
    risk_amount: float = 0.0
    risk_multiplier: float = 1.0
    feature_snapshot: Optional[FeatureSnapshot] = None
    estimated_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["regime"] = str(self.regime)
        d.pop("feature_snapshot", None)
        return d


@dataclass(slots=True)
class RejectedCandidate:
    """A candidate that did not become a trade, with the reason why."""

    symbol: str
    timestamp: datetime
    strategy: str
    score: float
    expected_edge: float
    reason: RejectReason
    regime: Regime
    detail: str = ""
    feature_snapshot: Optional[FeatureSnapshot] = None


@dataclass(slots=True)
class OrderRecord:
    """Local mirror of (and intent for) a broker order."""

    client_order_id: str
    alpaca_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    qty: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PLANNED
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_qty: float = 0.0
    average_fill_price: Optional[float] = None
    strategy: str = ""
    # Intent: why this order exists ("entry" | "stop" | "target")
    intent: str = "entry"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = str(self.status)
        d["submitted_at"] = iso(self.submitted_at) if self.submitted_at else None
        d["updated_at"] = iso(self.updated_at) if self.updated_at else None
        return d


@dataclass(slots=True)
class PositionRecord:
    """Local mirror of an open position, including protection state."""

    symbol: str
    qty: float
    average_entry: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    strategy: str = ""
    entry_time: Optional[datetime] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    trailing_state: dict[str, Any] = field(default_factory=dict)
    state: PositionState = PositionState.FLAT
    risk_amount: float = 0.0
    initial_stop_price: Optional[float] = None
    initial_risk_per_unit: float = 0.0
    feature_snapshot: Optional[FeatureSnapshot] = None
    regime: Optional[Regime] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "average_entry": self.average_entry,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "strategy": self.strategy,
            "entry_time": iso(self.entry_time) if self.entry_time else None,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "trailing_state": self.trailing_state,
            "state": str(self.state),
            "risk_amount": self.risk_amount,
            "regime": str(self.regime) if self.regime else None,
        }


@dataclass(slots=True)
class AccountState:
    """Snapshot of the broker account."""

    equity: float
    cash: float
    buying_power: float
    last_sync: datetime = field(default_factory=utcnow)

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "last_sync": iso(self.last_sync),
        }


@dataclass(slots=True)
class TradeRecord:
    """A completed round-trip trade."""

    trade_id: str
    symbol: str
    strategy: str
    entry_price: float
    exit_price: float
    qty: float
    fees: float
    slippage: float
    gross_pnl: float
    net_pnl: float
    entry_time: datetime
    exit_time: datetime
    holding_time_seconds: float
    exit_reason: ExitReason
    regime: Regime
    signal_score: float
    expected_edge: float
    estimated_cost: float
    actual_cost: float
    feature_snapshot: Optional[FeatureSnapshot] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "fees": self.fees,
            "slippage": self.slippage,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "entry_time": iso(self.entry_time),
            "exit_time": iso(self.exit_time),
            "holding_time_seconds": self.holding_time_seconds,
            "exit_reason": str(self.exit_reason),
            "regime": str(self.regime),
            "signal_score": self.signal_score,
            "expected_edge": self.expected_edge,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
        }


@dataclass(slots=True)
class StrategyHealth:
    """Rolling performance stats for a single strategy."""

    strategy: str
    trade_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_drawdown: float = 0.0
    recent_performance: float = 0.0
    weight: float = 1.0
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CostEstimate:
    """Breakdown of expected trading cost for a candidate trade."""

    entry_fee: float
    exit_fee: float
    slippage: float
    spread: float

    @property
    def total(self) -> float:
        return self.entry_fee + self.exit_fee + self.slippage + self.spread

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        return d


@dataclass(slots=True)
class EnsembleResult:
    """Combined view of all strategy signals for one symbol."""

    symbol: str
    action: Action
    weighted_score: float
    confidence: float
    agreement: float
    conflict: bool
    contributing: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class HealthReport:
    """System health snapshot."""

    status: str
    components: dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=utcnow)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "components": self.components,
            "last_update": iso(self.last_update),
        }


# ---------------------------------------------------------------------------
# Category contribution caps (spec section 22): prevents indicator stacking
# from artificially inflating confidence.
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY_CAPS: dict[SignalCategory, float] = {
    SignalCategory.TREND: 0.30,
    SignalCategory.MOMENTUM: 0.20,
    SignalCategory.VOLUME: 0.15,
    SignalCategory.STRUCTURE: 0.20,
    SignalCategory.RELATIVE_STRENGTH: 0.15,
}
