"""Canonical enums used across the whole system.

Keeping these in one place prevents stringly-typed drift between modules and
makes persisted state (SQLite) and logs unambiguous.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum that serialises cleanly to JSON/SQLite."""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


class TradingMode(StrEnum):
    """Explicit operating modes. PAPER is the safe default."""

    BACKTEST = "backtest"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        table = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return table[self.value]


class Action(StrEnum):
    """Signal / decision actions."""

    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"


class Regime(StrEnum):
    """Composite market regime classification."""

    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGE = "RANGE"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    PANIC = "PANIC"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


class TrendRegime(StrEnum):
    STRONG_UP = "STRONG_UP"
    WEAK_UP = "WEAK_UP"
    RANGE = "RANGE"
    WEAK_DOWN = "WEAK_DOWN"
    STRONG_DOWN = "STRONG_DOWN"
    UNKNOWN = "UNKNOWN"


class VolatilityRegime(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


class SignalCategory(StrEnum):
    """Feature categories used to prevent correlated-indicator stacking."""

    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    STRUCTURE = "STRUCTURE"
    RELATIVE_STRENGTH = "RELATIVE_STRENGTH"


class OrderStatus(StrEnum):
    PLANNED = "PLANNED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PROTECTED = "PROTECTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class PositionState(StrEnum):
    FLAT = "FLAT"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    PROTECTED = "PROTECTED"
    TRAILING = "TRAILING"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class TradingState(StrEnum):
    """Global trading state machine."""

    STARTING = "STARTING"
    SYNCING = "SYNCING"
    READY = "READY"
    TRADING = "TRADING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    RISK_LOCK = "RISK_LOCK"
    EMERGENCY = "EMERGENCY"
    STOPPED = "STOPPED"


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    EMERGENCY = "EMERGENCY"


class ExitReason(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    STRATEGY_INVALIDATION = "STRATEGY_INVALIDATION"
    REGIME_REVERSAL = "REGIME_REVERSAL"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"
    MANUAL = "MANUAL"
    RECONCILED = "RECONCILED"


class RejectReason(StrEnum):
    """Why a candidate trade did not proceed. Persisted for opportunity analysis."""

    NONE = "NONE"
    NO_SIGNAL = "NO_SIGNAL"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    EXPECTED_EDGE_TOO_SMALL = "EXPECTED_EDGE_TOO_SMALL"
    MARKET_PANIC = "MARKET_PANIC"
    BTC_CONFIRMATION_FAILED = "BTC_CONFIRMATION_FAILED"
    RELATIVE_STRENGTH_FAILED = "RELATIVE_STRENGTH_FAILED"
    VOLATILITY_TOO_HIGH = "VOLATILITY_TOO_HIGH"
    POSITION_LIMIT = "POSITION_LIMIT"
    SYMBOL_EXPOSURE_LIMIT = "SYMBOL_EXPOSURE_LIMIT"
    PORTFOLIO_EXPOSURE_LIMIT = "PORTFOLIO_EXPOSURE_LIMIT"
    DAILY_RISK_EXCEEDED = "DAILY_RISK_EXCEEDED"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"
    REGIME_NOT_ALLOWED = "REGIME_NOT_ALLOWED"
    DATA_STALE = "DATA_STALE"
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    LIQUIDITY_INSUFFICIENT = "LIQUIDITY_INSUFFICIENT"
    ML_VETO = "ML_VETO"
    KILL_SWITCH = "KILL_SWITCH"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"


class CircuitBreakerType(StrEnum):
    DAILY_LOSS = "DAILY_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    LOSING_STREAK = "LOSING_STREAK"
    API_FAILURES = "API_FAILURES"
    WEBSOCKET_DOWN = "WEBSOCKET_DOWN"
    STATE_MISMATCH = "STATE_MISMATCH"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    STRATEGY_DEGRADED = "STRATEGY_DEGRADED"
    DATA_STALE = "DATA_STALE"
