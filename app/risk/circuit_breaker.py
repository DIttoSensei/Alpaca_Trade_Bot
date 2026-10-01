"""Circuit breakers (spec sections 30-33, 131).

Tracks daily loss, max drawdown, losing streaks and infrastructure failures.
A tripped breaker blocks NEW entries but does not liquidate existing positions
unless a separate emergency rule requires it. Fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from app.config.risk_config import RiskConfig
from app.core.enums import CircuitBreakerType


@dataclass(slots=True)
class BreakerState:
    tripped: bool = False
    kind: Optional[CircuitBreakerType] = None
    detail: str = ""
    tripped_at: Optional[datetime] = None


class CircuitBreakerManager:
    def __init__(self, risk: RiskConfig) -> None:
        self.risk = risk
        self._day: date = datetime.now(timezone.utc).date()
        self._day_start_equity: Optional[float] = None
        self._realized_today: float = 0.0
        self._consecutive_losses: int = 0
        self._cooldown_remaining: int = 0
        self._peak_equity: Optional[float] = None
        self._api_failures: int = 0
        self.breakers: dict[CircuitBreakerType, BreakerState] = {}

    # -- equity tracking -------------------------------------------------
    def start_day(self, equity: float, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        today = now.date()
        if today != self._day:
            self._day = today
            self._day_start_equity = equity
            self._realized_today = 0.0
        if self._day_start_equity is None:
            self._day_start_equity = equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def update_equity(self, equity: float) -> None:
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    # -- trade outcomes --------------------------------------------------
    def record_trade_result(self, net_pnl: float) -> None:
        self._realized_today += net_pnl
        if net_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def record_api_failure(self) -> None:
        self._api_failures += 1
        if self._api_failures >= 5:
            self.trip(CircuitBreakerType.API_FAILURES, f"{self._api_failures} consecutive API failures")

    def record_api_success(self) -> None:
        self._api_failures = 0
        self.reset(CircuitBreakerType.API_FAILURES)

    # -- evaluation ------------------------------------------------------
    def evaluate(self, equity: float, unrealized: float = 0.0, now: Optional[datetime] = None) -> list[BreakerState]:
        now = now or datetime.now(timezone.utc)
        self.start_day(equity, now)
        self.update_equity(equity)
        tripped: list[BreakerState] = []

        # Daily loss limit (realized + unrealized).
        if self._day_start_equity and self._day_start_equity > 0:
            daily_pnl = (equity - self._day_start_equity)
            daily_pct = daily_pnl / self._day_start_equity
            if daily_pct <= -self.risk.max_daily_loss:
                st = self.trip(
                    CircuitBreakerType.DAILY_LOSS,
                    f"daily loss {daily_pct:.2%} <= -{self.risk.max_daily_loss:.2%}",
                    now,
                )
                tripped.append(st)

        # Max drawdown.
        if self._peak_equity and self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd >= self.risk.max_drawdown:
                st = self.trip(
                    CircuitBreakerType.MAX_DRAWDOWN,
                    f"drawdown {dd:.2%} >= {self.risk.max_drawdown:.2%}",
                    now,
                )
                tripped.append(st)

        # Losing streak -> reduce risk (NOT full stop). Represented as a
        # cooldown flag; the risk engine scales risk down.
        if self._consecutive_losses >= self.risk.max_consecutive_losses:
            self._cooldown_remaining = max(self._cooldown_remaining, self.risk.losing_streak_cooldown_trades)

        return tripped

    def trip(self, kind: CircuitBreakerType, detail: str = "", now: Optional[datetime] = None) -> BreakerState:
        state = BreakerState(tripped=True, kind=kind, detail=detail, tripped_at=now or datetime.now(timezone.utc))
        self.breakers[kind] = state
        return state

    def reset(self, kind: CircuitBreakerType) -> None:
        if kind in self.breakers:
            self.breakers[kind] = BreakerState(tripped=False, kind=kind)

    # -- queries ---------------------------------------------------------
    def entries_blocked(self) -> tuple[bool, str]:
        """True if any hard breaker blocks new entries."""
        for kind, state in self.breakers.items():
            if state.tripped and kind in (
                CircuitBreakerType.DAILY_LOSS,
                CircuitBreakerType.MAX_DRAWDOWN,
                CircuitBreakerType.API_FAILURES,
                CircuitBreakerType.WEBSOCKET_DOWN,
                CircuitBreakerType.STATE_MISMATCH,
                CircuitBreakerType.DATABASE_FAILURE,
                CircuitBreakerType.DATA_STALE,
            ):
                return True, f"{kind}: {state.detail}"
        return False, ""

    def risk_scale(self) -> float:
        """Risk multiplier applied to sizing. <1 during cooldown."""
        if self._cooldown_remaining > 0:
            return self.risk.losing_streak_risk_scale
        return 1.0

    def consume_cooldown(self) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def daily_realized(self) -> float:
        return self._realized_today

    def snapshot(self) -> dict:
        return {
            "consecutive_losses": self._consecutive_losses,
            "realized_today": round(self._realized_today, 4),
            "day_start_equity": self._day_start_equity,
            "peak_equity": self._peak_equity,
            "cooldown_remaining": self._cooldown_remaining,
            "breakers": {str(k): v.tripped for k, v in self.breakers.items()},
        }
