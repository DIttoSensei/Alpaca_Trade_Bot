"""System health aggregation (spec section 89).

Produces a single HealthReport from independently-checkable components so the
main loop can decide whether to keep trading, degrade, or halt. Fail-closed: a
component that cannot be verified is reported as unhealthy, never assumed fine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from app.config.settings import Settings
from app.core.enums import HealthStatus
from app.core.types import HealthReport, utcnow
from app.monitoring.alerts import AlertManager
from app.risk.circuit_breaker import CircuitBreakerManager
from app.state.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ComponentHealth:
    name: str
    healthy: bool
    detail: str = ""
    critical: bool = False


class HealthMonitor:
    def __init__(
        self,
        settings: Settings,
        state: StateManager,
        breakers: CircuitBreakerManager | None = None,
        alerts: AlertManager | None = None,
    ) -> None:
        self.settings = settings
        self.state = state
        self.breakers = breakers
        self.alerts = alerts
        self._last_status: HealthStatus | None = None

    def check(
        self,
        websocket_available: bool | None = None,
        websocket_stale: bool | None = None,
        data_fresh: dict[str, bool] | None = None,
        extra: dict[str, ComponentHealth] | None = None,
    ) -> HealthReport:
        components: dict[str, Any] = {}
        criticals: list[str] = []

        # 1) Database.
        db_ok = self.state.db.healthy()
        components["database"] = {"healthy": db_ok}
        if not db_ok:
            criticals.append("database")

        # 2) Account freshness.
        acct_age = self._account_age_seconds()
        acct_ok = acct_age is not None and acct_age <= max(120, self.settings.account_sync_interval_seconds * 4)
        components["account_sync"] = {"healthy": acct_ok, "age_seconds": acct_age}
        if not acct_ok:
            criticals.append("account_sync")

        # 3) Market data freshness.
        if data_fresh is not None:
            stale = [s for s, ok in data_fresh.items() if not ok]
            components["market_data"] = {"healthy": not stale, "stale": stale}
            if stale:
                criticals.append("market_data")

        # 4) Websocket (non-critical: REST fallback exists).
        if websocket_available is not None or websocket_stale is not None:
            ws_ok = bool(websocket_available) and not bool(websocket_stale)
            components["websocket"] = {
                "healthy": ws_ok,
                "available": websocket_available,
                "stale": websocket_stale,
            }

        # 5) Circuit breakers.
        if self.breakers is not None:
            blocked, why = self.breakers.entries_blocked()
            snapped = self.breakers.snapshot()
            components["circuit_breakers"] = {"blocking_entries": blocked, "detail": why, **snapped}
            if blocked:
                criticals.append("circuit_breakers")

        # 6) Caller-supplied components.
        if extra:
            for name, comp in extra.items():
                components[name] = {"healthy": comp.healthy, "detail": comp.detail}
                if not comp.healthy and comp.critical:
                    criticals.append(name)

        status = self._aggregate(criticals, components)
        report = HealthReport(status=str(status), components=components, last_update=utcnow())

        if status is not self._last_status:
            logger.log(
                logging.ERROR if status is HealthStatus.EMERGENCY else logging.WARNING,
                "health status changed: %s (critical=%s)",
                status,
                criticals,
            )
            if self.alerts is not None:
                self.alerts.on_health(status, ", ".join(criticals) or "ok")
            self._last_status = status
        return report

    def _account_age_seconds(self) -> Optional[float]:
        if self.state.account is None:
            return None
        return (utcnow() - self.state.account.last_sync).total_seconds()

    @staticmethod
    def _aggregate(criticals: list[str], components: dict[str, Any]) -> HealthStatus:
        if not criticals:
            return HealthStatus.HEALTHY
        # Database / account / state mismatch are emergency conditions.
        if "database" in criticals or "account_sync" in criticals:
            return HealthStatus.EMERGENCY
        return HealthStatus.DEGRADED

    def can_trade(self, report: HealthReport) -> bool:
        return report.status == str(HealthStatus.HEALTHY)
