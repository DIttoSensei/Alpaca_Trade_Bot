"""Watchdog: detect stalled components and react (spec sections 90, 91).

Components "feed" named heartbeats on each successful cycle. The watchdog checks
whether any expected heartbeat has gone quiet for longer than its budget and
reports the offenders. The main loop maps these to DEGRADED/EMERGENCY states and
can escalate to a pause.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.monitoring.alerts import AlertManager
from app.state.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WatchdogIssue:
    component: str
    age_seconds: Optional[float]
    budget_seconds: float
    critical: bool


class Watchdog:
    def __init__(
        self,
        state: StateManager,
        alerts: AlertManager | None = None,
        default_budget_seconds: float = 120.0,
    ) -> None:
        self.state = state
        self.alerts = alerts
        self.default_budget_seconds = default_budget_seconds
        self._budgets: dict[str, float] = {}
        self._critical: dict[str, bool] = {}

    def expect(self, component: str, budget_seconds: Optional[float] = None, critical: bool = False) -> None:
        self._budgets[component] = budget_seconds if budget_seconds is not None else self.default_budget_seconds
        self._critical[component] = critical

    def feed(self, component: str, detail: str = "") -> None:
        self.state.heartbeat(component, detail)

    def check(self) -> list[WatchdogIssue]:
        from app.core.types import utcnow

        now = utcnow()
        heartbeats = self.state.heartbeats()
        issues: list[WatchdogIssue] = []
        for component, budget in self._budgets.items():
            last = heartbeats.get(component)
            if last is None:
                issues.append(WatchdogIssue(component, None, budget, self._critical.get(component, False)))
                continue
            age = (now - last).total_seconds()
            if age > budget:
                issues.append(WatchdogIssue(component, age, budget, self._critical.get(component, False)))

        if issues and self.alerts is not None:
            critical = [i for i in issues if i.critical]
            if critical:
                self.alerts.critical(
                    "watchdog",
                    "; ".join(f"{i.component} quiet {i.age_seconds}s (budget {i.budget_seconds}s)" for i in critical),
                )
            else:
                self.alerts.warning(
                    "watchdog",
                    "; ".join(f"{i.component} quiet {i.age_seconds}s" for i in issues),
                )
        return issues

    @property
    def has_critical(self) -> bool:
        return any(i.critical for i in self.check())
