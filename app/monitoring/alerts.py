"""Alerting.

Telegram when configured, structured logging always. Alerts are de-duplicated
with a cooldown so a flapping component cannot spam the operator (spec section
90). Alerting must never block or crash trading: failures are swallowed and
logged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.enums import HealthStatus

logger = logging.getLogger("app.alerts")


class AlertLevel(str):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class Alert:
    level: str
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)

    def format(self) -> str:
        return f"[{self.level}] {self.title}: {self.message}"


class AlertManager:
    def __init__(
        self,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
        cooldown_seconds: float = 300.0,
        critical_cooldown_seconds: float = 60.0,
    ) -> None:
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.cooldown = cooldown_seconds
        self.critical_cooldown = critical_cooldown_seconds
        self._last_sent: dict[str, float] = {}

    def _cooled_down(self, key: str, level: str) -> bool:
        now = time.time()
        window = self.critical_cooldown if level == AlertLevel.CRITICAL else self.cooldown
        last = self._last_sent.get(key, 0.0)
        if now - last >= window:
            self._last_sent[key] = now
            return True
        return False

    def send(self, level: str, title: str, message: str) -> None:
        alert = Alert(level=level, title=title, message=message)
        log_fn = logger.critical if level == AlertLevel.CRITICAL else (
            logger.warning if level == AlertLevel.WARNING else logger.info
        )
        log_fn(alert.format())
        if not self._cooled_down(f"{level}:{title}", level):
            return
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(alert)

    def _send_telegram(self, alert: Alert) -> None:
        try:
            import urllib.parse
            import urllib.request

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = urllib.parse.urlencode(
                {"chat_id": self.telegram_chat_id, "text": alert.format()}
            ).encode()
            with urllib.request.urlopen(url, data=data, timeout=5) as resp:  # noqa: S310
                resp.read()
        except Exception as exc:  # noqa: BLE001 - alerts must never break trading
            logger.warning("telegram alert failed: %s", exc)

    # -- convenience -----------------------------------------------------
    def info(self, title: str, message: str) -> None:
        self.send(AlertLevel.INFO, title, message)

    def warning(self, title: str, message: str) -> None:
        self.send(AlertLevel.WARNING, title, message)

    def critical(self, title: str, message: str) -> None:
        self.send(AlertLevel.CRITICAL, title, message)

    def on_health(self, status: HealthStatus, detail: str) -> None:
        if status is HealthStatus.HEALTHY:
            return
        level = AlertLevel.CRITICAL if status is HealthStatus.EMERGENCY else AlertLevel.WARNING
        self.send(level, f"health:{status}", detail)
