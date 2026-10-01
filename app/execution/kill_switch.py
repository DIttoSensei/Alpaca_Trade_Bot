"""Kill switch (spec section 96).

A file-based switch so an operator can halt new entries (and optionally force
liquidation) without SSH-ing into a running process. The switch is checked before
every new entry. It is fail-closed: if we cannot read the filesystem we treat the
switch as engaged for entries.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KillSwitch:
    def __init__(self, path: Path | str, liquidate_on_engage: bool = False) -> None:
        self.path = Path(path)
        self.liquidate_on_engage = liquidate_on_engage
        self._engaged = False

    def engaged(self) -> bool:
        """True if the switch is engaged. Fail-closed on read errors."""
        if self._engaged:
            return True
        try:
            return self.path.exists()
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("kill switch unreadable (%s); treating as engaged", exc)
            return True

    def engage(self, reason: str = "") -> None:
        self._engaged = True
        logger.critical("KILL SWITCH engaged: %s", reason or "manual")
        try:
            self.path.write_text(reason or "engaged", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("could not write kill switch file: %s", exc)

    def release(self) -> None:
        self._engaged = False
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("could not remove kill switch file: %s", exc)
