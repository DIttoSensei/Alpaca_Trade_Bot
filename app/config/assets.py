"""Asset/timeframe context configuration.

Defines the multi-timeframe roles and which timeframes feed higher-timeframe
context. Kept separate from symbols so both can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import Timeframe


@dataclass(slots=True)
class ContextConfig:
    execution: Timeframe = Timeframe.M15
    confirmation: Timeframe = Timeframe.H1
    regime: Timeframe = Timeframe.H4
    macro: Timeframe = Timeframe.D1
    micro: Timeframe = Timeframe.M5

    def by_role(self, role: str) -> Timeframe | None:
        return {
            "execution": self.execution,
            "confirmation": self.confirmation,
            "regime": self.regime,
            "macro": self.macro,
            "micro": self.micro,
        }.get(role)

    def context_roles(self) -> list[Timeframe]:
        return [self.confirmation, self.regime, self.macro]

    def all_required(self) -> list[Timeframe]:
        return [self.micro, self.execution, self.confirmation, self.regime, self.macro]

    def to_dict(self) -> dict[str, str]:
        return {
            "execution": str(self.execution),
            "confirmation": str(self.confirmation),
            "regime": str(self.regime),
            "macro": str(self.macro),
            "micro": str(self.micro),
        }
