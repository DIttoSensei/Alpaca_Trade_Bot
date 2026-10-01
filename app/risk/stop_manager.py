"""Dynamic stop-loss, take-profit and trailing management (spec sections 26, 47).

Stops are volatility-adjusted (ATR x multiplier) with enforced min/max bounds.
Never use one universal percentage stop. Never move a stop farther away to
accommodate a losing position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config.risk_config import RiskConfig


@dataclass(slots=True)
class StopPlan:
    stop_price: float
    target_price: float
    stop_distance_pct: float
    risk_per_unit: float
    target_r_multiple: float


class StopManager:
    def __init__(self, risk: RiskConfig) -> None:
        self.risk = risk

    def initial_stop(
        self,
        entry_price: float,
        atr: float,
        side: str = "buy",
    ) -> StopPlan:
        """Compute the initial ATR-based stop and R-multiple target.

        stop_distance = ATR * multiplier, bounded by [min_stop_distance, max_stop_distance].
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        atr_pct = atr / entry_price if atr > 0 else self.risk.min_stop_distance
        stop_pct = atr_pct * self.risk.atr_stop_multiplier
        stop_pct = max(self.risk.min_stop_distance, min(self.risk.max_stop_distance, stop_pct))

        risk_per_unit = entry_price * stop_pct
        if side == "buy":
            stop_price = entry_price - risk_per_unit
            target_price = entry_price + risk_per_unit * self.risk.target_r_multiple
        else:  # short (future)
            stop_price = entry_price + risk_per_unit
            target_price = entry_price - risk_per_unit * self.risk.target_r_multiple

        return StopPlan(
            stop_price=max(0.0, stop_price),
            target_price=max(0.0, target_price),
            stop_distance_pct=stop_pct,
            risk_per_unit=risk_per_unit,
            target_r_multiple=self.risk.target_r_multiple,
        )

    def update_trailing(
        self,
        entry_price: float,
        current_price: float,
        initial_risk_per_unit: float,
        current_stop: float,
        atr: float,
        side: str = "buy",
        already_trailing: bool = False,
    ) -> tuple[float, str]:
        """Return (new_stop, trailing_state_name).

        - profit > breakeven_trigger_r -> move stop toward breakeven
        - profit > trailing_trigger_r -> activate ATR trailing
        Never loosens the stop.
        """
        if initial_risk_per_unit <= 0:
            return current_stop, "none"

        if side == "buy":
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price
        r_multiple = profit / initial_risk_per_unit

        new_stop = current_stop
        state = "none"

        # Move toward breakeven after 1R
        if r_multiple >= self.risk.breakeven_trigger_r:
            breakeven = entry_price
            if side == "buy":
                new_stop = max(new_stop, breakeven)
            else:
                new_stop = min(new_stop, breakeven)
            state = "breakeven"

        # Activate ATR trailing after trailing_trigger_r
        if r_multiple >= self.risk.trailing_trigger_r:
            trail_distance = atr * self.risk.trailing_atr_multiplier
            if side == "buy":
                candidate = current_price - trail_distance
                new_stop = max(new_stop, candidate)
            else:
                candidate = current_price + trail_distance
                new_stop = min(new_stop, candidate)
            state = "trailing"

        # Enforce monotonic tightening: never move a stop away (spec section 47).
        if side == "buy":
            new_stop = max(current_stop, new_stop)
        else:
            new_stop = min(current_stop, new_stop)

        if state == "none":
            state = "trailing" if already_trailing else "none"
        return new_stop, state
