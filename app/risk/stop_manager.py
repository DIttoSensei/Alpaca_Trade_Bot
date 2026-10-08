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
        trend_mode: bool = False,
    ) -> StopPlan:
        """Compute the initial ATR-based stop and R-multiple target.

        stop_distance = ATR * multiplier, bounded by [min_stop_distance, max_stop_distance].

        When ``trend_mode`` is set the target is left OPEN (``target_price`` 0.0,
        meaning "no fixed target") because capping a trend trade at 2R caps the
        edge; the trade is instead exited by the trailing stop or a trend exit.
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        atr_pct = atr / entry_price if atr > 0 else self.risk.min_stop_distance
        stop_pct = atr_pct * self.risk.atr_stop_multiplier
        stop_pct = max(self.risk.min_stop_distance, min(self.risk.max_stop_distance, stop_pct))

        risk_per_unit = entry_price * stop_pct
        if trend_mode:
            # No fixed target: ride the trend until the trailing/trend exit fires.
            stop_price = entry_price - risk_per_unit if side == "buy" else entry_price + risk_per_unit
            target_price = 0.0
            r_multiple = self.risk.trend_target_r_multiple
        elif side == "buy":
            stop_price = entry_price - risk_per_unit
            target_price = entry_price + risk_per_unit * self.risk.target_r_multiple
            r_multiple = self.risk.target_r_multiple
        else:  # short (future)
            stop_price = entry_price + risk_per_unit
            target_price = entry_price - risk_per_unit * self.risk.target_r_multiple
            r_multiple = self.risk.target_r_multiple

        return StopPlan(
            stop_price=max(0.0, stop_price),
            target_price=max(0.0, target_price),
            stop_distance_pct=stop_pct,
            risk_per_unit=risk_per_unit,
            target_r_multiple=r_multiple,
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
        trend_mode: bool = False,
    ) -> tuple[float, str]:
        """Return (new_stop, trailing_state_name).

        - profit > breakeven_trigger_r -> move stop toward breakeven
        - profit > trailing_trigger_r -> activate ATR trailing
        Never loosens the stop.

        In ``trend_mode`` the trailing distance uses the wider trend ATR multiple
        (and the trend trigger) so a trend position is not shaken out by noise.
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

        # Activate ATR trailing after the (mode-specific) trigger.
        trail_trigger = self.risk.trend_trailing_trigger_r if trend_mode else self.risk.trailing_trigger_r
        trail_mult = (
            self.risk.trend_trailing_atr_multiplier if trend_mode else self.risk.trailing_atr_multiplier
        )
        # r_multiple >= 0 is always true when the trigger is 0 (trail from entry).
        if r_multiple >= trail_trigger:
            trail_distance = atr * trail_mult
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
