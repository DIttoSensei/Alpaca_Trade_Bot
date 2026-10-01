"""Dynamic position sizing (spec sections 27-29).

Risk is defined in dollars/equity percentage, not position size. Final size is
the minimum of all applicable constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.risk_config import RiskConfig


@dataclass(slots=True)
class SizingResult:
    quantity: float
    notional: float
    risk_amount: float
    stop_distance_pct: float
    binding_constraint: str = ""


class PositionSizer:
    def __init__(self, risk: RiskConfig) -> None:
        self.risk = risk

    def size(
        self,
        equity: float,
        entry_price: float,
        stop_distance_pct: float,
        risk_multiplier: float = 1.0,
        symbol_exposure_limit: float | None = None,
        portfolio_remaining: float | None = None,
        buying_power: float | None = None,
        liquidity_limit_notional: float | None = None,
    ) -> SizingResult:
        if entry_price <= 0 or equity <= 0 or stop_distance_pct <= 0:
            return SizingResult(0.0, 0.0, 0.0, stop_distance_pct, "invalid_input")

        risk_amount = equity * self.risk.risk_per_trade * risk_multiplier
        # theoretical position: risk_amount / stop_distance_pct
        target_notional = risk_amount / stop_distance_pct

        constraints: list[tuple[str, float]] = [("risk_based", target_notional)]

        if symbol_exposure_limit is not None:
            constraints.append(("symbol_exposure", symbol_exposure_limit * equity))
        if portfolio_remaining is not None:
            constraints.append(("portfolio_exposure", max(0.0, portfolio_remaining)))
        if buying_power is not None:
            constraints.append(("buying_power", max(0.0, buying_power)))
        if liquidity_limit_notional is not None:
            constraints.append(("liquidity", max(0.0, liquidity_limit_notional)))

        binding, notional = min(constraints, key=lambda x: x[1])
        quantity = notional / entry_price if entry_price > 0 else 0.0
        return SizingResult(
            quantity=max(0.0, quantity),
            notional=max(0.0, notional),
            risk_amount=risk_amount,
            stop_distance_pct=stop_distance_pct,
            binding_constraint=binding,
        )
