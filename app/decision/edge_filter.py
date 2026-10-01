"""Cost / expected-edge engine (spec section 25).

Every candidate trade must pass through here. The bot must not enter simply
because a strategy says BUY; it must ask whether the expected opportunity is
large enough to justify trading after fees, spread, slippage and a risk buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.risk_config import RiskConfig
from app.config.settings import FeeModel
from app.core.types import CostEstimate, Quote


@dataclass(slots=True)
class EdgeResult:
    passed: bool
    expected_move: float
    cost: CostEstimate
    remaining_edge: float
    reason: str = ""


class EdgeFilter:
    def __init__(self, fees: FeeModel, risk: RiskConfig) -> None:
        self.fees = fees
        self.risk = risk

    def estimate_cost(self, quote: Quote | None = None) -> CostEstimate:
        # Fees are taker both ways (market entries assumed).
        entry_fee = self.fees.taker_fee
        exit_fee = self.fees.taker_fee
        slippage = self.fees.estimated_slippage
        # Use live spread if available, else the configured spread cost.
        if quote is not None and quote.mid > 0:
            spread = quote.spread_pct
        else:
            spread = self.fees.spread_cost
        return CostEstimate(entry_fee=entry_fee, exit_fee=exit_fee, slippage=slippage, spread=spread)

    def evaluate(
        self,
        expected_move: float,
        quote: Quote | None = None,
        require: bool | None = None,
        min_edge: float | None = None,
    ) -> EdgeResult:
        require = self.risk.require_cost_edge if require is None else require
        min_edge = self.risk.min_expected_edge if min_edge is None else min_edge
        cost = self.estimate_cost(quote)
        remaining = expected_move - cost.total - self.risk.risk_buffer

        if not require:
            return EdgeResult(True, expected_move, cost, remaining, "cost filter disabled")
        if remaining < min_edge:
            return EdgeResult(
                False,
                expected_move,
                cost,
                remaining,
                f"remaining edge {remaining:.4f} < required {min_edge:.4f}",
            )
        return EdgeResult(True, expected_move, cost, remaining, "edge sufficient")
