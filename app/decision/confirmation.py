"""Cross-asset confirmation: BTC market filter and relative-strength filter.

Spec sections 23-24. BTC is the primary market-regime reference. For altcoins
we require BOTH relative strength AND acceptable absolute regime -- relative
strength alone can mask a dangerous absolute market.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.risk_config import RiskConfig
from app.core.enums import Regime, RejectReason, SignalCategory
from app.core.types import FeatureSnapshot, MarketState


@dataclass(slots=True)
class ConfirmationResult:
    passed: bool
    reason: RejectReason = RejectReason.NONE
    detail: str = ""


class BTCFilter:
    def __init__(self, reference: str = "BTC/USD", require_confirmation: bool = True) -> None:
        self.reference = reference
        self.require_confirmation = require_confirmation

    def check(self, state: MarketState) -> ConfirmationResult:
        if not self.require_confirmation:
            return ConfirmationResult(True)
        if state.symbol == self.reference:
            return ConfirmationResult(True)

        btc = state.btc_features
        if btc is None:
            # No BTC data -> fail closed when confirmation is required.
            return ConfirmationResult(False, RejectReason.BTC_CONFIRMATION_FAILED, "no BTC data")

        ret = btc.values.get("return_6")
        vol_pct = btc.values.get("vol_percentile")
        rsi = btc.values.get("rsi")

        # Severe BTC panic -> disable altcoin entries (unless a recovery trade).
        if state.is_panic:
            return ConfirmationResult(False, RejectReason.MARKET_PANIC, "market panic")
        if ret is not None and ret <= -0.04 and vol_pct is not None and vol_pct >= 0.8:
            return ConfirmationResult(False, RejectReason.MARKET_PANIC, "BTC severe decline + high vol")
        if ret is not None and ret <= -0.06:
            return ConfirmationResult(False, RejectReason.BTC_CONFIRMATION_FAILED, "BTC sharp decline")

        # Weak but non-panic: allow if BTC not outright collapsing.
        if rsi is not None and rsi < 25 and ret is not None and ret < 0:
            return ConfirmationResult(False, RejectReason.BTC_CONFIRMATION_FAILED, "BTC weak momentum")

        return ConfirmationResult(True)


class RelativeStrengthFilter:
    def __init__(self, min_relative_strength: float = 0.0) -> None:
        self.min_relative_strength = min_relative_strength

    def check(self, state: MarketState, cfg: RiskConfig | None = None) -> ConfirmationResult:
        rs = state.relative_strength
        if rs is None:
            # No RS data: do not block on missing optional context.
            return ConfirmationResult(True)
        if rs < self.min_relative_strength:
            return ConfirmationResult(
                False,
                RejectReason.RELATIVE_STRENGTH_FAILED,
                f"relative strength {rs:.4f} < {self.min_relative_strength}",
            )
        return ConfirmationResult(True)
