"""Risk engine facade.

Ties together sizing, stops, exposure, correlation and circuit breakers into a
single ``plan`` call used by the decision composer. The risk engine ALWAYS has
authority over the strategy (spec sections 86, 120).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.risk_config import RiskConfig
from app.config.symbols import SymbolRegistry
from app.core.enums import RejectReason
from app.core.types import PositionRecord
from app.risk.circuit_breaker import CircuitBreakerManager
from app.risk.correlation_manager import CorrelationManager
from app.risk.exposure_manager import ExposureManager, ExposureSnapshot
from app.risk.position_sizing import PositionSizer, SizingResult
from app.risk.stop_manager import StopManager, StopPlan


@dataclass(slots=True)
class RiskPlan:
    allowed: bool
    reject_reason: RejectReason = RejectReason.NONE
    detail: str = ""
    stop_plan: StopPlan | None = None
    sizing: SizingResult | None = None
    effective_risk_multiplier: float = 1.0
    exposure: ExposureSnapshot | None = None
    correlation_scale: float = 1.0
    risk_amount: float = 0.0
    metadata: dict = field(default_factory=dict)


class RiskEngine:
    def __init__(self, risk: RiskConfig, symbols: SymbolRegistry) -> None:
        self.risk = risk
        self.symbols = symbols
        self.sizer = PositionSizer(risk)
        self.stops = StopManager(risk)
        self.exposure = ExposureManager(risk, symbols)
        self.correlation = CorrelationManager(risk)
        self.breakers = CircuitBreakerManager(risk)

    def strategy_risk_multiplier(self, strategy_name: str, base: float = 1.0) -> float:
        default = self.risk.strategy_risk_defaults.get(strategy_name, 1.0)
        return base * default

    def plan(
        self,
        equity: float,
        symbol: str,
        entry_price: float,
        atr: float,
        strategy_name: str,
        positions: dict[str, PositionRecord],
        return_series: dict[str, list[float]] | None = None,
        buying_power: float | None = None,
        recent_volume: float | None = None,
        base_risk_multiplier: float = 1.0,
        extreme_volatility: bool = False,
    ) -> RiskPlan:
        # 1) Hard breakers block all new entries.
        blocked, why = self.breakers.entries_blocked()
        if blocked:
            return RiskPlan(False, RejectReason.DAILY_RISK_EXCEEDED, why)

        # 2) Exposure limits.
        snap = self.exposure.snapshot(equity, positions)
        can, reason = self.exposure.can_open(snap, symbol, positions)
        if not can:
            return RiskPlan(
                False,
                RejectReason(reason) if reason in RejectReason.__members__ else RejectReason.PORTFOLIO_EXPOSURE_LIMIT,
                reason,
                exposure=snap,
            )

        # 3) Stop plan (ATR based).
        stop_plan = self.stops.initial_stop(entry_price, atr, side="buy")

        # 4) Risk multiplier stack: strategy * breakeven/cooldown * correlation * vol.
        risk_mult = self.strategy_risk_multiplier(strategy_name, base_risk_multiplier)
        risk_mult *= self.breakers.risk_scale()

        corr_scale = 1.0
        if return_series is not None:
            open_symbols = [s for s, p in positions.items() if p.qty != 0 and s != symbol]
            adj = self.correlation.adjustment(symbol, open_symbols, return_series)
            corr_scale = adj.scale
            risk_mult *= corr_scale

        if extreme_volatility:
            risk_mult *= self.risk.extreme_vol_risk_scale

        # 5) Sizing.
        symbol_room = self.exposure.symbol_room(snap, symbol)
        portfolio_room = self.exposure.remaining_portfolio_room(snap)
        liquidity_limit = None
        if recent_volume is not None and entry_price > 0:
            liquidity_limit = self.risk.max_order_size_vs_volume * recent_volume * entry_price

        sizing = self.sizer.size(
            equity=equity,
            entry_price=entry_price,
            stop_distance_pct=stop_plan.stop_distance_pct,
            risk_multiplier=max(0.0, risk_mult),
            symbol_exposure_limit=None,  # applied via room below
            portfolio_remaining=min(symbol_room, portfolio_room) if (symbol_room or portfolio_room) else None,
            buying_power=buying_power,
            liquidity_limit_notional=liquidity_limit,
        )
        sizing.risk_amount = equity * self.risk.risk_per_trade * max(0.0, risk_mult)

        if sizing.quantity <= 0:
            return RiskPlan(
                False,
                RejectReason.INSUFFICIENT_BUYING_POWER,
                "computed size is zero",
                stop_plan=stop_plan,
                sizing=sizing,
                exposure=snap,
            )

        return RiskPlan(
            allowed=True,
            stop_plan=stop_plan,
            sizing=sizing,
            effective_risk_multiplier=max(0.0, risk_mult),
            exposure=snap,
            correlation_scale=corr_scale,
            risk_amount=sizing.risk_amount,
            metadata={
                "binding_constraint": sizing.binding_constraint,
                "risk_multiplier": round(risk_mult, 4),
            },
        )
