"""Trade decision composer.

Converts an ensemble result into either a TradeDecision or a RejectedCandidate
with an explicit reason. This is the gatekeeper that enforces, in order:

  ensemble -> BTC filter -> relative strength -> expected-edge/cost filter -> risk

Every rejection is recorded so we can later ask "why did we miss this?" (spec
sections 77-78).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.risk_config import RiskConfig
from app.config.settings import FeeModel
from app.core.enums import Action, RejectReason, Regime
from app.core.types import (
    EnsembleResult,
    MarketState,
    PositionRecord,
    RejectedCandidate,
    TradeDecision,
)
from app.decision.confirmation import BTCFilter, RelativeStrengthFilter
from app.decision.edge_filter import EdgeFilter
from app.risk.engine import RiskEngine


@dataclass(slots=True)
class DecisionOutcome:
    decision: TradeDecision | None
    rejection: RejectedCandidate | None

    @property
    def accepted(self) -> bool:
        return self.decision is not None


class TradeDecisionComposer:
    def __init__(
        self,
        fees: FeeModel,
        risk: RiskConfig,
        risk_engine: RiskEngine,
        reference_symbol: str = "BTC/USD",
        require_btc_confirmation: bool = True,
        min_relative_strength: float = 0.0,
    ) -> None:
        self.edge = EdgeFilter(fees, risk)
        self.risk = risk
        self.risk_engine = risk_engine
        self.btc_filter = BTCFilter(reference_symbol, require_btc_confirmation)
        self.rs_filter = RelativeStrengthFilter(min_relative_strength)

    def _reject(
        self,
        state: MarketState,
        ensemble: EnsembleResult,
        reason: RejectReason,
        detail: str,
        expected_edge: float = 0.0,
        strategy: str = "",
    ) -> DecisionOutcome:
        return DecisionOutcome(
            decision=None,
            rejection=RejectedCandidate(
                symbol=state.symbol,
                timestamp=state.timestamp,
                strategy=strategy or ensemble.reason,
                score=ensemble.weighted_score,
                expected_edge=expected_edge,
                reason=reason,
                regime=state.regime,
                detail=detail,
                feature_snapshot=state.features,
            ),
        )

    def compose(
        self,
        state: MarketState,
        ensemble: EnsembleResult,
        equity: float,
        positions: dict[str, PositionRecord],
        return_series: dict[str, list[float]] | None = None,
        buying_power: float | None = None,
        recent_volume: float | None = None,
        ml_veto: bool = False,
    ) -> DecisionOutcome:
        # 0) Only long entries are considered in v2 (spec section 135).
        if ensemble.action is not Action.BUY:
            return self._reject(state, ensemble, RejectReason.NO_SIGNAL, f"ensemble action {ensemble.action}")

        # 0b) ML meta-filter has veto power only (spec section 85).
        if ml_veto:
            return self._reject(state, ensemble, RejectReason.ML_VETO, "ML meta-filter vetoed")

        # 1) Already in position for this symbol? no pyramiding.
        existing = positions.get(state.symbol)
        if existing is not None and existing.qty != 0:
            return self._reject(state, ensemble, RejectReason.DUPLICATE_POSITION, "position already open")

        # 2) Panic hard-stop.
        if state.is_panic:
            return self._reject(state, ensemble, RejectReason.MARKET_PANIC, "panic regime")

        # 3) BTC confirmation.
        btc_res = self.btc_filter.check(state)
        if not btc_res.passed:
            return self._reject(state, ensemble, btc_res.reason, btc_res.detail)

        # 4) Relative strength.
        rs_res = self.rs_filter.check(state)
        if not rs_res.passed:
            return self._reject(state, ensemble, rs_res.reason, rs_res.detail)

        # 5) Risk plan (gives stop plan and sizing).
        price = state.features.values.get("close")
        atr = state.features.values.get("atr")
        if price is None or atr is None or price <= 0:
            return self._reject(state, ensemble, RejectReason.NO_SIGNAL, "missing price/ATR")

        extreme_vol = False
        vol_pct = state.features.values.get("vol_percentile")
        if vol_pct is not None and vol_pct >= self.risk.extreme_atr_percentile:
            extreme_vol = True

        risk_plan = self.risk_engine.plan(
            equity=equity,
            symbol=state.symbol,
            entry_price=price,
            atr=atr,
            strategy_name=self._dominant_strategy(ensemble),
            positions=positions,
            return_series=return_series,
            buying_power=buying_power,
            recent_volume=recent_volume,
            base_risk_multiplier=1.0,
            extreme_volatility=extreme_vol,
        )
        if not risk_plan.allowed:
            reason = risk_plan.reject_reason
            return self._reject(state, ensemble, reason, risk_plan.detail)

        stop_plan = risk_plan.stop_plan
        assert stop_plan is not None and risk_plan.sizing is not None

        # 6) Expected edge / cost filter. Expected move ties to the intended
        # target distance (stop_distance * R multiple).
        expected_move = stop_plan.stop_distance_pct * stop_plan.target_r_multiple
        edge_res = self.edge.evaluate(expected_move, quote=state.quote)
        if not edge_res.passed:
            return self._reject(
                state,
                ensemble,
                RejectReason.EXPECTED_EDGE_TOO_SMALL,
                edge_res.reason,
                expected_edge=edge_res.remaining_edge,
            )

        strategy = self._dominant_strategy(ensemble)
        decision = TradeDecision(
            symbol=state.symbol,
            side="buy",
            quantity=risk_plan.sizing.quantity,
            entry_type="market",
            limit_price=None,
            stop_price=stop_plan.stop_price,
            target_price=stop_plan.target_price,
            strategy=strategy,
            score=ensemble.weighted_score,
            expected_edge=edge_res.remaining_edge,
            regime=state.regime,
            reason=ensemble.reason,
            risk_amount=risk_plan.risk_amount,
            risk_multiplier=risk_plan.effective_risk_multiplier,
            feature_snapshot=state.features,
            estimated_cost=edge_res.cost.total,
            metadata={
                "cost": edge_res.cost.as_dict(),
                "stop_distance_pct": stop_plan.stop_distance_pct,
                "binding_constraint": risk_plan.metadata.get("binding_constraint"),
                "correlation_scale": risk_plan.correlation_scale,
            },
        )
        return DecisionOutcome(decision=decision, rejection=None)

    @staticmethod
    def _dominant_strategy(ensemble: EnsembleResult) -> str:
        if not ensemble.contributing:
            return "ensemble"
        return max(ensemble.contributing.items(), key=lambda kv: kv[1])[0]
