"""Ensemble engine (spec sections 21, 95).

Combines strategy signals intelligently instead of blindly averaging. Uses
configurable weights, measures agreement/conflict, and resolves conflicts using
regime eligibility and historical expectancy (weights). Does NOT optimize
dozens of independent weights (spec section 21).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.strategy_config import StrategyConfig
from app.core.enums import Action
from app.core.types import EnsembleResult
from app.decision.signal_engine import SignalSet
from app.strategies.base import Strategy


class Ensemble:
    def __init__(self, strategies: dict[str, Strategy], config: StrategyConfig) -> None:
        self.strategies = strategies
        self.config = config
        self._health_weights: dict[str, float] = {}

    def set_health_weights(self, weights: dict[str, float]) -> None:
        """Inject adaptive weights (slow, bounded; from strategy health)."""
        self._health_weights = dict(weights)

    def combine(self, signal_set: SignalSet) -> EnsembleResult:
        cfg = self.config
        actionable = [s for s in signal_set.signals if s.action in (Action.BUY, Action.SELL, Action.EXIT)]

        if not actionable:
            return EnsembleResult(
                symbol=signal_set.symbol,
                action=Action.NO_SIGNAL,
                weighted_score=0.0,
                confidence=0.0,
                agreement=0.0,
                conflict=False,
                reason="no actionable signals",
            )

        # Directional votes -------------------------------------------------
        buy_score = 0.0
        sell_score = 0.0
        exit_score = 0.0
        contributing: dict[str, float] = {}
        total_weight = 0.0

        for sig in actionable:
            params = cfg.get(sig.strategy_name)
            base_weight = params.ensemble_weight if params else 1.0
            health = self._health_weights.get(sig.strategy_name, 1.0)
            weight = base_weight * health
            weighted = sig.score * weight
            contributing[sig.strategy_name] = round(weighted, 3)
            total_weight += weight
            if sig.action is Action.BUY:
                buy_score += weighted
            elif sig.action is Action.SELL:
                sell_score += weighted
            elif sig.action is Action.EXIT:
                exit_score += weighted

        # Conflict: simultaneous opposing entries, or an EXIT within a position
        conflict = (buy_score > 0 and sell_score > 0)
        # Exit logic outranks new entries (spec section 96).
        if exit_score > 0:
            dominant = Action.EXIT
            weighted_score = exit_score
        elif buy_score >= sell_score and buy_score > 0:
            dominant = Action.BUY
            weighted_score = buy_score
        elif sell_score > 0:
            dominant = Action.SELL
            weighted_score = sell_score
        else:
            dominant = Action.NO_SIGNAL
            weighted_score = 0.0

        denom = total_weight if total_weight > 0 else 1.0
        normalised_score = weighted_score / denom
        # agreement = share of actionable weight agreeing with the dominant side
        if dominant is Action.BUY:
            agreement = buy_score / denom
        elif dominant is Action.SELL:
            agreement = sell_score / denom
        elif dominant is Action.EXIT:
            agreement = exit_score / denom
        else:
            agreement = 0.0

        confidence = min(1.0, agreement) * min(1.0, normalised_score / max(1.0, cfg.ensemble_min_score))

        reason = f"dominant={dominant} score={normalised_score:.2f} agreement={agreement:.2f}"
        if conflict:
            reason += " (conflict detected)"

        # Threshold gating
        if dominant in (Action.BUY, Action.SELL):
            if normalised_score < cfg.ensemble_min_score or agreement < cfg.ensemble_min_confidence:
                dominant = Action.NO_SIGNAL
                reason += " -> below ensemble thresholds"

        return EnsembleResult(
            symbol=signal_set.symbol,
            action=dominant,
            weighted_score=round(normalised_score, 3),
            confidence=round(confidence, 3),
            agreement=round(agreement, 3),
            conflict=conflict,
            contributing=contributing,
            reason=reason,
        )
