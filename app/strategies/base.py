"""Strategy base class (spec sections 14, 106, 22).

Every strategy implements the same interface and returns a Signal. Strategies
NEVER place orders; they only produce signals.

Scoring uses CATEGORY-CAPPED contributions so correlated indicators cannot
stack to inflate confidence (spec section 22). Each feature belongs to a
category (TREND, MOMENTUM, VOLATILITY, VOLUME, STRUCTURE, RELATIVE_STRENGTH);
a raw contribution is added within its category, and each category's total
contribution is capped before summing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState, Signal


@dataclass(slots=True)
class Capability:
    """Strategy capability matrix (spec section 106)."""

    allowed_regimes: tuple[Regime, ...]
    supported_symbols: tuple[str, ...] = ()  # empty -> all symbols
    minimum_data: int = 220
    risk_multiplier: float = 1.0

    def allows_regime(self, regime: Regime) -> bool:
        return regime in self.allowed_regimes

    def supports_symbol(self, symbol: str) -> bool:
        return not self.supported_symbols or symbol in self.supported_symbols


class ScoreBuilder:
    """Accumulates category-scoped contributions and applies caps."""

    def __init__(self, caps: dict[SignalCategory, float] | None = None, max_score: float = 9.0) -> None:
        self.caps = caps or {
            SignalCategory.TREND: 0.30,
            SignalCategory.MOMENTUM: 0.20,
            SignalCategory.VOLUME: 0.15,
            SignalCategory.STRUCTURE: 0.20,
            SignalCategory.RELATIVE_STRENGTH: 0.15,
        }
        self.max_score = max_score
        self._raw: dict[SignalCategory, float] = {}
        self._reasons: list[str] = []
        self._items: list[tuple[str, SignalCategory, float]] = []

    def add(self, reason: str, category: SignalCategory, value: float) -> "ScoreBuilder":
        self._raw[category] = self._raw.get(category, 0.0) + value
        self._items.append((reason, category, value))
        if value != 0:
            self._reasons.append(reason)
        return self

    def raw_category_totals(self) -> dict[SignalCategory, float]:
        return dict(self._raw)

    def capped_scores(self) -> tuple[float, dict[str, float]]:
        """Return (score, per-category capped contribution in points)."""
        total_raw = sum(max(0.0, v) for v in self._raw.values()) or 1.0
        capped_total = 0.0
        per_category: dict[str, float] = {}
        for category, raw in self._raw.items():
            cap = self.caps.get(category, 0.0)
            # category's share of the positive raw score, limited by cap
            share = (max(0.0, raw) / total_raw) if total_raw > 0 else 0.0
            effective_share = min(share, cap)
            contribution = effective_share * self.max_score
            capped_total += contribution
            per_category[str(category)] = round(raw, 3)
        # Preserve sign: overall direction from the raw total.
        return round(min(capped_total, self.max_score), 3), per_category

    @property
    def reasons(self) -> str:
        return "; ".join(self._reasons)


class Strategy:
    """Base strategy interface."""

    name: str = "base"
    capability: Capability

    def __init__(self, params=None) -> None:
        self.params = params

    # -- to be implemented by subclasses ---------------------------------
    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        raise NotImplementedError

    # -- public ----------------------------------------------------------
    def evaluate(self, state: MarketState) -> Signal:
        caps = getattr(self.params, "category_caps", None) if self.params else None
        max_score = getattr(self.params, "max_score", 9.0) if self.params else 9.0
        threshold = getattr(self.params, "signal_threshold", 6.0) if self.params else 6.0

        builder = ScoreBuilder(caps=caps, max_score=max_score)

        # Regime gate: a strategy must not operate outside its designed regime.
        if self.capability and not self.capability.allows_regime(state.regime):
            return Signal(
                action=Action.NO_SIGNAL,
                confidence=0.0,
                score=0.0,
                strategy_name=self.name,
                reason=f"regime {state.regime} not allowed",
            )
        if self.capability and not self.capability.supports_symbol(state.symbol):
            return Signal(
                action=Action.NO_SIGNAL,
                confidence=0.0,
                score=0.0,
                strategy_name=self.name,
                reason="symbol not supported",
            )

        action = self._evaluate(state, builder)
        score, per_category = builder.capped_scores()
        confidence = min(1.0, score / max_score) if max_score else 0.0

        if action in (Action.BUY, Action.SELL) and score < threshold:
            # strong directional setup but insufficient confidence
            return Signal(
                action=Action.NO_SIGNAL,
                confidence=confidence,
                score=score,
                strategy_name=self.name,
                reason=f"score {score} below threshold {threshold}",
                category_scores={k: float(v) for k, v in per_category.items()},
            )

        return Signal(
            action=action,
            confidence=confidence,
            score=score,
            strategy_name=self.name,
            reason=builder.reasons,
            category_scores={k: float(v) for k, v in per_category.items()},
            metadata={"threshold": threshold, "risk_multiplier": self.capability.risk_multiplier},
        )

    # -- helpers for subclasses ------------------------------------------
    @staticmethod
    def _get(state: MarketState, key: str, default: float | None = None) -> float | None:
        return state.features.values.get(key, default)

    @staticmethod
    def _ctx(state: MarketState, timeframe: str, key: str, default: float | None = None) -> float | None:
        snap = state.context.get(timeframe)
        if snap is None:
            return default
        return snap.values.get(key, default)
