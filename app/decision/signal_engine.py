"""Signal engine: runs the regime-eligible strategies for a symbol.

Produces a list of Signals. It does not decide; the ensemble does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.strategy_config import StrategyConfig
from app.core.types import MarketState, Signal
from app.strategies import eligible_strategies


@dataclass(slots=True)
class SignalSet:
    symbol: str
    regime: str
    signals: list[Signal] = field(default_factory=list)

    def actionable(self) -> list[Signal]:
        from app.core.enums import Action

        return [s for s in self.signals if s.action in (Action.BUY, Action.SELL, Action.EXIT)]


class SignalEngine:
    def __init__(self, strategies: dict, config: StrategyConfig) -> None:
        self.strategies = strategies
        self.config = config

    def evaluate(self, state: MarketState) -> SignalSet:
        names = eligible_strategies(state.regime, self.config)
        signals: list[Signal] = []
        for name in names:
            strategy = self.strategies.get(name)
            if strategy is None:
                continue
            try:
                signals.append(strategy.evaluate(state))
            except Exception as exc:  # noqa: BLE001 - a broken strategy must not kill the bot
                from app.core.enums import Action

                signals.append(
                    Signal(
                        action=Action.NO_SIGNAL,
                        confidence=0.0,
                        score=0.0,
                        strategy_name=name,
                        reason=f"strategy error: {exc}",
                    )
                )
        return SignalSet(symbol=state.symbol, regime=str(state.regime), signals=signals)
