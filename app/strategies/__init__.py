"""Strategy engines. Strategies produce Signals only; they never place orders.

Regime -> allowed-strategy mapping is defined here (spec section 20), driven by
each strategy's capability matrix.
"""

from __future__ import annotations

from app.config.strategy_config import StrategyConfig
from app.core.enums import Regime
from app.core.types import MarketState, Signal
from app.strategies.base import Capability, ScoreBuilder, Strategy
from app.strategies.breakout import BreakoutStrategy
from app.strategies.legacy import LegacyTrendPullbackStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.recovery import RecoveryStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy

STRATEGY_CLASSES = {
    "trend_pullback": TrendPullbackStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "recovery": RecoveryStrategy,
    "legacy_trend_pullback": LegacyTrendPullbackStrategy,
}

# Which strategies may consider each regime (spec section 20). This is derived
# from capability matrices but stated explicitly for clarity/auditing.
REGIME_STRATEGY_MAP: dict[Regime, list[str]] = {
    Regime.STRONG_UPTREND: ["trend_pullback", "momentum", "breakout"],
    Regime.WEAK_UPTREND: ["trend_pullback", "momentum", "breakout"],
    Regime.RANGE: ["mean_reversion", "breakout"],
    Regime.LOW_VOLATILITY: ["mean_reversion", "breakout"],
    Regime.HIGH_VOLATILITY: ["mean_reversion"],
    Regime.STRONG_DOWNTREND: [],
    Regime.WEAK_DOWNTREND: [],
    Regime.PANIC: [],
    Regime.RECOVERY: ["recovery", "trend_pullback"],
    Regime.UNKNOWN: [],
}


def build_strategies(config: StrategyConfig) -> dict[str, Strategy]:
    """Instantiate enabled strategies with their params."""
    strategies: dict[str, Strategy] = {}
    switch = {
        "trend_pullback": config.trend,
        "breakout": config.breakout,
        "mean_reversion": config.mean_reversion,
        "momentum": config.momentum,
        "recovery": config.recovery,
    }
    for name, cls in STRATEGY_CLASSES.items():
        if name == "legacy_trend_pullback":
            continue  # benchmark only; instantiated explicitly
        if not switch.get(name, False):
            continue
        params = config.get(name)
        if params is not None and not params.enabled:
            continue
        strategies[name] = cls(params=params)
    return strategies


def eligible_strategies(regime: Regime, config: StrategyConfig) -> list[str]:
    names = REGIME_STRATEGY_MAP.get(regime, [])
    out: list[str] = []
    for name in names:
        params = config.get(name)
        if params is not None and params.enabled:
            out.append(name)
    return out


__all__ = [
    "Strategy",
    "Capability",
    "ScoreBuilder",
    "TrendPullbackStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "RecoveryStrategy",
    "LegacyTrendPullbackStrategy",
    "STRATEGY_CLASSES",
    "REGIME_STRATEGY_MAP",
    "build_strategies",
    "eligible_strategies",
    "Signal",
    "MarketState",
]
