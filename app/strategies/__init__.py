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
from app.strategies.grid import GridStrategy
from app.strategies.legacy import LegacyTrendPullbackStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.recovery import RecoveryStrategy
from app.strategies.trend_daily import DailyTrendStrategy
from app.strategies.trend_pullback import TrendPullbackStrategy

STRATEGY_CLASSES = {
    "trend_pullback": TrendPullbackStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "recovery": RecoveryStrategy,
    "grid": GridStrategy,
    "trend_daily": DailyTrendStrategy,
    "legacy_trend_pullback": LegacyTrendPullbackStrategy,
}

# Which strategies may consider each regime (spec section 20). This is derived
# from capability matrices but stated explicitly for clarity/auditing.
#
# Grid is ONLY eligible in RANGE / LOW_VOLATILITY (safety: never grid a trend).
#
# ``trend_daily`` is added broadly on purpose. It is an ENTRY-gated, cross-
# timeframe trend follower: it fires only when the DAILY series is in a genuine
# bull structure (close>SMA200, EMA20>EMA50, ...). The per-symbol regime is read
# on the execution timeframe and is frequently RANGE/UNKNOWN even while the daily
# trend is up, so restricting the daily strategy to only *_UPTREND regimes would
# hold the position as the intraday regime wobbles and destroy the trend.
REGIME_STRATEGY_MAP: dict[Regime, list[str]] = {
    Regime.STRONG_UPTREND: ["trend_pullback", "momentum", "breakout", "trend_daily"],
    Regime.WEAK_UPTREND: ["trend_pullback", "momentum", "breakout", "trend_daily"],
    Regime.RANGE: ["mean_reversion", "breakout", "grid", "trend_daily"],
    Regime.LOW_VOLATILITY: ["mean_reversion", "breakout", "grid", "trend_daily"],
    Regime.HIGH_VOLATILITY: ["mean_reversion", "trend_daily"],
    Regime.STRONG_DOWNTREND: ["trend_daily"],
    Regime.WEAK_DOWNTREND: ["trend_daily"],
    Regime.PANIC: [],
    Regime.RECOVERY: ["recovery", "trend_pullback", "trend_daily"],
    Regime.UNKNOWN: ["trend_daily"],
}


def build_strategies(config: StrategyConfig) -> dict[str, Strategy]:
    """Instantiate enabled strategies with their params.

    Grid is EXCLUDED here on purpose: it is stateful across many levels and does
    not fit the single ``evaluate()->Signal`` ensemble path. It is driven
    separately by the grid runner (backtest) / dedicated grid management, so it
    must not be folded into the per-bar streaming ensemble.
    """
    strategies: dict[str, Strategy] = {}
    switch = {
        "trend_pullback": config.trend,
        "breakout": config.breakout,
        "mean_reversion": config.mean_reversion,
        "momentum": config.momentum,
        "recovery": config.recovery,
        "grid": config.grid,
        "trend_daily": config.daily_trend,
    }
    for name, cls in STRATEGY_CLASSES.items():
        if name == "legacy_trend_pullback":
            continue  # benchmark only; instantiated explicitly
        if name == "grid":
            continue  # stateful; driven by the grid runner, not the ensemble
        if not switch.get(name, False):
            continue
        params = config.get(name)
        if params is not None and not params.enabled:
            continue
        strategies[name] = cls(params=params)
    return strategies


def grid_enabled(config: StrategyConfig) -> bool:
    """True when the grid strategy is switched on and enabled."""
    return bool(config.grid) and getattr(config.grid, "enabled", False)


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
    "GridStrategy",
    "DailyTrendStrategy",
    "LegacyTrendPullbackStrategy",
    "STRATEGY_CLASSES",
    "REGIME_STRATEGY_MAP",
    "build_strategies",
    "grid_enabled",
    "eligible_strategies",
    "Signal",
    "MarketState",
]
