"""Strategy configuration.

Thresholds here are STARTING POINTS, explicitly NOT assumed optimal. They must be
tested and tuned through the walk-forward/backtest pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import Regime, SignalCategory


@dataclass(slots=True)
class RegimeConfig:
    """Regime scoring thresholds (spec sections 10-13)."""

    sma_long_period: int = 200
    adx_trend_threshold: float = 20.0
    adx_strong_threshold: float = 25.0

    strong_up_threshold: int = 5
    weak_up_threshold: int = 3
    weak_down_threshold: int = -3
    strong_down_threshold: int = -5

    # Volatility percentile bands
    low_vol_percentile: float = 0.25
    high_vol_percentile: float = 0.75
    extreme_vol_percentile: float = 0.90

    # Panic detection
    panic_atr_percentile: float = 0.95
    panic_return_threshold: float = -0.06  # 4h return
    panic_volume_ratio: float = 2.5
    panic_correlation_threshold: float = 0.85

    # Recovery detection
    recovery_vol_falling_periods: int = 6
    recovery_momentum_lookback: int = 6


@dataclass(slots=True)
class StrategyParams:
    """Generic, per-strategy tunables. Keep the count small (spec section 67)."""

    enabled: bool = True
    signal_threshold: float = 6.0
    risk_multiplier: float = 1.0
    max_score: float = 9.0
    allowed_regimes: tuple[Regime, ...] = ()
    supported_symbols: tuple[str, ...] = ()  # empty -> all
    minimum_data: int = 220
    # weights used by ensemble
    ensemble_weight: float = 1.0
    # lookback knobs
    lookback: int = 50
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyConfig:
    # Master enable switches
    trend: bool = True
    breakout: bool = True
    mean_reversion: bool = True
    momentum: bool = True
    recovery: bool = True

    # Per-strategy params
    trend_pullback: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            signal_threshold=6.0,
            risk_multiplier=1.0,
            ensemble_weight=1.0,
            allowed_regimes=(Regime.STRONG_UPTREND, Regime.WEAK_UPTREND),
        )
    )
    breakout: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            signal_threshold=6.0,
            risk_multiplier=0.8,
            ensemble_weight=0.9,
            allowed_regimes=(
                Regime.STRONG_UPTREND,
                Regime.WEAK_UPTREND,
                Regime.RANGE,
                Regime.LOW_VOLATILITY,
            ),
        )
    )
    mean_reversion: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            signal_threshold=6.0,
            risk_multiplier=0.6,
            ensemble_weight=0.8,
            allowed_regimes=(Regime.RANGE, Regime.LOW_VOLATILITY, Regime.HIGH_VOLATILITY),
        )
    )
    momentum: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            signal_threshold=6.0,
            risk_multiplier=1.0,
            ensemble_weight=1.0,
            allowed_regimes=(Regime.STRONG_UPTREND, Regime.WEAK_UPTREND),
        )
    )
    recovery: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            signal_threshold=6.0,
            risk_multiplier=0.4,
            ensemble_weight=0.7,
            allowed_regimes=(Regime.RECOVERY,),
        )
    )

    # Category contribution caps (spec section 22)
    category_caps: dict[SignalCategory, float] = field(
        default_factory=lambda: {
            SignalCategory.TREND: 0.30,
            SignalCategory.MOMENTUM: 0.20,
            SignalCategory.VOLUME: 0.15,
            SignalCategory.STRUCTURE: 0.20,
            SignalCategory.RELATIVE_STRENGTH: 0.15,
        }
    )

    # Ensemble behavior
    ensemble_min_score: float = 6.0
    ensemble_min_confidence: float = 0.5

    # Relative strength
    min_relative_strength: float = 0.0
    require_btc_confirmation: bool = True

    # Adaptive weighting (spec section 35): slow, bounded
    adaptive_weighting: bool = True
    min_sample_for_adaptive: int = 30
    max_weight_change_per_update: float = 0.10
    weight_update_cooldown_trades: int = 10

    def all_params(self) -> dict[str, StrategyParams]:
        return {
            "trend_pullback": self.trend_pullback,
            "breakout": self.breakout,
            "mean_reversion": self.mean_reversion,
            "momentum": self.momentum,
            "recovery": self.recovery,
            "legacy_trend_pullback": StrategyParams(
                signal_threshold=6.0, risk_multiplier=1.0, ensemble_weight=0.0
            ),
        }

    def get(self, name: str) -> StrategyParams | None:
        return self.all_params().get(name)


def load_strategy_config() -> StrategyConfig:
    return StrategyConfig()
