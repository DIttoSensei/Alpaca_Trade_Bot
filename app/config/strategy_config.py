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
class GridParams:
    """Grid-trading tunables (spec addendum: grid strategy).

    Grid is stateful across many price levels rather than a single
    ``evaluate()->Signal`` call, so it gets its own parameter object. It still
    lives in ``StrategyConfig`` so it is configured alongside every other
    strategy and inherits the same risk/exposure governance.
    """

    enabled: bool = True
    # Number of equal price levels the range is divided into.
    grid_levels: int = 10
    # Range half-width in ATRs: upper = ref + ATR*mult, lower = ref - ATR*mult.
    range_multiplier: float = 2.0
    # Recompute (cancel + replace) the grid every N execution bars.
    recenter_interval: int = 96
    # Fraction of equity committed to the WHOLE grid at once.
    capital_per_grid: float = 0.15
    # Max buy levels allowed to be concurrently filled (safety #3).
    max_filled_levels: int = 5
    # Adjacent-level gap must clear round-trip cost by at least this multiple.
    fee_safety_multiple: float = 3.0
    # ATR period used to size the range.
    atr_period: int = 14
    # Slow SMA period on the regime timeframe used as the range centre.
    reference_sma_period: int = 100
    # Hard-bound trigger: price beyond the bound by > mult * range width means
    # the ranging assumption has failed regardless of the regime detector.
    hard_bound_multiplier: float = 1.5


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
    # Master enable switches.
    #
    # The five original strategies are DISABLED by default: all five were
    # backtested over 730 days of real BTC/ETH/SOL history and confirmed
    # unprofitable, so the default configuration must not trade them. They
    # remain fully implemented and can be re-enabled explicitly.
    trend: bool = False
    breakout: bool = False
    mean_reversion: bool = False
    momentum: bool = False
    recovery: bool = False
    # Grid is the only strategy enabled by default.
    grid: bool = True

    # Per-strategy params
    trend_pullback: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            enabled=False,
            signal_threshold=6.0,
            risk_multiplier=1.0,
            ensemble_weight=1.0,
            allowed_regimes=(Regime.STRONG_UPTREND, Regime.WEAK_UPTREND),
        )
    )
    breakout: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            enabled=False,
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
            enabled=False,
            signal_threshold=6.0,
            risk_multiplier=0.6,
            ensemble_weight=0.8,
            allowed_regimes=(Regime.RANGE, Regime.LOW_VOLATILITY, Regime.HIGH_VOLATILITY),
        )
    )
    momentum: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            enabled=False,
            signal_threshold=6.0,
            risk_multiplier=1.0,
            ensemble_weight=1.0,
            allowed_regimes=(Regime.STRONG_UPTREND, Regime.WEAK_UPTREND),
        )
    )
    recovery: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            enabled=False,
            signal_threshold=6.0,
            risk_multiplier=0.4,
            ensemble_weight=0.7,
            allowed_regimes=(Regime.RECOVERY,),
        )
    )
    grid: GridParams = field(default_factory=GridParams)

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

    def all_params(self) -> dict[str, StrategyParams | GridParams]:
        return {
            "trend_pullback": self.trend_pullback,
            "breakout": self.breakout,
            "mean_reversion": self.mean_reversion,
            "momentum": self.momentum,
            "recovery": self.recovery,
            "grid": self.grid,
            "legacy_trend_pullback": StrategyParams(
                signal_threshold=6.0, risk_multiplier=1.0, ensemble_weight=0.0
            ),
        }

    def get(self, name: str) -> StrategyParams | GridParams | None:
        return self.all_params().get(name)


def load_strategy_config() -> StrategyConfig:
    return StrategyConfig()
