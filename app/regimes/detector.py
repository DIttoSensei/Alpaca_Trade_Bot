"""Composite market regime detector.

Combines trend, volatility, panic and recovery assessments into a single
``Regime``. This is one of the most important components (spec section 10).

The detector is stateful only with respect to panic history (to enable recovery
detection). It is otherwise a pure function of the inputs, so backtest and live
share identical behaviour.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.config.strategy_config import RegimeConfig
from app.core.enums import Regime, TrendRegime, VolatilityRegime
from app.regimes.market_state import PanicAssessment, RecoveryAssessment, detect_panic, detect_recovery
from app.regimes.trend_regime import score_trend
from app.regimes.volatility_regime import classify_volatility


@dataclass(slots=True)
class RegimeResult:
    regime: Regime
    trend: TrendRegime
    volatility: VolatilityRegime
    trend_score: int
    panic: PanicAssessment
    recovery: RecoveryAssessment
    details: dict = field(default_factory=dict)


class RegimeDetector:
    def __init__(self, cfg: Optional[RegimeConfig] = None) -> None:
        self.cfg = cfg or RegimeConfig()
        self._panic_history: deque[bool] = deque(maxlen=24)
        self._last_regime: Regime = Regime.UNKNOWN

    def _had_panic_recently(self, window: int = 8) -> bool:
        recent = list(self._panic_history)[-window:]
        return any(recent)

    def detect(
        self,
        features: dict[str, float],
        btc_features: Optional[dict[str, float]] = None,
        recent_returns: Optional[list[float]] = None,
        vol_series: Optional[list[float]] = None,
        momentum_series: Optional[list[float]] = None,
        higher_low: bool = False,
        timestamp: Optional[datetime] = None,
        is_reference: bool = False,
    ) -> RegimeResult:
        trend = score_trend(features, self.cfg)
        vol = classify_volatility(features, self.cfg)

        # BTC drawdown acceleration: reference asset in steep decline
        btc_drawdown_accel = False
        btc_stabilizing = True
        if btc_features is not None:
            btc_ret = btc_features.get("return_6")
            btc_vol = btc_features.get("vol_percentile")
            btc_drawdown_accel = (
                btc_ret is not None
                and btc_ret <= self.cfg.panic_return_threshold / 2
                and btc_vol is not None
                and btc_vol >= self.cfg.high_vol_percentile
            )
            btc_stabilizing = btc_ret is not None and btc_ret > self.cfg.panic_return_threshold / 2

        panic = detect_panic(
            features,
            self.cfg,
            recent_returns=recent_returns,
            btc_drawdown_accel=btc_drawdown_accel,
        )
        self._panic_history.append(panic.is_panic)

        recovery = detect_recovery(
            features,
            self.cfg,
            had_panic_recently=self._had_panic_recently(),
            vol_series=vol_series,
            momentum_series=momentum_series,
            higher_low=higher_low,
            btc_stabilizing=btc_stabilizing,
        )

        regime = self._combine(trend.regime, vol.regime, panic, recovery)
        self._last_regime = regime
        return RegimeResult(
            regime=regime,
            trend=trend.regime,
            volatility=vol.regime,
            trend_score=trend.score,
            panic=panic,
            recovery=recovery,
            details={
                "trend_contributions": trend.contributions,
                "vol_percentile": vol.percentile,
                "panic_triggers": panic.triggers,
                "recovery_conditions": recovery.conditions,
            },
        )

    def _combine(
        self,
        trend: TrendRegime,
        vol: VolatilityRegime,
        panic: PanicAssessment,
        recovery: RecoveryAssessment,
    ) -> Regime:
        # PANIC dominates.
        if panic.is_panic:
            return Regime.PANIC
        # Recovery requires prior panic (already gated inside detection).
        if recovery.is_recovery:
            return Regime.RECOVERY
        # Extreme volatility without panic -> high volatility regime.
        if vol is VolatilityRegime.EXTREME:
            return Regime.HIGH_VOLATILITY
        if vol is VolatilityRegime.LOW and trend is TrendRegime.RANGE:
            return Regime.LOW_VOLATILITY

        mapping = {
            TrendRegime.STRONG_UP: Regime.STRONG_UPTREND,
            TrendRegime.WEAK_UP: Regime.WEAK_UPTREND,
            TrendRegime.RANGE: Regime.RANGE,
            TrendRegime.WEAK_DOWN: Regime.WEAK_DOWNTREND,
            TrendRegime.STRONG_DOWN: Regime.STRONG_DOWNTREND,
            TrendRegime.UNKNOWN: Regime.UNKNOWN,
        }
        if vol is VolatilityRegime.HIGH and trend is TrendRegime.RANGE:
            return Regime.HIGH_VOLATILITY
        return mapping.get(trend, Regime.UNKNOWN)

    def reset(self) -> None:
        self._panic_history.clear()
        self._last_regime = Regime.UNKNOWN
