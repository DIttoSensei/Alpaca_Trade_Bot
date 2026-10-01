"""Panic and recovery detection (spec sections 12-13).

PANIC is a distinct, dangerous state: no "buy the dip" unless a specifically
tested recovery strategy permits it.

RECOVERY is distinct from RANGE and UPTREND and requires its own validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.config.strategy_config import RegimeConfig


@dataclass(slots=True)
class PanicAssessment:
    is_panic: bool
    triggers: dict[str, bool] = field(default_factory=dict)
    severity: float = 0.0


def detect_panic(
    features: dict[str, float],
    cfg: RegimeConfig,
    recent_returns: Sequence[float] | None = None,
    btc_drawdown_accel: bool = False,
    correlation_spike: bool = False,
) -> PanicAssessment:
    """Multi-condition panic detector.

    Conditions considered:
    - very high ATR percentile
    - large negative return
    - volume spike
    - rapid multi-candle decline
    - BTC drawdown acceleration (passed in)
    - cross-asset correlation spike (passed in)
    """
    triggers: dict[str, bool] = {}

    vol_pct = features.get("vol_percentile")
    triggers["atr_extreme"] = vol_pct is not None and vol_pct >= cfg.panic_atr_percentile

    ret = features.get("return_6", features.get("return_3"))
    triggers["large_negative_return"] = ret is not None and ret <= cfg.panic_return_threshold

    vol_ratio = features.get("volume_ratio")
    triggers["volume_spike"] = vol_ratio is not None and vol_ratio >= cfg.panic_volume_ratio

    if recent_returns:
        neg = [r for r in recent_returns if r < 0]
        # rapid multi-candle decline: >= 80% of bars down and cumulative steep
        cum = sum(recent_returns)
        triggers["rapid_decline"] = (len(neg) >= 4 and len(neg) / len(recent_returns) >= 0.8
                                     and cum <= cfg.panic_return_threshold)
    else:
        triggers["rapid_decline"] = False

    triggers["btc_drawdown_accel"] = btc_drawdown_accel
    triggers["correlation_spike"] = correlation_spike

    # Panic requires the primary volatility/return signals plus corroboration.
    primary = triggers["large_negative_return"] and (
        triggers["atr_extreme"] or triggers["volume_spike"] or triggers["rapid_decline"]
    )
    corroborated = primary or (
        triggers["atr_extreme"] and triggers["rapid_decline"]
    ) or (triggers["btc_drawdown_accel"] and triggers["correlation_spike"])

    severity = sum(1 for v in triggers.values() if v) / max(1, len(triggers))
    return PanicAssessment(is_panic=corroborated, triggers=triggers, severity=severity)


@dataclass(slots=True)
class RecoveryAssessment:
    is_recovery: bool
    conditions: dict[str, bool] = field(default_factory=dict)


def detect_recovery(
    features: dict[str, float],
    cfg: RegimeConfig,
    had_panic_recently: bool,
    vol_series: Sequence[float] | None = None,
    momentum_series: Sequence[float] | None = None,
    higher_low: bool = False,
    btc_stabilizing: bool = True,
) -> RecoveryAssessment:
    """Recovery requires: prior panic, falling volatility, higher low, momentum
    turning up and volume confirmation (spec section 13)."""
    conditions: dict[str, bool] = {}
    conditions["had_panic"] = had_panic_recently

    # volatility declining over the last N periods
    if vol_series and len(vol_series) > cfg.recovery_vol_falling_periods:
        recent = vol_series[-cfg.recovery_vol_falling_periods :]
        conditions["vol_falling"] = recent[-1] < recent[0]
    else:
        pct = features.get("vol_percentile")
        conditions["vol_falling"] = pct is not None and pct < cfg.high_vol_percentile

    # momentum recovering
    if momentum_series and len(momentum_series) >= 2:
        conditions["momentum_recovery"] = momentum_series[-1] > momentum_series[-2]
    else:
        hist = features.get("macd_hist")
        rsi = features.get("rsi")
        conditions["momentum_recovery"] = (hist is not None and hist > 0) or (
            rsi is not None and rsi > 45
        )

    conditions["higher_low"] = higher_low
    conditions["btc_stabilizing"] = btc_stabilizing
    vr = features.get("volume_ratio")
    conditions["volume_confirmation"] = vr is not None and vr >= 1.0

    # require: had panic + volatility falling + momentum recovery (+1 corroboration)
    core = conditions["had_panic"] and conditions["vol_falling"] and conditions["momentum_recovery"]
    corroboration = conditions["higher_low"] or conditions["volume_confirmation"] or conditions["btc_stabilizing"]
    return RecoveryAssessment(is_recovery=core and corroboration, conditions=conditions)
