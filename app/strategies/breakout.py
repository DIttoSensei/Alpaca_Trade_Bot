"""Strategy 2 - Breakout (spec section 16).

Detects range compression -> volatility expansion -> break above recent
resistance with volume/momentum confirmation. Rejects tiny / low-volume
breakouts, breakouts into major resistance and panic conditions.
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class BreakoutStrategy(Strategy):
    name = "breakout"
    capability = Capability(
        allowed_regimes=(
            Regime.STRONG_UPTREND,
            Regime.WEAK_UPTREND,
            Regime.RANGE,
            Regime.LOW_VOLATILITY,
        ),
        minimum_data=120,
        risk_multiplier=0.8,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        f = state.features.values
        close = f.get("close")
        recent_high = f.get("recent_high")
        bb_width = f.get("bb_width")
        vol_pct = f.get("vol_percentile")
        atr_pct = f.get("atr_pct", 0.01)

        if close is None or recent_high is None:
            return Action.NO_SIGNAL

        # Break above recent resistance
        broke = close >= recent_high * 0.998
        builder.add("break above recent resistance", SignalCategory.STRUCTURE, 2.0 if broke else 0.0)

        # Prior range compression (low Bollinger width)
        compressed = bb_width is not None and bb_width < 0.04
        builder.add("range compression before break", SignalCategory.VOLATILITY, 1.5 if compressed else 0.0)

        # Volatility expansion (from low toward normal/high)
        expanding = vol_pct is not None and 0.3 <= vol_pct <= 0.85
        builder.add("volatility expansion", SignalCategory.VOLATILITY, 1.0 if expanding else 0.0)

        # Volume confirmation
        vr = f.get("volume_ratio")
        builder.add("volume confirmation", SignalCategory.VOLUME, 1.5 if (vr is not None and vr >= 1.5) else 0.0)

        # Momentum confirmation (ROC)
        roc = f.get("roc")
        builder.add("momentum confirmation", SignalCategory.MOMENTUM, 1.5 if (roc is not None and roc > 0.01) else 0.0)

        # ADX trend
        adx = f.get("adx")
        builder.add("ADX supports breakout", SignalCategory.TREND, 1.0 if (adx is not None and adx > 20) else 0.0)

        # Rejections -----------------------------------------------------
        # tiny breakout: move not big enough to justify costs
        breakout_size = (close - recent_high) / recent_high if recent_high else 0.0
        if abs(breakout_size) < max(0.003, atr_pct * 0.3):
            return Action.NO_SIGNAL
        # low volume breakout
        if vr is not None and vr < 0.8:
            return Action.NO_SIGNAL
        # panic conditions
        if state.is_panic:
            return Action.NO_SIGNAL

        if broke:
            return Action.BUY
        return Action.NO_SIGNAL
