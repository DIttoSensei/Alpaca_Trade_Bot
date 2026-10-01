"""Strategy 4 - Momentum continuation (spec section 18).

Distinct from breakout: strong trend + strong momentum + relative strength +
volume confirmation + controlled continuation. Avoids buying after an already
extreme vertical move (overextension filter).
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class MomentumStrategy(Strategy):
    name = "momentum"
    capability = Capability(
        allowed_regimes=(Regime.STRONG_UPTREND, Regime.WEAK_UPTREND),
        minimum_data=220,
        risk_multiplier=1.0,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        f = state.features.values
        close = f.get("close")
        ema20 = f.get("ema20")
        ema50 = f.get("ema50")
        adx = f.get("adx")
        roc = f.get("roc")
        rsi = f.get("rsi")
        atr_pct = f.get("atr_pct", 0.01)
        vol_ratio = f.get("volume_ratio")

        # Strong trend
        trend = (
            close is not None and ema20 is not None and ema50 is not None
            and close > ema20 > ema50
        )
        builder.add("strong trend structure", SignalCategory.TREND, 2.0 if trend else -1.0)

        # Strong momentum
        builder.add("positive ROC", SignalCategory.MOMENTUM, 1.5 if (roc is not None and roc > 0.02) else 0.0)

        # Relative strength vs BTC
        rs = state.relative_strength
        builder.add("positive relative strength", SignalCategory.RELATIVE_STRENGTH, 1.5 if (rs is not None and rs > 0) else 0.0)

        # Volume confirmation
        builder.add("volume confirmation", SignalCategory.VOLUME, 1.0 if (vol_ratio is not None and vol_ratio >= 1.0) else 0.0)

        # ADX strong
        builder.add("ADX strong", SignalCategory.TREND, 1.0 if (adx is not None and adx > 25) else 0.0)

        # Overextension filter: reject if RSI extreme or far above EMA20 relative to ATR
        overextended = False
        if rsi is not None and rsi > 78:
            overextended = True
        if close is not None and ema20 is not None and ema20 > 0:
            extension = (close - ema20) / ema20
            if extension > max(0.05, atr_pct * 3):
                overextended = True
        if overextended:
            return Action.NO_SIGNAL

        if trend and (roc is not None and roc > 0):
            return Action.BUY
        return Action.NO_SIGNAL
