"""Strategy 3 - Mean Reversion (spec section 17).

Only active when the market is NOT strongly trending. A large negative move is
NOT automatically a buying opportunity: a strong downtrend disables this.
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    capability = Capability(
        allowed_regimes=(Regime.RANGE, Regime.LOW_VOLATILITY, Regime.HIGH_VOLATILITY),
        minimum_data=60,
        risk_multiplier=0.6,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        f = state.features.values
        close = f.get("close")
        zscore = f.get("zscore")
        bb_lower = f.get("bb_lower")
        bb_upper = f.get("bb_upper")
        rsi = f.get("rsi")
        adx = f.get("adx")
        return_dev = f.get("return_deviation")

        if close is None or zscore is None:
            return Action.NO_SIGNAL

        # Hard gate: never mean-revert into a strong trend or panic.
        if adx is not None and adx > 30:
            return Action.NO_SIGNAL
        if state.is_panic:
            return Action.NO_SIGNAL

        # Price significantly displaced below mean
        displaced = zscore < -2
        builder.add("price displaced below mean (z<-2)", SignalCategory.STRUCTURE, 2.0 if displaced else 0.0)

        # Bollinger deviation (below lower band)
        below_bb = bb_lower is not None and close < bb_lower
        builder.add("below lower Bollinger band", SignalCategory.VOLATILITY, 1.5 if below_bb else 0.0)

        # RSI oversold but recovering (not collapsing)
        rsi_ok = rsi is not None and 25 <= rsi <= 45
        builder.add("RSI oversold/recovering", SignalCategory.MOMENTUM, 1.5 if rsi_ok else 0.0)

        # Low ADX (range conditions)
        builder.add("ADX low (range)", SignalCategory.TREND, 1.0 if (adx is not None and adx < 20) else 0.0)

        # Short-term return extreme but reverting
        reverting = return_dev is not None and return_dev < 0
        builder.add("short-term return reverting", SignalCategory.MOMENTUM, 1.0 if reverting else 0.0)

        if displaced and (below_bb or rsi_ok):
            return Action.BUY
        return Action.NO_SIGNAL
