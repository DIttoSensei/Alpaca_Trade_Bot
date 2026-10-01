"""Legacy trend-pullback benchmark (spec section 111).

Preserves the original bot's logic as a benchmark so the V2 system can be
compared against it on identical datasets and costs. Do NOT delete working
functionality before establishing a baseline.
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class LegacyTrendPullbackStrategy(Strategy):
    """Simple RSI + EMA + ADX combination, representative of the original bot.

    Deliberately kept modest: the original approach was an indicator
    conjunction without regime awareness or cost filtering.
    """

    name = "legacy_trend_pullback"
    capability = Capability(
        allowed_regimes=(
            Regime.STRONG_UPTREND,
            Regime.WEAK_UPTREND,
            Regime.RANGE,
            Regime.UNKNOWN,
        ),
        minimum_data=200,
        risk_multiplier=1.0,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        f = state.features.values
        close = f.get("close")
        ema20 = f.get("ema20")
        ema50 = f.get("ema50")
        sma200 = f.get("sma200")
        rsi = f.get("rsi")
        adx = f.get("adx")

        up = (
            close is not None and ema20 is not None and ema50 is not None and sma200 is not None
            and close > ema50 and ema20 > ema50 and close > sma200
        )
        builder.add("price/EMA trend up", SignalCategory.TREND, 2.0 if up else 0.0)

        # Original style: enter on a modest RSI dip within an uptrend
        rsi_dip = rsi is not None and 40 <= rsi <= 55
        builder.add("RSI dip", SignalCategory.MOMENTUM, 2.0 if rsi_dip else 0.0)

        builder.add("ADX > 20", SignalCategory.STRUCTURE, 2.0 if (adx is not None and adx > 20) else 0.0)

        if up and rsi_dip:
            return Action.BUY
        return Action.NO_SIGNAL
