"""Strategy 1 - Trend Pullback (spec section 15).

Evolution of the legacy strategy. Does NOT require every condition to be
perfect; uses scoring. Only trades above a configurable threshold.
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class TrendPullbackStrategy(Strategy):
    name = "trend_pullback"
    capability = Capability(
        allowed_regimes=(Regime.STRONG_UPTREND, Regime.WEAK_UPTREND),
        minimum_data=220,
        risk_multiplier=1.0,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        f = state.features.values

        # Higher timeframe trend positive (+2)
        h1_close = self._ctx(state, "1h", "close")
        h1_ema50 = self._ctx(state, "1h", "ema50")
        h4_close = self._ctx(state, "4h", "close")
        h4_sma50 = self._ctx(state, "4h", "sma50")
        htf_positive = False
        if h1_close is not None and h1_ema50 is not None:
            htf_positive = h1_close > h1_ema50
        if h4_close is not None and h4_sma50 is not None:
            htf_positive = htf_positive or h4_close > h4_sma50
        builder.add("higher timeframe trend positive", SignalCategory.TREND, 2.0 if htf_positive else -2.0)

        # Trend confirmation: price above SMA200 and EMA alignment (+2)
        close = f.get("close")
        sma200 = f.get("sma200")
        ema20 = f.get("ema20")
        ema50 = f.get("ema50")
        trend_ok = (
            close is not None and sma200 is not None and close > sma200
            and ema20 is not None and ema50 is not None and ema20 > ema50
        )
        builder.add("trend confirmation (price>sma200, ema20>ema50)", SignalCategory.TREND, 2.0 if trend_ok else -1.0)

        # ADX supports trend (+1 handled via category; here as structure)
        adx = f.get("adx")
        if adx is not None and adx > 20:
            builder.add("ADX supports trend", SignalCategory.STRUCTURE, 1.0)

        # EMA pullback: price near/at EMA20 or EMA50 (+1)
        pullback = False
        if close is not None and ema20 is not None and ema20 > 0:
            dist = (close - ema20) / ema20
            pullback = -0.02 <= dist <= 0.01
        if close is not None and ema50 is not None and ema50 > 0:
            dist50 = (close - ema50) / ema50
            pullback = pullback or (-0.01 <= dist50 <= 0.02)
        builder.add("price pulled toward moving average", SignalCategory.STRUCTURE, 1.0 if pullback else 0.0)

        # RSI recovery: RSI was low, now turning up (+1)
        rsi = f.get("rsi")
        rsi_recovery = rsi is not None and 30 <= rsi <= 55
        builder.add("RSI recovering", SignalCategory.MOMENTUM, 1.0 if rsi_recovery else 0.0)

        # MACD recovery: histogram rising / positive (+1)
        hist = f.get("macd_hist")
        macd_recovery = hist is not None and hist > -0.0005
        builder.add("MACD recovering", SignalCategory.MOMENTUM, 1.0 if macd_recovery else 0.0)

        # Volume confirmation (+1)
        vr = f.get("volume_ratio")
        builder.add("volume confirmation", SignalCategory.VOLUME, 1.0 if (vr is not None and vr >= 1.0) else 0.0)

        # BTC confirmation (+1)
        btc_ok = True
        if state.btc_features is not None:
            btc_ret = state.btc_features.values.get("return_6")
            btc_ok = btc_ret is not None and btc_ret > -0.02
        builder.add("BTC confirmation", SignalCategory.RELATIVE_STRENGTH, 1.0 if btc_ok else 0.0)

        if not trend_ok and not htf_positive:
            return Action.NO_SIGNAL
        return Action.BUY
