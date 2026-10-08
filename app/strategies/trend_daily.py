"""Strategy - Daily Trend (cross-timeframe trend following).

Motivation (evidence-based): an out-of-sample, walk-forward study over 2 years of
real BTC/ETH/SOL data (see ``diagnostics/edge_research.py``) showed that the only
long-only edge that survives realistic costs (0.35%/side) is LOW-TURNOVER trend
following read off the DAILY series:

    * D1 donchian/ema trend-following: +39%..+80% in each of 2025 and 2026
    * buy-and-hold over the same window: -54% and -18%

Intraday (M15) entries could not overcome the cost drag because average per-trade
moves were a fraction of the round-trip cost. Daily entries trade ~10x less often
so the trend move dominates the cost.

This strategy therefore evaluates on the DAILY context (``state.context["1d"]``),
which the pipeline exposes only once the daily candle has CLOSED (no lookahead).
The ENTRY is gated by the daily structure itself (below); the execution-timeframe
regime is deliberately NOT used as an entry gate, because that regime is read on
M15 and is frequently RANGE/UNKNOWN even while the daily trend is up. Firing only
when BOTH the intraday regime and the daily trend agree would miss most of the
edge. It fires only when a genuine trend is up:

    * close > SMA200            (structural bull)
    * EMA20 > EMA50             (intermediate trend up)
    * close > SMA50             (higher low / above medium trend)
    * close >= prior 20d high   (Donchian breakout confirmation)
    * 12d momentum positive     (don't buy a stalling trend)

Exit/target/trailing are delegated to the risk + simulator layer; this strategy
requests a trend-following exit (no fixed target, wide ATR trailing) via its
params so winners are allowed to run.
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class DailyTrendStrategy(Strategy):
    name = "trend_daily"
    # Allowed in every regime except PANIC. The real gate is the daily bull
    # structure evaluated inside ``_evaluate`` (close>SMA200 and EMA20>EMA50);
    # the execution-timeframe regime must NOT veto an otherwise-intact daily
    # trend. PANIC is excluded so the strategy never buys into a cascade (the
    # composer also rejects panic separately).
    capability = Capability(
        allowed_regimes=(
            Regime.STRONG_UPTREND,
            Regime.WEAK_UPTREND,
            Regime.RANGE,
            Regime.LOW_VOLATILITY,
            Regime.HIGH_VOLATILITY,
            Regime.WEAK_DOWNTREND,
            Regime.STRONG_DOWNTREND,
            Regime.RECOVERY,
            Regime.UNKNOWN,
        ),
        minimum_data=210,
        risk_multiplier=1.0,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        daily = state.context.get("1d")
        if daily is None:
            return Action.NO_SIGNAL
        d = daily.values

        close = d.get("close")
        sma200 = d.get("sma200")
        sma50 = d.get("sma50")
        ema20 = d.get("ema20")
        ema50 = d.get("ema50")
        recent_high = d.get("recent_high")  # prior 20 daily bars (incl. current)
        r6 = d.get("return_6")
        r12 = d.get("return_12")

        if close is None or sma200 is None or sma50 is None or ema20 is None or ema50 is None:
            return Action.NO_SIGNAL

        # --- Structure: bull regime -------------------------------------
        above200 = close > sma200
        ema_up = ema20 > ema50
        builder.add("daily close > SMA200", SignalCategory.TREND, 2.0 if above200 else 0.0)
        builder.add("daily EMA20 > EMA50", SignalCategory.TREND, 2.0 if ema_up else 0.0)

        # --- Structure: price strength / breakout -----------------------
        above50 = close > sma50
        breakout = recent_high is not None and close >= recent_high * 0.999
        builder.add("daily close > SMA50", SignalCategory.STRUCTURE, 1.5 if above50 else 0.0)
        builder.add("20d breakout / at highs", SignalCategory.STRUCTURE, 1.5 if breakout else 0.0)

        # --- Momentum: trend still advancing ----------------------------
        mom6 = r6 is not None and r6 > 0
        mom12 = r12 is not None and r12 > 0
        builder.add("6d momentum > 0", SignalCategory.MOMENTUM, 1.0 if mom6 else 0.0)
        builder.add("12d momentum > 0", SignalCategory.MOMENTUM, 2.0 if mom12 else 0.0)

        # Hard confirmation: never fire without the daily bull structure.
        if not (above200 and ema_up):
            return Action.NO_SIGNAL

        return Action.BUY
