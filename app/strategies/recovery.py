"""Strategy 5 - Recovery (spec sections 13, 19).

Only activated after PANIC and subsequent stabilization. Uses substantially
smaller risk than normal trend entries until validated (risk_multiplier 0.4).
"""

from __future__ import annotations

from app.core.enums import Action, Regime, SignalCategory
from app.core.types import MarketState
from app.strategies.base import Capability, ScoreBuilder, Strategy


class RecoveryStrategy(Strategy):
    name = "recovery"
    capability = Capability(
        allowed_regimes=(Regime.RECOVERY,),
        minimum_data=120,
        risk_multiplier=0.4,
    )

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        f = state.features.values
        rsi = f.get("rsi")
        hist = f.get("macd_hist")
        vol_pct = f.get("vol_percentile")
        bump = state.extra.get("recovery_conditions", {})

        # panic ended (state is RECOVERY, so panic flag false)
        builder.add("panic ended", SignalCategory.VOLATILITY, 2.0 if not state.is_panic else -3.0)

        # volatility falling
        vol_falling = bool(bump.get("vol_falling")) or (vol_pct is not None and vol_pct < 0.75)
        builder.add("volatility falling", SignalCategory.VOLATILITY, 1.5 if vol_falling else 0.0)

        # higher low
        builder.add("higher low formed", SignalCategory.STRUCTURE, 1.5 if bump.get("higher_low") else 0.0)

        # momentum recovery
        mom = bump.get("momentum_recovery") or (hist is not None and hist > 0)
        builder.add("momentum recovering", SignalCategory.MOMENTUM, 1.5 if mom else 0.0)

        # BTC stabilizing
        builder.add("BTC stabilizing", SignalCategory.RELATIVE_STRENGTH, 1.0 if bump.get("btc_stabilizing", True) else 0.0)

        # volume confirmation
        builder.add("volume confirmation", SignalCategory.VOLUME, 1.0 if bump.get("volume_confirmation") else 0.0)

        if state.is_panic:
            return Action.NO_SIGNAL
        if mom and vol_falling:
            return Action.BUY
        return Action.NO_SIGNAL
