"""Execution monitor: position protection and exit decisions.

Runs on every execution tick. For each open position it:

* updates price/unrealised P&L,
* tightens the stop (breakeven / ATR trailing) via the StopManager,
* decides whether a hard exit is warranted (stop breached, target hit, regime
  reversal, strategy invalidation, kill switch, portfolio risk).

The monitor never submits orders itself: it returns ``ExitInstruction`` objects
that the caller routes through the OrderManager (spec sections 47, 66, 90).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.config.risk_config import RiskConfig
from app.core.enums import ExitReason, Regime, TradingState
from app.core.types import MarketState, PositionRecord, utcnow
from app.risk.stop_manager import StopManager

logger = logging.getLogger(__name__)

# Regimes under which a long position should be abandoned outright.
BEARISH_REGIMES = {Regime.STRONG_DOWNTREND, Regime.PANIC}


@dataclass(slots=True)
class ExitInstruction:
    symbol: str
    qty: float
    reason: ExitReason
    detail: str = ""
    new_stop: Optional[float] = None


@dataclass(slots=True)
class PositionUpdate:
    position: PositionRecord
    new_stop: Optional[float] = None
    trailing_state: str = "none"
    exit_instruction: Optional[ExitInstruction] = None
    metadata: dict = field(default_factory=dict)


class ExecutionMonitor:
    def __init__(self, risk: RiskConfig, stops: StopManager | None = None) -> None:
        self.risk = risk
        self.stops = stops or StopManager(risk)

    def update_position(
        self,
        position: PositionRecord,
        current_price: float,
        atr: Optional[float],
        state: MarketState | None = None,
        trading_state: TradingState = TradingState.TRADING,
        kill_switch: bool = False,
        side: str = "buy",
    ) -> PositionUpdate:
        position.current_price = current_price
        if position.average_entry > 0:
            position.unrealized_pnl = (current_price - position.average_entry) * position.qty

        # 1) Hard exits take priority, cheapest/most certain first.
        exit_instruction = self._hard_exit(position, current_price, state, trading_state, kill_switch, side)
        if exit_instruction is not None:
            return PositionUpdate(position=position, exit_instruction=exit_instruction)

        # 2) Otherwise tighten the stop.
        new_stop: Optional[float] = None
        trailing_state = position.trailing_state.get("state", "none") if position.trailing_state else "none"
        if (
            atr is not None
            and atr > 0
            and position.initial_risk_per_unit > 0
            and position.stop_price is not None
        ):
            new_stop, trailing_state = self.stops.update_trailing(
                entry_price=position.average_entry,
                current_price=current_price,
                initial_risk_per_unit=position.initial_risk_per_unit,
                current_stop=position.stop_price,
                atr=atr,
                side=side,
                already_trailing=trailing_state == "trailing",
            )
            if new_stop is not None and abs(new_stop - position.stop_price) > 1e-12:
                position.stop_price = new_stop
                position.trailing_state = {
                    "state": trailing_state,
                    "updated_at": utcnow().isoformat(),
                }
        return PositionUpdate(
            position=position,
            new_stop=new_stop,
            trailing_state=trailing_state,
        )

    def _hard_exit(
        self,
        position: PositionRecord,
        current_price: float,
        state: MarketState | None,
        trading_state: TradingState,
        kill_switch: bool,
        side: str,
    ) -> Optional[ExitInstruction]:
        # Kill switch / emergency: flatten everything.
        if kill_switch:
            return ExitInstruction(position.symbol, position.qty, ExitReason.EMERGENCY_SHUTDOWN, "kill switch")
        if trading_state in (TradingState.EMERGENCY,):
            return ExitInstruction(position.symbol, position.qty, ExitReason.EMERGENCY_SHUTDOWN, "emergency")

        if side == "buy":
            if position.stop_price is not None and current_price <= position.stop_price:
                return ExitInstruction(position.symbol, position.qty, ExitReason.STOP_LOSS, "stop breached")
            if position.target_price is not None and current_price >= position.target_price:
                return ExitInstruction(position.symbol, position.qty, ExitReason.TAKE_PROFIT, "target reached")

        if state is not None:
            if state.regime in BEARISH_REGIMES:
                return ExitInstruction(
                    position.symbol, position.qty, ExitReason.REGIME_REVERSAL, f"regime {state.regime}"
                )
            features = state.features.values
            if position.strategy and self._strategy_invalidated(position.strategy, features):
                return ExitInstruction(
                    position.symbol, position.qty, ExitReason.STRATEGY_INVALIDATION, "entry thesis broken"
                )
        return None

    @staticmethod
    def _strategy_invalidated(strategy: str, features: dict) -> bool:
        """Cheap, conservative invalidation checks per strategy family.

        Uses the exact feature keys emitted by app.indicators (ema20/ema50/rsi/
        bb_upper). A missing feature is treated as "not invalidated" so we never
        exit purely because of absent data.
        """
        if strategy in ("trend_pullback", "momentum", "legacy_trend_pullback"):
            ema_fast = features.get("ema20")
            ema_slow = features.get("ema50")
            rsi = features.get("rsi")
            if ema_fast is not None and ema_slow is not None and ema_fast < ema_slow:
                return True
            if rsi is not None and rsi < 35:
                return True
        elif strategy == "breakout":
            close = features.get("close")
            recent_high = features.get("recent_high")
            if close is not None and recent_high is not None and close < recent_high * 0.98:
                return True
        return False

    def reconcile_stop(self, position: PositionRecord, atr: float) -> Optional[ExitInstruction]:
        """Safety net: if a position has no stop, derive one from entry and ATR."""
        if position.stop_price is not None or atr <= 0 or position.average_entry <= 0:
            return None
        plan = self.stops.initial_stop(position.average_entry, atr, side="buy")
        position.stop_price = plan.stop_price
        position.initial_stop_price = position.initial_stop_price or plan.stop_price
        position.initial_risk_per_unit = position.initial_risk_per_unit or plan.risk_per_unit
        if position.target_price is None:
            position.target_price = plan.target_price
        return None
