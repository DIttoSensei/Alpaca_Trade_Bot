"""Backtest fill/cost simulation.

We deliberately model the costs that eat live returns: taker fees on both sides,
slippage and half the spread. Optimistic fill assumptions are the fastest way to
produce a backtest that cannot be traded, so here they are explicit and
conservative (spec sections 25, 63, 64).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.config.risk_config import RiskConfig
from app.config.settings import FeeModel
from app.core.enums import ExitReason, Regime
from app.core.types import (
    Candle,
    FeatureSnapshot,
    PositionRecord,
    TradeDecision,
    TradeRecord,
)
from app.risk.stop_manager import StopManager


@dataclass(slots=True)
class SimConfig:
    starting_equity: float = 10_000.0
    apply_slippage: bool = True
    apply_spread: bool = True
    fill_on_next_open: bool = True
    # If a bar touches both stop and target, assume the stop filled (worst case).
    pessimistic_intrabar: bool = True


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    qty: float
    entry_price: float
    entry_index: int
    entry_time: datetime
    stop_price: float
    target_price: float
    initial_risk_per_unit: float
    strategy: str
    regime: Regime
    feature_snapshot: Optional[FeatureSnapshot]
    expected_edge: float
    estimated_cost: float
    risk_amount: float
    entry_fee: float
    highest_price: float = 0.0
    trailing_state: str = "none"
    # Trend-following context (daily-trend strategy): no fixed target, a wide
    # ATR trailing stop, and a close-below-trend-SMA exit.
    trend_mode: bool = False
    trend_atr: float = 0.0
    trend_sma: float = 0.0

    def to_record(self) -> PositionRecord:
        return PositionRecord(
            symbol=self.symbol,
            qty=self.qty,
            average_entry=self.entry_price,
            current_price=self.entry_price,
            strategy=self.strategy,
            entry_time=self.entry_time,
            stop_price=self.stop_price,
            target_price=self.target_price,
            initial_stop_price=self.stop_price,
            initial_risk_per_unit=self.initial_risk_per_unit,
            risk_amount=self.risk_amount,
            feature_snapshot=self.feature_snapshot,
            regime=self.regime,
            trailing_state={"state": self.trailing_state},
        )


class CostModel:
    """Deterministic cost model shared with the live edge filter assumptions."""

    def __init__(self, fees: FeeModel, sim: SimConfig) -> None:
        self.fees = fees
        self.sim = sim

    def buy_fill(self, reference_price: float) -> float:
        adjustment = 0.0
        if self.sim.apply_slippage:
            adjustment += self.fees.estimated_slippage
        if self.sim.apply_spread:
            adjustment += self.fees.spread_cost / 2.0
        return reference_price * (1.0 + adjustment)

    def sell_fill(self, reference_price: float) -> float:
        adjustment = 0.0
        if self.sim.apply_slippage:
            adjustment += self.fees.estimated_slippage
        if self.sim.apply_spread:
            adjustment += self.fees.spread_cost / 2.0
        return reference_price * (1.0 - adjustment)

    def taker_fee(self, notional: float) -> float:
        return abs(notional) * self.fees.taker_fee


class Simulator:
    """Cost-aware fill simulator.

    Implemented as a plain class rather than a ``slots=True`` dataclass because
    it owns mutable runtime state created in ``__init__``; a slotted dataclass
    cannot be assigned attributes outside its declared fields.
    """

    def __init__(self, sim: SimConfig, risk: RiskConfig, fees: FeeModel) -> None:
        self.sim = sim
        self.risk = risk
        self.fees = fees
        self.costs = CostModel(self.fees, self.sim)
        self.stops = StopManager(self.risk)
        self.positions: dict[str, OpenPosition] = {}
        self.equity: float = self.sim.starting_equity
        self.cash: float = self.sim.starting_equity
        self.realized_pnl: float = 0.0
        self.trades: list[TradeRecord] = []
        self.trade_seq: int = 0

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.positions.clear()
        self.equity = self.sim.starting_equity
        self.cash = self.sim.starting_equity
        self.realized_pnl = 0.0
        self.trades = []
        self.trade_seq = 0

    # -- entering --------------------------------------------------------
    def open_from_decision(
        self,
        decision: TradeDecision,
        reference_price: float,
        index: int,
        timestamp: datetime,
    ) -> OpenPosition:
        fill = self.costs.buy_fill(reference_price)
        qty = decision.quantity
        notional = fill * qty
        fee = self.costs.taker_fee(notional)
        self.cash -= notional + fee
        direction_offset = fill - reference_price
        meta = decision.metadata or {}
        trend_mode = bool(meta.get("trend_mode", False))
        # ``target_price == 0.0`` is the sentinel for "no fixed target" (trend
        # mode). Only substitute a concrete target when one was intended.
        if decision.target_price:
            target_price = decision.target_price
        elif trend_mode:
            target_price = 0.0
        else:
            target_price = fill * (1 + self.risk.target_r_multiple * self.risk.min_stop_distance)
        pos = OpenPosition(
            symbol=decision.symbol,
            qty=qty,
            entry_price=fill,
            entry_index=index,
            entry_time=timestamp,
            stop_price=decision.stop_price or fill * (1 - self.risk.max_stop_distance),
            target_price=target_price,
            initial_risk_per_unit=max(1e-12, fill - (decision.stop_price or fill)),
            strategy=decision.strategy,
            regime=decision.regime,
            feature_snapshot=decision.feature_snapshot,
            expected_edge=decision.expected_edge,
            estimated_cost=decision.estimated_cost,
            risk_amount=decision.risk_amount,
            entry_fee=fee,
            highest_price=fill,
            trend_mode=trend_mode,
            trend_atr=float(meta.get("trend_atr", 0.0) or 0.0),
            trend_sma=float(meta.get("trend_sma", 0.0) or 0.0),
        )
        # Recompute risk per unit against the actual fill (slippage changes it).
        pos.initial_risk_per_unit = max(1e-12, fill - pos.stop_price)
        self.positions[decision.symbol] = pos
        return pos

    # -- marking ---------------------------------------------------------
    def mark_to_market(self, prices: dict[str, float]) -> None:
        """equity = cash (already net of entry cost + fees) + market value held."""
        self.equity = self.cash + sum(
            prices.get(s, p.entry_price) * p.qty for s, p in self.positions.items()
        )

    @property
    def open_risk(self) -> float:
        return sum(
            max(0.0, (p.entry_price - p.stop_price) * p.qty) for p in self.positions.values()
        )

    # -- exiting ---------------------------------------------------------
    def check_bar_exits(self, symbol: str, bar: Candle) -> list[TradeRecord]:
        """Process stop/target/trend exits against a bar's high/low/close."""
        pos = self.positions.get(symbol)
        if pos is None:
            return []
        pos.highest_price = max(pos.highest_price, bar.high)

        stop_hit = bar.low <= pos.stop_price
        # A target of 0.0 means "no fixed target" (trend mode).
        has_target = pos.target_price > 0
        target_hit = has_target and bar.high >= pos.target_price
        if not stop_hit and not target_hit:
            # Trend exit: price closes back below the trend SMA (canonical trend
            # invalidation). Evaluated on the CLOSED bar, so no look-ahead.
            if pos.trend_mode and pos.trend_sma > 0 and bar.close < pos.trend_sma:
                return [
                    self._close(
                        pos, bar.close, ExitReason.STRATEGY_INVALIDATION, bar.timestamp, bar
                    )
                ]
            self._trail(pos, bar)
            return []

        if stop_hit and target_hit and self.sim.pessimistic_intrabar:
            return [self._close(pos, pos.stop_price, self._stop_reason(pos), bar.timestamp, bar)]
        if target_hit:
            return [self._close(pos, pos.target_price, ExitReason.TAKE_PROFIT, bar.timestamp, bar)]
        return [self._close(pos, pos.stop_price, self._stop_reason(pos), bar.timestamp, bar)]

    @staticmethod
    def _stop_reason(pos: OpenPosition) -> ExitReason:
        """Distinguish a trailing/protected exit from an initial stop-out.

        Once a stop has been trailed to breakeven or beyond (``stop_price >=
        entry_price`` on a long, or ``trailing_state`` is active) hitting it books
        a profit, not a loss. Labelling it ``TRAILING_STOP`` stops those exits
        from being misread as "STOP_LOSS showing a gain".
        """
        if pos.trailing_state != "none" or pos.stop_price >= pos.entry_price:
            return ExitReason.TRAILING_STOP
        return ExitReason.STOP_LOSS

    def close_position(
        self,
        symbol: str,
        reference_price: float,
        reason: ExitReason,
        timestamp: datetime,
        bar: Optional[Candle] = None,
    ) -> Optional[TradeRecord]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        return self._close(pos, reference_price, reason, timestamp, bar)

    def _trail(self, pos: OpenPosition, bar: Candle) -> None:
        new_stop, state = self.stops.update_trailing(
            entry_price=pos.entry_price,
            current_price=bar.close,
            initial_risk_per_unit=pos.initial_risk_per_unit,
            current_stop=pos.stop_price,
            atr=self._atr_of(bar, pos),
            side="buy",
            already_trailing=pos.trailing_state == "trailing",
            trend_mode=pos.trend_mode,
        )
        if new_stop > pos.stop_price:
            pos.stop_price = new_stop
            pos.trailing_state = state

    def _atr_of(self, bar: Candle, pos: Optional[OpenPosition] = None) -> float:
        """Volatility used for trailing, in the position's own timeframe.

        Bars carry no ATR, so a single-bar range is used as a FLOOR. On the M15
        execution timeframe that range is tiny, and trailing on it churned
        positions out on noise -- the documented bug. We therefore (a) prefer the
        DAILY ATR captured at entry for trend positions, and (b) floor the
        intraday proxy at the stop distance implied by ``min_stop_distance``.
        """
        if pos is not None and pos.trend_mode and pos.trend_atr > 0:
            return pos.trend_atr
        floor = (pos.entry_price * self.risk.min_stop_distance) if pos is not None else 0.0
        return max(floor, 1e-12, (bar.high - bar.low))

    def _close(
        self,
        pos: OpenPosition,
        stop_reference: float,
        reason: ExitReason,
        timestamp: datetime,
        bar: Optional[Candle],
    ) -> TradeRecord:
        fill = self.costs.sell_fill(stop_reference)
        notional = fill * pos.qty
        exit_fee = self.costs.taker_fee(notional)
        proceeds = notional - exit_fee
        self.cash += proceeds

        gross = (fill - pos.entry_price) * pos.qty
        net = gross - pos.entry_fee - exit_fee
        self.realized_pnl += net
        # Slippage = exit-side market impact versus the intended exit reference.
        slippage_cost = abs(fill - stop_reference) * pos.qty
        self.trade_seq += 1
        holding = max(0.0, (timestamp - pos.entry_time).total_seconds())
        trade = TradeRecord(
            trade_id=f"{pos.symbol}-{pos.entry_index}-{self.trade_seq}",
            symbol=pos.symbol,
            strategy=pos.strategy,
            entry_price=pos.entry_price,
            exit_price=fill,
            qty=pos.qty,
            fees=pos.entry_fee + exit_fee,
            slippage=slippage_cost,
            gross_pnl=gross,
            net_pnl=net,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            holding_time_seconds=holding,
            exit_reason=reason,
            regime=pos.regime,
            signal_score=0.0,
            expected_edge=pos.expected_edge,
            estimated_cost=pos.estimated_cost,
            actual_cost=pos.entry_fee + exit_fee + slippage_cost,
            feature_snapshot=pos.feature_snapshot,
        )
        self.trades.append(trade)
        del self.positions[pos.symbol]
        return trade
