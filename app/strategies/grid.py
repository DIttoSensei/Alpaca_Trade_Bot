"""Grid trading strategy (stateful, level-based).

Unlike the other strategies, grid trading is NOT a single ``evaluate()->Signal``
call: it maintains a ladder of resting orders across many price levels and fills
them against each bar's high/low. This module therefore implements the ladder as
a self-contained, deterministic state machine plus an explicit, documented
backtest bar processor. It reuses the existing cost model (``FeeModel``), the
existing risk/exposure governance (called by the runner) and the existing SQLite
checkpoint layer for durability.

Mechanism
---------
* Range centred on a slow reference (SMA on the regime timeframe)::

      upper = ref + ATR * range_multiplier
      lower = ref - ATR * range_multiplier

* The range is divided into ``grid_levels`` equal levels. A resting BUY sits at
  every level at or below the current price. When a buy fills, a SELL rests one
  level up; when that sell fills, a buy is re-armed at the original level. This
  is spot / long-inventory only: a "sell" always exits previously-bought
  inventory, never a naked short.
* The grid only operates in RANGE / LOW_VOLATILITY regimes.
* The grid re-centres (recompute range, cancel + replace levels) on a bar
  interval or when price leaves the range.

Backtest fill-ordering assumption (READ BEFORE TRUSTING RESULTS)
----------------------------------------------------------------
A single candle can span several grid levels, so close-only evaluation would
silently miscount/misorder fills. We therefore MUST use each bar's high AND low.
Because we have no tick data, touch order within a bar is ambiguous. We resolve
it with the documented WORST-CASE ordering for a long-inventory grid: each bar
is assumed to have moved to its LOW first and then to its HIGH.

* Down-leg (low first): BUY levels are filled low-to-high (the deepest / most
  favourable fill first). Only ``max_filled_levels`` buys may fill per bar.
* Up-leg (high second): SELL levels are filled high-to-low, booking the profit
  of inventory acquired on the down-leg.

This is deliberately conservative: it books the best buy and only then the
best sell, which is the least favourable net for a range-fade but matches the
pessimistic intrabar convention already used elsewhere in the simulator (a bar
that touches both stop and target is assumed to have hit the stop first). The
alternative (assume the high was reached first) would flatter results and is
NOT used. Set ``GridBacktester``'s ``fill_order`` to record which assumption is
active; the value is asserted in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from app.config.risk_config import RiskConfig
from app.config.settings import FeeModel
from app.config.strategy_config import GridParams
from app.core.enums import Action, ExitReason, Regime, SignalCategory
from app.core.types import Candle, MarketState, Signal, TradeRecord, iso, utcnow
from app.strategies.base import Capability, ScoreBuilder, Strategy

logger = logging.getLogger(__name__)

# Documented intrabar fill ordering for the grid backtest.
FILL_ORDER_LOW_FIRST = "low_first_then_high"


# --------------------------------------------------------------------------- #
# Range / level geometry
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GridRange:
    """A resolved grid range with its N price levels."""

    ref: float
    upper: float
    lower: float
    levels: list[float]  # ascending, N entries from lower..upper
    width: float
    spacing: float

    def in_range(self, price: float) -> bool:
        return self.lower <= price <= self.upper


def build_range(
    ref: float,
    atr: float,
    *,
    levels: int,
    range_multiplier: float,
) -> GridRange:
    """Build an ``levels``-step grid centred on ``ref`` with ATR-scaled width."""
    n = max(2, int(levels))
    half = max(1e-12, atr * range_multiplier)
    upper = ref + half
    lower = max(1e-12, ref - half)
    step = (upper - lower) / (n - 1)
    rungs = [lower + step * i for i in range(n)]
    return GridRange(ref=ref, upper=upper, lower=lower, levels=rungs, width=upper - lower, spacing=step)


def fee_safety_ok(spacing: float, price: float, fees: FeeModel, multiple: float) -> bool:
    """True when the level gap clears round-trip cost by ``multiple``.

    Round-trip cost is 2x taker fee + expected slippage + spread (Fraction of
    notional). The gap between adjacent levels must exceed ``multiple`` times
    that cost, otherwise every completed grid cycle is a guaranteed per-cycle
    loss and the whole grid is skipped (safety #2).
    """
    if price <= 0:
        return False
    cost_frac = fees.round_trip_cost(include_slippage=True, include_spread=True)
    return (spacing / price) >= (multiple * cost_frac)


# --------------------------------------------------------------------------- #
# Ladder state (durable)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LadderLevel:
    """One rung of the grid ladder.

    ``qty`` is the per-level order size (computed once from capital_per_grid);
    ``inventory`` is the quantity currently held having been bought at this rung
    and not yet sold back up one level.
    """

    price: float
    qty: float
    inventory: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"price": self.price, "qty": self.qty, "inventory": self.inventory}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LadderLevel":
        return cls(price=float(data["price"]), qty=float(data["qty"]), inventory=float(data.get("inventory", 0.0)))


@dataclass(slots=True)
class GridState:
    """Full durable state of one symbol's grid."""

    symbol: str
    ref: float = 0.0
    upper: float = 0.0
    lower: float = 0.0
    width: float = 0.0
    spacing: float = 0.0
    levels: list[LadderLevel] = field(default_factory=list)
    last_recenter_bar: int = -10**9
    recenter_count: int = 0
    regime_exit_flattens: int = 0
    hard_bound_triggers: int = 0
    active: bool = True

    @property
    def inventory_qty(self) -> float:
        return sum(lvl.inventory for lvl in self.levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ref": self.ref,
            "upper": self.upper,
            "lower": self.lower,
            "width": self.width,
            "spacing": self.spacing,
            "levels": [lvl.to_dict() for lvl in self.levels],
            "last_recenter_bar": self.last_recenter_bar,
            "recenter_count": self.recenter_count,
            "regime_exit_flattens": self.regime_exit_flattens,
            "hard_bound_triggers": self.hard_bound_triggers,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GridState":
        st = cls(symbol=str(data["symbol"]))
        st.ref = float(data.get("ref", 0.0))
        st.upper = float(data.get("upper", 0.0))
        st.lower = float(data.get("lower", 0.0))
        st.width = float(data.get("width", 0.0))
        st.spacing = float(data.get("spacing", 0.0))
        st.levels = [LadderLevel.from_dict(d) for d in data.get("levels", [])]
        st.last_recenter_bar = int(data.get("last_recenter_bar", -10**9))
        st.recenter_count = int(data.get("recenter_count", 0))
        st.regime_exit_flattens = int(data.get("regime_exit_flattens", 0))
        st.hard_bound_triggers = int(data.get("hard_bound_triggers", 0))
        st.active = bool(data.get("active", True))
        return st


def dump_state(state: GridState) -> dict[str, Any]:
    return state.to_dict()


def load_state(data: dict[str, Any]) -> GridState:
    return GridState.from_dict(data)


class GridStrategy(Strategy):
    """Stateful grid strategy.

    ``evaluate()`` (inherited) is intentionally inert — grid fills are driven by
    ``GridBacktester`` against bar high/low rather than the per-bar signal path.
    The class exists so grid is a first-class strategy (config, risk governance,
    regime capabilities) and so its intrabar semantics are documented in one
    place.
    """

    name = "grid"
    capability = Capability(
        allowed_regimes=(Regime.RANGE, Regime.LOW_VOLATILITY),
        minimum_data=220,
        risk_multiplier=1.0,
    )

    def __init__(self, params: GridParams | None = None) -> None:
        super().__init__(params or GridParams())

    def _evaluate(self, state: MarketState, builder: ScoreBuilder) -> Action:
        # No per-bar directional opinion: grid P&L comes from level cycling.
        builder.add("grid manages resting levels", SignalCategory.STRUCTURE, 0.0)
        return Action.NO_SIGNAL


# --------------------------------------------------------------------------- #
# Backtest engine (bar high/low, deterministic intrabar ordering)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GridFill:
    """A single grid fill (one level touched).

    ``spacing`` / ``lower`` capture the ladder geometry in force at fill time so
    a consumer can prove the fill landed on a configured level (safety check in
    ``scripts/backtest_grid.py``): ``(level_price - lower) / spacing`` must be an
    integer.
    """

    bar_index: int
    timestamp: datetime
    side: str  # "buy" | "sell"
    level_price: float
    fill_price: float
    qty: float
    spacing: float = 0.0
    lower: float = 0.0


@dataclass(slots=True)
class GridBacktestResult:
    symbol: str
    trades: list[TradeRecord] = field(default_factory=list)
    fills: list[GridFill] = field(default_factory=list)
    regime_exit_flattens: int = 0
    hard_bound_triggers: int = 0
    recenter_count: int = 0
    peak_committed: float = 0.0
    levels_placed_skipped: int = 0
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    starting_equity: float = 10_000.0
    final_equity: float = 10_000.0


class GridBacktester:
    """Runs one symbol's grid over a candle series (backtest-only)."""

    def __init__(
        self,
        symbol: str,
        params: GridParams,
        fees: FeeModel,
        risk: RiskConfig,
        *,
        starting_equity: float = 10_000.0,
        apply_slippage: bool = True,
        apply_spread: bool = True,
        fill_order: str = FILL_ORDER_LOW_FIRST,
    ) -> None:
        self.symbol = symbol
        self.p = params
        self.fees = fees
        self.risk = risk
        self.equity0 = starting_equity
        self.apply_slippage = apply_slippage
        self.apply_spread = apply_spread
        self.fill_order = fill_order
        self.state: Optional[GridState] = None
        self.cash = starting_equity
        self._trade_seq = 0
        self._open_batches: list[dict[str, Any]] = []
        self.result = GridBacktestResult(symbol=symbol, starting_equity=starting_equity)

    # -- cost model (mirrors app.backtest.simulator.CostModel) --------------
    def _buy_fill(self, ref: float) -> float:
        adj = 0.0
        if self.apply_slippage:
            adj += self.fees.estimated_slippage
        if self.apply_spread:
            adj += self.fees.spread_cost / 2.0
        return ref * (1.0 + adj)

    def _sell_fill(self, ref: float) -> float:
        adj = 0.0
        if self.apply_slippage:
            adj += self.fees.estimated_slippage
        if self.apply_spread:
            adj += self.fees.spread_cost / 2.0
        return ref * (1.0 - adj)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fees.taker_fee

    # -- range setup --------------------------------------------------------
    def _recenter(
        self,
        ref: float,
        atr: float,
        price: float,
        bar_index: int,
        *,
        skip_fee_check: bool = False,
    ) -> bool:
        """(Re)build the ladder. Returns False if the grid is not viable.

        Safety #2: if the spacing does not clear round-trip cost by the required
        multiple, skip the whole grid rather than trade at a guaranteed loss.
        ``skip_fee_check`` is a backtest/test affordance that forces a ladder onto
        a deliberately tight synthetic range; production never sets it.
        """
        rng = build_range(
            ref,
            atr,
            levels=self.p.grid_levels,
            range_multiplier=self.p.range_multiplier,
        )
        if not skip_fee_check and not fee_safety_ok(
            rng.spacing, price, self.fees, self.p.fee_safety_multiple
        ):
            self.result.levels_placed_skipped += 1
            logger.debug(
                "grid %s: spacing %.6f does not clear %.1fx round-trip cost; grid skipped",
                self.symbol,
                rng.spacing,
                self.p.fee_safety_multiple,
            )
            if self.state is not None:
                self.state.active = False
            return False

        # Capital per level: whole-grid capital / number of buy-capable levels.
        capital = self.p.capital_per_grid * self.equity0
        n_levels = len(rng.levels)
        cap_per_level = capital / max(1, n_levels)

        levels: list[LadderLevel] = []
        for rung in rng.levels:
            qty = cap_per_level / rung if rung > 0 else 0.0
            levels.append(LadderLevel(price=rung, qty=qty))

        # Preserve existing inventory (and its acquisition batches) across a
        # re-center by mapping each OLD rung to the NEAREST new rung. Without
        # this, price drift would silently orphan held inventory.
        if self.state is not None and self.state.levels:
            new_prices = [lvl.price for lvl in levels]
            remap: dict[float, float] = {}
            for old in self.state.levels:
                nearest = min(new_prices, key=lambda p: abs(p - old.price))
                remap[round(old.price, 10)] = nearest
                if old.inventory > 0:
                    for lvl in levels:
                        if abs(lvl.price - nearest) < 1e-9:
                            lvl.inventory += old.inventory
                            break
            for batch in self._open_batches:
                nearest = remap.get(round(batch["level_price"], 10))
                if nearest is not None:
                    batch["level_price"] = nearest

        if self.state is None:
            self.state = GridState(symbol=self.symbol)
        st = self.state
        st.ref, st.upper, st.lower = rng.ref, rng.upper, rng.lower
        st.width, st.spacing = rng.width, rng.spacing
        st.levels = levels
        st.last_recenter_bar = bar_index
        st.recenter_count += 1
        st.active = True
        self.result.recenter_count = st.recenter_count
        return True

    # -- inventory helpers --------------------------------------------------
    def _flatten(self, price: float, timestamp: datetime, bar_index: int, reason: ExitReason) -> list[TradeRecord]:
        """Market-sell all held inventory, booking round-trip trades."""
        out: list[TradeRecord] = []
        if self.state is None:
            return out
        for lvl in self.state.levels:
            if lvl.inventory <= 0:
                continue
            out.extend(self._sell_inventory(lvl, price, timestamp, bar_index, reason))
        return out

    def _sell_inventory(
        self,
        src: LadderLevel,
        sell_price: float,
        timestamp: datetime,
        bar_index: int,
        reason: ExitReason,
    ) -> list[TradeRecord]:
        """Sell the inventory held ON ``src`` (the rung it was bought at).

        ``sell_price`` is the price actually realised (one level up on a normal
        take-profit cycle, or the market price on a flatten). The acquisition
        batches recorded for ``src`` are consumed FIFO; each consumed batch
        produces one completed ``TradeRecord`` so grid P&L is attributable
        level-by-level. Decrementing ``src.inventory`` (NOT the sell rung's) is
        what implements "re-buy at the original level once the sell fills".
        """
        out: list[TradeRecord] = []
        remaining = src.inventory
        while remaining > 1e-12:
            batch = next(
                (
                    b
                    for b in self._open_batches
                    if abs(b["level_price"] - src.price) < 1e-9 and b["remaining"] > 1e-12
                ),
                None,
            )
            if batch is None:
                break
            take = min(remaining, batch["remaining"])
            fill = self._sell_fill(sell_price)
            notional = fill * take
            exit_fee = self._fee(notional)
            self.cash += notional - exit_fee
            gross = (fill - batch["entry_price"]) * take
            fees_total = batch["fee_alloc"] * (take / batch["qty"]) + exit_fee
            net = gross - fees_total
            self._trade_seq += 1
            holding = max(0.0, (timestamp - batch["entry_time"]).total_seconds())
            rec = TradeRecord(
                trade_id=f"{self.symbol}-grid-{self._trade_seq}",
                symbol=self.symbol,
                strategy="grid",
                entry_price=batch["entry_price"],
                exit_price=fill,
                qty=take,
                fees=fees_total,
                slippage=abs(fill - sell_price) * take,
                gross_pnl=gross,
                net_pnl=net,
                entry_time=batch["entry_time"],
                exit_time=timestamp,
                holding_time_seconds=holding,
                exit_reason=reason,
                regime=batch["regime"],
                signal_score=0.0,
                expected_edge=batch["expected_edge"],
                estimated_cost=batch["estimated_cost"],
                actual_cost=fees_total + abs(fill - sell_price) * take,
            )
            self.result.trades.append(rec)
            out.append(rec)
            batch["remaining"] -= take
            batch["fee_alloc"] -= fees_total - exit_fee
            src.inventory -= take
            remaining -= take
        if src.inventory < 1e-12:
            src.inventory = 0.0
        # Drop fully consumed batches.
        self._open_batches = [b for b in self._open_batches if b["remaining"] > 1e-12]
        return out

    def _new_batch(
        self,
        lvl: LadderLevel,
        qty: float,
        entry_fill: float,
        timestamp: datetime,
        regime: Regime,
        expected_edge: float,
        estimated_cost: float,
        fee: float,
    ) -> None:
        self._open_batches.append(
            {
                "level_price": lvl.price,
                "qty": qty,
                "remaining": qty,
                "entry_price": entry_fill,
                "entry_time": timestamp,
                "fee_alloc": fee,
                "regime": regime,
                "expected_edge": expected_edge,
                "estimated_cost": estimated_cost,
            }
        )

    # -- per-bar processing -------------------------------------------------
    def process_bar(
        self,
        bar: Candle,
        index: int,
        regime: Regime,
    ) -> None:
        """Advance the grid over one bar using its high AND low.

        See the module docstring for the documented worst-case ordering:
        the bar is treated as moving to its LOW first (buy leg) and then to its
        HIGH (sell leg). Fills are counted and committed capital is tracked.
        """
        if self.state is None or not self.state.active:
            return
        expected_edge = 0.0
        estimated_cost = self.fees.round_trip_cost()
        committed = self._committed()

        # -- Safety #1b: hard-bound trigger ----------------------------------
        # Price beyond the bound by more than hard_bound_multiplier * range width
        # means the ranging assumption failed regardless of the regime detector.
        low_break = self.state.lower - self.p.hard_bound_multiplier * self.state.width
        high_break = self.state.upper + self.p.hard_bound_multiplier * self.state.width
        if bar.low <= low_break or bar.high >= high_break:
            if self.state.inventory_qty > 0:
                self._flatten(bar.close, bar.timestamp, index, ExitReason.REGIME_REVERSAL)
                self.state.regime_exit_flattens += 1
            self.state.hard_bound_triggers += 1
            self.result.regime_exit_flattens = self.state.regime_exit_flattens
            self.result.hard_bound_triggers = self.state.hard_bound_triggers
            self.state.active = False
            return

        # -- Down-leg: fill BUY levels low->high -----------------------------
        filled_buys = 0
        portfolio_room = max(0.0, self.risk.max_symbol_exposure * self.equity0 - committed)
        for lvl in self.state.levels:
            if filled_buys >= self.p.max_filled_levels:
                break
            if lvl.inventory > 0:
                continue  # already holding this rung: it is now a sell level
            # A resting buy below the open fills only if the bar's low reached
            # it AND the rung sat at or below the bar's OPEN. Bounding by the
            # open stops the down-leg from "buying" rungs the price never
            # descended to (a rung above the whole bar was never touched).
            if bar.low <= lvl.price <= bar.open:
                # Respect capital cap first, then the portfolio risk budget.
                notional_cap = self.p.capital_per_grid * self.equity0
                if committed + lvl.qty * lvl.price > notional_cap + 1e-9:
                    continue
                if committed + lvl.qty * lvl.price > portfolio_room + 1e-9:
                    break
                fill = self._buy_fill(lvl.price)
                notional = fill * lvl.qty
                fee = self._fee(notional)
                self.cash -= notional + fee
                lvl.inventory += lvl.qty
                committed += lvl.qty * lvl.price
                filled_buys += 1
                self._new_batch(
                    lvl, lvl.qty, fill, bar.timestamp, regime, expected_edge, estimated_cost, fee
                )
                self.result.fills.append(
                    GridFill(
                        bar_index=index,
                        timestamp=bar.timestamp,
                        side="buy",
                        level_price=lvl.price,
                        fill_price=fill,
                        qty=lvl.qty,
                        spacing=self.state.spacing,
                        lower=self.state.lower,
                    )
                )
                self.result.peak_committed = max(self.result.peak_committed, committed)

        # -- Up-leg: fill SELL levels high->low ------------------------------
        # A sell always rests ONE level ABOVE the rung its inventory was bought
        # at. Iterate adjacent rung pairs from the top down: inventory sitting on
        # rung k-1 is sold at rung k's price when the bar's high reaches it. This
        # is the documented worst-case ordering (after the low was touched).
        levels = self.state.levels
        for k in range(len(levels) - 1, 0, -1):
            src = levels[k - 1]
            if src.inventory <= 0:
                continue
            sell_at = levels[k]
            if bar.high >= sell_at.price:
                recs = self._sell_inventory(
                    src, sell_at.price, bar.timestamp, index, ExitReason.TAKE_PROFIT
                )
                if recs:
                    sold_qty = sum(r.qty for r in recs)
                    self.result.fills.append(
                        GridFill(
                            bar_index=index,
                            timestamp=bar.timestamp,
                            side="sell",
                            level_price=sell_at.price,
                            fill_price=sell_at.price,
                            qty=sold_qty,
                            spacing=self.state.spacing,
                            lower=self.state.lower,
                        )
                    )

    def _committed(self) -> float:
        if self.state is None:
            return 0.0
        return sum(lvl.inventory * lvl.price for lvl in self.state.levels)

    def mark_to_market(self, price: float, timestamp: datetime) -> None:
        if self.state is None:
            equity = self.cash
        else:
            equity = self.cash + self.state.inventory_qty * price
        self.result.equity_curve.append((timestamp, equity))
        self.result.final_equity = equity


# --------------------------------------------------------------------------- #
# Portfolio runner (reuses feature store + regime detector + indicators)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GridPortfolioResult:
    per_symbol: dict[str, GridBacktestResult] = field(default_factory=dict)
    starting_equity: float = 10_000.0
    fill_order: str = FILL_ORDER_LOW_FIRST

    @property
    def trades(self) -> list[TradeRecord]:
        out: list[TradeRecord] = []
        for r in self.per_symbol.values():
            out.extend(r.trades)
        out.sort(key=lambda t: t.entry_time)
        return out

    @property
    def final_equity(self) -> float:
        # Sum each symbol's isolated equity (each grid is capital-siloed by
        # construction, so this is a conservative, additive portfolio view).
        return sum(r.final_equity for r in self.per_symbol.values())

    def summary(self) -> dict[str, Any]:
        trades = self.trades
        net = sum(t.net_pnl for t in trades)
        regime_flattens = sum(r.regime_exit_flattens for r in self.per_symbol.values())
        hard_bound = sum(r.hard_bound_triggers for r in self.per_symbol.values())
        peak = max((r.peak_committed for r in self.per_symbol.values()), default=0.0)
        # Each symbol's grid is capital-siloed, so the portfolio capital base is
        # the per-symbol starting equity times the number of symbols that ran.
        base = self.starting_equity * max(1, len(self.per_symbol))
        return {
            "trades": len(trades),
            "net_pnl": net,
            "return_pct": (net / base) * 100.0 if base else 0.0,
            "base_equity": base,
            "regime_exit_flattens": regime_flattens,
            "hard_bound_triggers": hard_bound,
            "peak_committed": peak,
            "fill_order": self.fill_order,
        }


def _latest_complete(times: list[datetime], ts: datetime) -> int:
    """Index of the latest bucket whose close time is at or before ``ts``."""
    import bisect

    return bisect.bisect_right(times, ts) - 1


def run_grid_backtest(
    candles_by_key: dict[tuple[str, Timeframe], Sequence[Candle]],
    config: Any = None,
    *,
    starting_equity: float = 10_000.0,
    collection_timeframe: Timeframe | None = None,
    state_store: Any = None,
    apply_slippage: bool = True,
    apply_spread: bool = True,
) -> GridPortfolioResult:
    """Run the grid strategy in isolation over historical candles.

    Reuses the exact shared stack (``build_stack``) for the regime detector and
    feature store, and the existing cost model. Grid is run per symbol so its
    many-level state never collides with the streaming strategy ensemble.

    ``state_store`` (an ``app.state.checkpoints.CheckpointStore``) is optional;
    when supplied the ladder state is written through so a restart mid-grid can
    recover it (idempotency: state is keyed by symbol, so re-running is safe).
    """
    from datetime import timedelta

    from app.assembly import build_stack
    from app.data.candle_store import CandleStore, aggregate
    from app.indicators.core import atr as _atr
    from app.indicators.core import sma as _sma
    from app.state.checkpoints import KEY_GRID_STATE

    cfg = config
    out = GridPortfolioResult(starting_equity=starting_equity)

    for (symbol, tf), bars in candles_by_key.items():
        if collection_timeframe is not None and tf != collection_timeframe:
            continue
        bars = sorted(bars, key=lambda c: c.timestamp)
        if not bars:
            continue

        params = (cfg.strategies.grid if cfg is not None else GridParams())
        if not getattr(params, "enabled", False):
            continue

        stack = build_stack(cfg, candles=CandleStore(max_bars=10**9))
        store = stack.candles
        store.merge(bars)
        stack.features.invalidate()

        regime_tf = stack.context.regime
        regime_bars = aggregate(bars, regime_tf)
        if not regime_bars:
            continue
        span = timedelta(seconds=regime_tf.seconds)
        close_times = [b.timestamp + span for b in regime_bars]
        closes = [b.close for b in regime_bars]
        sma_series = _sma(closes, params.reference_sma_period)
        atr_series = _atr(
            [b.high for b in regime_bars],
            [b.low for b in regime_bars],
            closes,
            params.atr_period,
        )

        fees = cfg.settings.fees if cfg is not None else FeeModel()
        risk = cfg.risk if cfg is not None else RiskConfig()
        engine = GridBacktester(
            symbol, params, fees, risk,
            starting_equity=starting_equity,
            apply_slippage=apply_slippage,
            apply_spread=apply_spread,
            fill_order=FILL_ORDER_LOW_FIRST,
        )

        warmup = max(220, params.reference_sma_period + 2)

        for i, bar in enumerate(bars):
            if i < warmup:
                continue
            timestamp = bar.timestamp

            # Regime from the shared, causal detector (exec features).
            state, _rr = stack.pipeline.build_state(symbol, timestamp, i)
            regime = state.regime
            in_grid_regime = regime in (Regime.RANGE, Regime.LOW_VOLATILITY)

            if not in_grid_regime:
                # Safety #1: leave the grid immediately; flatten held inventory.
                if engine.state is not None and engine.state.inventory_qty > 0:
                    engine._flatten(bar.close, timestamp, i, ExitReason.REGIME_REVERSAL)
                    engine.state.regime_exit_flattens += 1
                    engine.result.regime_exit_flattens = engine.state.regime_exit_flattens
                if engine.state is not None:
                    engine.state.active = False
                engine.mark_to_market(bar.close, timestamp)
                continue

            # Latest COMPLETE regime-timeframe bucket (no lookahead).
            j = _latest_complete(close_times, timestamp)
            if j < 0:
                engine.mark_to_market(bar.close, timestamp)
                continue
            ref = sma_series[j]
            atr_v = atr_series[j]
            if ref is None or atr_v is None or atr_v <= 0:
                engine.mark_to_market(bar.close, timestamp)
                continue

            st = engine.state
            need_recenter = (
                st is None
                or not st.active
                or (i - st.last_recenter_bar) >= params.recenter_interval
                or bar.close > st.upper
                or bar.close < st.lower
            )
            if need_recenter:
                engine._recenter(float(ref), float(atr_v), bar.close, i)

            if engine.state is not None and engine.state.active:
                engine.process_bar(bar, i, regime)
            engine.mark_to_market(bar.close, timestamp)

        # Flatten any residual inventory at the final price.
        if engine.state is not None and engine.state.inventory_qty > 0:
            last = bars[-1]
            engine._flatten(last.close, last.timestamp, len(bars) - 1, ExitReason.MANUAL)

        # Persist ladder state (best-effort; never fatal to a backtest).
        if state_store is not None and engine.state is not None:
            try:
                state_store.save(f"{KEY_GRID_STATE}:{symbol}", engine.state.to_dict())
            except Exception:  # noqa: BLE001
                logger.debug("grid state persistence skipped for %s", symbol)

        out.per_symbol[symbol] = engine.result

    return out
