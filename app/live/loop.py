"""Live trading loop (SHADOW / PAPER / LIVE).

One process, one authoritative state machine, no divergence from the backtest:
the loop reuses ``app.assembly.build_stack`` for all decision-making, and only
adds the operational concerns that do not exist in a backtest (broker I/O,
reconciliation, health, watchdog, alerting).

Safety model:
  * the broker is authoritative; we reconcile at startup and periodically;
  * the kill switch is checked before every entry and is fail-closed;
  * entries are gated behind health, circuit breakers and the trading state;
  * order intent is written durably before any broker call (OrderManager);
  * SHADOW mode makes decisions and records them but submits nothing.

Position lifecycle (deliberately conservative):
  1. a decision is planned and the entry order is submitted;
  2. the entry is tracked in ``_entry_info`` but is NOT written to the position
     book until the broker confirms a fill (so a not-yet-filled order can never
     be mistaken for a closed position by account sync);
  3. once filled, a PositionRecord is materialised and managed on every tick;
  4. when the broker closes it, the round trip is recorded as a TradeRecord.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from app.assembly import StrategyStack, build_stack
from app.broker.account_sync import AccountSync
from app.broker.alpaca_client import AlpacaClient
from app.broker.market_data import AlpacaMarketData, MarketDataPort
from app.broker.trading_client import AlpacaTradingClient
from app.broker.websocket_manager import WebsocketManager
from app.config import BotConfig, load_config
from app.core.enums import (
    ExitReason,
    PositionState,
    Regime,
    TradingMode,
    TradingState,
)
from app.core.types import (
    PositionRecord,
    Quote,
    TradeRecord,
    utcnow,
)
from app.data.candle_store import aggregate
from app.data.historical_loader import HistoricalLoader, ParquetCache
from app.execution.broker import Broker, BrokerError
from app.execution.execution_monitor import ExecutionMonitor
from app.execution.kill_switch import KillSwitch
from app.execution.order_manager import OrderManager
from app.execution.order_planner import OrderPlan, OrderPlanner, PlanRejection
from app.execution.reconciliation import Reconciler
from app.monitoring.alerts import AlertManager
from app.monitoring.health import HealthMonitor
from app.monitoring.metrics import MetricsRegistry
from app.monitoring.watchdog import Watchdog
from app.risk.circuit_breaker import CircuitBreakerManager
from app.state.database import Database
from app.state.state_manager import StateManager

logger = logging.getLogger(__name__)

# Order statuses meaning "the entry will never fill".
_DEAD_ENTRY_STATUSES = {"REJECTED", "CANCELLED"}


@dataclass
class LiveContext:
    """Everything the loop needs, built once and shared."""

    config: BotConfig
    stack: StrategyStack
    database: Database
    state: StateManager
    broker: Broker
    market_data: MarketDataPort
    loader: HistoricalLoader
    order_manager: OrderManager
    order_planner: OrderPlanner
    monitor: ExecutionMonitor
    reconciler: Reconciler
    breakers: CircuitBreakerManager
    account_sync: AccountSync
    alerts: AlertManager
    health: HealthMonitor
    watchdog: Watchdog
    metrics: MetricsRegistry
    kill_switch: KillSwitch
    websocket: Optional[WebsocketManager] = None


def build_live_context(
    config: BotConfig | None = None,
    broker: Broker | None = None,
    market_data: MarketDataPort | None = None,
    websocket: Optional[WebsocketManager] = None,
) -> LiveContext:
    """Assemble the live context. Broker/market-data can be injected for tests."""
    cfg = config or load_config()
    settings = cfg.settings

    # Database + state (single source of truth, restart-safe).
    database = Database(settings.paths.db_path)
    state = StateManager(database)

    # Decision stack (identical to the backtest).
    stack = build_stack(cfg)

    # Broker adapters (lazy; no credentials needed until a call is made).
    client = AlpacaClient(
        settings.alpaca_api_key,
        settings.alpaca_secret_key,
        settings.alpaca_base_url,
        settings.alpaca_data_url,
        retry_attempts=settings.retry_attempts,
        retry_base_delay=settings.retry_base_delay_seconds,
        retry_max_delay=settings.retry_max_delay_seconds,
    )
    alpaca_md = AlpacaMarketData(client)
    if market_data is None:
        market_data = MarketDataPort(alpaca_md)
    if broker is None:
        broker = AlpacaTradingClient(client, quotes=market_data)
    if websocket is None and settings.alpaca_api_key and settings.alpaca_secret_key:
        websocket = WebsocketManager(
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
            list(cfg.symbols.enabled()),
            timeframes=[cfg.symbols.execution_timeframe()],
        )

    cache = ParquetCache(settings.paths.data)
    loader = HistoricalLoader(market_data, cache=cache)

    order_planner = OrderPlanner(cfg.symbols)
    order_manager = OrderManager(
        broker,
        state,
        order_planner,
        retry_attempts=settings.retry_attempts,
        retry_base_delay=settings.retry_base_delay_seconds,
        retry_max_delay=settings.retry_max_delay_seconds,
    )
    monitor = ExecutionMonitor(cfg.risk)
    reconciler = Reconciler()
    breakers = CircuitBreakerManager(cfg.risk)
    alerts = AlertManager(
        telegram_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
    )
    account_sync = AccountSync(
        broker, state, breakers=breakers, interval_seconds=settings.account_sync_interval_seconds
    )
    health = HealthMonitor(settings, state, breakers=breakers, alerts=alerts)
    watchdog = Watchdog(state, alerts=alerts)
    metrics = MetricsRegistry()
    kill_switch = KillSwitch(settings.paths.kill_switch, liquidate_on_engage=False)

    return LiveContext(
        config=cfg,
        stack=stack,
        database=database,
        state=state,
        broker=broker,
        market_data=market_data,
        loader=loader,
        order_manager=order_manager,
        order_planner=order_planner,
        monitor=monitor,
        reconciler=reconciler,
        breakers=breakers,
        account_sync=account_sync,
        alerts=alerts,
        health=health,
        watchdog=watchdog,
        metrics=metrics,
        kill_switch=kill_switch,
        websocket=websocket,
    )


@dataclass
class _EntryInfo:
    """Tracks an entry order from submission until it is filled or abandoned."""

    client_order_id: str
    qty: float
    entry_price: float
    stop_price: float
    target_price: float
    strategy: str
    entry_time: datetime
    regime: Optional[Regime]
    expected_edge: float
    estimated_cost: float
    risk_amount: float = 0.0
    feature_snapshot: object = None
    filled: bool = False


@dataclass
class TradingLoop:
    """The main live loop. Call ``run()`` or ``run_once()``."""

    ctx: LiveContext
    poll_seconds: float = 15.0
    reconcile_seconds: float = 300.0
    _stop: bool = field(default=False, init=False)
    _entry_info: dict[str, _EntryInfo] = field(default_factory=dict, init=False)
    _last_eval_bar: dict[str, datetime] = field(default_factory=dict, init=False)
    _last_reconcile: Optional[datetime] = field(default=None, init=False)

    # -- properties ------------------------------------------------------
    @property
    def mode(self) -> TradingMode:
        return self.ctx.config.settings.mode

    @property
    def shadow(self) -> bool:
        return self.mode is TradingMode.SHADOW

    @property
    def symbols(self) -> list[str]:
        return self.ctx.config.symbols.enabled()

    @property
    def exec_tf(self):
        return self.ctx.stack.context.execution

    # -- lifecycle -------------------------------------------------------
    def request_stop(self, *_args) -> None:  # signal handler
        logger.warning("stop requested; finishing current cycle")
        self._stop = True

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError, AttributeError):  # not on main thread / unsupported
                pass

    def bootstrap(self) -> None:
        """Warm history, reconcile with the broker, and set initial state."""
        ctx = self.ctx
        exec_tf = self.exec_tf
        ctx.state.set_trading_state(TradingState.SYNCING)
        ctx.watchdog.expect("loop", budget_seconds=self.poll_seconds * 4, critical=True)
        ctx.watchdog.expect(
            "market_data", budget_seconds=ctx.config.settings.stale_data_threshold_seconds * 2
        )
        ctx.watchdog.expect(
            "account_sync",
            budget_seconds=ctx.config.settings.account_sync_interval_seconds * 4,
            critical=True,
        )

        start = utcnow() - timedelta(days=180)
        for symbol in self.symbols:
            try:
                report = ctx.loader.update(symbol, exec_tf, ctx.stack.candles, start=start)
                logger.info(
                    "warmup %s %s: fetched=%d stored=%d valid=%s",
                    symbol,
                    exec_tf,
                    report.fetched,
                    report.stored,
                    report.valid,
                )
            except Exception as exc:  # noqa: BLE001 - never hard-fail on data warmup
                logger.error("warmup failed for %s: %s", symbol, exc)
            self._rebuild_higher_timeframes(symbol)

        # Broker truth over local belief.
        try:
            recon = ctx.reconciler.reconcile(ctx.broker, ctx.state, auto_repair=True)
            if not recon.clean:
                ctx.alerts.warning("reconcile", f"startup issues: {recon.as_dict()['issues']}")
        except Exception as exc:  # noqa: BLE001
            logger.error("startup reconcile failed: %s", exc)

        ctx.state.set_trading_state(TradingState.READY)

    def run(self, max_iterations: int | None = None) -> None:
        ctx = self.ctx
        ctx.state.set_trading_state(TradingState.STARTING)
        if self.mode is TradingMode.LIVE and not ctx.config.settings.live_start_allowed():
            raise RuntimeError("LIVE mode requires the explicit double-lock (LIVE_TRADING_CONFIRMED)")
        ctx.config.settings.validate_or_raise()

        self._install_signal_handlers()
        self.bootstrap()

        # Break the circuit-breaker day at the current equity on start.
        if ctx.state.equity > 0:
            ctx.breakers.start_day(ctx.state.equity)

        iteration = 0
        ctx.alerts.info("startup", f"loop starting in {self.mode} mode")
        while not self._stop:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - the loop must survive a bad cycle
                logger.exception("cycle error: %s", exc)
                ctx.alerts.critical("loop", f"cycle error: {exc}")
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            # Sleep in small slices so a stop request is honoured promptly.
            slept = 0.0
            while slept < self.poll_seconds and not self._stop:
                time.sleep(min(1.0, self.poll_seconds - slept))
                slept += 1.0

        ctx.state.set_trading_state(TradingState.STOPPED)
        ctx.alerts.info("shutdown", "loop stopped")

    # -- single cycle ----------------------------------------------------
    def run_once(self) -> None:
        ctx = self.ctx
        now = utcnow()

        # 1) Kill switch (fail-closed) gates all new entries.
        killed = ctx.kill_switch.engaged()
        if killed:
            ctx.state.set_trading_state(TradingState.EMERGENCY)

        # 2) Account synchronisation.
        if ctx.account_sync.due(now):
            result = ctx.account_sync.sync(update_positions=True)
            if result.ok:
                ctx.watchdog.feed("account_sync", f"equity={result.equity:.2f}")
            else:
                ctx.alerts.warning("account_sync", result.error)

        # 3) Refresh market data (websocket first, REST authoritative).
        data_fresh = self._refresh_data()

        # 4) Health evaluation -> trading state.
        ws_available = getattr(ctx.websocket, "available", None) if ctx.websocket else None
        ws_stale = getattr(ctx.websocket, "stale", None) if ctx.websocket else None
        report = ctx.health.check(
            websocket_available=ws_available,
            websocket_stale=ws_stale,
            data_fresh=data_fresh,
        )
        healthy = ctx.health.can_trade(report)

        blocked, why = ctx.breakers.entries_blocked()
        if killed:
            ctx.state.set_trading_state(TradingState.EMERGENCY)
        elif blocked:
            ctx.state.set_trading_state(TradingState.RISK_LOCK)
        elif not healthy:
            ctx.state.set_trading_state(
                TradingState.EMERGENCY if report.status == "EMERGENCY" else TradingState.DEGRADED
            )
        else:
            ctx.state.set_trading_state(TradingState.TRADING)

        # 5) Periodic reconciliation against broker truth.
        if self._last_reconcile is None or (now - self._last_reconcile) >= timedelta(
            seconds=self.reconcile_seconds
        ):
            try:
                recon = ctx.reconciler.reconcile(ctx.broker, ctx.state, auto_repair=True)
                self._last_reconcile = now
                if recon.critical:
                    ctx.state.set_trading_state(TradingState.EMERGENCY)
                    ctx.alerts.critical("reconcile", str(recon.as_dict()["issues"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("reconcile failed: %s", exc)

        # 6) Update pending entry orders; materialise fills; detect closures.
        self._sync_entries()
        self._detect_closed_positions()

        # 7) Manage existing positions (stops/exits) then consider entries.
        for symbol in self.symbols:
            self._manage_position(symbol, now)

        can_enter = (
            ctx.state.trading_state is TradingState.TRADING
            and healthy
            and not blocked
            and not killed
        )
        if can_enter:
            for symbol in self.symbols:
                self._consider_entry(symbol, now)

        # 8) Heartbeat.
        ctx.watchdog.feed("loop", f"state={ctx.state.trading_state}")

    # -- data ------------------------------------------------------------
    def _rebuild_higher_timeframes(self, symbol: str) -> None:
        exec_tf = self.exec_tf
        bars = self.ctx.stack.candles.get(symbol, exec_tf)
        if not bars:
            return
        for role in ("confirmation", "regime", "macro"):
            tf = self.ctx.stack.context.by_role(role)
            if tf is None or tf == exec_tf:
                continue
            agg = aggregate(bars, tf)
            if agg:
                self.ctx.stack.candles.merge(agg)
        self.ctx.stack.features.invalidate(symbol)

    def _refresh_data(self) -> dict[str, bool]:
        ctx = self.ctx
        exec_tf = self.exec_tf
        settings = ctx.config.settings
        fresh: dict[str, bool] = {}
        for symbol in self.symbols:
            try:
                ctx.loader.update(symbol, exec_tf, ctx.stack.candles)
            except Exception as exc:  # noqa: BLE001
                logger.warning("data update failed for %s: %s", symbol, exc)
            self._rebuild_higher_timeframes(symbol)
            fresh[symbol] = ctx.stack.candles.is_fresh(
                symbol, exec_tf, settings.stale_data_threshold_seconds
            )
        if any(fresh.values()):
            ctx.watchdog.feed("market_data", f"fresh={sum(fresh.values())}/{len(fresh)}")
        return fresh

    def _quote(self, symbol: str) -> Optional[Quote]:
        if self.ctx.websocket is not None:
            q = self.ctx.websocket.latest_quote(symbol)
            if q is not None:
                return q
        try:
            return self.ctx.market_data.get_quote(symbol)
        except Exception:  # noqa: BLE001
            return None

    def _price(self, symbol: str) -> Optional[float]:
        q = self._quote(symbol)
        if q is not None and q.mid > 0:
            return q.mid
        try:
            price = self.ctx.market_data.get_last_price(symbol)
            if price:
                return price
        except Exception:  # noqa: BLE001
            pass
        bar = self.ctx.stack.candles.last(symbol, self.exec_tf)
        return bar.close if bar else None

    # -- positions -------------------------------------------------------
    def _sync_entries(self) -> None:
        """Poll pending entry orders and materialise fills into the position book."""
        ctx = self.ctx
        for symbol, info in list(self._entry_info.items()):
            try:
                order = ctx.order_manager.sync_order(info.client_order_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("entry order sync failed for %s: %s", symbol, exc)
                order = ctx.state.orders.get(info.client_order_id)
            if order is None:
                continue

            if not info.filled and order.filled_qty > 0:
                info.filled = True
                info.qty = order.filled_qty
                if order.average_fill_price:
                    info.entry_price = order.average_fill_price
                self._materialise_position(symbol, info)
                ctx.alerts.info("fill", f"{symbol} {info.strategy} qty={info.qty:.8f}")
            elif not info.filled and str(order.status) in _DEAD_ENTRY_STATUSES:
                logger.info("entry for %s did not fill (%s); dropping", symbol, order.status)
                self._entry_info.pop(symbol, None)
                self._last_eval_bar.pop(symbol, None)

    def _materialise_position(self, symbol: str, info: _EntryInfo) -> None:
        ctx = self.ctx
        position = PositionRecord(
            symbol=symbol,
            qty=info.qty,
            average_entry=info.entry_price,
            current_price=info.entry_price,
            strategy=info.strategy,
            entry_time=info.entry_time,
            stop_price=info.stop_price,
            target_price=info.target_price,
            initial_stop_price=info.stop_price,
            initial_risk_per_unit=max(1e-12, info.entry_price - info.stop_price),
            risk_amount=info.risk_amount,
            feature_snapshot=info.feature_snapshot,  # type: ignore[arg-type]
            regime=info.regime,
            state=PositionState.OPEN,
        )
        ctx.state.upsert_position(position)

    def _manage_position(self, symbol: str, now: datetime) -> None:
        ctx = self.ctx
        position = ctx.state.positions.get(symbol)
        if position is None or position.qty == 0:
            return
        price = self._price(symbol)
        if price is None:
            return

        # Latest completed execution-timeframe state (for invalidation checks).
        state_result = None
        bars = ctx.stack.candles.get(symbol, self.exec_tf)
        if bars:
            idx = len(bars) - 1
            ts = bars[idx].timestamp
            btc = ctx.stack.candles.get(ctx.stack.reference_symbol, self.exec_tf)
            btc_index = len(btc) - 1 if btc else None
            try:
                pipe = ctx.stack.pipeline.evaluate(
                    symbol=symbol,
                    timestamp=ts,
                    exec_index=idx,
                    equity=ctx.state.equity,
                    positions=ctx.state.positions,
                    btc_index=btc_index,
                )
                state_result = pipe.state
            except Exception as exc:  # noqa: BLE001
                logger.debug("state build for %s failed: %s", symbol, exc)

        atr = state_result.features.values.get("atr") if state_result is not None else None

        update = ctx.monitor.update_position(
            position=position,
            current_price=price,
            atr=atr,
            state=state_result,
            trading_state=ctx.state.trading_state,
            kill_switch=ctx.kill_switch.engaged(),
            side="buy",
        )
        ctx.state.upsert_position(update.position)

        instruction = update.exit_instruction
        if instruction is None:
            return
        if self.shadow:
            logger.info("[SHADOW] would exit %s (%s)", symbol, instruction.reason)
            return
        try:
            ctx.order_manager.execute_exit(
                instruction.symbol,
                instruction.qty,
                strategy=position.strategy,
                intent="exit",
                now=now,
            )
            ctx.alerts.info("exit", f"{instruction.symbol} {instruction.reason}")
        except BrokerError as exc:
            ctx.alerts.critical("exit", f"{instruction.symbol} exit failed: {exc}")

    def _consider_entry(self, symbol: str, now: datetime) -> None:
        ctx = self.ctx
        if symbol in ctx.state.positions or symbol in self._entry_info:
            return
        bars = ctx.stack.candles.get(symbol, self.exec_tf)
        if not bars:
            return
        idx = len(bars) - 1
        ts = bars[idx].timestamp
        if self._last_eval_bar.get(symbol) == ts:
            return  # already evaluated this completed bar

        # Warmup requirement: core features must be computable.
        if not ctx.stack.features.is_ready(symbol, self.exec_tf, minimum=200):
            return

        btc = ctx.stack.candles.get(ctx.stack.reference_symbol, self.exec_tf)
        btc_index = len(btc) - 1 if btc else None
        try:
            pipe = ctx.stack.pipeline.evaluate(
                symbol=symbol,
                timestamp=ts,
                exec_index=idx,
                equity=ctx.state.equity,
                positions=ctx.state.positions,
                buying_power=ctx.state.account.buying_power if ctx.state.account else None,
                btc_index=btc_index,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("evaluation failed for %s: %s", symbol, exc)
            return

        self._last_eval_bar[symbol] = ts
        self._record_decision(symbol, ts, pipe)

        outcome = pipe.outcome
        if outcome.decision is None:
            return

        plan = ctx.order_planner.plan(outcome.decision, now=now)
        if isinstance(plan, PlanRejection):
            logger.info("plan rejected %s: %s (%s)", symbol, plan.reason, plan.detail)
            return

        if self.shadow:
            logger.info(
                "[SHADOW] would enter %s %s qty=%.8f stop=%.4f target=%.4f",
                symbol,
                plan.strategy,
                plan.entry.qty,
                plan.stop_price,
                plan.target_price,
            )
            return

        try:
            ctx.order_manager.execute_entry(plan)
        except BrokerError as exc:
            ctx.alerts.critical("entry", f"{symbol} entry failed: {exc}")
            return

        # Track the entry; the position book is only updated once filled.
        decision = outcome.decision
        ref_price = None
        if decision.feature_snapshot is not None:
            ref_price = decision.feature_snapshot.values.get("close")
        ref_price = ref_price or plan.metadata.get("reference_price") or plan.stop_price
        self._entry_info[symbol] = _EntryInfo(
            client_order_id=plan.entry.client_order_id,
            qty=plan.entry.qty,
            entry_price=ref_price,
            stop_price=plan.stop_price,
            target_price=plan.target_price,
            strategy=plan.strategy,
            entry_time=now,
            regime=decision.regime,
            expected_edge=decision.expected_edge,
            estimated_cost=decision.estimated_cost,
            risk_amount=decision.risk_amount,
            feature_snapshot=decision.feature_snapshot,
        )

    def _detect_closed_positions(self) -> None:
        """Filled positions the broker has since closed are recorded as trades."""
        ctx = self.ctx
        for symbol, info in list(self._entry_info.items()):
            if info.filled and symbol not in ctx.state.positions:
                self._entry_info.pop(symbol, None)
                self._last_eval_bar.pop(symbol, None)
                self._record_trade(symbol, info, ctx)

    def _record_trade(self, symbol: str, info: _EntryInfo, ctx: LiveContext) -> None:
        exit_price = self._price(symbol) or info.entry_price
        fees_rate = ctx.config.settings.fees.taker_fee
        gross = (exit_price - info.entry_price) * info.qty
        fees = (info.entry_price + exit_price) * info.qty * fees_rate
        net = gross - fees
        now = utcnow()
        trade = TradeRecord(
            trade_id=f"{symbol}-{int(info.entry_time.timestamp())}",
            symbol=symbol,
            strategy=info.strategy,
            entry_price=info.entry_price,
            exit_price=exit_price,
            qty=info.qty,
            fees=fees,
            slippage=0.0,
            gross_pnl=gross,
            net_pnl=net,
            entry_time=info.entry_time,
            exit_time=now,
            holding_time_seconds=max(0.0, (now - info.entry_time).total_seconds()),
            exit_reason=ExitReason.RECONCILED,
            regime=info.regime or Regime.UNKNOWN,
            signal_score=0.0,
            expected_edge=info.expected_edge,
            estimated_cost=info.estimated_cost,
            actual_cost=fees,
            feature_snapshot=info.feature_snapshot,  # type: ignore[arg-type]
        )
        try:
            ctx.state.record_trade(trade)
            ctx.breakers.record_trade_result(net)
            ctx.metrics.incr("trades.closed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to record trade for %s: %s", symbol, exc)

    def _record_decision(self, symbol: str, ts: datetime, pipe) -> None:
        try:
            outcome = pipe.outcome
            if outcome.decision is not None:
                d = outcome.decision
                self.ctx.state.record_signal(
                    {
                        "symbol": symbol,
                        "timestamp": ts.isoformat(),
                        "regime": str(pipe.state.regime),
                        "strategy": d.strategy,
                        "decision": "ACCEPTED",
                        "score": d.score,
                        "confidence": d.risk_multiplier,
                        "expected_edge": d.expected_edge,
                        "estimated_cost": d.estimated_cost,
                        "reject_reason": None,
                        "detail": d.reason,
                        "feature_snapshot": d.feature_snapshot.as_dict()
                        if d.feature_snapshot
                        else None,
                    }
                )
            elif outcome.rejection is not None:
                r = outcome.rejection
                self.ctx.state.record_signal(
                    {
                        "symbol": symbol,
                        "timestamp": ts.isoformat(),
                        "regime": str(r.regime),
                        "strategy": r.strategy,
                        "decision": "REJECTED",
                        "score": r.score,
                        "confidence": 1.0,
                        "expected_edge": r.expected_edge,
                        "estimated_cost": None,
                        "reject_reason": str(r.reason),
                        "detail": r.detail,
                        "feature_snapshot": r.feature_snapshot.as_dict()
                        if r.feature_snapshot
                        else None,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("record_decision failed: %s", exc)
