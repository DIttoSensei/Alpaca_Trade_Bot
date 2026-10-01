"""Crypto websocket manager.

Streams trades/quotes/bars for the configured symbols and feeds the CandleStores.
Designed to fail safely:

* if the socket is down, ``available`` is False and the loop falls back to REST;
* reconnection uses exponential backoff with a cap;
* staleness is detectable by consumers via ``last_message_at``.

The websocket is an optimisation, never a correctness dependency (spec section 90).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from app.core.enums import Timeframe
from app.core.types import Candle, Quote, ensure_utc, utcnow
from app.data.candle_store import CandleStore

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    connected: bool = False
    reconnects: int = 0
    last_message_at: Optional[datetime] = None
    last_error: str = ""
    subscribed: list[str] = field(default_factory=list)


class WebsocketManager:
    def __init__(
        self,
        api_key: str | None,
        secret_key: str | None,
        symbols: list[str],
        timeframes: Optional[list[Timeframe]] = None,
        max_backoff: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = list(symbols)
        self.timeframes = timeframes or [Timeframe.M5, Timeframe.M15]
        self.max_backoff = max_backoff
        self.state = StreamState()
        self._stores: dict[tuple[str, Timeframe], CandleStore] = {}
        self._quotes: dict[str, Quote] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_bar: Optional[Callable[[Candle], None]] = None

    # -- stores / accessors ---------------------------------------------
    def store(self, symbol: str, timeframe: Timeframe) -> CandleStore:
        key = (symbol, timeframe)
        if key not in self._stores:
            self._stores[key] = CandleStore()
        return self._stores[key]

    def latest_quote(self, symbol: str) -> Optional[Quote]:
        return self._quotes.get(symbol)

    @property
    def available(self) -> bool:
        return self.state.connected

    @property
    def stale(self) -> bool:
        if self.state.last_message_at is None:
            return True
        return (utcnow() - self.state.last_message_at).total_seconds() > 180

    def set_bar_callback(self, cb: Callable[[Candle], None]) -> None:
        self._on_bar = cb

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.api_key or not self.secret_key:
            logger.warning("websocket not started: missing credentials")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ws-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread:
            self._thread.join(timeout=5.0)
        self.state.connected = False

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: BLE001
            logger.error("websocket thread crashed: %s", exc)
            self.state.connected = False
            self.state.last_error = str(exc)

    async def _main(self) -> None:
        from alpaca.data.live.crypto import CryptoDataStream

        self._loop = asyncio.get_running_loop()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                stream = CryptoDataStream(self.api_key, self.secret_key)
                stream.subscribe_trades(self._handle_trade, *self.symbols)
                stream.subscribe_quotes(self._handle_quote, *self.symbols)
                self.state.connected = True
                self.state.subscribed = list(self.symbols)
                backoff = 1.0
                logger.info("websocket connected for %s", ", ".join(self.symbols))
                await stream._run_forever()
            except Exception as exc:  # noqa: BLE001
                self.state.connected = False
                self.state.last_error = str(exc)
                self.state.reconnects += 1
                logger.warning("websocket error: %s; reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(self.max_backoff, backoff * 2)
        self.state.connected = False

    # -- handlers --------------------------------------------------------
    def _touch(self) -> None:
        self.state.last_message_at = utcnow()

    async def _handle_trade(self, trade) -> None:
        self._touch()
        symbol = getattr(trade, "symbol", None)
        price = getattr(trade, "price", None)
        if symbol is None or price is None:
            return
        self._quotes.setdefault(
            symbol, Quote(symbol=symbol, bid=float(price), ask=float(price), timestamp=utcnow())
        )
        self._append_tick(symbol, float(price), float(getattr(trade, "size", 0.0) or 0.0))

    async def _handle_quote(self, quote) -> None:
        self._touch()
        symbol = getattr(quote, "symbol", None)
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        if symbol is None or bid is None or ask is None:
            return
        self._quotes[symbol] = Quote(
            symbol=symbol,
            bid=float(bid),
            ask=float(ask),
            bid_size=float(getattr(quote, "bid_size", 0.0) or 0.0),
            ask_size=float(getattr(quote, "ask_size", 0.0) or 0.0),
            timestamp=ensure_utc(getattr(quote, "timestamp", None) or utcnow()),
        )

    def _append_tick(self, symbol: str, price: float, size: float) -> None:
        """Aggregate ticks into the finest configured timeframe store.

        We only build partial bars here as a latency optimisation; authoritative
        candles come from REST/aggregation so higher-timeframe completeness rules
        are never violated.
        """
        for tf in self.timeframes:
            store = self.store(symbol, tf)
            now = utcnow()
            bucket = int(now.timestamp()) // tf.seconds * tf.seconds
            ts = datetime.fromtimestamp(bucket, tz=now.tzinfo)
            last = store.latest_timestamp(symbol, tf)
            if last is not None and ensure_utc(last) == ts:
                candle = store.last(symbol, tf)
                if candle is not None:
                    candle.high = max(candle.high, price)
                    candle.low = min(candle.low, price)
                    candle.close = price
                    candle.volume += size
            else:
                candle = Candle(
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=tf,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                )
                store.append(candle)
                if self._on_bar is not None:
                    try:
                        self._on_bar(candle)
                    except Exception:  # noqa: BLE001
                        logger.exception("bar callback failed")
