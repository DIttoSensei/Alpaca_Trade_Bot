"""Alpaca market-data adapter.

Implements the data layer's ``HistoricalProvider`` protocol and provides quotes,
so both the backtest engine and the live loop use identical data semantics. The
adapter converts alpaca-py bar objects into canonical ``Candle`` objects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from app.broker.alpaca_client import AlpacaClient
from app.broker.rate_limiter import data_limiter
from app.core.enums import Timeframe
from app.core.types import Candle, Quote, ensure_utc, utcnow
from app.data.candle_store import CandleStore

logger = logging.getLogger(__name__)

_ALPACA_TIMEFRAME: dict[str, tuple[int, str]] = {
    "1m": (1, "Min"),
    "5m": (5, "Min"),
    "15m": (15, "Min"),
    "1h": (1, "Hour"),
    "4h": (4, "Hour"),
    "1d": (1, "Day"),
}


def _to_alpaca_timeframe(tf: Timeframe) -> Any:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour, "Day": TimeFrameUnit.Day}
    amount, unit = _ALPACA_TIMEFRAME[tf.value]
    return TimeFrame(amount, unit_map[unit])


class AlpacaMarketData:
    def __init__(self, client: AlpacaClient, chunk_days: int = 30) -> None:
        self.client = client
        self.chunk_days = chunk_days

    # -- historical bars -------------------------------------------------
    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[Candle]:
        from alpaca.data.requests import CryptoBarsRequest

        end = ensure_utc(end or utcnow())
        cursor = ensure_utc(start)
        out: dict[datetime, Candle] = {}
        step = timedelta(days=self.chunk_days)
        while cursor < end:
            chunk_end = min(cursor + step, end)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=_to_alpaca_timeframe(timeframe),
                start=cursor,
                end=chunk_end,
            )
            barset = self.client.call(
                self.client.data.get_crypto_bars, request, limiter=data_limiter()
            )
            for bar in self._iter_bars(barset, symbol):
                candle = self._to_candle(bar, symbol, timeframe)
                if candle is not None:
                    out[candle.timestamp] = candle
            cursor = chunk_end
        candles = [out[k] for k in sorted(out)]
        logger.debug("fetched %d %s bars for %s", len(candles), timeframe, symbol)
        return candles

    @staticmethod
    def _iter_bars(barset: Any, symbol: str):
        if barset is None:
            return
        data = getattr(barset, "data", None) or barset
        try:
            bars = data[symbol]
        except (KeyError, TypeError):
            bars = data
        for bar in bars or []:
            yield bar

    @staticmethod
    def _to_candle(bar: Any, symbol: str, timeframe: Timeframe) -> Optional[Candle]:
        ts = getattr(bar, "timestamp", None)
        if ts is None:
            return None
        try:
            return Candle(
                timestamp=ensure_utc(ts),
                symbol=symbol,
                timeframe=timeframe,
                open=float(getattr(bar, "open")),
                high=float(getattr(bar, "high")),
                low=float(getattr(bar, "low")),
                close=float(getattr(bar, "close")),
                volume=float(getattr(bar, "volume", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    # -- quotes ----------------------------------------------------------
    def get_last_price(self, symbol: str) -> Optional[float]:
        from alpaca.data.requests import CryptoLatestTradeRequest

        request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
        try:
            tradeset = self.client.call(
                self.client.data.get_crypto_latest_trade, request, limiter=data_limiter()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("latest trade failed for %s: %s", symbol, exc)
            return None
        trade = self._pick(tradeset, symbol)
        if trade is None:
            return None
        price = getattr(trade, "price", None)
        return float(price) if price is not None else None

    def get_quote(self, symbol: str) -> Optional[Quote]:
        from alpaca.data.requests import CryptoLatestQuoteRequest

        request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        try:
            quoteset = self.client.call(
                self.client.data.get_crypto_latest_quote, request, limiter=data_limiter()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("latest quote failed for %s: %s", symbol, exc)
            return None
        quote = self._pick(quoteset, symbol)
        if quote is None:
            # Fall back to last trade with an assumed tiny spread.
            price = self.get_last_price(symbol)
            if price is None:
                return None
            return Quote(symbol=symbol, bid=price, ask=price, timestamp=utcnow())
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        if bid is None or ask is None:
            return None
        return Quote(
            symbol=symbol,
            bid=float(bid),
            ask=float(ask),
            bid_size=float(getattr(quote, "bid_size", 0.0) or 0.0),
            ask_size=float(getattr(quote, "ask_size", 0.0) or 0.0),
            timestamp=ensure_utc(getattr(quote, "timestamp", None) or utcnow()),
        )

    @staticmethod
    def _pick(payload: Any, symbol: str) -> Any:
        if payload is None:
            return None
        data = getattr(payload, "data", None)
        container = data if data is not None else payload
        if isinstance(container, dict):
            return container.get(symbol)
        return container


class MarketDataPort:
    """Facade used by the live loop and backtest runner."""

    def __init__(self, provider: AlpacaMarketData) -> None:
        self.provider = provider
        self._stores: dict[tuple[str, Timeframe], CandleStore] = {}

    def store(self, symbol: str, timeframe: Timeframe) -> CandleStore:
        key = (symbol, timeframe)
        if key not in self._stores:
            self._stores[key] = CandleStore()
        return self._stores[key]

    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[Candle]:
        return self.provider.fetch_bars(symbol, timeframe, start, end)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self.provider.get_quote(symbol)

    def get_last_price(self, symbol: str) -> Optional[float]:
        return self.provider.get_last_price(symbol)
