"""Historical data loading, caching and incremental update.

Handles:
- pagination (Alpaca returns pagination tokens; keep requesting until exhausted)
- parquet cache persistence (don't re-download the whole history every backtest)
- incremental update: find latest local timestamp, request only missing data,
  validate, append, deduplicate (spec sections 92-94)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

from app.core.enums import Timeframe
from app.core.types import Candle, ensure_utc, utcnow
from app.data.candle_store import CandleStore
from app.data.data_validator import dedupe, validate_series

logger = logging.getLogger(__name__)


class HistoricalProvider(Protocol):
    """Anything that can fetch crypto bars (implemented by the Alpaca adapter)."""

    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[Candle]:
        """Return all candles in [start, end], handling pagination internally."""
        ...


class ParquetCache:
    """Simple on-disk cache of candle series as parquet files.

    Layout: data/<SYMBOL>/<timeframe>.parquet
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: Timeframe) -> Path:
        safe = symbol.replace("/", "").replace("-", "")
        return self.root / safe / f"{timeframe}.parquet"

    def save(self, symbol: str, timeframe: Timeframe, candles: Sequence[Candle]) -> Path:
        import pandas as pd

        path = self._path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "timestamp": ensure_utc(c.timestamp),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        )
        frame.to_parquet(path, index=False)
        return path

    def load(self, symbol: str, timeframe: Timeframe) -> list[Candle]:
        import pandas as pd

        path = self._path(symbol, timeframe)
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        out: list[Candle] = []
        for row in frame.itertuples(index=False):
            ts = row.timestamp
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            out.append(
                Candle(
                    timestamp=ensure_utc(ts),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                )
            )
        return out

    def latest_timestamp(self, symbol: str, timeframe: Timeframe) -> Optional[datetime]:
        bars = self.load(symbol, timeframe)
        return bars[-1].timestamp if bars else None


@dataclass(slots=True)
class LoadReport:
    symbol: str
    timeframe: Timeframe
    fetched: int
    stored: int
    from_ts: Optional[datetime]
    to_ts: Optional[datetime]
    valid: bool
    issues: list[str]


class HistoricalLoader:
    def __init__(self, provider: HistoricalProvider, cache: Optional[ParquetCache] = None) -> None:
        self.provider = provider
        self.cache = cache

    def load_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[Candle]:
        end = end or utcnow()
        candles = self.provider.fetch_bars(symbol, timeframe, start, end)
        candles = dedupe(candles)
        result = validate_series(candles, timeframe)
        if not result.ok:
            logger.warning("historical validation issues for %s %s: %s", symbol, timeframe, result.details[:5])
        if self.cache is not None and candles:
            self.cache.save(symbol, timeframe, candles)
        return candles

    def update(
        self,
        symbol: str,
        timeframe: Timeframe,
        store: CandleStore,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        overlap_bars: int = 3,
    ) -> LoadReport:
        """Incrementally update the store with only missing data.

        Overlaps by a few bars to catch late/revised candles, then dedupes.
        """
        end = end or utcnow()
        cached = self.cache.load(symbol, timeframe) if self.cache is not None else []
        last_ts = None
        if cached:
            last_ts = cached[-1].timestamp
        existing_last = store.latest_timestamp(symbol, timeframe)
        if existing_last and (last_ts is None or existing_last > last_ts):
            last_ts = existing_last

        fetch_start = (
            last_ts - timedelta(seconds=timeframe.seconds * overlap_bars)
            if last_ts is not None
            else (start or (end - timedelta(days=365)))
        )
        fetched = self.provider.fetch_bars(symbol, timeframe, fetch_start, end)
        fetched = dedupe(fetched)
        result = validate_series(fetched, timeframe)
        stored_before = store.count(symbol, timeframe)
        if cached and self.cache is not None:
            store.merge(cached)
        store.merge(fetched)
        stored_after = store.count(symbol, timeframe)
        if self.cache is not None:
            self.cache.save(symbol, timeframe, store.get(symbol, timeframe))

        return LoadReport(
            symbol=symbol,
            timeframe=timeframe,
            fetched=len(fetched),
            stored=stored_after - stored_before,
            from_ts=fetched[0].timestamp if fetched else None,
            to_ts=fetched[-1].timestamp if fetched else None,
            valid=result.ok,
            issues=[str(i) for i in result.issues],
        )

    def backfill(
        self,
        symbols: Iterable[str],
        timeframes: Iterable[Timeframe],
        store: CandleStore,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[LoadReport]:
        reports: list[LoadReport] = []
        for symbol in symbols:
            for tf in timeframes:
                try:
                    reports.append(self.update(symbol, tf, store, start=start, end=end))
                except Exception as exc:  # noqa: BLE001
                    logger.error("backfill failed for %s %s: %s", symbol, tf, exc)
        return reports
