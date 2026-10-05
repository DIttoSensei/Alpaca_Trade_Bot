"""In-memory (and optionally parquet-backed) candle storage.

Holds OHLCV series keyed by (symbol, timeframe) and supports:
- append/merge with dedupe on (symbol, timeframe, timestamp)
- multi-timeframe aggregation from a base timeframe
- freshness tracking (last candle close age)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from app.core.enums import Timeframe
from app.core.types import Candle, ensure_utc, utcnow
from app.data.data_validator import dedupe


class CandleStore:
    """Thread-safe store for candle series."""

    def __init__(self, max_bars: int = 5000) -> None:
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], list[Candle]] = defaultdict(list)
        self._max_bars = max_bars

    def set_capacity(self, max_bars: int) -> None:
        """Raise the retention cap (never lowers it).

        The backtest engine addresses candles by ABSOLUTE index into its own
        copy of the series; the store must therefore retain the identical
        series. ``merge`` silently drops the oldest bars past ``max_bars``,
        which would misalign every ``snapshot_at(index)`` read. The engine calls
        this once before loading so the whole history is kept. Live keeps its
        bounded default (only the recent tail is needed there).
        """
        with self._lock:
            if max_bars > self._max_bars:
                self._max_bars = max_bars

    # -- write -----------------------------------------------------------
    def merge(self, candles: Iterable[Candle], update_freshness: bool = True) -> int:
        """Merge candles with dedupe. Returns number of stored bars for the key."""
        grouped: dict[tuple[str, str], list[Candle]] = defaultdict(list)
        for c in candles:
            grouped[(c.symbol, str(c.timeframe))].append(c)
        with self._lock:
            for key, group in grouped.items():
                existing = self._data[key]
                combined = dedupe([*existing, *group])
                if len(combined) > self._max_bars:
                    combined = combined[-self._max_bars :]
                self._data[key] = combined
        return 0

    def append(self, candle: Candle) -> None:
        self.merge([candle])

    def peek(self, symbol: str, timeframe: Timeframe) -> Sequence[Candle]:
        """Read-only view of the stored series WITHOUT copying.

        ``get`` returns a fresh list (an O(n) copy); ``peek`` returns the live
        internal list. Callers must treat it as read-only. Used on hot read paths
        (e.g. the backtest feature loop) so per-bar reads stay O(1) and the whole
        run remains O(n).
        """
        with self._lock:
            return self._data.get((symbol, str(timeframe)), [])

    # -- read ------------------------------------------------------------
    def get(self, symbol: str, timeframe: Timeframe) -> list[Candle]:
        with self._lock:
            return list(self._data.get((symbol, str(timeframe)), []))

    def last(self, symbol: str, timeframe: Timeframe) -> Candle | None:
        bars = self.get(symbol, timeframe)
        return bars[-1] if bars else None

    def count(self, symbol: str, timeframe: Timeframe) -> int:
        with self._lock:
            return len(self._data.get((symbol, str(timeframe)), []))

    def latest_timestamp(self, symbol: str, timeframe: Timeframe) -> datetime | None:
        bar = self.last(symbol, timeframe)
        return bar.timestamp if bar else None

    def has_minimum(self, symbol: str, timeframe: Timeframe, minimum: int) -> bool:
        return self.count(symbol, timeframe) >= minimum

    # -- freshness -------------------------------------------------------
    def age_seconds(self, symbol: str, timeframe: Timeframe, now: datetime | None = None) -> float | None:
        """Age, in seconds, of the last candle's CLOSE relative to now."""
        bar = self.last(symbol, timeframe)
        if bar is None:
            return None
        now = ensure_utc(now or utcnow())
        close_time = ensure_utc(bar.timestamp) + timedelta(seconds=timeframe.seconds)
        return (now - close_time).total_seconds()

    def is_fresh(
        self,
        symbol: str,
        timeframe: Timeframe,
        threshold_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        age = self.age_seconds(symbol, timeframe, now)
        if age is None:
            return False
        return age <= threshold_seconds

    def symbols(self) -> list[str]:
        with self._lock:
            return sorted({sym for sym, _ in self._data.keys()})

    def keys(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._data.keys())


def aggregate(candles: Sequence[Candle], target: Timeframe) -> list[Candle]:
    """Aggregate a lower-timeframe series into a higher timeframe.

    Buckets candles by their aligned open time; only emits COMPLETE buckets
    (a bucket is complete when it has accumulated >= its expected number of
    source bars). This guarantees no partial/forming higher-timeframe candle is
    ever exposed to strategy code (spec section 61).
    """
    if not candles:
        return []
    src_tf = candles[0].timeframe
    if src_tf == target:
        return list(candles)
    ratio = target.seconds // src_tf.seconds
    if ratio <= 0:
        return list(candles)

    buckets: dict[datetime, list[Candle]] = defaultdict(list)
    for c in candles:
        ts = ensure_utc(c.timestamp)
        epoch = int(ts.timestamp())
        bucket_start = epoch - (epoch % target.seconds)
        buckets[datetime.fromtimestamp(bucket_start, tz=ts.tzinfo)].append(c)

    out: list[Candle] = []
    for start in sorted(buckets.keys()):
        group = sorted(buckets[start], key=lambda x: x.timestamp)
        if len(group) < ratio:
            # incomplete highest bucket -> skip
            continue
        group = group[:ratio]
        out.append(
            Candle(
                timestamp=start,
                symbol=group[0].symbol,
                timeframe=target,
                open=group[0].open,
                high=max(g.high for g in group),
                low=min(g.low for g in group),
                close=group[-1].close,
                volume=sum(g.volume for g in group),
            )
        )
    return out
