"""Feature engine.

Calculates features ONCE and caches them. A new candle triggers an incremental
recompute; we never recompute every indicator from scratch on every tick (spec
section 124).

The FeatureStore is the single producer of ``FeatureSnapshot`` objects, which
are the ONLY feature values strategy code may see. Both live and backtest use
the same code path, preventing divergence (spec section 59).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.core.enums import Timeframe
from app.core.types import Candle, FeatureSnapshot, ensure_utc
from app.data.candle_store import CandleStore
from app.indicators import compute_frame

# Names we expose as a stable, ordered feature vector. Extra names are still
# available but this list guarantees consistency for ML and logging.
CORE_FEATURES = [
    "sma20", "sma50", "sma100", "sma200",
    "ema9", "ema20", "ema50", "ema200",
    "adx", "di_plus", "di_minus", "slope20",
    "rsi", "macd", "macd_signal", "macd_hist", "roc",
    "return_1", "return_3", "return_6", "return_12",
    "atr", "atr_pct", "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "stddev20", "vol_percentile",
    "volume_sma5", "volume_sma20", "volume_sma50",
    "volume_ratio", "obv", "obv_slope", "volume_momentum",
    "zscore", "dist_from_ema20", "dist_from_bb_mid", "return_deviation",
    "recent_high", "recent_low", "range_width", "breakout_distance",
    "swing_high", "swing_low", "range_efficiency", "vwap",
]


@dataclass(slots=True)
class _CacheEntry:
    last_timestamp: Optional[datetime]
    count: int
    frame: dict[str, list]


class FeatureStore:
    def __init__(self, candles: CandleStore) -> None:
        self._candles = candles
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], _CacheEntry] = {}

    # -- frame computation ----------------------------------------------
    def _frame(self, symbol: str, timeframe: Timeframe) -> _CacheEntry:
        key = (symbol, str(timeframe))
        bars = self._candles.get(symbol, timeframe)
        with self._lock:
            entry = self._cache.get(key)
            last_ts = bars[-1].timestamp if bars else None
            if entry is not None and entry.count == len(bars) and entry.last_timestamp == last_ts:
                return entry
            frame = compute_frame(bars) if bars else {}
            entry = _CacheEntry(last_timestamp=last_ts, count=len(bars), frame=frame)
            self._cache[key] = entry
            return entry

    def invalidate(self, symbol: str | None = None) -> None:
        with self._lock:
            if symbol is None:
                self._cache.clear()
            else:
                for key in list(self._cache):
                    if key[0] == symbol:
                        del self._cache[key]

    # -- snapshot builders ----------------------------------------------
    @staticmethod
    def _flat(frame: dict[str, list], index: int) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, series in frame.items():
            if index < 0 or index >= len(series):
                continue
            v = series[index]
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                values[name] = fv
        return values

    def snapshot_at(
        self,
        symbol: str,
        timeframe: Timeframe,
        index: int,
        timestamp: Optional[datetime] = None,
    ) -> FeatureSnapshot:
        """Snapshot for a specific bar index (used by the backtester)."""
        entry = self._frame(symbol, timeframe)
        bars = self._candles.get(symbol, timeframe)
        if bars and 0 <= index < len(bars):
            ts = ensure_utc(bars[index].timestamp)
        else:
            ts = ensure_utc(timestamp) if timestamp else ensure_utc(bars[-1].timestamp)
        values = self._flat(entry.frame, index)
        # Inject the raw bar OHLC so regime/strategy code has the current price
        # without recomputing it. These are point-in-time values, never future.
        if bars and 0 <= index < len(bars):
            bar = bars[index]
            values.setdefault("close", bar.close)
            values.setdefault("open", bar.open)
            values.setdefault("high", bar.high)
            values.setdefault("low", bar.low)
            values.setdefault("volume", bar.volume)
        return FeatureSnapshot(symbol=symbol, timestamp=ts, timeframe=timeframe, values=values)

    def latest_snapshot(self, symbol: str, timeframe: Timeframe) -> FeatureSnapshot:
        """Snapshot for the most recent (completed) bar."""
        bars = self._candles.get(symbol, timeframe)
        index = len(bars) - 1 if bars else -1
        return self.snapshot_at(symbol, timeframe, index)

    def has_minimum(self, symbol: str, timeframe: Timeframe, minimum: int) -> bool:
        return self._candles.has_minimum(symbol, timeframe, minimum)

    def is_ready(self, symbol: str, timeframe: Timeframe, minimum: int = 200) -> bool:
        """Whether enough warm-up bars exist AND core features are present."""
        if not self._candles.has_minimum(symbol, timeframe, minimum):
            return False
        snap = self.latest_snapshot(symbol, timeframe)
        # require at least the long trend features to be computable
        return all(k in snap.values for k in ("sma200", "rsi", "adx", "atr_pct"))

    # -- relative strength (spec section 9 & 24) ------------------------
    def relative_strength(
        self,
        symbol: str,
        reference: str,
        timeframe: Timeframe,
        lookback_bars: int = 4,
    ) -> Optional[float]:
        """asset return over N bars minus reference return over N bars."""
        asset = self._candles.get(symbol, timeframe)
        ref = self._candles.get(reference, timeframe)
        if len(asset) <= lookback_bars or len(ref) <= lookback_bars:
            return None
        a_ret = _ret(asset, lookback_bars)
        r_ret = _ret(ref, lookback_bars)
        if a_ret is None or r_ret is None:
            return None
        return a_ret - r_ret


def _ret(bars: list[Candle], lookback: int) -> Optional[float]:
    if len(bars) <= lookback:
        return None
    prev = bars[-1 - lookback].close
    if prev == 0:
        return None
    return (bars[-1].close - prev) / prev
