"""Inspect the cached parquet candle data: coverage, gaps, price ranges.

Read-only diagnostics. Does not touch the trading stack.
"""

from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402


def main() -> int:
    cfg = load_config()
    cache = ParquetCache(cfg.settings.paths.data)
    print("=" * 78)
    print("CACHED DATA INVENTORY")
    print("=" * 78)
    for symbol in ("BTC/USD", "ETH/USD", "SOL/USD"):
        for tf in (Timeframe.M15, Timeframe.H1):
            try:
                bars = cache.load(symbol, tf)
            except Exception as exc:  # noqa: BLE001
                print(f"{symbol:<9} {tf.value:<4} ERROR {exc}")
                continue
            if not bars:
                print(f"{symbol:<9} {tf.value:<4} (empty / not cached)")
                continue
            first = bars[0].timestamp.astimezone(timezone.utc)
            last = bars[-1].timestamp.astimezone(timezone.utc)
            span_days = (last - first).total_seconds() / 86400.0
            expected = int(span_days * 86400 / tf.seconds) + 1
            dupes = 0
            gaps = 0
            max_gap_bars = 0
            prev = None
            closes = []
            for b in bars:
                if prev is not None:
                    dt = (b.timestamp - prev).total_seconds()
                    if dt == 0:
                        dupes += 1
                    elif dt > tf.seconds:
                        g = int(dt // tf.seconds)
                        gaps += 1
                        max_gap_bars = max(max_gap_bars, g)
                prev = b.timestamp
                closes.append(b.close)
            print(
                f"{symbol:<9} {tf.value:<4} bars={len(bars):<7} "
                f"{first.date()} -> {last.date()}  span={span_days:6.1f}d "
                f"expected~{expected:<7} gaps={gaps} max_gap_bars={max_gap_bars} "
                f"dupes={dupes}"
            )
            print(
                f"{'':<14} close min={min(closes):,.2f} max={max(closes):,.2f} "
                f"first={closes[0]:,.2f} last={closes[-1]:,.2f} "
                f"total_move={(closes[-1]/closes[0]-1)*100:+.1f}%"
            )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
