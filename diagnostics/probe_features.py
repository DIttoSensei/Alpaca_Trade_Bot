"""Probe: dump real feature values + regime contributions at a sample bar."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.assembly import build_stack  # noqa: E402
from app.config import load_config  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.candle_store import CandleStore  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402

cfg = load_config()
tf = Timeframe.H1
cache = ParquetCache(cfg.settings.paths.data)
bars = sorted(cache.load("BTC/USD", tf), key=lambda c: c.timestamp)
print("bars:", len(bars))

store = CandleStore(max_bars=10**9)
stack = build_stack(cfg, candles=store)
store.merge(bars)
stack.features.invalidate()

i = len(bars) // 2
snap = stack.features.snapshot_at("BTC/USD", tf, i, bars[i].timestamp)
v = snap.values
print("snapshot keys:", len(v))
for k in ["close", "sma200", "ema20", "ema50", "ema200", "adx", "di_plus", "di_minus",
          "rsi", "macd_hist", "return_6", "atr_pct", "vol_percentile", "slope20"]:
    print(f"  {k:<14} = {v.get(k)}")

state, rr = stack.pipeline.build_state("BTC/USD", bars[i].timestamp, i)
print("regime:", rr.regime, "trend:", rr.trend, "vol:", rr.volatility, "trend_score:", rr.trend_score)
print("details:", rr.details)
print("state features count:", len(state.features.values))
