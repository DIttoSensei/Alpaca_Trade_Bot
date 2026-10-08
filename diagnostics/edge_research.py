"""Edge research harness v2: search for a genuine long-only crypto edge.

Rationale: on M15 the 0.35%/side cost with high turnover is fatal, so the only
place an edge can survive is LOW-turnover trend-following on H4/D1, where average
moves are many multiples of cost. This harness aggregates the real M15 history to
H4 and D1 and backtests classic, evidence-based long-only edges with a TRAIN/TEST
split.

Execution model (honest): signal evaluated on a CLOSED bar, executed at the NEXT
bar's open; stop/target resolved against the next bar's high/low pessimistically
(stop first); fees+slippage charged on both sides. One position per symbol
(independent sleeve); portfolio return is the product of sleeves.

Outputs diagnostics/edge_research.txt.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.core.enums import Timeframe  # noqa: E402
from app.data.candle_store import aggregate  # noqa: E402
from app.data.historical_loader import ParquetCache  # noqa: E402
from app.indicators.core import ema, sma  # noqa: E402

TAKER = 0.0025
SLIP = 0.0010
COST = TAKER + SLIP  # per side


@dataclass
class Metric:
    name: str
    ret: float
    dd: float
    trades: int
    win: float
    pf: float
    expo: float


def _atr(highs, lows, closes, period=14):
    n = len(closes)
    out = [None] * n
    trs = []
    for i in range(n):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    run = 0.0
    for i in range(n):
        run += trs[i]
        if i >= period:
            run -= trs[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def _roll_max(s, n):
    return [None if i < n else max(s[i - n : i]) for i in range(len(s))]


def _roll_min(s, n):
    return [None if i < n else min(s[i - n : i]) for i in range(len(s))]


def simulate(bars, start, entries, exits, stop_pct=None, trail_pct=None, target_pct=None):
    n = len(bars)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = wins = in_mkt = 0
    gw = gl = 0.0
    pos = None
    for i in range(start, n - 1):
        nxt = bars[i + 1]
        if pos is None:
            if entries[i]:
                entry = nxt.open * (1 + COST)
                pos = {"entry": entry, "peak": entry, "stop": None, "target": None}
                if stop_pct and stop_pct[i] is not None:
                    pos["stop"] = entry * (1 - stop_pct[i])
                if target_pct and target_pct[i] is not None:
                    pos["target"] = entry * (1 + target_pct[i])
                trades += 1
            continue
        in_mkt += 1
        pos["peak"] = max(pos["peak"], nxt.high)
        exit_price = None
        if pos["stop"] is not None and nxt.low <= pos["stop"]:
            exit_price = pos["stop"]
        elif pos["target"] is not None and nxt.high >= pos["target"]:
            exit_price = pos["target"]
        elif trail_pct and trail_pct[i] is not None:
            stop = pos["peak"] * (1 - trail_pct[i])
            if nxt.low <= stop:
                exit_price = stop
        if exit_price is None and exits[i]:
            exit_price = nxt.close
        if exit_price is not None:
            fill = exit_price * (1 - COST)
            r = fill / pos["entry"] - 1.0
            equity *= (1 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            if r >= 0:
                wins += 1
                gw += r
            else:
                gl += -r
            pos = None
    if pos is not None:
        fill = bars[-1].close * (1 - COST)
        r = fill / pos["entry"] - 1.0
        equity *= (1 + r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        if r >= 0:
            wins += 1
            gw += r
        else:
            gl += -r
    return equity, max_dd, trades, wins, gw, gl, in_mkt / max(1, n)


def evaluate(bars_by_sym, strat_fn, test_frac, name):
    eq = 1.0
    dds = []
    tr = wins = 0
    gw = gl = 0.0
    expo = []
    per_sym = []
    for sym, bars in bars_by_sym.items():
        start = int(len(bars) * (1 - test_frac))
        r = strat_fn(bars, start)
        if r is None:
            continue
        e, dd, t, w, g_w, g_l, ex = r
        eq *= e
        dds.append(dd)
        tr += t
        wins += w
        gw += g_w
        gl += g_l
        expo.append(ex)
        per_sym.append(f"{sym.split('/')[0]}={ (e-1)*100:+.1f}%")
    pf = gw / gl if gl > 0 else float("inf")
    return Metric(name, (eq - 1) * 100, max(dds) * 100 if dds else 0,
                  tr, (wins / tr * 100) if tr else 0, pf,
                  sum(expo) / len(expo) * 100 if expo else 0), per_sym


# ---- strategies -------------------------------------------------------------

def buy_hold(bars, start):
    n = len(bars)
    entries = [False] * n
    exits = [False] * n
    entries[start] = True
    exits[n - 2] = True
    return simulate(bars, start, entries, exits)


def tsmom(period):
    def f(bars, start):
        closes = [b.close for b in bars]
        s = sma(closes, period)
        n = len(bars)
        en = [False] * n
        ex = [False] * n
        for i in range(start, n - 1):
            if s[i] is None:
                continue
            above = closes[i] > s[i]
            prev = s[i - 1] is not None and closes[i - 1] > s[i - 1]
            if above and not prev:
                en[i] = True
            if not above and prev:
                ex[i] = True
        return simulate(bars, start, en, ex)
    return f


def donchian(n_en, n_ex):
    def f(bars, start):
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        hh = _roll_max(highs, n_en)
        ll = _roll_min(lows, n_ex)
        n = len(bars)
        en = [False] * n
        ex = [False] * n
        for i in range(start, n - 1):
            if hh[i] is not None and closes[i] > hh[i]:
                en[i] = True
            if ll[i] is not None and closes[i] < ll[i]:
                ex[i] = True
        return simulate(bars, start, en, ex)
    return f


def turtle(n_en=55, atr_mult=2.0, trail_mult=3.0):
    def f(bars, start):
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        hh = _roll_max(highs, n_en)
        a = _atr(highs, lows, closes, 14)
        n = len(bars)
        en = [False] * n
        ex = [False] * n
        stop_pct = [None] * n
        trail = [None] * n
        for i in range(n):
            if a[i] is not None and closes[i] > 0:
                stop_pct[i] = max(0.01, (a[i] * atr_mult) / closes[i])
                trail[i] = (a[i] * trail_mult) / closes[i]
        for i in range(start, n - 1):
            if hh[i] is not None and closes[i] > hh[i]:
                en[i] = True
        return simulate(bars, start, en, ex, stop_pct=stop_pct, trail_pct=trail)
    return f


def ema_cross(fast, slow):
    def f(bars, start):
        closes = [b.close for b in bars]
        ef = ema(closes, fast)
        es = ema(closes, slow)
        n = len(bars)
        en = [False] * n
        ex = [False] * n
        for i in range(start, n - 1):
            if ef[i] is None or es[i] is None:
                continue
            up = ef[i] > es[i]
            prev = ef[i - 1] is not None and es[i - 1] is not None and ef[i - 1] > es[i - 1]
            if up and not prev:
                en[i] = True
            if not up and prev:
                ex[i] = True
        return simulate(bars, start, en, ex)
    return f


def donchian_filtered(n_en, n_ex, sma_period):
    """Classic Turtle-with-filter: long on n_en-bar high ONLY when close>SMA.",
    exit on n_ex-bar low (or when close<SMA)."""
    def f(bars, start):
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        hh = _roll_max(highs, n_en)
        ll = _roll_min(lows, n_ex)
        sf = sma(closes, sma_period)
        n = len(bars)
        en = [False] * n
        ex = [False] * n
        for i in range(start, n - 1):
            if sf[i] is None:
                continue
            filt = closes[i] > sf[i]
            if filt and hh[i] is not None and closes[i] > hh[i]:
                en[i] = True
            if (ll[i] is not None and closes[i] < ll[i]) or not filt:
                ex[i] = True
        return simulate(bars, start, en, ex)
    return f


STRATS = [
    ("buy_hold", buy_hold),
    ("donchian_20_10", donchian(20, 10)),
    ("donchian_55_20", donchian(55, 20)),
    ("donch_filt_55_20", donchian_filtered(55, 20, 200)),
    ("donch_filt_20_10", donchian_filtered(20, 10, 200)),
    ("turtle_55_atr", turtle(55, 2.0, 3.0)),
    ("tsmom_sma100", tsmom(100)),
    ("ema_cross_20_50", ema_cross(20, 50)),
]


def main() -> int:
    cfg = load_config()
    base_tf = cfg.symbols.timeframes.get("execution", Timeframe.M15)
    cache = ParquetCache(cfg.settings.paths.data)
    m15 = {}
    for sym in cfg.symbols.enabled():
        try:
            b = sorted(cache.load(sym, base_tf), key=lambda c: c.timestamp)
        except Exception:  # noqa: BLE001
            continue
        if b:
            m15[sym] = b

    lines = []
    for tf in (Timeframe.H4, Timeframe.D1):
        by_sym = {sym: aggregate(b, tf) for sym, b in m15.items()}
        by_sym = {s: b for s, b in by_sym.items() if len(b) > 260}
        print(f"[edge] tf={tf} bars={ {s: len(b) for s, b in by_sym.items()} }")
        for test_frac, label in ((0.4, "TRAIN 60% / TEST 40% -> reporting TEST (last 40%)"),):
            lines.append("=" * 100)
            lines.append(f"EDGE RESEARCH v2 -- {tf} -- {label}  (cost/side {COST*100:.2f}%, next-open fills)")
            lines.append("=" * 100)
            lines.append(f"{'strategy':<20}{'ret%':>10}{'maxDD%':>9}{'trades':>8}{'win%':>8}{'PF':>8}{'expo%':>8}")
            lines.append("-" * 100)
            for name, fn in STRATS:
                m, per = evaluate(by_sym, fn, test_frac, name)
                pf = "inf" if math.isinf(m.pf) else f"{m.pf:.2f}"
                lines.append(f"{name:<20}{m.ret:>10.1f}{m.dd:>9.1f}{m.trades:>8}{m.win:>8.1f}{pf:>8}{m.expo:>8.1f}")
            lines.append("")

        # Per-year robustness on D1 for the leading candidates.
        if tf is Timeframe.D1:
            lines.append("-" * 100)
            lines.append("PER-YEAR (D1, full history, each calendar year evaluated independently)")
            lines.append(f"{'strategy':<20}{'2024%':>12}{'2025%':>12}{'2026%':>12}")
            lines.append("-" * 100)
            for name, fn in STRATS:
                yr_ret = {}
                for year in (2024, 2025, 2026):
                    eq = 1.0
                    for sym, bars in by_sym.items():
                        yb = [b for b in bars if b.timestamp.year == year]
                        if len(yb) < 60:
                            continue
                        r = fn(yb, 0)
                        if r is not None:
                            eq *= r[0]
                    yr_ret[year] = (eq - 1) * 100
                lines.append(f"{name:<20}{yr_ret[2024]:>11.1f}%{yr_ret[2025]:>11.1f}%{yr_ret[2026]:>11.1f}%")
            lines.append("")

    text = "\n".join(lines)
    (Path(__file__).resolve().parent / "edge_research.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
