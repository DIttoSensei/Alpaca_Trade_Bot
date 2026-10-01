"""Indicator primitives.

Implemented in-house (numpy only) so the bot has no TA-Lib dependency and runs
cheaply on a small VM. Every function accepts plain sequences of floats and
returns a list aligned to the input length, using ``None`` for warm-up periods.

These are pure functions: no hidden state, no lookahead.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

Number = Optional[float]


def _f(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


def sma(values: Sequence[float], period: int) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    if period <= 0 or len(v) < period:
        return out
    running = sum(v[:period])
    out[period - 1] = running / period
    for i in range(period, len(v)):
        running += v[i] - v[i - period]
        out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    if period <= 0 or len(v) < period:
        return out
    k = 2.0 / (period + 1.0)
    # seed with SMA of first `period`
    seed = sum(v[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(v)):
        prev = v[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def rolling_std(values: Sequence[float], period: int) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    if period <= 1 or len(v) < period:
        return out
    window = v[:period]
    mean = sum(window) / period
    var = sum((x - mean) ** 2 for x in window) / period
    out[period - 1] = math.sqrt(var)
    for i in range(period, len(v)):
        window = v[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / period
        out[i] = math.sqrt(var)
    return out


def rolling_sum(values: Sequence[float], period: int) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    if period <= 0 or len(v) < period:
        return out
    running = sum(v[:period])
    out[period - 1] = running
    for i in range(period, len(v)):
        running += v[i] - v[i - period]
        out[i] = running
    return out


def rolling_max(values: Sequence[float], period: int) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    if period <= 0 or len(v) < period:
        return out
    for i in range(period - 1, len(v)):
        out[i] = max(v[i - period + 1 : i + 1])
    return out


def rolling_min(values: Sequence[float], period: int) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    if period <= 0 or len(v) < period:
        return out
    for i in range(period - 1, len(v)):
        out[i] = min(v[i - period + 1 : i + 1])
    return out


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[Number]:
    h, low, c = _f(highs), _f(lows), _f(closes)
    n = len(c)
    out: list[Number] = [None] * n
    for i in range(n):
        if i == 0:
            out[i] = h[i] - low[i]
        else:
            out[i] = max(h[i] - low[i], abs(h[i] - c[i - 1]), abs(low[i] - c[i - 1]))
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[Number]:
    tr = true_range(highs, lows, closes)
    # Wilder's smoothing
    return _wilder_smooth(tr, period)


def _wilder_smooth(values: Sequence[Number], period: int) -> list[Number]:
    out: list[Number] = [None] * len(values)
    if len(values) < period or period <= 0:
        return out
    seed_vals = [v for v in values[:period] if v is not None]
    if len(seed_vals) < period:
        return out
    prev = sum(seed_vals) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        val = values[i]
        if val is None:
            continue
        prev = (prev * (period - 1) + val) / period
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[Number]:
    v = _f(values)
    n = len(v)
    out: list[Number] = [None] * n
    if n <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = v[i] - v[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, n):
        change = v[i] - v[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[Number], list[Number], list[Number]]:
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: list[Number] = [None] * len(values)
    for i in range(len(values)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    # signal = EMA of macd_line (ignoring leading Nones)
    valid = [(i, x) for i, x in enumerate(macd_line) if x is not None]
    signal_line: list[Number] = [None] * len(values)
    if len(valid) >= signal:
        seq = [x for _, x in valid]
        sig = ema(seq, signal)
        for (idx, _), s in zip(valid, sig):
            signal_line[idx] = s
    hist: list[Number] = [None] * len(values)
    for i in range(len(values)):
        if macd_line[i] is not None and signal_line[i] is not None:
            hist[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, hist


def roc(values: Sequence[float], period: int = 10) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    for i in range(period, len(v)):
        if v[i - period] != 0:
            out[i] = (v[i] - v[i - period]) / v[i - period]
    return out


def obv(closes: Sequence[float], volumes: Sequence[float]) -> list[Number]:
    c, vol = _f(closes), _f(volumes)
    out: list[Number] = [None] * len(c)
    if not c:
        return out
    running = 0.0
    out[0] = running
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            running += vol[i]
        elif c[i] < c[i - 1]:
            running -= vol[i]
        out[i] = running
    return out


def adx_di(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> tuple[list[Number], list[Number], list[Number]]:
    """Return (adx, di_plus, di_minus) using Wilder's smoothing."""
    h, low, c = _f(highs), _f(lows), _f(closes)
    n = len(c)
    plus_dm: list[float] = [0.0] * n
    minus_dm: list[float] = [0.0] * n
    tr = true_range(h, low, c)
    for i in range(1, n):
        up_move = h[i] - h[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    smoothed_tr = _wilder_smooth([x if x is not None else 0.0 for x in tr], period)
    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)
    di_plus: list[Number] = [None] * n
    di_minus: list[Number] = [None] * n
    dx: list[Number] = [None] * n
    for i in range(n):
        st = smoothed_tr[i]
        if st and st > 0 and smoothed_plus[i] is not None and smoothed_minus[i] is not None:
            dp = 100.0 * (smoothed_plus[i] / st)
            dm = 100.0 * (smoothed_minus[i] / st)
            di_plus[i] = dp
            di_minus[i] = dm
            denom = dp + dm
            dx[i] = (100.0 * abs(dp - dm) / denom) if denom > 0 else 0.0
    adx = _wilder_smooth([x if x is not None else 0.0 for x in dx], period)
    # align: adx needs period more bars; nulls already introduced by smoothing
    return adx, di_plus, di_minus


def zscore(values: Sequence[float], period: int = 20) -> list[Number]:
    v = _f(values)
    out: list[Number] = [None] * len(v)
    for i in range(period - 1, len(v)):
        window = v[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / period
        sd = math.sqrt(var)
        out[i] = 0.0 if sd == 0 else (v[i] - mean) / sd
    return out


def bollinger(
    values: Sequence[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[list[Number], list[Number], list[Number]]:
    mid = sma(values, period)
    sd = rolling_std(values, period)
    upper: list[Number] = [None] * len(values)
    lower: list[Number] = [None] * len(values)
    for i in range(len(values)):
        if mid[i] is not None and sd[i] is not None:
            upper[i] = mid[i] + num_std * sd[i]
            lower[i] = mid[i] - num_std * sd[i]
    return upper, mid, lower


def percentile_rank(values: Sequence[Optional[float]], lookback: int) -> list[Number]:
    """Rank of the current value within the trailing lookback window [0,1]."""
    v = list(values)
    out: list[Number] = [None] * len(v)
    for i in range(lookback, len(v)):
        window = [x for x in v[i - lookback + 1 : i + 1] if x is not None]
        cur = v[i]
        if cur is None or len(window) < max(5, lookback // 4):
            continue
        below = sum(1 for x in window if x <= cur)
        out[i] = below / len(window)
    return out


def slope(values: Sequence[Optional[float]], period: int) -> list[Number]:
    """Simple linear-regression slope over the trailing window, normalised."""
    v = list(values)
    out: list[Number] = [None] * len(v)
    for i in range(period - 1, len(v)):
        window = v[i - period + 1 : i + 1]
        if any(x is None for x in window):
            continue
        xs = list(range(period))
        mean_x = sum(xs) / period
        mean_y = sum(window) / period  # type: ignore[arg-type]
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, window))  # type: ignore[arg-type]
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0 or mean_y == 0:
            out[i] = 0.0
        else:
            out[i] = (num / den) / abs(mean_y)
    return out


def last_valid(values: Sequence[Number]) -> Optional[float]:
    for x in reversed(values):
        if x is not None:
            return x
    return None


def previous_valid(values: Sequence[Number], offset: int = 1) -> Optional[float]:
    seen = 0
    for x in reversed(values):
        if x is not None:
            seen += 1
            if seen == offset + 1:
                return x
    return None
