"""Candle validation.

Validates candles before they become usable. Critically, it distinguishes BAD
DATA from a REAL EXTREME MARKET EVENT: a genuine crash is unusual but valid and
must not be silently deleted (spec section 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

from app.core.enums import Timeframe
from app.core.types import Candle


class DataIssue(str, Enum):
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    OHLC_INCONSISTENT = "OHLC_INCONSISTENT"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    GAP = "GAP"
    ZERO_VOLUME = "ZERO_VOLUME"
    LATE_CANDLE = "LATE_CANDLE"
    ABNORMAL_SPIKE = "ABNORMAL_SPIKE"


# Issues that make a candle UNUSABLE (bad data).
FATAL_ISSUES = {
    DataIssue.INVALID_TIMESTAMP,
    DataIssue.NON_POSITIVE_PRICE,
    DataIssue.NEGATIVE_VOLUME,
    DataIssue.OHLC_INCONSISTENT,
    DataIssue.DUPLICATE_TIMESTAMP,
}

# Issues that are suspicious but may reflect real markets.
NON_FATAL_ISSUES = {
    DataIssue.ZERO_VOLUME,
    DataIssue.ABNORMAL_SPIKE,
    DataIssue.LATE_CANDLE,
    DataIssue.GAP,
    DataIssue.OUT_OF_ORDER,
}


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: list[DataIssue] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def has_fatal(self) -> bool:
        return any(i in FATAL_ISSUES for i in self.issues)


def validate_candle(
    candle: Candle,
    timeframe: Timeframe,
    now: datetime | None = None,
    expect_complete: bool = True,
) -> ValidationResult:
    """Validate a single candle."""
    issues: list[DataIssue] = []
    details: list[str] = []

    if candle.timestamp is None:
        issues.append(DataIssue.INVALID_TIMESTAMP)
        details.append("timestamp is None")
    elif expect_complete and now is not None:
        # A candle whose close time is in the future is not yet complete.
        close_time = candle.timestamp + timedelta(seconds=timeframe.seconds)
        if close_time > now:
            issues.append(DataIssue.LATE_CANDLE)
            details.append("candle close time is in the future (incomplete)")

    o, h, l, c = candle.open, candle.high, candle.low, candle.close
    if min(o, h, l, c) <= 0:
        issues.append(DataIssue.NON_POSITIVE_PRICE)
        details.append(f"non-positive price ohlc=({o},{h},{l},{c})")

    if candle.volume < 0:
        issues.append(DataIssue.NEGATIVE_VOLUME)
        details.append(f"negative volume {candle.volume}")
    elif candle.volume == 0:
        issues.append(DataIssue.ZERO_VOLUME)
        details.append("zero volume (possible thin market)")

    if not (
        h >= max(o, c)
        and l <= min(o, c)
        and h >= l
    ):
        issues.append(DataIssue.OHLC_INCONSISTENT)
        details.append("OHLC ordering violated")

    ok = not any(i in FATAL_ISSUES for i in issues)
    return ValidationResult(ok=ok, issues=issues, details=details)


def validate_series(
    candles: Sequence[Candle],
    timeframe: Timeframe,
    max_gap_multiplier: float = 3.0,
    spike_threshold: float = 0.35,
) -> ValidationResult:
    """Validate and cross-check an ordered series of candles.

    Detects duplicates, out-of-order entries, gaps and abnormal single-bar
    spikes. A large move is flagged but not treated as fatal.
    """
    issues: list[DataIssue] = []
    details: list[str] = []
    if not candles:
        return ValidationResult(ok=False, issues=[DataIssue.INVALID_TIMESTAMP], details=["empty series"])

    expected_delta = timedelta(seconds=timeframe.seconds)
    seen: set[datetime] = set()
    prev: Candle | None = None

    for i, candle in enumerate(candles):
        res = validate_candle(candle, timeframe)
        for issue in res.issues:
            if issue in FATAL_ISSUES:
                issues.append(issue)
                details.append(f"[{i}] {issue}: {'; '.join(res.details)}")

        key = candle.timestamp
        if key in seen:
            issues.append(DataIssue.DUPLICATE_TIMESTAMP)
            details.append(f"[{i}] duplicate timestamp {key}")
        seen.add(key)

        if prev is not None:
            delta = candle.timestamp - prev.timestamp
            if delta < timedelta(0):
                issues.append(DataIssue.OUT_OF_ORDER)
                details.append(f"[{i}] out of order ({delta})")
            elif delta > expected_delta * max_gap_multiplier:
                issues.append(DataIssue.GAP)
                details.append(f"[{i}] gap of {delta} (> {expected_delta * max_gap_multiplier})")

            # abnormal spike: single-bar move beyond threshold
            if prev.close > 0:
                move = abs(candle.close - prev.close) / prev.close
                if move > spike_threshold:
                    issues.append(DataIssue.ABNORMAL_SPIKE)
                    details.append(f"[{i}] spike {move:.2%} (flag, may be real)")

        prev = candle

    ok = not any(i in FATAL_ISSUES for i in issues)
    return ValidationResult(ok=ok, issues=issues, details=details)


def dedupe(candles: Sequence[Candle]) -> list[Candle]:
    """Merge candles using (symbol, timeframe, timestamp) as the unique key.

    Later entries win (assumed fresher). Returns a timestamp-sorted list. Used
    when reconnecting and merging historical bars into local storage.
    """
    by_key: dict[tuple[str, str, datetime], Candle] = {}
    for c in candles:
        by_key[(c.symbol, str(c.timeframe), c.timestamp)] = c
    return sorted(by_key.values(), key=lambda x: x.timestamp)
