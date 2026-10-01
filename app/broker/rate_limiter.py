"""Client-side rate limiting.

Alpaca enforces request limits. Rather than discover them by getting throttled
(and risking a burst that trips circuit breakers), we self-limit with a token
bucket. Thread-safe because market-data and trading clients may be called from
different threads/tasks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token-bucket limiter.

    ``rate`` tokens are added per second up to ``capacity``. ``acquire`` blocks
    (polling) until a token is available or the timeout elapses.
    """

    def __init__(self, rate: float, capacity: int | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(1, int(rate)))
        self._bucket = _Bucket(tokens=self.capacity, updated=time.monotonic())
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._bucket.updated
        if elapsed > 0:
            self._bucket.tokens = min(self.capacity, self._bucket.tokens + elapsed * self.rate)
            self._bucket.updated = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._bucket.tokens >= tokens:
                self._bucket.tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self.try_acquire(tokens):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, tokens / self.rate if self.rate else 0.05))

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._bucket.tokens


# Sensible defaults: trading endpoints are stricter than data endpoints.
def trading_limiter() -> RateLimiter:
    return RateLimiter(rate=3.0, capacity=6)


def data_limiter() -> RateLimiter:
    return RateLimiter(rate=8.0, capacity=16)


def websocket_limiter() -> RateLimiter:
    # Crypto market data websockets allow a small number of subscriptions.
    return RateLimiter(rate=1.0, capacity=3)
