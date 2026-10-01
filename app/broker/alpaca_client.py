"""Thin, defensive wrapper around alpaca-py REST clients.

Responsibilities:
* lazy-construct trading/data clients (so importing the app needs no credentials);
* apply client-side rate limiting;
* retry transient failures with exponential backoff;
* translate alpaca-py exceptions into the execution layer's BrokerError types.

Nothing in this module returns alpaca objects to callers; the concrete broker
adapter (``trading_client``) does the normalisation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

from app.broker.rate_limiter import RateLimiter, data_limiter, trading_limiter
from app.execution.broker import BrokerError, RejectedOrderError, TransientBrokerError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes we treat as retryable.
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_REJECT_STATUS = {400, 401, 403, 404, 409, 422}


class AlpacaClient:
    def __init__(
        self,
        api_key: str | None,
        secret_key: str | None,
        base_url: str,
        data_url: str,
        retry_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 16.0,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url
        self.retry_attempts = max(1, retry_attempts)
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.trading_limiter: RateLimiter = trading_limiter()
        self.data_limiter: RateLimiter = data_limiter()
        self._trading: Any = None
        self._data: Any = None

    # -- lazy clients ----------------------------------------------------
    def require_credentials(self) -> None:
        if not self.api_key or not self.secret_key:
            raise BrokerError("Alpaca credentials are not configured (ALPACA_API_KEY/SECRET_KEY)")

    @property
    def trading(self) -> Any:
        if self._trading is None:
            self.require_credentials()
            from alpaca.trading.client import TradingClient  # lazy import

            self._trading = TradingClient(self.api_key, self.secret_key)
        return self._trading

    @property
    def data(self) -> Any:
        if self._data is None:
            self.require_credentials()
            from alpaca.data.historical import CryptoHistoricalDataClient  # lazy import

            self._data = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        return self._data

    # -- call wrapper ----------------------------------------------------
    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for attr in ("status_code", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        return None

    @classmethod
    def _classify(cls, exc: Exception) -> BrokerError:
        name = type(exc).__name__.lower()
        status = cls._status_code(exc)
        if status in _TRANSIENT_STATUS:
            return TransientBrokerError(f"transient broker error (HTTP {status}): {exc}")
        if status in _REJECT_STATUS:
            return RejectedOrderError(f"broker rejected request (HTTP {status}): {exc}")
        if "timeout" in name or "connection" in name or "network" in name:
            return TransientBrokerError(f"network error: {exc}")
        if "apierror" in name or "apierror" in name:
            # alpaca APIError without a recognised code: treat as transient.
            return TransientBrokerError(f"api error: {exc}")
        return BrokerError(str(exc))

    def call(
        self,
        fn: Callable[..., T],
        *args: Any,
        limiter: RateLimiter | None = None,
        retries: int | None = None,
        **kwargs: Any,
    ) -> T:
        limiter = limiter or self.trading_limiter
        attempts = retries if retries is not None else self.retry_attempts
        last: Exception | None = None
        for attempt in range(attempts):
            if not limiter.acquire(timeout=self.retry_max_delay):
                logger.warning("rate limiter timed out; proceeding cautiously")
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - translated below
                classified = self._classify(exc)
                if isinstance(classified, RejectedOrderError):
                    raise classified from exc
                last = classified
                if attempt < attempts - 1:
                    delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
                    logger.warning(
                        "broker call %s failed (attempt %d/%d): %s; retry in %.1fs",
                        getattr(fn, "__name__", "call"),
                        attempt + 1,
                        attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        assert last is not None
        raise last
