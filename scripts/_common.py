"""Shared helpers for the operational scripts.

Kept deliberately thin: the scripts are operator entrypoints, not a second
implementation of the system. All real logic lives in ``app``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from app.broker.alpaca_client import AlpacaClient
from app.broker.market_data import AlpacaMarketData
from app.config import BotConfig, load_config
from app.core.enums import Timeframe
from app.core.types import Candle, utcnow
from app.data.historical_loader import HistoricalLoader, ParquetCache


def build_provider(cfg: BotConfig) -> AlpacaMarketData:
    s = cfg.settings
    client = AlpacaClient(
        s.alpaca_api_key,
        s.alpaca_secret_key,
        s.alpaca_base_url,
        s.alpaca_data_url,
        retry_attempts=s.retry_attempts,
        retry_base_delay=s.retry_base_delay_seconds,
        retry_max_delay=s.retry_max_delay_seconds,
    )
    return AlpacaMarketData(client)


def load_history(
    cfg: BotConfig,
    days: int = 365,
    symbols: Optional[list[str]] = None,
) -> tuple[dict[tuple[str, Timeframe], list[Candle]], Timeframe]:
    """Load execution-timeframe history for the enabled universe.

    Returns ``(candles_by_key, execution_timeframe)``. Bars are cached to parquet
    so repeat runs do not re-download the same window.
    """
    provider = build_provider(cfg)
    cache = ParquetCache(cfg.settings.paths.data)
    loader = HistoricalLoader(provider, cache=cache)
    exec_tf = cfg.symbols.execution_timeframe()
    start = utcnow() - timedelta(days=days)

    universe = symbols or cfg.symbols.enabled()
    candles_by_key: dict[tuple[str, Timeframe], list[Candle]] = {}
    for symbol in universe:
        bars = loader.load_range(symbol, exec_tf, start)
        if bars:
            candles_by_key[(symbol, exec_tf)] = bars
    return candles_by_key, exec_tf


def load_config_or_default() -> BotConfig:
    return load_config()
