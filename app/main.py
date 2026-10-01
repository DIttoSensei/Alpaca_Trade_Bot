"""Unified entrypoint for Crypto Trading Bot V2.

Dispatches on the configured ``TradingMode``:

  * ``backtest`` -> fetch history, run the shared backtest engine, write a report;
  * ``shadow``   -> run the live loop but submit nothing (decisions are logged);
  * ``paper``    -> run the live loop against the Alpaca paper endpoint (default);
  * ``live``     -> run against real money, behind the explicit double-lock.

Usage::

    python -m app.main                # uses MODE / settings from the environment
    python -m app.main --mode shadow
    python -m app.main --mode backtest --days 365 --no-charts

Nothing here makes trading decisions: it only wires configuration, logging and
the loop, keeping ``main`` a thin, auditable shell.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta

from app.config import load_config
from app.core.enums import TradingMode
from app.core.types import utcnow

logger = logging.getLogger("app.main")


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once (console + optional file)."""
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)

    # Mirror logs to a file inside the configured logs directory when possible.
    try:
        cfg = load_config()
        log_path = cfg.settings.paths.logs / "bot.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)
    except Exception:  # noqa: BLE001 - logging must never prevent startup
        pass


def _run_live(mode: TradingMode, max_iterations: int | None) -> int:
    from app.live.loop import TradingLoop, build_live_context

    ctx = build_live_context()
    # Override the mode resolved from the environment with the CLI choice.
    ctx.config.settings.mode = mode
    if mode is TradingMode.LIVE and not ctx.config.settings.live_start_allowed():
        logger.error(
            "Refusing to start in LIVE mode: set LIVE_TRADING_CONFIRMED=true and a "
            "non-paper ALPACA_BASE_URL to enable the second lock."
        )
        return 2

    loop = TradingLoop(ctx)
    if ctx.websocket is not None:
        try:
            ctx.websocket.start()
        except Exception as exc:  # noqa: BLE001 - REST fallback remains
            logger.warning("websocket start failed (%s); relying on REST polling", exc)
    try:
        loop.run(max_iterations=max_iterations)
    finally:
        if ctx.websocket is not None:
            try:
                ctx.websocket.stop()
            except Exception:  # noqa: BLE001
                pass
    return 0


def _run_backtest(cfg, days: int, charts: bool) -> int:
    from app.backtest import BacktestConfig, run_backtest, save_report
    from app.broker.alpaca_client import AlpacaClient
    from app.broker.market_data import AlpacaMarketData
    from app.data.candle_store import CandleStore
    from app.data.historical_loader import HistoricalLoader, ParquetCache

    settings = cfg.settings
    client = AlpacaClient(
        settings.alpaca_api_key,
        settings.alpaca_secret_key,
        settings.alpaca_base_url,
        settings.alpaca_data_url,
        retry_attempts=settings.retry_attempts,
        retry_base_delay=settings.retry_base_delay_seconds,
        retry_max_delay=settings.retry_max_delay_seconds,
    )
    provider = AlpacaMarketData(client)
    cache = ParquetCache(settings.paths.data)
    loader = HistoricalLoader(provider, cache=cache)

    exec_tf = cfg.symbols.execution_timeframe()
    start = utcnow() - timedelta(days=days)
    candles_by_key: dict[tuple[str, object], list] = {}
    for symbol in cfg.symbols.enabled():
        bars = loader.load_range(symbol, exec_tf, start)
        if not bars:
            logger.warning("no historical bars for %s %s", symbol, exec_tf)
            continue
        candles_by_key[(symbol, exec_tf)] = bars
        logger.info("loaded %d bars for %s %s", len(bars), symbol, exec_tf)

    if not candles_by_key:
        logger.error("No data loaded; cannot run the backtest (check credentials).")
        return 1

    result = run_backtest(
        candles_by_key,
        config=cfg,
        backtest=BacktestConfig(starting_equity=10_000.0),
        collection_timeframe=exec_tf,
    )
    paths = save_report(result, settings.paths.reports, label="backtest", charts=charts)
    logger.info("backtest complete: %d trades", len(result.trades))
    print(f"Report: {paths.text_path}")
    print(f"JSON:   {paths.json_path}")
    if paths.trades_csv:
        print(f"Trades: {paths.trades_csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crypto Trading Bot V2")
    parser.add_argument(
        "--mode",
        choices=[str(m) for m in TradingMode],
        default=None,
        help="override the configured trading mode",
    )
    parser.add_argument("--days", type=int, default=365, help="backtest history window in days")
    parser.add_argument("--no-charts", action="store_true", help="skip matplotlib charts")
    parser.add_argument("--max-iterations", type=int, default=None, help="stop after N loop cycles")
    parser.add_argument("--log-level", default=None, help="override LOG_LEVEL")
    args = parser.parse_args(argv)

    # Resolve the mode before logging so the mode can influence behaviour.
    cfg = load_config()
    mode = TradingMode(args.mode) if args.mode else cfg.settings.mode
    setup_logging(args.log_level)

    logger.info("Crypto Trading Bot V2 starting in %s mode", mode)
    cfg.settings.validate_or_raise()

    if mode is TradingMode.BACKTEST:
        return _run_backtest(cfg, days=args.days, charts=not args.no_charts)
    return _run_live(mode, max_iterations=args.max_iterations)


if __name__ == "__main__":
    sys.exit(main())
