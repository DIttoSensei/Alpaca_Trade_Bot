"""Backtesting: engine, simulator, metrics, walk-forward, Monte Carlo, reports.

The same decision stack used live (``app.assembly.build_stack``) drives the
backtest engine, so a strategy that works in the backtest is the strategy that
runs in production.
"""

from app.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    EquityPoint,
    run_backtest,
)
from app.backtest.monte_carlo import (
    MonteCarloResult,
    bootstrap_returns,
    shuffle_trades,
)
from app.backtest.report import (
    ReportPaths,
    build_report,
    render_text,
    save_report,
    write_trades_csv,
)
from app.backtest.simulator import CostModel, OpenPosition, SimConfig, Simulator
from app.backtest.walk_forward import (
    WalkForwardResult,
    WalkForwardWindow,
    run_walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "EquityPoint",
    "run_backtest",
    "SimConfig",
    "Simulator",
    "CostModel",
    "OpenPosition",
    "WalkForwardResult",
    "WalkForwardWindow",
    "run_walk_forward",
    "MonteCarloResult",
    "bootstrap_returns",
    "shuffle_trades",
    "ReportPaths",
    "build_report",
    "render_text",
    "save_report",
    "write_trades_csv",
]
