"""Risk configuration (spec sections 26-35, 47)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RiskConfig:
    # Position risk
    risk_per_trade: float = 0.005  # 0.5% equity risked per trade
    max_portfolio_risk: float = 0.02  # 2% aggregate open risk

    # Exposure limits
    max_symbol_exposure: float = 0.20
    max_total_exposure: float = 0.60
    max_altcoin_exposure: float = 0.40
    max_positions: int = 3

    # Circuit breakers
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    max_consecutive_losses: int = 5
    losing_streak_risk_scale: float = 0.5
    losing_streak_cooldown_trades: int = 3

    # Stops (ATR based, spec section 26)
    atr_stop_multiplier: float = 1.8
    min_stop_distance: float = 0.008  # 0.8%
    max_stop_distance: float = 0.08  # 8.0%

    # Take profit
    target_r_multiple: float = 2.0  # target = 2R

    # Trailing (spec section 47)
    breakeven_trigger_r: float = 1.0
    trailing_trigger_r: float = 1.5
    trailing_atr_multiplier: float = 1.0

    # Trend-following exit (daily-trend strategy). A mean-reverting fixed target
    # caps winners and destroys a trend edge, so the daily strategy requests a
    # trailing exit with NO fixed target: it rides the trend and is stopped out
    # only when price closes back below the trend SMA (the canonical donchian/
    # SMA trend exit). Wide ATR trailing keeps the position through pullbacks.
    trend_trailing_atr_multiplier: float = 3.0
    trend_trailing_trigger_r: float = 1.0  # trail only after 1R is banked
    trend_exit_sma_period: int = 50
    # Nominal R-multiple used ONLY to size the expected move for the cost/edge
    # gate; a trend trade has no fixed target price, but the filter still needs a
    # representative horizon (stop_distance * this) to compare against costs.
    trend_target_r_multiple: float = 6.0

    # Cost / edge filter (spec section 25)
    require_cost_edge: bool = True
    min_expected_edge: float = 0.010  # 1.0% net of costs
    risk_buffer: float = 0.002  # extra cushion subtracted from expected edge
    max_slippage: float = 0.003

    # Liquidity (spec section 102)
    max_order_size_vs_volume: float = 0.01  # <=1% of recent bar volume
    min_volume_ratio: float = 0.2

    # Volatility (spec section 103)
    extreme_atr_percentile: float = 0.95
    extreme_vol_risk_scale: float = 0.5

    # Correlation (spec section 28)
    correlation_lookback: int = 60
    correlation_cluster_threshold: float = 0.7
    correlated_risk_penalty: float = 0.5

    # Data quality
    require_market_data_health: bool = True
    require_btc_confirmation: bool = True

    # Emergency
    emergency_liquidation: bool = False  # never default to liquidation

    # Fleet-wide caps keyed by strategy name -> risk multiplier default
    strategy_risk_defaults: dict[str, float] = field(
        default_factory=lambda: {
            "trend_pullback": 1.0,
            "breakout": 0.8,
            "mean_reversion": 0.6,
            "momentum": 1.0,
            "recovery": 0.4,
            "trend_daily": 1.0,
        }
    )


def load_risk_config() -> RiskConfig:
    return RiskConfig()
