"""Configuration package: settings, symbols, strategy and risk configuration."""

from app.config.risk_config import RiskConfig, load_risk_config
from app.config.settings import FeeModel, Paths, Settings, get_settings, load_settings
from app.config.strategy_config import (
    RegimeConfig,
    StrategyConfig,
    StrategyParams,
    load_strategy_config,
)
from app.config.symbols import (
    ALL_TIMEFRAMES,
    DEFAULT_SYMBOLS,
    DEFAULT_TIMEFRAMES,
    SymbolConfig,
    SymbolRegistry,
    load_symbols,
)

__all__ = [
    "FeeModel",
    "Paths",
    "Settings",
    "get_settings",
    "load_settings",
    "RegimeConfig",
    "StrategyConfig",
    "StrategyParams",
    "load_strategy_config",
    "SymbolConfig",
    "SymbolRegistry",
    "load_symbols",
    "DEFAULT_SYMBOLS",
    "DEFAULT_TIMEFRAMES",
    "ALL_TIMEFRAMES",
]


class BotConfig:
    """Aggregate of all configuration, passed around the system."""

    def __init__(
        self,
        settings: Settings | None = None,
        symbols: SymbolRegistry | None = None,
        strategies: StrategyConfig | None = None,
        risk: RiskConfig | None = None,
        regime: RegimeConfig | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.symbols = symbols or load_symbols()
        self.strategies = strategies or load_strategy_config()
        self.risk = risk or load_risk_config()
        self.regime = regime or RegimeConfig()

    def config_hash(self) -> str:
        """Deterministic hash of configuration for experiment reproducibility."""
        import hashlib
        import json

        payload = {
            "mode": str(self.settings.mode),
            "risk": self.risk.__dict__ if hasattr(self.risk, "__dict__") else str(self.risk),
            "fees": {
                "maker": self.settings.fees.maker_fee,
                "taker": self.settings.fees.taker_fee,
                "slippage": self.settings.fees.estimated_slippage,
                "spread": self.settings.fees.spread_cost,
            },
            "symbols": list(self.symbols.symbols.keys()),
        }
        raw = json.dumps(payload, default=str, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_config() -> BotConfig:
    return BotConfig()
