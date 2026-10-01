"""Global settings and environment loading.

All operational parameters live in configuration, never buried in strategy code.
Secrets come from environment variables only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.core.enums import TradingMode

# Load .env from the project root if present.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class FeeModel:
    """Configurable trading cost model.

    Alpaca crypto fees are volume-tiered, so these must NOT be hardcoded as
    fixed truths throughout the program. Values are fractions (0.0025 = 0.25%).
    """

    maker_fee: float = 0.0015
    taker_fee: float = 0.0025
    estimated_slippage: float = 0.0010
    spread_cost: float = 0.0005

    def taker_round_trip(self) -> float:
        """Total taker cost for a round trip (entry + exit) excluding spread/slip."""
        return self.taker_fee * 2.0

    def round_trip_cost(self, include_slippage: bool = True, include_spread: bool = True) -> float:
        total = self.taker_round_trip()
        if include_slippage:
            total += self.estimated_slippage
        if include_spread:
            total += self.spread_cost
        return total


@dataclass(slots=True)
class Paths:
    root: Path
    data: Path
    logs: Path
    reports: Path
    models: Path
    db_path: Path
    kill_switch: Path


@dataclass(slots=True)
class Settings:
    # Mode and safety
    mode: TradingMode = TradingMode.PAPER
    live_confirmed: bool = False

    # Credentials
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"

    # Alerts
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # Paths
    paths: Paths = field(default_factory=lambda: _default_paths())

    # Costs
    fees: FeeModel = field(default_factory=FeeModel)

    # Loop cadence
    account_sync_interval_seconds: int = 30
    reconcile_interval_seconds: int = 300
    health_check_interval_seconds: int = 15
    stale_data_threshold_seconds: int = 180

    # Retry policy
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 16.0

    # Optional ML meta-filter (veto-only). Disabled by default: the core bot is
    # fully rule-based and must never depend on an optional model.
    ml_enabled: bool = False
    ml_veto_threshold: float = 0.5

    def __post_init__(self) -> None:
        for path in (self.paths.data, self.paths.logs, self.paths.reports):
            path.mkdir(parents=True, exist_ok=True)

    # -- derived helpers -------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self.mode is TradingMode.LIVE

    def live_start_allowed(self) -> bool:
        """Live mode requires an explicit double-lock (spec section 74)."""
        if not self.is_live:
            return True
        return self.live_confirmed and (self.alpaca_base_url or "").find("paper") == -1

    def validate_or_raise(self) -> None:
        if self.mode is TradingMode.LIVE:
            if not self.live_confirmed:
                raise RuntimeError(
                    "Refusing to start LIVE trading: LIVE_TRADING_CONFIRMED is not 'true'."
                )
            if "paper" in (self.alpaca_base_url or ""):
                raise RuntimeError(
                    "Refusing to start LIVE trading against the paper endpoint. "
                    "Set ALPACA_BASE_URL=https://api.alpaca.markets"
                )


def _default_paths() -> Paths:
    root = _PROJECT_ROOT
    data = Path(_env("DATA_DIR", str(root / "data")) or (root / "data"))
    logs = Path(_env("LOG_DIR", str(root / "logs")) or (root / "logs"))
    reports = Path(_env("REPORT_DIR", str(root / "reports")) or (root / "reports"))
    models = Path(_env("MODELS_DIR", str(root / "models")) or (root / "models"))
    db_path = Path(_env("DB_PATH", str(data / "bot.db")) or (data / "bot.db"))
    kill = root / (_env("KILL_SWITCH_FILE", "KILL_SWITCH") or "KILL_SWITCH")
    return Paths(
        root=root,
        data=data,
        logs=logs,
        reports=reports,
        models=models,
        db_path=db_path,
        kill_switch=kill,
    )


def load_settings() -> Settings:
    """Build Settings from environment variables."""
    mode_raw = (_env("TRADING_MODE", "paper") or "paper").strip().lower()
    try:
        mode = TradingMode(mode_raw)
    except ValueError:
        mode = TradingMode.PAPER

    settings = Settings(
        mode=mode,
        live_confirmed=_env_bool("LIVE_TRADING_CONFIRMED", False),
        alpaca_api_key=_env("ALPACA_API_KEY"),
        alpaca_secret_key=_env("ALPACA_SECRET_KEY"),
        alpaca_base_url=_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        or "https://paper-api.alpaca.markets",
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
        account_sync_interval_seconds=_env_int("ACCOUNT_SYNC_INTERVAL", 30),
        reconcile_interval_seconds=_env_int("RECONCILE_INTERVAL", 300),
        health_check_interval_seconds=_env_int("HEALTH_CHECK_INTERVAL", 15),
        stale_data_threshold_seconds=_env_int("STALE_DATA_THRESHOLD", 180),
        retry_attempts=_env_int("RETRY_ATTEMPTS", 3),
        ml_enabled=_env_bool("ML_ENABLED", False),
        ml_veto_threshold=_env_float("ML_VETO_THRESHOLD", 0.5),
    )
    return settings


# A module-level lazily-initialised singleton for convenience.
_SETTINGS: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _SETTINGS
    if _SETTINGS is None or reload:
        _SETTINGS = load_settings()
    return _SETTINGS
