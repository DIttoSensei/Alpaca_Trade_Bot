"""Symbol configuration.

Never hardcode ``if symbol == "BTC/USD"`` in strategy code. Use SymbolConfig.
Adding a new pair is a configuration change, not an architectural change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import Timeframe


@dataclass(slots=True)
class SymbolConfig:
    symbol: str
    enabled: bool = True
    is_reference: bool = False  # BTC is the market reference asset
    base: str = ""
    quote: str = "USD"
    # Per-symbol overrides (None -> use global defaults)
    min_order_size: float | None = None
    max_spread: float | None = None
    min_volume_24h: float | None = None
    tick_size: float = 0.00000001


def _make(symbol: str, **kwargs) -> SymbolConfig:
    base, _, quote = symbol.partition("/")
    return SymbolConfig(symbol=symbol, base=base, quote=quote or "USD", **kwargs)


# Default universe (spec section 1 & 3). Add pairs here; no code changes needed.
DEFAULT_SYMBOLS: dict[str, SymbolConfig] = {
    "BTC/USD": _make("BTC/USD", is_reference=True),
    "ETH/USD": _make("ETH/USD"),
    "SOL/USD": _make("SOL/USD"),
}

# Timeframe roles (spec section 8).
DEFAULT_TIMEFRAMES: dict[str, Timeframe] = {
    "execution": Timeframe.M15,
    "confirmation": Timeframe.H1,
    "regime": Timeframe.H4,
    "macro": Timeframe.D1,
    "micro": Timeframe.M5,
}

ALL_TIMEFRAMES: list[Timeframe] = [
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
]


@dataclass(slots=True)
class SymbolRegistry:
    """Lookup helper over the configured universe."""

    symbols: dict[str, SymbolConfig] = field(default_factory=lambda: dict(DEFAULT_SYMBOLS))
    timeframes: dict[str, Timeframe] = field(default_factory=lambda: dict(DEFAULT_TIMEFRAMES))

    def enabled(self) -> list[str]:
        return [s for s, cfg in self.symbols.items() if cfg.enabled]

    def get(self, symbol: str) -> SymbolConfig | None:
        return self.symbols.get(symbol)

    def reference_symbol(self) -> str | None:
        for s, cfg in self.symbols.items():
            if cfg.is_reference:
                return s
        return None

    def add(self, cfg: SymbolConfig) -> None:
        self.symbols[cfg.symbol] = cfg

    def execution_timeframe(self) -> Timeframe:
        return self.timeframes["execution"]


def load_symbols() -> SymbolRegistry:
    # In a full deployment this would merge overrides from a YAML file.
    return SymbolRegistry()
