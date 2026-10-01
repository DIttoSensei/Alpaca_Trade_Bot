"""Portfolio exposure management (spec sections 28-29, 102-103).

Understands that BTC/ETH/SOL are not independent risks. Tracks gross/net,
BTC and altcoin exposure, and enforces limits. Provides the remaining
portfolio room used by the position sizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.risk_config import RiskConfig
from app.config.symbols import SymbolRegistry
from app.core.types import PositionRecord


@dataclass(slots=True)
class ExposureSnapshot:
    equity: float
    gross_exposure: float
    net_exposure: float
    btc_exposure: float
    altcoin_exposure: float
    per_symbol: dict[str, float] = field(default_factory=dict)

    @property
    def gross_pct(self) -> float:
        return self.gross_exposure / self.equity if self.equity else 0.0

    @property
    def btc_pct(self) -> float:
        return self.btc_exposure / self.equity if self.equity else 0.0

    @property
    def altcoin_pct(self) -> float:
        return self.altcoin_exposure / self.equity if self.equity else 0.0


class ExposureManager:
    def __init__(self, risk: RiskConfig, symbols: SymbolRegistry) -> None:
        self.risk = risk
        self.symbols = symbols

    def snapshot(self, equity: float, positions: dict[str, PositionRecord]) -> ExposureSnapshot:
        per_symbol: dict[str, float] = {}
        gross = 0.0
        net = 0.0
        btc = 0.0
        alt = 0.0
        ref = self.symbols.reference_symbol()
        for symbol, pos in positions.items():
            if pos.qty == 0:
                continue
            price = pos.current_price or pos.average_entry
            notional = abs(pos.qty * price)
            signed = pos.qty * price
            per_symbol[symbol] = notional
            gross += notional
            net += signed
            if symbol == ref:
                btc += notional
            else:
                alt += notional
        return ExposureSnapshot(
            equity=equity,
            gross_exposure=gross,
            net_exposure=net,
            btc_exposure=btc,
            altcoin_exposure=alt,
            per_symbol=per_symbol,
        )

    def remaining_portfolio_room(self, snap: ExposureSnapshot) -> float:
        """Notional room before hitting max_total_exposure."""
        limit = self.risk.max_total_exposure * snap.equity
        return max(0.0, limit - snap.gross_exposure)

    def symbol_room(self, snap: ExposureSnapshot, symbol: str) -> float:
        limit = self.risk.max_symbol_exposure * snap.equity
        current = snap.per_symbol.get(symbol, 0.0)
        return max(0.0, limit - current)

    def can_open(self, snap: ExposureSnapshot, symbol: str, positions: dict[str, PositionRecord]) -> tuple[bool, str]:
        open_count = sum(1 for p in positions.values() if p.qty != 0)
        if symbol not in positions and open_count >= self.risk.max_positions:
            return False, "POSITION_LIMIT"
        if snap.gross_exposure >= self.risk.max_total_exposure * snap.equity:
            return False, "PORTFOLIO_EXPOSURE_LIMIT"
        ref = self.symbols.reference_symbol()
        if symbol != ref:
            alt_limit = self.risk.max_altcoin_exposure * snap.equity
            if snap.altcoin_exposure >= alt_limit:
                return False, "PORTFOLIO_EXPOSURE_LIMIT"
        if snap.per_symbol.get(symbol, 0.0) >= self.risk.max_symbol_exposure * snap.equity:
            return False, "SYMBOL_EXPOSURE_LIMIT"
        return True, ""
