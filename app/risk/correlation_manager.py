"""Correlation-aware risk adjustment (spec section 28).

If BTC/ETH/SOL are highly correlated and all signaled simultaneously, treat
them as one large crypto-market bet and reduce combined risk rather than
filling every available slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config.risk_config import RiskConfig


def returns_from_closes(closes: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] != 0:
            out.append((closes[i] - closes[i - 1]) / closes[i - 1])
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a = a[-n:]
    b = b[-n:]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


@dataclass(slots=True)
class CorrelationAdjustment:
    scale: float
    cluster_size: int
    max_correlation: float
    reason: str = ""


class CorrelationManager:
    def __init__(self, risk: RiskConfig) -> None:
        self.risk = risk

    def adjustment(
        self,
        candidate_symbol: str,
        open_symbols: list[str],
        return_series: dict[str, list[float]],
    ) -> CorrelationAdjustment:
        """Reduce risk when the candidate is highly correlated with already-open
        positions in the same cluster."""
        if not open_symbols:
            return CorrelationAdjustment(1.0, 0, 0.0, "no open positions")
        cand = return_series.get(candidate_symbol)
        if not cand:
            return CorrelationAdjustment(1.0, 0, 0.0, "no candidate history")

        high_corr: list[str] = []
        max_corr = 0.0
        for sym in open_symbols:
            other = return_series.get(sym)
            if not other:
                continue
            c = pearson(cand, other)
            max_corr = max(max_corr, abs(c))
            if abs(c) >= self.risk.correlation_cluster_threshold:
                high_corr.append(sym)

        if not high_corr:
            return CorrelationAdjustment(1.0, 0, max_corr, "low correlation")

        # Scale down by the penalty raised to the cluster size, floored to 0.25.
        scale = max(0.25, self.risk.correlated_risk_penalty ** len(high_corr))
        return CorrelationAdjustment(
            scale=scale,
            cluster_size=len(high_corr),
            max_correlation=max_corr,
            reason=f"correlated with {high_corr} (max |rho|={max_corr:.2f})",
        )
