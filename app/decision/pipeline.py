"""Shared trading pipeline.

Builds MarketState from the feature store (using ONLY completed candles at time
t) and runs the full chain: regime -> strategies -> ensemble -> decision.

Live and backtest both call this exact code, guaranteeing that the backtest and
the live bot make the same decisions from the same information.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.config.assets import ContextConfig
from app.core.enums import Regime, Timeframe
from app.core.types import EnsembleResult, FeatureSnapshot, MarketState, PositionRecord
from app.data.candle_store import CandleStore
from app.data.feature_store import FeatureStore
from app.decision.ensemble import Ensemble
from app.decision.signal_engine import SignalEngine
from app.decision.trade_decision import DecisionOutcome, TradeDecisionComposer
from app.regimes.detector import RegimeDetector, RegimeResult


@dataclass(slots=True)
class PipelineResult:
    state: MarketState
    regime: RegimeResult
    ensemble: EnsembleResult
    outcome: DecisionOutcome


class TradingPipeline:
    def __init__(
        self,
        candles: CandleStore,
        features: FeatureStore,
        regime_detector: RegimeDetector,
        signal_engine: SignalEngine,
        ensemble: Ensemble,
        composer: TradeDecisionComposer,
        context: ContextConfig | None = None,
        reference_symbol: str = "BTC/USD",
        meta_filter=None,
    ) -> None:
        self.candles = candles
        self.features = features
        self.regime_detector = regime_detector
        self.signal_engine = signal_engine
        self.ensemble = ensemble
        self.composer = composer
        self.context = context or ContextConfig()
        self.reference_symbol = reference_symbol
        # Optional veto-only ML meta-filter. Never required for the core stack.
        self.meta_filter = meta_filter

    @property
    def ml_active(self) -> bool:
        return self.meta_filter is not None and getattr(self.meta_filter, "enabled", False)

    # -- index helpers ---------------------------------------------------
    def _ctx_index(self, symbol: str, timeframe: Timeframe, at_or_before: datetime) -> int:
        bars = self.candles.get(symbol, timeframe)
        if not bars:
            return -1
        times = [b.timestamp for b in bars]
        # last bar whose timestamp <= at_or_before (bars are complete buckets)
        idx = bisect.bisect_right(times, at_or_before) - 1
        return idx

    def build_state(
        self,
        symbol: str,
        timestamp: datetime,
        exec_index: int,
        btc_features: Optional[FeatureSnapshot] = None,
        relative_strength: Optional[float] = None,
        quote=None,
    ) -> tuple[MarketState, RegimeResult]:
        exec_tf = self.context.execution
        features = self.features.snapshot_at(symbol, exec_tf, exec_index, timestamp)

        # Higher-timeframe context snapshots (completed candles only).
        context: dict[str, FeatureSnapshot] = {}
        for key in ("confirmation", "regime", "macro"):
            tf = self.context.by_role(key)
            if tf is None:
                continue
            idx = self._ctx_index(symbol, tf, timestamp)
            if idx >= 0:
                context[str(tf)] = self.features.snapshot_at(symbol, tf, idx, timestamp)

        # Series for panic/recovery detection.
        recent_returns = self._recent_returns(symbol, exec_tf, exec_index, n=6)
        vol_series = self._series(symbol, exec_tf, "atr_pct", exec_index, n=12)
        momentum_series = self._series(symbol, exec_tf, "macd_hist", exec_index, n=6)
        higher_low = self._higher_low(symbol, exec_tf, exec_index)

        btc_feats = btc_features.values if btc_features is not None else None
        regime_result = self.regime_detector.detect(
            features=features.values,
            btc_features=btc_feats,
            recent_returns=recent_returns,
            vol_series=vol_series,
            momentum_series=momentum_series,
            higher_low=higher_low,
            timestamp=timestamp,
            is_reference=(symbol == self.reference_symbol),
        )

        state = MarketState(
            symbol=symbol,
            timestamp=timestamp,
            regime=regime_result.regime,
            features=features,
            context=context,
            btc_features=btc_features,
            relative_strength=relative_strength,
            quote=quote,
            is_panic=regime_result.regime is Regime.PANIC,
            is_recovery=regime_result.regime is Regime.RECOVERY,
            extra={"recovery_conditions": regime_result.recovery.conditions},
        )
        return state, regime_result

    def _recent_returns(self, symbol: str, tf: Timeframe, index: int, n: int = 6) -> list[float]:
        bars = self.candles.get(symbol, tf)
        out: list[float] = []
        for i in range(max(1, index - n + 1), index + 1):
            if i <= 0 or i >= len(bars):
                continue
            prev = bars[i - 1].close
            if prev:
                out.append((bars[i].close - prev) / prev)
        return out

    def _series(self, symbol: str, tf: Timeframe, name: str, index: int, n: int) -> list[float]:
        entry = self.features._frame(symbol, tf)  # internal, cached frame
        series = entry.frame.get(name)
        if not series:
            return []
        lo = max(0, index - n + 1)
        out = []
        for i in range(lo, min(index + 1, len(series))):
            v = series[i]
            if v is not None:
                out.append(float(v))
        return out

    def _higher_low(self, symbol: str, tf: Timeframe, index: int, lookback: int = 10) -> bool:
        bars = self.candles.get(symbol, tf)
        if index < lookback * 2 or index >= len(bars):
            return False
        recent_low = min(b.low for b in bars[index - lookback + 1 : index + 1])
        prior_low = min(b.low for b in bars[index - 2 * lookback + 1 : index - lookback + 1])
        return recent_low > prior_low

    # -- full evaluation -------------------------------------------------
    def evaluate(
        self,
        symbol: str,
        timestamp: datetime,
        exec_index: int,
        equity: float,
        positions: dict[str, PositionRecord],
        return_series: dict[str, list[float]] | None = None,
        buying_power: float | None = None,
        recent_volume: float | None = None,
        btc_index: int | None = None,
        ml_veto: bool = False,
    ) -> PipelineResult:
        # BTC reference snapshot / relative strength.
        btc_features = None
        relative_strength = None
        if symbol != self.reference_symbol:
            btc_features = self.features.snapshot_at(self.reference_symbol, self.context.execution, btc_index if btc_index is not None else -1, timestamp)
            relative_strength = self.features.relative_strength(
                symbol, self.reference_symbol, self.context.regime, lookback_bars=4
            )

        state, regime_result = self.build_state(
            symbol, timestamp, exec_index, btc_features=btc_features, relative_strength=relative_strength
        )
        signal_set = self.signal_engine.evaluate(state)
        ensemble = self.ensemble.combine(signal_set)

        # The ML meta-filter is evaluated HERE, on the full candidate signal (not
        # only after the rule-based gates pass), so the training set contains the
        # realistic distribution the filter will face live.
        metadata: dict = {}
        if not ml_veto and self.ml_active:
            decision = self.meta_filter.evaluate(state.features, state.regime)
            ml_veto = decision.veto
            metadata["ml"] = decision.as_dict()

        outcome = self.composer.compose(
            state=state,
            ensemble=ensemble,
            equity=equity,
            positions=positions,
            return_series=return_series,
            buying_power=buying_power,
            recent_volume=recent_volume,
            ml_veto=ml_veto,
        )
        if outcome.decision is not None and metadata:
            outcome.decision.metadata.update(metadata)
        elif outcome.rejection is not None and metadata:
            outcome.rejection.detail = f"{outcome.rejection.detail} | {metadata['ml']['reason']}"
        return PipelineResult(state=state, regime=regime_result, ensemble=ensemble, outcome=outcome)
