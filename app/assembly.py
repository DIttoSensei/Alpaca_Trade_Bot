"""Shared construction of the decision stack.

The single most important anti-divergence measure: the backtest engine and the
live loop build the EXACT same objects here, in the same order, with the same
configuration. If it runs in the backtest it runs live and vice versa
(spec sections 59, 92).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.config.assets import ContextConfig
from app.config.risk_config import RiskConfig
from app.config.strategy_config import StrategyConfig
from app.config.symbols import SymbolRegistry
from app.config import BotConfig, load_config
from app.core.enums import Timeframe
from app.core.types import Candle
from app.data.candle_store import CandleStore
from app.data.feature_store import FeatureStore
from app.decision.ensemble import Ensemble
from app.decision.pipeline import TradingPipeline
from app.decision.signal_engine import SignalEngine
from app.decision.trade_decision import TradeDecisionComposer
from app.regimes.detector import RegimeDetector
from app.risk.engine import RiskEngine
from app.ml.meta_filter import MetaFilterModel
from app.strategies import Strategy, build_strategies


@dataclass
class StrategyStack:
    """All decision components, wired together identically for live/backtest."""

    config: BotConfig
    candles: CandleStore
    features: FeatureStore
    strategies: dict[str, Strategy]
    signal_engine: SignalEngine
    ensemble: Ensemble
    regime_detector: RegimeDetector
    risk_engine: RiskEngine
    composer: TradeDecisionComposer
    pipeline: TradingPipeline
    context: ContextConfig
    reference_symbol: str

    @property
    def symbols(self) -> SymbolRegistry:
        return self.config.symbols

    @property
    def risk(self) -> RiskConfig:
        return self.config.risk

    @property
    def strategy_config(self) -> StrategyConfig:
        return self.config.strategies


def build_context(symbols: SymbolRegistry) -> ContextConfig:
    tf = symbols.timeframes
    return ContextConfig(
        execution=tf.get("execution", Timeframe.M15),
        confirmation=tf.get("confirmation", Timeframe.H1),
        regime=tf.get("regime", Timeframe.H4),
        macro=tf.get("macro", Timeframe.D1),
        micro=tf.get("micro", Timeframe.M5),
    )


def build_meta_filter(cfg: BotConfig):
    """Load the optional ML meta-filter from the models directory.

    Returns ``None`` when disabled or when no usable model exists, in which case
    the pipeline behaves exactly like the rule-based core. This is intentionally
    fail-open: an optional enhancement must never block trading.
    """
    settings = cfg.settings
    if not getattr(settings, "ml_enabled", False):
        return None
    try:
        return MetaFilterModel.load_default(
            settings.paths.models, threshold=settings.ml_veto_threshold
        )
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("meta-filter disabled (%s)", exc)
        return None


def build_stack(
    config: BotConfig | None = None,
    candles: CandleStore | None = None,
) -> StrategyStack:
    cfg = config or load_config()
    candles = candles or CandleStore()
    features = FeatureStore(candles)
    strategies = build_strategies(cfg.strategies)
    signal_engine = SignalEngine(strategies, cfg.strategies)
    ensemble = Ensemble(strategies, cfg.strategies)
    regime_detector = RegimeDetector(cfg.regime)
    risk_engine = RiskEngine(cfg.risk, cfg.symbols)
    context = build_context(cfg.symbols)
    reference_symbol = cfg.symbols.reference_symbol() or "BTC/USD"

    composer = TradeDecisionComposer(
        fees=cfg.settings.fees,
        risk=cfg.risk,
        risk_engine=risk_engine,
        reference_symbol=reference_symbol,
        require_btc_confirmation=cfg.strategies.require_btc_confirmation,
        min_relative_strength=cfg.strategies.min_relative_strength,
    )

    meta_filter = build_meta_filter(cfg)

    pipeline = TradingPipeline(
        candles=candles,
        features=features,
        regime_detector=regime_detector,
        signal_engine=signal_engine,
        ensemble=ensemble,
        composer=composer,
        context=context,
        reference_symbol=reference_symbol,
        meta_filter=meta_filter,
    )

    return StrategyStack(
        config=cfg,
        candles=candles,
        features=features,
        strategies=strategies,
        signal_engine=signal_engine,
        ensemble=ensemble,
        regime_detector=regime_detector,
        risk_engine=risk_engine,
        composer=composer,
        pipeline=pipeline,
        context=context,
        reference_symbol=reference_symbol,
    )


def warm_stack(stack: StrategyStack, series: Iterable[Candle]) -> None:
    """Load historical candles into the stack's store (used by backtest/live bootstrap)."""
    stack.candles.merge(list(series))
    stack.features.invalidate()


def warm_from_map(stack: StrategyStack, by_key: dict[tuple[str, Timeframe], list[Candle]]) -> None:
    for (_symbol, _tf), candles in by_key.items():
        stack.candles.merge(candles)
    stack.features.invalidate()
