# Profitability & Integration Diagnostics — Honest Report

Data source: real cached candles (`data/`), 730 days, symbols BTC/ETH/SOL.
Execution timeframe: **M15** (the configured `context.execution`).
Cost model (baseline): taker 0.25%/side + slippage 0.10% + spread 0.05% => **0.65% round-trip**.
Harness: `diagnostics/validate.py`, `diagnostics/signal_census.py`, `diagnostics/scenario_sweep.py`.

---

## 1. Headline verdict

**The bot is not profitable on real data, and the cause is not fees — it is the absence of a
gross edge.** Every strategy arm loses money, and it still loses money with **zero** transaction
costs. This corroborates the project's own documented finding that the bot "loses money in
backtest"; the diagnostics below confirm it independently and explain *why*.

---

## 2. Corrected methodology (important)

An earlier iteration reported "all arms zero trades / 51,919 NO_SIGNAL". That was a
**harness/data-timeframe mismatch**, not bot behaviour:

- The strategy stack resolves features on `context.execution`, default **M15** (`app/config/assets.py:16`).
- The harness fed it **H1** candles, so `snapshot_at(M15)` returned an empty feature map.
- With all features `None`, `score_trend` yields score 0 -> `RANGE`, and the panic/recovery
  detectors saw empty inputs -> `UNKNOWN`.
- Net effect: 100% `RANGE`, empty `state.features`, so no strategy could fire.

Re-running everything on the **configured M15 execution timeframe** (with HTF context re-aggregated
exactly as `BacktestEngine._preload` does) produces healthy behaviour.

**Signal census (M15, 209,230 symbol-bars, post-warmup):**

| regime | share |
|---|---|
| WEAK_UPTREND | 17.9% |
| WEAK_DOWNTREND | 17.5% |
| HIGH_VOLATILITY | 16.6% |
| STRONG_UPTREND | 15.6% |
| STRONG_DOWNTREND | 13.3% |
| RANGE | 10.2% |
| LOW_VOLATILITY | 8.8% |
| PANIC / RECOVERY | ~0.03% |

`trend_pullback` fires **28,664 actionable** signals (max score 8.10); momentum fires 39,880 raw
BUYs but none clear the 6.0 threshold; breakout never fires; mean_reversion rarely. So the engine
*wants* to trade — the problem is purely whether those trades make money.

---

## 3. Backtest results on the correct timeframe (M15, 730 days)

| arm | trades | win% | gross | net | ret% | PF | maxDD% | P(profit) | P(ruin) |
|---|---|---|---|---|---|---|---|---|---|
| current_bot (legacy) | 475 | 30.9% | -1,403 | -5,038 | -50.4 | 0.38 | 50.4% | 0% | 58% |
| v2_trend_only | 356 | 25.3% | -2,385 | -5,003 | -50.0 | 0.31 | 50.2% | 0% | 54% |
| v2_multi | 356 | 25.3% | -2,385 | -5,003 | -50.0 | 0.31 | 50.2% | 0% | 54% |
| only_trend_pullback | 356 | 25.3% | -2,385 | -5,003 | -50.0 | 0.31 | 50.2% | 0% | 54% |
| only_breakout | 0 | — | 0 | 0 | 0 | — | 0 | 0% | 0% |
| only_mean_reversion | 0 | — | 0 | 0 | 0 | — | 0 | 0% | 0% |
| only_momentum | 183 | 23.5% | -1,560 | -3,066 | -30.7 | 0.30 | 30.8% | 0% | 0% |
| grid | 36 | 0% | 0 | -64 | -0.2 | inf | 0 | 0% | 0% |

Per-year (net): `v2_trend_only` 2024 = -2,726, 2025 = -2,277 — negative in **both** periods.
Exit mix: STOP_LOSS 227, TAKE_PROFIT 78, TRAILING_STOP 51. `v2_multi` rejections: NO_SIGNAL 34,234,
EXPECTED_EDGE_TOO_SMALL 7,348, RELATIVE_STRENGTH_FAILED 2,866.

---

## 4. The decisive test — scenario sweep (threshold x target-R x fee regime)

`diagnostics/scenario_sweep.py` isolates `trend_pullback` (the only frequently-firing strategy)
across entry threshold, target-R and fee models:

| scenario | trades | win% | gross | net | PF |
|---|---|---|---|---|---|
| baseline thr6 R2 | 356 | 25.3% | -2,385 | -5,003 | 0.31 |
| thr7 R2 | 342 | 23.7% | -2,553 | -5,017 | 0.28 |
| thr8 R2 | 52 | 26.9% | -393 | -887 | 0.33 |
| thr6 R3 | 512 | 30.5% | -1,319 | -5,052 | 0.37 |
| thr6 R4 | 519 | 30.1% | -1,274 | -5,062 | 0.39 |
| LOWFEE thr7 R3 | 1,314 | 28.5% | -3,107 | -5,007 | 0.65 |
| **ZEROCOST thr6 R2** | 3,440 | 31.7% | **-1,146** | -1,146 | 0.97 |
| **ZEROCOST thr7 R3** | 2,357 | 29.3% | **-1,377** | -1,377 | 0.94 |
| **ZEROCOST thr8 R4** | 275 | 29.1% | **-105** | -105 | 0.96 |

**No configuration is net-positive — not even with zero fees, zero slippage and zero spread.**
The profit factor converges to ~0.95 at zero cost (a fair coin would be 1.0). This proves the raw
signal has **no gross edge**; transaction costs merely amplify an already-losing system
(gross -2,385 -> net -5,003 is ~2.1x the loss).

---

## 5. Why it loses

1. **No directional edge.** Win rate 25–32% with a 2–4R target implies the entry logic does not
   predict direction. At zero cost the expectancy is still negative.
2. **Cost structure is punitive relative to the edge.** With 0.65% round-trip and an ATR-based stop
   (`atr_stop_multiplier=1.8`, `min_stop_distance=0.8%`), the cost is a large fraction of R;
   turnover is high enough that costs consume the account.
3. **The edge filter is doing its job but is insufficient.** `EXPECTED_EDGE_TOO_SMALL` rejected 7,348
   signals (good — it removes the worst trades) yet the survivors still lose, so the underlying
   scoring model is not discriminative.
4. **Ensemble/v2 has no advantage over legacy.** `v2_trend_only` ~= `v2_multi` ~= legacy in loss
   magnitude; the extra confirmation layers change *which* trades are taken, not the sign of P&L.
5. **Grid arm is inert.** 36 trades, 26 "flattens", effectively flat/negative (-64) — no rescue.

---

## 6. Live / paper integration reality

- **Backtest and live are genuinely unified.** `app/main.py:118` (backtest) and `build_live_context`
  (paper/live) both call the identical `build_stack` and the shared `TradingPipeline`. Same features,
  same regime detector, same strategies — no train/serve divergence.
- **The `grid` arm is NOT wired into execution.** It exists only in the comparison harness
  (`scripts/compare.py`); `build_stack` does not construct it. It cannot trade live regardless of its
  (flat) backtest result.
- **What can actually trade live:** `trend_pullback` (frequent) and occasionally `momentum`; the
  others are effectively gated off by regime or threshold. Since all of these are net-negative in
  backtest, **enabling live/paper trading today is expected to lose money.**

---

## 7. Recommendations

1. **Do not run this live.** Keep `PAPER`/`SHADOW`; the backtest parity guarantees live would
   reproduce the -50% equity paths.
2. **Attack the edge first, not the costs.** Since zero-cost P&L is negative, fee reduction alone
   cannot make this profitable. The scoring model needs a genuine predictive signal (validated
   features, out-of-sample testing, purged walk-forward) before any parameter tuning.
3. **Establish a profitability gate in CI.** Add a regression check that fails when a candidate
   configuration's zero-cost expectancy <= 0, so "profitable-looking" results cannot hide behind
   cost framing.
4. **Re-validate the timeframe wiring.** The M15-vs-H1 bug is easy to re-introduce; the harness now
   defaults to `symbols.execution_timeframe()`. Keep execution-timeframe consistency asserted in
   tests.

## 8. Artifacts

- `diagnostics/validation_results.txt` — full arm metrics, per-year, exits, rejections.
- `diagnostics/signal_census.txt` — regime mix and per-strategy signal counts.
- `diagnostics/scenario_sweep.txt` — threshold x target-R x fee sweep.
- Tools: `diagnostics/signal_census.py`, `diagnostics/scenario_sweep.py`, `diagnostics/probe_features.py`, `diagnostics/validate.py`.
