# Crypto Trading Bot V2

A production-grade, regime-aware, multi-strategy crypto trading bot for
[Alpaca](https://alpaca.markets/). It is designed to run cheaply on a
low-resource VM, to fail closed, and to be crash/restart safe. The **same
decision stack** powers backtests and live trading, so a strategy that looks
good on historical data is exercised by identical code in production.

> **Trading is risky.** This software is provided as-is, with no warranty and no
> guarantee of profit. Always start in `shadow`, then `paper`, and only then
> consider `live` — behind the explicit double lock described below.

---

## Highlights

- **Regime-aware.** A composite detector classifies the market (strong/weak
  up/down trends, range, high/low volatility, panic, recovery) and only the
  strategies suited to the current regime are allowed to fire.
- **Multi-strategy ensemble.** `trend_pullback`, `breakout`, `mean_reversion`,
  `momentum` and `recovery` produce complementary signals that are combined by
  a category-capped, cost-aware ensemble. The original strategy is preserved as
  the benchmark `legacy_trend_pullback`.
- **Costs are first-class.** Fees, spread and slippage are modelled before a
  trade is placed; an edge filter rejects candidates whose expected edge does
  not clear realistic round-trip costs.
- **Risk-first execution.** ATR-based dynamic stops and position sizing,
  exposure and correlation limits, and circuit breakers that fail closed.
- **Crash-safe state.** SQLite (WAL) is the single source of truth, with
  write-through updates, checkpoints and recovery on boot. The broker is always
  authoritative; a reconciler repairs drift.
- **Idempotent orders.** Deterministic client order IDs prevent duplicate orders
  across retries and restarts.
- **Honest validation.** Walk-forward/out-of-sample testing, Monte Carlo and
  bootstrap resampling, and an experiment registry that records every run.
- **Optional ML meta-filter.** A veto-only classifier (scikit-learn) can reject
  low-probability setups. It is fail-open: an absent or broken model never
  blocks trading, and it is trained on the exact feature snapshots produced at
  inference time (no train/serve skew).

---

## Project layout

```
crypto-trading-bot/
├── app/
│   ├── main.py               # unified CLI entrypoint (dispatches on mode)
│   ├── assembly.py           # builds the shared decision stack
│   ├── backtest/             # engine, simulator, metrics, walk-forward, Monte Carlo, report
│   ├── broker/               # Alpaca clients, market data, account sync, websockets, rate limiter
│   ├── config/               # settings, symbols, strategy config, risk config
│   ├── core/                 # enums + canonical dataclasses (Candle, Signal, TradeDecision, ...)
│   ├── data/                 # candle store, validator, historical loader, feature store
│   ├── decision/             # signal engine, ensemble, confirmation, edge filter, composer, pipeline
│   ├── execution/            # order planner/manager, execution monitor, reconciliation, kill switch
│   ├── indicators/           # in-house numpy indicators (no TA-Lib)
│   ├── live/                 # the live loop (SHADOW / PAPER / LIVE)
│   ├── ml/                   # optional veto-only meta-filter + experiment registry
│   ├── monitoring/           # health, watchdog, alerts, metrics
│   ├── regimes/              # trend/volatility regimes, panic, recovery, composite detector
│   ├── risk/                 # sizing, stops, exposure, correlation, circuit breakers
│   ├── state/                # SQLite database, schema, checkpoints, state manager
│   └── strategies/           # trend_pullback, breakout, mean_reversion, momentum, recovery, legacy
├── scripts/                  # operator entrypoints (see below)
├── tests/                    # pytest suite
├── data/                     # parquet cache + SQLite DB (runtime)
├── logs/                     # log files (runtime)
└── reports/                  # backtest reports + experiment registry (runtime)
```

---

## Architecture at a glance

```
market data ──▶ feature store ──▶ regime detector
                                      │
                                      ▼
                          signal engine (eligible strategies)
                                      │
                                      ▼
                              ensemble + confirmation
                                      │
                                      ▼
                        edge filter (fees/spread/slippage)
                                      │
                                      ▼
                      ML meta-filter (optional, veto-only)
                                      │
                                      ▼
                        trade decision composer
                                      │
                                      ▼
                risk engine (sizing, stops, exposure, breakers)
                                      │
                                      ▼
                     order planner ──▶ order manager ──▶ broker
                                      │
                                      ▼
                    execution monitor + reconciliation (broker authoritative)
```

The **same** `TradingPipeline` / `TradeDecisionComposer` / `BacktestEngine` path
is used by the live loop, so behaviour is consistent across modes.

---

## Installation

Requires **Python 3.11+**.

```bash
cd crypto-trading-bot
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Indicators are implemented in-house with numpy, so **no TA-Lib** is required.

---

## Configuration

Copy the example environment file and fill in your keys:

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

Key variables:

| Variable | Purpose |
| --- | --- |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Broker/data credentials. |
| `ALPACA_BASE_URL` | Paper endpoint by default; live endpoint only when going live. |
| `TRADING_MODE` | `backtest` \| `shadow` \| `paper` \| `live` (default `paper`). |
| `LIVE_TRADING_CONFIRMED` | Must be `true` **and** the base URL must be the live endpoint for `live` mode (double lock). |
| `KILL_SWITCH_FILE` | If this file exists in the project root, no new orders are placed. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional alerting. |
| `DATA_DIR` / `LOG_DIR` / `REPORT_DIR` / `DB_PATH` | Optional path overrides. |
| `ML_ENABLED` / `ML_VETO_THRESHOLD` | Optional ML meta-filter toggle and veto threshold. |

Configuration for instruments, strategies and risk lives in
[`app/config/`](crypto-trading-bot/app/config/__init__.py) and is aggregated into
[`BotConfig`](crypto-trading-bot/app/config/__init__.py:39).

> **Never commit your real `.env`.** It is already covered by `.gitignore`.

---

## Trading modes

| Mode | What it does |
| --- | --- |
| `backtest` | Fetches history, runs the shared engine, writes a report. No orders. |
| `shadow` | Runs the live loop, makes and logs decisions, **submits nothing**. |
| `paper` | Runs the live loop against the Alpaca **paper** endpoint. |
| `live` | Runs against real money, only behind the explicit double lock. |

`paper` is the default. `shadow` is the recommended first run to confirm data,
regime detection and decisions before touching an account.

### Double lock for live trading

`live` mode starts only when **both** hold:

1. `LIVE_TRADING_CONFIRMED=true` (case-insensitive), and
2. `TRADING_MODE=live` with a non-paper `ALPACA_BASE_URL`.

Otherwise [`live_start_allowed()`](crypto-trading-bot/app/config/settings.py:140)
returns `False` and the bot refuses to start.

---

## Usage

### Unified entrypoint

```bash
# Run in the mode resolved from .env
python -m app.main

# Backtest 1 year of history, skipping charts
python -m app.main --mode backtest --days 365 --no-charts

# Shadow run: decide but never submit, stop after 20 cycles
python -m app.main --mode shadow --max-iterations 20
```

CLI flags: `--mode`, `--days`, `--no-charts`, `--max-iterations`, `--log-level`.

### Operator scripts

| Script | Purpose |
| --- | --- |
| [`scripts/backtest.py`](crypto-trading-bot/scripts/backtest.py) | Full backtest with optional `--walk-forward` and `--monte-carlo`. |
| [`scripts/train.py`](crypto-trading-bot/scripts/train.py) | Train the optional ML meta-filter from a backtest's trades. |
| [`scripts/paper.py`](crypto-trading-bot/scripts/paper.py) | Run the live loop in `shadow` or `paper`. |
| [`scripts/reconcile.py`](crypto-trading-bot/scripts/reconcile.py) | Reconcile local state with the broker (`--repair` to fix). |
| [`scripts/healthcheck.py`](crypto-trading-bot/scripts/healthcheck.py) | Bounded health check for cron/systemd/k8s (exit codes 0/1/2). |
| [`scripts/compare.py`](crypto-trading-bot/scripts/compare.py) | Head-to-head report: current bot vs V2 trend-only vs V2 multi-strategy (see [`FINAL_REPORT.md`](crypto-trading-bot/FINAL_REPORT.md)). |

Examples:

```bash
python scripts/backtest.py --days 730 --walk-forward --monte-carlo
python scripts/train.py --days 730 --model gradient_boosting
python scripts/paper.py --mode shadow --max-iterations 5
python scripts/reconcile.py --repair
python scripts/healthcheck.py --json --sync-account
```

---

## Backtesting & validation

Every backtest produces a text report, a JSON payload, a trades CSV and (unless
disabled) an equity-curve chart under `reports/`. Validation options:

- **Walk-forward / out-of-sample** (`--walk-forward`) splits the timeline into
  rolling train/test windows to expose overfitting.
- **Monte Carlo / bootstrap** (`--monte-carlo`) resamples trade returns to
  estimate the probability of profit and of ruin.

The engine is **O(n) in the number of bars**: it pre-computes the indicator frame
once per `(symbol, timeframe)` up front and the per-bar loop only resolves
point-in-time snapshots. Because every indicator is causal, this is equivalent to
recomputing on each bar but with no look-ahead and without the cost. (The full
frame is held in memory for the run, so memory scales with history.)

Goals are expressed in **risk-adjusted expectancy after realistic costs**, not
raw win rate.

---

## ML meta-filter

The meta-filter is an **optional, veto-only** classifier that predicts the
probability a candidate setup wins after costs. If it is absent, untrained or
errors at inference, it **fails open** (never blocks trades).

```bash
# 1. Generate trades + train (writes models/meta_filter.joblib)
python scripts/train.py --days 730 --model gradient_boosting

# 2. Enable it
ML_ENABLED=true ML_VETO_THRESHOLD=0.5 python -m app.main --mode backtest
```

Training uses a **chronological** split (never shuffles across time) and reports
precision/recall/F1/ROC-AUC plus the realised veto rate and win-rate lift. The
feature vector is built from the exact [`FeatureSnapshot`](crypto-trading-bot/app/ml/features.py)
captured at decision time, so there is no train/serve skew.

Runs are recorded in the experiment registry at
`reports/experiments.jsonl` via [`ExperimentRegistry`](crypto-trading-bot/app/ml/registry.py).

---

## State, recovery & safety

- **SQLite (WAL)** is the primary store — not JSON. State writes are
  write-through so an unexpected shutdown loses at most the current cycle.
- **Reconcile on boot.** The broker is authoritative; the loop reconciles
  positions/orders at start, and `scripts/reconcile.py` can repair drift any
  time (the recommended first action after an unclean shutdown).
- **Idempotent orders.** Deterministic client order IDs prevent duplicates on
  retry or restart.
- **Position lifecycle.** A position is only materialised in the book after a
  broker-confirmed fill, so a pending order can never be mistaken for a closed
  position.
- **Kill switch.** Create the `KILL_SWITCH` file in the project root to halt new
  orders; the switch is fail-closed on read errors.
- **Circuit breakers.** Daily-loss, drawdown, consecutive-loss and API-failure
  breakers block entries (and can scale risk down) automatically.
- **Monitoring.** Heartbeats, a watchdog for stalled components, health
  aggregation and optional Telegram alerts.

---

## Testing

```bash
python -m pytest tests -q
```

The suite covers indicator correctness, candle validation/dedupe/aggregation,
ML feature-vector stability and sanitisation, meta-filter veto/fail-open and
save/load, meta-filter training (chronological split, lift metrics, guards) and
the experiment registry. scikit-learn–dependent tests are skipped automatically
when it is not installed.

---

## Development notes

- **No lookahead.** Higher-timeframe features use only *completed* candles.
- **Deterministic costs.** The backtest cost model mirrors the live edge-filter
  assumptions so simulated and live economics agree.
- **Fail closed.** Safety-critical paths (kill switch, reconciliation, breakers)
  default to blocking rather than allowing a trade.
- **Lean footprint.** numpy-only indicators and a minimal dependency set keep the
  bot affordable on a small VM.

---

## License

Released under the MIT License. See the repository for details.

## Disclaimer

Nothing here is financial advice. Markets can and do move against any strategy.
You are solely responsible for any capital you deploy and for complying with the
terms of your broker and applicable regulations.
