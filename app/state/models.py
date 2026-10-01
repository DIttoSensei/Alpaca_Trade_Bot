"""SQLite schema definition.

The database is the bot's durable memory. JSON files are explicitly NOT used as
the primary database. All writes that pair an order with its broker identity are
done in a transaction so we never end up with a broker order the DB doesn't know
about (spec section 127).
"""

from __future__ import annotations

SCHEMA_VERSION = 1

SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS account_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        account_equity REAL NOT NULL,
        cash REAL NOT NULL,
        buying_power REAL NOT NULL,
        last_sync TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY,
        qty REAL NOT NULL,
        average_entry REAL NOT NULL,
        current_price REAL NOT NULL DEFAULT 0,
        unrealized_pnl REAL NOT NULL DEFAULT 0,
        strategy TEXT,
        entry_time TEXT,
        stop_price REAL,
        target_price REAL,
        initial_stop_price REAL,
        initial_risk_per_unit REAL DEFAULT 0,
        risk_amount REAL DEFAULT 0,
        trailing_state TEXT,
        state TEXT,
        regime TEXT,
        feature_snapshot TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        client_order_id TEXT PRIMARY KEY,
        alpaca_order_id TEXT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        qty REAL NOT NULL,
        limit_price REAL,
        stop_price REAL,
        status TEXT NOT NULL,
        intent TEXT DEFAULT 'entry',
        filled_qty REAL DEFAULT 0,
        average_fill_price REAL,
        strategy TEXT,
        submitted_at TEXT,
        updated_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_orders_alpaca ON orders (alpaca_order_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, status)",
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        qty REAL NOT NULL,
        fees REAL NOT NULL DEFAULT 0,
        slippage REAL NOT NULL DEFAULT 0,
        gross_pnl REAL NOT NULL,
        net_pnl REAL NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT NOT NULL,
        holding_time_seconds REAL NOT NULL,
        exit_reason TEXT NOT NULL,
        regime TEXT NOT NULL,
        signal_score REAL NOT NULL,
        expected_edge REAL NOT NULL DEFAULT 0,
        estimated_cost REAL NOT NULL DEFAULT 0,
        actual_cost REAL NOT NULL DEFAULT 0,
        feature_snapshot TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy)",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades (exit_time)",
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        regime TEXT NOT NULL,
        strategy TEXT,
        decision TEXT NOT NULL,
        score REAL,
        confidence REAL,
        expected_edge REAL,
        estimated_cost REAL,
        reject_reason TEXT,
        detail TEXT,
        feature_snapshot TEXT,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_signals_decision ON signals (decision)",
    """
    CREATE TABLE IF NOT EXISTS heartbeats (
        component TEXT PRIMARY KEY,
        last_seen TEXT NOT NULL,
        detail TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        event TEXT NOT NULL,
        message TEXT,
        data TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_event ON events (event, timestamp)",
    """
    CREATE TABLE IF NOT EXISTS strategy_health (
        strategy TEXT PRIMARY KEY,
        trade_count INTEGER DEFAULT 0,
        win_rate REAL DEFAULT 0,
        profit_factor REAL DEFAULT 0,
        expectancy REAL DEFAULT 0,
        average_win REAL DEFAULT 0,
        average_loss REAL DEFAULT 0,
        max_drawdown REAL DEFAULT 0,
        recent_performance REAL DEFAULT 0,
        weight REAL DEFAULT 1,
        enabled INTEGER DEFAULT 1,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        git_commit TEXT,
        config_hash TEXT,
        dataset_hash TEXT,
        mode TEXT,
        created_at TEXT NOT NULL,
        summary TEXT
    )
    """,
]
