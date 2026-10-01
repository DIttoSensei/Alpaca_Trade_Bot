"""Healthcheck for schedulers and operators.

Performs a bounded, read-mostly check of the system: database, broker
connectivity, account freshness and circuit-breaker state. Exits non-zero when
the system is not fit to trade so it can be wired into cron / systemd / k8s.

Example::

    python scripts/healthcheck.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.live.loop import build_live_context  # noqa: E402

logger = logging.getLogger("scripts.healthcheck")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check system health")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    parser.add_argument("--sync-account", action="store_true",
                        help="refresh the account snapshot before checking")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    ctx = build_live_context()

    connectivity_ok = True
    connectivity_error = None
    if args.sync_account:
        try:
            ctx.account_sync.sync(update_positions=True)
        except Exception as exc:  # noqa: BLE001
            connectivity_ok = False
            connectivity_error = str(exc)
            logger.error("account sync failed: %s", exc)

    report = ctx.health.check()
    blocked, why = ctx.breakers.entries_blocked()

    payload = {
        "status": report.status,
        "can_trade": ctx.health.can_trade(report),
        "database_healthy": ctx.state.db.healthy(),
        "broker_connectivity": connectivity_ok,
        "broker_error": connectivity_error,
        "entries_blocked": blocked,
        "entries_blocked_reason": why,
        "components": report.components,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"status:          {payload['status']}")
        print(f"can_trade:       {payload['can_trade']}")
        print(f"database:        {'ok' if payload['database_healthy'] else 'FAIL'}")
        print(f"broker:          {'ok' if connectivity_ok else 'FAIL ' + str(connectivity_error)}")
        print(f"entries_blocked: {blocked} ({why})")

    # Exit codes: 0 healthy, 1 degraded, 2 emergency/unusable.
    if report.status == "HEALTHY" and connectivity_ok:
        return 0
    if report.status == "DEGRADED":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
