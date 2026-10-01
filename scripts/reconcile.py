"""Reconcile local state against the broker and optionally repair it.

The broker is authoritative. This is safe to run any time and is the recommended
first action after an unexpected shutdown.

Example::

    python scripts/reconcile.py --repair
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.live.loop import build_live_context  # noqa: E402

logger = logging.getLogger("scripts.reconcile")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile bot state with the broker")
    parser.add_argument("--repair", action="store_true", help="apply repairs to local state")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    ctx = build_live_context()
    try:
        report = ctx.reconciler.reconcile(ctx.broker, ctx.state, auto_repair=args.repair)
    except Exception as exc:  # noqa: BLE001
        logger.error("reconciliation failed: %s", exc)
        return 1

    print(json.dumps(report.as_dict(), indent=2, default=str))
    if report.critical:
        logger.error("critical discrepancies found; trading should remain halted")
        return 2
    if not report.clean:
        logger.warning("reconciliation completed with %d non-critical issue(s)",
                       len(report.issues))
        return 0
    logger.info("state is consistent with the broker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
