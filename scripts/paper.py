"""Run the live loop in SHADOW or PAPER mode.

SHADOW makes and logs decisions but submits nothing; PAPER submits to the
Alpaca paper endpoint. Both use the exact decision stack the backtest uses.

Example::

    python scripts/paper.py --mode shadow --max-iterations 5
    python scripts/paper.py --mode paper
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.enums import TradingMode  # noqa: E402
from app.live.loop import TradingLoop, build_live_context  # noqa: E402

logger = logging.getLogger("scripts.paper")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bot in shadow/paper mode")
    parser.add_argument("--mode", choices=["shadow", "paper"], default="shadow")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="stop after N cycles (useful for a dry run)")
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    ctx = build_live_context()
    # Enforce the operator's intent explicitly; paper NEVER means live.
    ctx.config.settings.mode = (
        TradingMode.SHADOW if args.mode == "shadow" else TradingMode.PAPER
    )

    loop = TradingLoop(ctx)
    if args.poll_seconds:
        loop.poll_seconds = args.poll_seconds

    if ctx.websocket is not None:
        try:
            ctx.websocket.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("websocket unavailable (%s); using REST", exc)
    try:
        loop.run(max_iterations=args.max_iterations)
    finally:
        if ctx.websocket is not None:
            try:
                ctx.websocket.stop()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
