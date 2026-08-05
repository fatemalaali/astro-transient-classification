"""Ingest + classify. Run this in one window, ``demo.run_api`` in another.

    python -m demo.run_consumer                        # live Kafka (default)
    python -m demo.run_consumer --mode replay          # pinned offsets
    python -m demo.run_consumer --mode catchup --yes   # drain the backlog
    python -m demo.run_consumer --mode rest            # no credentials needed
    python -m demo.run_consumer --mode offline --loop  # archived .avro, no network

Runs as a **separate process** from the API on purpose. The poll loop must
never be blocked by an HTTP handler, a crash in the web layer must not lose
stream position, and ``uvicorn --reload`` spawns worker subprocesses — which on
Windows would instantiate one Kafka consumer per worker in the same consumer
group, producing duplicate rows and partition churn.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow `python demo/run_consumer.py` as well as `python -m demo.run_consumer`.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.config import MODES, get_settings  # noqa: E402
from demo.ingest import ConsumerService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo.run_consumer",
        description="Consume ZTF alerts, classify them, write to the demo database.",
    )
    parser.add_argument("--mode", choices=MODES, help="override DEMO_MODE")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N alerts"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="proceed even if the backlog exceeds the confirmation threshold",
    )
    parser.add_argument(
        "--stubs",
        action="store_true",
        help="use seeded stub branches instead of the trained models",
    )
    parser.add_argument(
        "--topics", help="comma-separated topic override (Kafka modes only)"
    )
    parser.add_argument(
        "--replay-path", help="offline mode: folder or file of archived .avro alerts"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="offline mode: >0 replays the real inter-alert gaps divided by this",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="offline mode: restart from the beginning when exhausted",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.mode:
        os.environ["DEMO_MODE"] = args.mode
    if args.stubs:
        os.environ["DEMO_USE_STUBS"] = "1"
    if args.topics:
        os.environ["DEMO_TOPICS"] = args.topics

    settings = get_settings(refresh=True)
    settings.ensure_dirs()

    source_kwargs: dict = {}
    if settings.mode == "offline":
        if args.replay_path:
            source_kwargs["path"] = Path(args.replay_path)
        source_kwargs["speed"] = args.speed
        source_kwargs["loop"] = args.loop

    logging.getLogger("demo").info(
        "mode=%s topics=%s stubs=%s db=%s",
        settings.mode,
        ",".join(settings.topics) if settings.mode not in ("rest", "offline") else "-",
        settings.use_stubs,
        settings.db_path,
    )

    service = ConsumerService(settings, limit=args.limit, **source_kwargs)
    return service.run(assume_yes=args.yes)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
