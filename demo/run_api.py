"""Serve the dashboard and the read-only API.

    python -m demo.run_api

``--reload`` is off by default and should stay off during a demo: uvicorn's
reloader spawns worker subprocesses, and on Windows that means the module is
re-imported per worker. It is harmless for this read-only service but would be
fatal for the consumer, so the two entry points keep the same discipline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.config import get_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo.run_api", description="Serve the demo dashboard and API."
    )
    parser.add_argument("--host", help="override DEMO_API_HOST")
    parser.add_argument("--port", type=int, help="override DEMO_API_PORT")
    parser.add_argument("--reload", action="store_true", help="development only")
    parser.add_argument("--open", action="store_true", help="open a browser window")
    parser.add_argument(
        "--stubs", action="store_true", help="report stub models on /api/models"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.stubs:
        os.environ["DEMO_USE_STUBS"] = "1"

    settings = get_settings(refresh=True)
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    if not settings.db_path.exists():
        logging.getLogger("demo.api").warning(
            "no database at %s yet — start the consumer in another window: "
            "python -m demo.run_consumer --mode offline",
            settings.db_path,
        )

    url = f"http://{host}:{port}/"
    logging.getLogger("demo.api").info("dashboard on %s", url)
    if args.open:
        webbrowser.open(url)

    import uvicorn

    uvicorn.run(
        "demo.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
