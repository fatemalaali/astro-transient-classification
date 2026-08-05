"""Shared FastAPI dependencies.

The API is strictly **read-only**: it opens the database read-only where
possible so a bug in a route can never corrupt the consumer's data, and it
holds no long-lived connection, because SQLite connections are not safe to
share across threads.
"""

from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import HTTPException

from demo.config import Settings, get_settings
from demo.storage import db as store


def settings() -> Settings:
    return get_settings()


def get_db() -> Iterator[sqlite3.Connection]:
    config = get_settings()
    if not config.db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"no database at {config.db_path}. Start the consumer first: "
                "python -m demo.run_consumer --mode offline  (or --mode rest)"
            ),
        )
    conn = store.connect(config, readonly=True)
    try:
        yield conn
    finally:
        conn.close()
