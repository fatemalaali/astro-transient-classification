"""SQLite storage: schema, connections, and one-time bootstrap."""

from demo.storage.bootstrap import bootstrap
from demo.storage.db import connect, init_db

__all__ = ["bootstrap", "connect", "init_db"]
