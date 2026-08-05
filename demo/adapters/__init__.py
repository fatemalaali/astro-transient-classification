"""Ingestion adapters. All produce :class:`demo.models.NormalisedAlert`.

``FinkKafkaAdapter`` is the primary path; ``RestPollAdapter`` and
``ReplayAdapter`` are fallbacks. Imports are lazy so that a machine without
``fink-client`` installed can still run the REST and replay paths.
"""

from __future__ import annotations

import threading
from typing import Any

from demo.config import Settings


def build_source(settings: Settings, shutdown: threading.Event, **kwargs: Any):
    """Return the adapter that ``settings.mode`` selects."""
    if settings.mode in ("live", "replay", "catchup"):
        from demo.adapters.fink_kafka import FinkKafkaAdapter

        return FinkKafkaAdapter(settings, shutdown=shutdown, **kwargs)
    if settings.mode == "rest":
        from demo.adapters.fink_rest import RestPollAdapter

        return RestPollAdapter(settings, shutdown=shutdown, **kwargs)
    if settings.mode == "offline":
        from demo.adapters.replay import ReplayAdapter

        return ReplayAdapter(settings, shutdown=shutdown, **kwargs)
    raise ValueError(f"unknown mode {settings.mode!r}")
