"""Health tracking for the live-stream indicator.

The Kafka consumer is not thread-safe for concurrent use, so lag is sampled on
the **polling thread** and handed to the writer thread through this object. The
writer thread owns the only database connection, which keeps SQLite's
single-writer model intact.

The counters here are the honesty counters: dropped alerts and decode failures
are displayed rather than hidden, because a demo that silently discards data
while looking healthy is worse than one that admits a gap.
"""

from __future__ import annotations

import threading
import time

from demo.models import SourceHealth


class HealthTracker:
    """Thread-safe counters plus the most recent source snapshot."""

    def __init__(self, sample_interval_s: float = 2.0) -> None:
        self._lock = threading.Lock()
        self._snapshot: SourceHealth | None = None
        self._last_sample = 0.0
        self.sample_interval_s = sample_interval_s
        self.queue_depth = 0
        self.dropped_total = 0
        self.decode_failures = 0
        self.consumed_total = 0

    # --- polling thread ------------------------------------------------ #
    def due_for_sample(self) -> bool:
        return (time.monotonic() - self._last_sample) >= self.sample_interval_s

    def sample(self, source) -> None:
        """Take a health snapshot from the source. Polling thread only."""
        try:
            snapshot = source.health()
        except Exception as exc:  # pragma: no cover - a lag query must never kill us
            snapshot = SourceHealth(
                connected=False,
                mode=getattr(source, "name", "unknown"),
                error=f"{type(exc).__name__}: {exc}",
            )
        with self._lock:
            self._snapshot = snapshot
            self._last_sample = time.monotonic()

    def record_drop(self) -> None:
        with self._lock:
            self.dropped_total += 1

    def record_consumed(self) -> None:
        with self._lock:
            self.consumed_total += 1

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self.queue_depth = depth

    def set_decode_failures(self, count: int) -> None:
        with self._lock:
            self.decode_failures = count

    # --- writer thread ------------------------------------------------- #
    def snapshot(self) -> tuple[SourceHealth | None, dict]:
        with self._lock:
            return self._snapshot, {
                "queue_depth": self.queue_depth,
                "dropped_total": self.dropped_total,
                "decode_failures": self.decode_failures,
                "consumed_total": self.consumed_total,
            }
