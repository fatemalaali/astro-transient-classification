"""The ingest process: poll a source, enqueue, let the worker classify.

Two threads. The **polling thread** (this module's ``run``) owns the source and
never blocks on inference. The **worker thread** owns the database. A bounded
queue between them is the backpressure boundary.

Backpressure policy, recorded rather than hidden: when the queue is full the
*newest* record is dropped and ``dropped_total`` increments, which the live
indicator displays. Unbounded queueing would turn a transient stall into memory
growth while the UI still looked healthy; sampling would silently bias the class
distribution on screen, which is precisely the kind of quiet distortion this
thesis argues against.

Shutdown is graceful: SIGINT/SIGTERM (and SIGBREAK on Windows) set an event, the
poll loop finishes its current record, the queue drains with a deadline, offsets
commit, and the consumer closes.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time

from demo.adapters import build_source
from demo.config import Settings
from demo.ingest.health import HealthTracker
from demo.ingest.worker import InferenceWorker
from demo.models import NormalisedAlert

log = logging.getLogger("demo.consumer")

DRAIN_DEADLINE_S = 10.0


class ConsumerService:
    """Owns the source, the queue, the worker and the shutdown protocol."""

    def __init__(self, settings: Settings, limit: int | None = None, **source_kwargs):
        self.settings = settings
        self.shutdown = threading.Event()
        self.tracker = HealthTracker()
        self.queue: "queue.Queue[NormalisedAlert]" = queue.Queue(
            maxsize=settings.queue_maxsize
        )
        self.source = build_source(
            settings, self.shutdown, limit=limit, **source_kwargs
        )
        self.worker = InferenceWorker(
            settings, self.queue, self.shutdown, self.tracker
        )
        # Heartbeat. Without this, a consumer on an idle stream never samples
        # health, writes no rows, and the dashboard declares it dead after 60s —
        # which is exactly what happens during Palomar daylight, when a healthy
        # consumer is *supposed* to be quiet.
        self.source.on_idle = self._heartbeat

    def _heartbeat(self) -> None:
        """Sample source health from the polling thread. Cheap and rate-limited.

        Must run here rather than on a timer thread: the Kafka consumer is not
        safe for concurrent use, so only the thread that polls it may query lag.
        """
        if self.tracker.due_for_sample():
            self.tracker.sample(self.source)
            self.tracker.set_queue_depth(self.queue.qsize())
            self.tracker.set_decode_failures(
                getattr(self.source, "_decode_failures", 0)
            )

    # ------------------------------------------------------------------ #
    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("signal %s received — shutting down", signum)
            self.shutdown.set()

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # pragma: no cover - non-main thread
                    pass

    # ------------------------------------------------------------------ #
    def preflight(self, assume_yes: bool = False) -> bool:
        """Report the backlog before the first poll, and gate on it.

        On a first connection there can be up to four days of accumulated
        alerts across all subscribed topics. ``live`` mode seeks to the end so
        this never bites, but ``catchup`` genuinely replays and must be
        deliberate.
        """
        if self.settings.mode != "catchup":
            return True
        backlog = {}
        try:
            backlog = self.source.backlog()
        except Exception as exc:
            log.warning("could not read the backlog: %s", exc)
            return True
        total = sum(backlog.values())
        if not total:
            return True
        log.warning("backlog before first poll: %s (total %d)", backlog, total)
        if total > self.settings.backlog_confirm_threshold and not assume_yes:
            log.error(
                "backlog of %d exceeds DEMO_BACKLOG_CONFIRM_THRESHOLD=%d; "
                "re-run with --yes to proceed, or use --mode live to skip it",
                total,
                self.settings.backlog_confirm_threshold,
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    def run(self, assume_yes: bool = False) -> int:
        self.settings.ensure_dirs()
        self.install_signal_handlers()
        if not self.preflight(assume_yes=assume_yes):
            return 2

        self.worker.start()
        processed = 0
        max_backlog = (
            self.settings.max_backlog if self.settings.mode == "catchup" else None
        )

        try:
            for alert in self.source.stream():
                if self.shutdown.is_set():
                    break
                self.tracker.record_consumed()

                if self.queue.full():
                    # Drop the newest and say so. Never silently.
                    self.tracker.record_drop()
                    log.warning(
                        "queue full (%d) — dropped %s; inference is behind the stream",
                        self.settings.queue_maxsize,
                        alert.stamp_key(),
                    )
                else:
                    self.queue.put(alert)
                    # Commit only after a safe handoff, so a crash replays the
                    # in-flight alert rather than losing it.
                    self.source.commit(alert)
                    processed += 1

                self.tracker.set_queue_depth(self.queue.qsize())
                self.tracker.set_decode_failures(
                    getattr(self.source, "_decode_failures", 0)
                )
                if self.tracker.due_for_sample():
                    self.tracker.sample(self.source)

                if max_backlog is not None and processed >= max_backlog:
                    log.info("reached --max-backlog=%d; stopping", max_backlog)
                    break
        except KeyboardInterrupt:  # pragma: no cover
            log.info("interrupted")
        except Exception:
            log.exception("poll loop failed")
            return 1
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        log.info("draining %d queued alert(s)...", self.queue.qsize())
        deadline = time.monotonic() + DRAIN_DEADLINE_S
        while not self.queue.empty() and time.monotonic() < deadline:
            time.sleep(0.2)
        if not self.queue.empty():
            log.warning(
                "%d alert(s) still queued at the drain deadline", self.queue.qsize()
            )
        self.shutdown.set()
        self.worker.join(timeout=DRAIN_DEADLINE_S)
        try:
            self.tracker.sample(self.source)
        except Exception:  # pragma: no cover
            pass
        self.source.close()
        log.info("shutdown complete")
