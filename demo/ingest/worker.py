"""The inference worker: dequeue -> classify -> persist.

Runs in its own thread and owns the **only** database connection, which is what
keeps SQLite's single-writer model intact while the API reads concurrently
through WAL.

Separating this from the poll loop matters: inference must never be able to
stall the Kafka poll, because lag is a metric the demo displays. If the model
gets slow, the queue depth rises visibly instead of the stream quietly falling
behind.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from demo.config import Settings
from demo.ingest.health import HealthTracker
from demo.inference import InferenceEngine
from demo.models import NormalisedAlert, iso
from demo.storage import db as store

log = logging.getLogger("demo.worker")


class InferenceWorker(threading.Thread):
    """Consumes normalised alerts, writes classified rows."""

    #: How often to retry alerts that are missing their tabular branch.
    UPGRADE_INTERVAL_S = 60.0
    #: How many to attempt per pass. Small, so a backlog never blocks the loop.
    UPGRADE_BATCH = 10
    #: Minimum gap between attempts on the same object. ALeRCE featurises a new
    #: transient on its own cadence (hours), so retrying sooner is wasted work.
    UPGRADE_RETRY_OBJECT_S = 900.0

    def __init__(
        self,
        settings: Settings,
        work_queue: "queue.Queue[NormalisedAlert]",
        shutdown: threading.Event,
        tracker: HealthTracker,
        engine: InferenceEngine | None = None,
    ) -> None:
        super().__init__(name="inference-worker", daemon=True)
        self.settings = settings
        self.queue = work_queue
        self.shutdown = shutdown
        self.tracker = tracker
        self.engine = engine or InferenceEngine(settings)
        self.processed = 0
        self.errors = 0
        self._conn = None
        self._last_health_write = 0.0
        self._last_upgrade_pass = 0.0
        self._upgrade_attempts: dict[str, float] = {}
        self.upgraded_total = 0

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._conn = store.init_db(self.settings)
        log.info("worker started; database at %s", self.settings.db_path)
        try:
            while not self.shutdown.is_set() or not self.queue.empty():
                try:
                    alert = self.queue.get(timeout=0.5)
                except queue.Empty:
                    self._write_health()
                    # Idle time is free time: retry the tabular branch for
                    # alerts that arrived without features. No manual backfill.
                    self._upgrade_pass()
                    continue
                try:
                    self.handle(alert)
                except Exception:
                    self.errors += 1
                    log.exception("failed to handle alert %s", alert.stamp_key())
                finally:
                    self.queue.task_done()
                self._write_health()
        finally:
            self._write_health(force=True)
            if self._conn is not None:
                self._conn.close()
            self.engine.close()
            log.info(
                "worker stopped after %d alert(s), %d error(s)",
                self.processed,
                self.errors,
            )

    # ------------------------------------------------------------------ #
    def handle(self, alert: NormalisedAlert) -> None:
        stamp_path = self.save_stamps(alert)
        prediction = self.engine.classify(alert)

        store.save_alert(self._conn, alert, stamp_path=stamp_path)
        store.save_photometry(self._conn, alert)
        store.save_prediction(self._conn, prediction)
        self._conn.commit()

        self.processed += 1
        self.log_line(alert, prediction)

        log.info(
            "%s %s -> %s (%.2f) mode=%s lat=%.0fms features=%s",
            alert.topic or alert.source,
            alert.object_id,
            prediction.predicted_class or "unclassified",
            prediction.confidence or 0.0,
            prediction.fusion_mode,
            prediction.t_pipeline_ms,
            prediction.feature_provenance.source
            if prediction.feature_provenance
            else "n/a",
        )

    def save_stamps(self, alert: NormalisedAlert) -> str | None:
        """Persist the triplet as .npy beside the database.

        Stamps stay out of SQLite: a float32 triplet is ~47 KB, and keeping the
        database small is what makes it portable enough to copy to a USB stick
        before a viva.
        """
        stack = alert.stamp_stack()
        if stack is None:
            return None
        try:
            self.settings.stamps_dir.mkdir(parents=True, exist_ok=True)
            path: Path = self.settings.stamps_dir / f"{alert.stamp_key()}.npy"
            np.save(path, stack.astype(np.float32))
            return str(path.name)
        except Exception:  # pragma: no cover
            log.exception("could not save stamps for %s", alert.stamp_key())
            return None

    def log_line(self, alert: NormalisedAlert, prediction) -> None:
        """Structured per-alert JSON, one line, appended to inference.log."""
        record = {
            "object_id": alert.object_id,
            "candid": alert.candid,
            "source": alert.source,
            "topic": alert.topic,
            "partition": alert.partition,
            "offset": alert.offset,
            "kafka_ts": iso(alert.kafka_ts_utc),
            "emitted_utc": iso(alert.emitted_utc),
            "received_utc": iso(alert.received_utc),
            "n_det": alert.n_det,
            "n_nondet": alert.n_nondet,
            "cutout_status": alert.cutout_status,
            "feature_provenance": (
                prediction.feature_provenance.source
                if prediction.feature_provenance
                else None
            ),
            "n_features_present": (
                prediction.feature_provenance.n_present
                if prediction.feature_provenance
                else None
            ),
            "p_tab": _tolist(prediction.p_tab),
            "p_img": _tolist(prediction.p_img),
            "p_fused": _tolist(prediction.p_fused),
            "predicted_class": prediction.predicted_class,
            "confidence": prediction.confidence,
            "fusion_mode": prediction.fusion_mode,
            "branch_disagree": prediction.branch_disagree,
            "fusion_flips": prediction.fusion_flips,
            "t_stamp_ms": round(prediction.t_stamp_ms, 3),
            "t_tab_ms": round(prediction.tabular.elapsed_ms, 3)
            if prediction.tabular
            else None,
            "t_img_ms": round(prediction.image.elapsed_ms, 3)
            if prediction.image
            else None,
            "t_fuse_ms": round(prediction.fusion.elapsed_ms, 3)
            if prediction.fusion
            else None,
            "t_pipeline_ms": round(prediction.t_pipeline_ms, 3),
            "t_broker_to_classified_ms": prediction.t_broker_to_classified_ms,
            "t_emitted_to_classified_s": prediction.t_emitted_to_classified_s,
            "model_versions": prediction.model_versions,
            "split_id": prediction.split_id,
        }
        try:
            with open(self.settings.inference_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception:  # pragma: no cover
            log.debug("could not append to the inference log")

    def _upgrade_pass(self) -> None:
        """Re-classify a few alerts that are missing their tabular branch.

        Two things routinely leave an alert image-only at ingest: the feature
        service being unreachable, and — far more often — the object being too
        new for ALeRCE to have featurised it yet. Both resolve themselves given
        time, so the fix is to keep trying rather than to make it a manual step.

        Runs only when the queue is empty, so it can never delay a live alert,
        and processes a small batch at a time so a large backlog is worked off
        gradually instead of blocking the loop. Runs in the worker thread, which
        owns the only database connection, so the single-writer model holds.
        """
        now = time.monotonic()
        if (now - self._last_upgrade_pass) < self.UPGRADE_INTERVAL_S:
            return
        self._last_upgrade_pass = now

        status = self.engine.resolver.status()
        if status["state"] == "disabled":
            return
        if status["state"] == "blocked":
            # The resolver's own breaker decides when to retry the network;
            # asking it here would just burn requests against a closed door.
            return

        # Skip objects tried recently. Most pending alerts are brand-new
        # transients ALeRCE has not featurised yet, and they stay that way for
        # hours — re-fetching them every minute would burn a request per object
        # per pass and, because the pending list is ordered, would starve older
        # alerts that might now be resolvable. The cooldown both throttles and
        # rotates.
        now_wall = time.monotonic()
        pending = [
            (candid, oid)
            for candid, oid in store.upgradeable_candids(self._conn)
            if (now_wall - self._upgrade_attempts.get(oid, 0.0))
            >= self.UPGRADE_RETRY_OBJECT_S
        ][: self.UPGRADE_BATCH]
        if not pending:
            return

        upgraded = 0
        for candid, _oid in pending:
            self._upgrade_attempts[_oid] = now_wall
            if self.shutdown.is_set() or not self.queue.empty():
                break  # live alerts always take priority
            alert = store.load_alert(self._conn, candid, self.settings)
            if alert is None:
                continue
            prediction = self.engine.classify(alert)
            if prediction.fusion_mode == "both":
                store.save_prediction(self._conn, prediction)
                upgraded += 1
                self._upgrade_attempts.pop(alert.object_id, None)
                log.info(
                    "upgraded %s to two-branch fusion -> %s (%.2f)",
                    alert.object_id,
                    prediction.predicted_class,
                    prediction.confidence or 0.0,
                )
        if upgraded:
            self._conn.commit()
            self.upgraded_total += upgraded

    def _write_health(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_health_write) < 2.0:
            return
        self._last_health_write = now
        snapshot, counters = self.tracker.snapshot()
        if snapshot is None or self._conn is None:
            return
        try:
            store.save_health(
                self._conn,
                snapshot,
                alerce=self.engine.resolver.status(),
                **counters,
            )
            self._conn.commit()
        except Exception:  # pragma: no cover
            log.debug("health write failed", exc_info=True)


def _tolist(arr) -> list | None:
    return None if arr is None else [round(float(v), 6) for v in np.asarray(arr).ravel()]
