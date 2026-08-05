"""ReplayAdapter — offline ingestion from saved Avro files.

Kafka offset rewind covers *reproducibility*; this covers *no network at all*.
It is the demo-day insurance policy: with a folder of alerts archived by
``fink_consumer --save``, the whole pipeline runs with the cable pulled.

Alerts saved by ``fink_consumer`` are Avro **object container** files (schema
embedded in the header), so ``fastavro.reader`` reads them directly and
fink-client is not required. ``fink_client.avro_utils.AlertReader`` is used when
available because it also handles the multi-alert files the client writes.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

from demo.adapters.fink_kafka import FinkKafkaAdapter
from demo.config import Settings
from demo.models import NormalisedAlert, SourceHealth, utcnow

log = logging.getLogger("demo.replay")


def read_avro(path: Path) -> list[dict]:
    """Read one ``.avro`` container file into a list of alert dicts."""
    try:
        from fink_client.avro_utils import AlertReader

        return list(AlertReader(str(path)).to_list())
    except Exception:
        pass
    try:
        import fastavro

        with open(path, "rb") as handle:
            return list(fastavro.reader(handle))
    except Exception as exc:
        log.warning("could not read %s: %s", path.name, exc)
        return []


class ReplayAdapter:
    """Streams archived alerts, optionally pacing them like the original stream."""

    name = "replay"

    def __init__(
        self,
        settings: Settings,
        path: Path | None = None,
        speed: float = 0.0,
        shutdown: threading.Event | None = None,
        limit: int | None = None,
        loop: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        speed
            0 = as fast as possible (development). >0 = replay the real
            inter-alert gaps divided by ``speed``, capped at 5 s, so a recorded
            session *looks* live in a demo without anyone waiting minutes
            between alerts.
        loop
            Restart from the beginning when exhausted — keeps an offline demo
            alive indefinitely.
        """
        self.settings = settings
        self.path = Path(path or settings.raw_alerts_dir)
        self.speed = speed
        self.shutdown = shutdown or threading.Event()
        self.limit = limit
        self.loop = loop
        self._last_alert_utc: datetime | None = None
        self._emitted = 0
        self.on_idle = None  # heartbeat; see FinkKafkaAdapter

    def files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        return sorted(self.path.glob("*.avro"))

    def _records(self) -> Iterator[NormalisedAlert]:
        for path in self.files():
            if self.shutdown.is_set():
                return
            for alert in read_avro(path):
                record = FinkKafkaAdapter.normalise(alert, msg=None, topic=None)
                # Rewrite the provenance: this came off disk, not off Kafka.
                record.source = "replay"
                record.topic = record.topic or f"replay:{path.stem}"
                record.raw_packet_ref = str(path)
                yield record

    def stream(self) -> Iterator[NormalisedAlert]:
        previous_jd: float | None = None
        while not self.shutdown.is_set():
            emitted_this_pass = 0
            for record in self._records():
                if self.shutdown.is_set():
                    return
                if self.limit is not None and self._emitted >= self.limit:
                    return
                if self.speed > 0 and previous_jd is not None:
                    gap = (record.jd - previous_jd) * 86400.0 / self.speed
                    if gap > 0:
                        self.shutdown.wait(min(gap, 5.0))
                previous_jd = record.jd
                record.received_utc = utcnow()
                self._last_alert_utc = record.received_utc
                self._emitted += 1
                emitted_this_pass += 1
                yield record
            if not self.loop or emitted_this_pass == 0:
                return
            log.info("replay exhausted after %d alerts; looping", self._emitted)
            previous_jd = None
            self.shutdown.wait(2.0)

    def commit(self, record: NormalisedAlert) -> None:
        """No-op: nothing to commit when reading from disk."""

    def health(self) -> SourceHealth:
        return SourceHealth(
            connected=True,
            mode="offline",
            topics=("replay",),
            last_alert_utc=self._last_alert_utc,
            is_live_stream=False,  # never claim "live" for a recording
        )

    def close(self) -> None:
        pass
