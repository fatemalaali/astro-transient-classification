"""FinkKafkaAdapter — the primary ingestion path.

Wraps ``fink_client.consumer.AlertConsumer``. Two things about that class shape
this module (both verified against fink-client 11.0 source, 2026-08-05):

1. ``survey`` is a **required** third positional argument. The example on
   https://doc.ztf.fink-broker.org/services/livestream/ predates it (that page
   is tested against v8.8) and would raise ``TypeError`` here.

2. ``AlertConsumer.poll()`` discards the underlying ``confluent_kafka.Message``,
   so partition, offset and the broker timestamp are unrecoverable from it. We
   therefore drive ``consumer._consumer.poll()`` ourselves and hand the message
   to the public ``consumer.process_message()``. That is one level of private
   access, isolated to :meth:`_poll_once` and guarded by a pinned
   ``fink-client==11.0`` in requirements-demo.txt.

The Avro schema travels in the Kafka message **key**, not the value, which is
why schema drift surfaces as an ``IndexError`` from deep inside fastavro rather
than as a clean error. :meth:`_poll_once` catches it, quarantines the payload
and keeps the loop alive.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from demo.adapters.base import (
    big_int,
    build_lightcurve,
    extract_broker_meta,
    extract_candidate_fields,
    num,
)
from demo.adapters.cutouts import decode_fink_packet
from demo.adapters.offsets import ReplayManifest, make_on_assign, query_lag
from demo.config import Settings
from demo.models import NormalisedAlert, SourceHealth, parse_iso, utcnow

log = logging.getLogger("demo.fink_kafka")


def load_kafka_config(topics: tuple[str, ...] = ()) -> dict:
    """``ztf_credentials.yml`` -> the dict ``AlertConsumer`` expects.

    Shared by the adapter and by ``scripts/record_replay_manifest.py`` so there
    is exactly one place that knows the translation. See
    :meth:`FinkKafkaAdapter._load_config` for why each line is the way it is.
    """
    from fink_client.configuration import load_credentials

    try:
        raw = dict(load_credentials("ztf"))
    except TypeError:  # pragma: no cover - pre-v10 signature
        raw = dict(load_credentials())

    conf: dict[str, Any] = {
        "bootstrap.servers": raw["servers"],
        "group.id": raw["group_id"],
    }
    if raw.get("password") is not None:
        # Both keys are required for _get_kafka_config to switch SASL on.
        conf["username"] = raw.get("username")
        conf["password"] = raw["password"]

    registered = tuple(raw.get("mytopics") or ())
    if topics and registered and set(registered) != set(topics):
        log.warning(
            "configured topics differ from those registered in "
            "ztf_credentials.yml.\n  configured : %s\n  registered : %s\n"
            "Subscribing to the configured set; re-run fink_client_register "
            "if that is not what you want.",
            ", ".join(topics),
            ", ".join(registered),
        )
    return conf


class FinkKafkaAdapter:
    """Long-lived Kafka consumer producing :class:`NormalisedAlert` records."""

    name = "fink_kafka"

    def __init__(
        self,
        settings: Settings,
        shutdown: threading.Event | None = None,
        limit: int | None = None,
    ) -> None:
        self.settings = settings
        self.shutdown = shutdown or threading.Event()
        self.limit = limit
        self._consumer: Any = None
        self._connected = False
        self._error: str | None = None
        self._last_alert_utc: datetime | None = None
        self._decode_failures = 0
        self._consumed = 0
        self._consecutive_errors = 0
        #: Called on every poll that returns nothing. The consumer service
        #: uses it as a heartbeat so an idle stream still reports health.
        self.on_idle = None

    # ------------------------------------------------------------------ #
    # connection
    # ------------------------------------------------------------------ #
    def _load_config(self) -> dict:
        """Read ``~/.finkclient/ztf_credentials.yml`` and translate it for Kafka.

        This mirrors ``fink_client/scripts/fink_consumer.py`` exactly, which is
        the only reliable specification — the published documentation does not
        match the client's behaviour. What the client actually sends is::

            {"bootstrap.servers": conf["servers"], "group.id": conf["group_id"]}

        and it adds ``password`` **only** when the stored value is not ``None``.

        Three consequences, each of which cost a debugging cycle:

        1. The YAML key is ``servers``; ``_get_kafka_config`` reads only
           ``bootstrap.servers`` and silently falls back to
           ``localhost:9093,9094,9095`` otherwise. The failure looks like a dead
           local Kafka rather than a config error.
        2. **No SASL is negotiated at all** on the current ZTF deployment.
           ``_get_kafka_config`` enables SASL only when *both* ``username`` and
           ``password`` are present, and the client never sends either when the
           password is null. This is consistent with the documented handshake
           error, whose broker-supported-mechanisms list is empty — the broker
           is not offering SASL. Access is controlled by the credentials Fink
           issues out of band, not by a Kafka-level challenge.
        3. Passing ``password: None`` explicitly is worse than omitting it:
           librdkafka rejects it with
           ``sasl.username and sasl.password must be set``.

        So the username is deliberately **not** forwarded. It identifies you to
        Fink and is what their ACLs are keyed on; it is not a Kafka credential.
        """
        conf = load_kafka_config(self.settings.topics)

        if self.settings.mode == "replay":
            # A fresh group id means no committed offsets, so the pinned
            # manifest is the only thing deciding where we start.
            run_id = time.strftime("%Y%m%d%H%M%S")
            conf["group.id"] = f"{conf['group.id']}-replay-{run_id}"
        return conf

    def connect(self) -> None:
        from fink_client.consumer import AlertConsumer

        conf = self._load_config()
        manifest = None
        if self.settings.mode == "replay":
            if not self.settings.replay_manifest.exists():
                raise FileNotFoundError(
                    f"replay mode needs {self.settings.replay_manifest}; "
                    "generate it with scripts/record_replay_manifest.py"
                )
            manifest = ReplayManifest.load(self.settings.replay_manifest)
            if manifest.total_limit and self.limit is None:
                self.limit = manifest.total_limit

        topics = list(self.settings.topics)
        log.info("connecting to Fink Kafka, mode=%s topics=%s", self.settings.mode, topics)
        self._consumer = AlertConsumer(
            topics,
            conf,
            "ztf",
            on_assign=make_on_assign(self.settings.mode, manifest),
        )
        self._connected = True
        self._error = None

    # ------------------------------------------------------------------ #
    # polling
    # ------------------------------------------------------------------ #
    def _poll_once(self) -> NormalisedAlert | None:
        """One poll cycle. Returns None on timeout or a handled error."""
        try:
            msg = self._consumer._consumer.poll(self.settings.poll_timeout_s)
        except Exception as exc:
            self._handle_transport_error(exc)
            return None

        if msg is None:
            self._consecutive_errors = 0
            return None  # idle: normal, ZTF only observes at night

        if msg.error():
            self._handle_kafka_message_error(msg)
            return None

        try:
            topic, alert, _key = self._consumer.process_message(msg)
        except Exception as exc:
            # IndexError from fastavro means the schema in the key does not
            # match the payload. Quarantine and carry on — a schema change must
            # never take the demo down mid-viva.
            self._decode_failures += 1
            self._quarantine(msg, exc)
            return None

        self._consecutive_errors = 0
        if alert is None:
            return None

        record = self.normalise(alert, msg, topic)
        self._last_alert_utc = record.received_utc
        self._consumed += 1
        return record

    def _handle_transport_error(self, exc: Exception) -> None:
        self._consecutive_errors += 1
        self._error = f"{type(exc).__name__}: {exc}"
        backoff = min(2**self._consecutive_errors, 60)
        log.warning(
            "kafka transport error (%d consecutive): %s — backing off %ss",
            self._consecutive_errors,
            exc,
            backoff,
        )
        if self._consecutive_errors >= 5:
            log.error("rebuilding the consumer after repeated transport errors")
            self.close()
            try:
                self.connect()
                self._consecutive_errors = 0
            except Exception:
                log.exception("reconnect failed")
        self.shutdown.wait(backoff)

    @staticmethod
    def _handle_kafka_message_error(msg: Any) -> None:
        err = msg.error()
        try:
            from confluent_kafka import KafkaError

            if err.code() == KafkaError._PARTITION_EOF:
                return  # informational: we caught up with this partition
        except Exception:  # pragma: no cover
            pass
        log.warning("kafka message error on %s: %s", msg.topic(), err)

    def _quarantine(self, msg: Any, exc: Exception) -> None:
        """Persist an undecodable payload plus its schema key for inspection.

        Equivalent to ``fink_consumer --dump_schema``, but automatic and
        non-fatal.
        """
        outdir: Path = self.settings.bad_alerts_dir
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%f")
        try:
            (outdir / f"payload_{stamp}.avro").write_bytes(msg.value() or b"")
            key = msg.key()
            if key:
                text = key.decode("utf8") if isinstance(key, bytes) else str(key)
                (outdir / f"schema_{stamp}.json").write_text(text, encoding="utf-8")
            (outdir / f"error_{stamp}.txt").write_text(
                f"{type(exc).__name__}: {exc}\ntopic={msg.topic()} "
                f"partition={msg.partition()} offset={msg.offset()}\n",
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover - disk problems must not cascade
            log.exception("failed to quarantine a bad alert")
        log.error(
            "avro decode failed (%d total) on %s[%s]@%s: %s — quarantined in %s",
            self._decode_failures,
            msg.topic(),
            msg.partition(),
            msg.offset(),
            exc,
            outdir,
        )

    def stream(self) -> Iterator[NormalisedAlert]:
        if self._consumer is None:
            self.connect()
        # Tick once before the first poll so the dashboard shows "connected"
        # immediately, rather than after the first alert — which on an idle
        # night could be hours away.
        if self.on_idle:
            self.on_idle()
        while not self.shutdown.is_set():
            if self.limit is not None and self._consumed >= self.limit:
                log.info("reached limit of %d alerts; stopping", self.limit)
                return
            record = self._poll_once()
            if record is not None:
                yield record
            elif self.on_idle:
                # An idle poll is the NORMAL state: ZTF observes only at night
                # from Palomar, so most of the day returns nothing. The heartbeat
                # has to happen here, or a perfectly healthy consumer writes no
                # health rows at all and the dashboard declares it dead.
                self.on_idle()

    def commit(self, record: NormalisedAlert) -> None:
        """Commit the offset for a record, after it has been safely enqueued.

        Committing after the handoff (rather than on receipt) means a crash
        replays the in-flight alert instead of losing it.
        """
        if self._consumer is None or record.partition is None or record.offset is None:
            return
        try:
            from confluent_kafka import TopicPartition

            self._consumer._consumer.commit(
                offsets=[
                    TopicPartition(record.topic, record.partition, record.offset + 1)
                ],
                asynchronous=True,
            )
        except Exception:  # pragma: no cover - commit races are non-fatal
            log.debug("offset commit failed for %s", record.stamp_key())

    # ------------------------------------------------------------------ #
    # normalisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalise(
        alert: dict, msg: Any = None, topic: str | None = None
    ) -> NormalisedAlert:
        """Fink Avro alert dict -> NormalisedAlert."""
        candidate = alert.get("candidate") or {}
        fields = extract_candidate_fields(candidate)
        detections, nondetections = build_lightcurve(
            candidate, alert.get("prv_candidates")
        )
        cutouts, cutout_status = decode_fink_packet(alert)

        kafka_ts = None
        partition = offset = None
        if msg is not None:
            try:
                partition, offset = msg.partition(), msg.offset()
                ts_type, ts_ms = msg.timestamp()
                if ts_type and ts_ms and ts_ms > 0:
                    kafka_ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            except Exception:  # pragma: no cover
                pass

        return NormalisedAlert(
            object_id=str(alert.get("objectId")),
            candid=big_int(alert.get("candid"))
            or big_int(candidate.get("candid"))
            or 0,
            source="fink_kafka",
            topic=topic,
            partition=partition,
            offset=offset,
            jd=float(num(candidate.get("jd")) or 0.0),
            kafka_ts_utc=kafka_ts,
            broker_ingest_utc=parse_iso(alert.get("timestamp")),
            received_utc=utcnow(),
            detections=detections,
            nondetections=nondetections,
            cutouts=cutouts,
            cutout_status=cutout_status,
            broker_meta=extract_broker_meta(alert),
            **fields,
        )

    # ------------------------------------------------------------------ #
    def health(self) -> SourceHealth:
        lag: dict[str, int] = {}
        committed: dict[str, int] = {}
        if self._consumer is not None and self._connected:
            lag, committed = query_lag(self._consumer._consumer, self.settings.topics)
        return SourceHealth(
            connected=self._connected,
            mode=self.settings.mode,
            topics=self.settings.topics,
            lag_by_topic=lag,
            committed_by_topic=committed,
            last_alert_utc=self._last_alert_utc,
            error=self._error,
            is_live_stream=self.settings.mode in ("live", "catchup"),
        )

    def backlog(self) -> dict[str, int]:
        """Total lag per topic, for the pre-flight backlog warning."""
        if self._consumer is None:
            self.connect()
            # An assignment only exists after the first poll, so provoke one.
            self._consumer._consumer.poll(min(self.settings.poll_timeout_s, 5.0))
        lag, _ = query_lag(self._consumer._consumer, self.settings.topics)
        return lag

    def save_raw(self, alert: dict, record: NormalisedAlert) -> str | None:
        """Archive the decoded packet as JSON so the trace view can show it.

        The Avro bytes themselves are not re-encodable without the writer
        schema, so we store the decoded dict minus the cutout payloads (which
        already live as .npy next to it).
        """
        try:
            outdir = self.settings.raw_alerts_dir
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"{record.stamp_key()}.json"
            trimmed = {
                k: v
                for k, v in alert.items()
                if k not in ("cutoutScience", "cutoutTemplate", "cutoutDifference")
            }
            path.write_text(
                json.dumps(trimmed, indent=1, default=str), encoding="utf-8"
            )
            return str(path.relative_to(self.settings.data_dir.parent.parent))
        except Exception:  # pragma: no cover
            return None

    def close(self) -> None:
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:  # pragma: no cover
                log.debug("consumer close raised", exc_info=True)
        self._consumer = None
        self._connected = False
