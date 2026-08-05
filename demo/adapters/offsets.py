"""Kafka offset policy: replay manifests and lag queries.

Kafka's own offset rewind (``fink_consumer -start_at earliest``) gets you *a*
replay, not *the same* replay: a shared consumer group's committed offsets move
between runs, so two "replays" can cover different alerts. A pinned manifest —
an explicit ``(topic, partition, offset)`` set committed to git — is what makes
"the demo is reproducible" a claim rather than a hope.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from demo.models import iso, utcnow

log = logging.getLogger("demo.offsets")


@dataclass(frozen=True, slots=True)
class PinnedOffset:
    topic: str
    partition: int
    offset: int
    limit: int | None = None  # stop after this many alerts from this partition


@dataclass
class ReplayManifest:
    """A pinned starting position per topic-partition, plus a total alert cap."""

    created_utc: str
    total_limit: int | None
    offsets: tuple[PinnedOffset, ...]

    @classmethod
    def load(cls, path: Path) -> "ReplayManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            created_utc=raw.get("created_utc", ""),
            total_limit=raw.get("total_limit"),
            offsets=tuple(
                PinnedOffset(
                    topic=o["topic"],
                    partition=int(o["partition"]),
                    offset=int(o["offset"]),
                    limit=o.get("limit"),
                )
                for o in raw.get("offsets", [])
            ),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "created_utc": self.created_utc,
                    "total_limit": self.total_limit,
                    "offsets": [
                        {
                            "topic": o.topic,
                            "partition": o.partition,
                            "offset": o.offset,
                            **({"limit": o.limit} if o.limit is not None else {}),
                        }
                        for o in self.offsets
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def lookup(self, topic: str, partition: int) -> int | None:
        for o in self.offsets:
            if o.topic == topic and o.partition == partition:
                return o.offset
        return None

    @staticmethod
    def new(offsets: list[PinnedOffset], total_limit: int | None) -> "ReplayManifest":
        return ReplayManifest(
            created_utc=iso(utcnow()) or "",
            total_limit=total_limit,
            offsets=tuple(offsets),
        )


def make_on_assign(
    mode: str, manifest: ReplayManifest | None = None
) -> Callable[[Any, list], None] | None:
    """Build the ``on_assign`` callback that implements the offset policy.

    ``live``    seek every partition to the end. This is the single most
                important line of defence against the 4-day first-connection
                backlog: a fresh consumer never replays days of alerts by
                accident.
    ``replay``  seek to the pinned offsets in the manifest; partitions absent
                from the manifest are parked at the end so they contribute
                nothing.
    ``catchup`` leave the assignment alone — committed offsets (or
                ``auto.offset.reset``) decide, and the caller throttles with
                ``--max-backlog``.
    """
    if mode == "catchup":
        return None

    def on_assign(consumer: Any, partitions: list) -> None:
        # confluent_kafka is imported lazily so this module stays importable
        # (and testable) on a machine with no Kafka stack installed.
        from confluent_kafka import OFFSET_END

        for tp in partitions:
            if mode == "replay" and manifest is not None:
                pinned = manifest.lookup(tp.topic, tp.partition)
                tp.offset = OFFSET_END if pinned is None else pinned
            else:
                tp.offset = OFFSET_END
        try:
            consumer.assign(partitions)
        except Exception:  # pragma: no cover - librdkafka state edge case
            log.exception("assign failed for %s", partitions)
        log.info(
            "offset policy %s applied to %d partition(s): %s",
            mode,
            len(partitions),
            ", ".join(f"{tp.topic}[{tp.partition}]@{tp.offset}" for tp in partitions),
        )

    return on_assign


def query_lag(consumer: Any, topics: tuple[str, ...]) -> tuple[dict, dict]:
    """Return ``(lag_by_topic, committed_by_topic)`` for the live indicator.

    Uses the same quantities as ``fink_consumer --display_statistics``:
    ``Committed`` is where we are, ``Lag`` is what remains upstream. Any
    failure returns empty dicts — a lag query must never take the demo down.
    """
    lag: dict[str, int] = {}
    committed: dict[str, int] = {}
    try:
        from confluent_kafka import TopicPartition
    except ImportError:
        return lag, committed

    try:
        assignment = consumer.assignment()
    except Exception:
        return lag, committed
    if not assignment:
        return lag, committed

    for tp in assignment:
        if topics and tp.topic not in topics:
            continue
        try:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(tp.topic, tp.partition), timeout=5.0, cached=False
            )
            positions = consumer.committed(
                [TopicPartition(tp.topic, tp.partition)], timeout=5.0
            )
            pos = positions[0].offset if positions else 0
            if pos is None or pos < 0:
                pos = low
            lag[tp.topic] = lag.get(tp.topic, 0) + max(0, int(high) - int(pos))
            committed[tp.topic] = committed.get(tp.topic, 0) + max(0, int(pos))
        except Exception:  # pragma: no cover - transient broker errors
            log.debug("lag query failed for %s[%s]", tp.topic, tp.partition)
    return lag, committed
