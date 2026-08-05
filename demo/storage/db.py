"""SQLite access layer.

One writer (the consumer process), N readers (the API). WAL mode is what makes
that safe without a server. Every connection sets the same pragmas, because
``journal_mode`` is persistent but ``foreign_keys`` and ``busy_timeout`` are
per-connection.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from demo.config import CLASS_NAMES, Settings
from demo.models import NormalisedAlert, SourceHealth, iso, utcnow

log = logging.getLogger("demo.db")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"


def connect(settings: Settings, readonly: bool = False) -> sqlite3.Connection:
    """Open a connection with the pragmas this demo depends on."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.db_path
    if readonly and path.exists():
        conn = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False,
            timeout=10.0,
        )
    else:
        conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db(settings: Settings) -> sqlite3.Connection:
    """Create the schema if absent and return a writable connection."""
    conn = connect(settings)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _migrate(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations, so an existing demo.db keeps working.

    CREATE TABLE IF NOT EXISTS does nothing when the table already exists, so
    columns added after a database was first created need an explicit ALTER.
    Keep these additive and nullable — a demo database is not worth a migration
    framework, but silently losing one is not acceptable either.
    """
    for table, column, decl in (("stream_health", "alerce_json", "TEXT"),):
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            log.info("migrated: added %s.%s", table, column)
    conn.commit()


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #
def _proba_columns(proba: np.ndarray | None) -> tuple[float | None, ...]:
    if proba is None:
        return (None, None, None)
    values = np.asarray(proba, dtype=float).ravel()
    return tuple(float(values[i]) if i < values.size else None for i in range(3))


def save_alert(
    conn: sqlite3.Connection, alert: NormalisedAlert, stamp_path: str | None = None
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO alerts (
            candid, object_id, source, topic, kafka_partition, kafka_offset,
            jd, mjd, emitted_utc, kafka_ts_utc, broker_ingest_utc, received_utc,
            ra, dec, fid, magpsf, sigmapsf, diffmaglim, isdiffpos,
            distnr, magnr, sgscore1, distpsnr1, neargaia, rb, drb,
            ndethist, n_det, n_nondet, cutout_status, stamp_path,
            raw_packet_ref, broker_meta_json
        ) VALUES (?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?,?,
                  ?,?,?,?,?,?,?)
        """,
        (
            alert.candid, alert.object_id, alert.source, alert.topic,
            alert.partition, alert.offset,
            alert.jd, alert.mjd, iso(alert.emitted_utc), iso(alert.kafka_ts_utc),
            iso(alert.broker_ingest_utc), iso(alert.received_utc),
            alert.ra, alert.dec, alert.fid, alert.magpsf, alert.sigmapsf,
            alert.diffmaglim, alert.isdiffpos,
            alert.distnr, alert.magnr, alert.sgscore1, alert.distpsnr1,
            alert.neargaia, alert.rb, alert.drb,
            alert.ndethist, alert.n_det, alert.n_nondet, alert.cutout_status,
            stamp_path, alert.raw_packet_ref,
            json.dumps(alert.broker_meta, default=str),
        ),
    )


def save_photometry(conn: sqlite3.Connection, alert: NormalisedAlert) -> None:
    rows: list[tuple] = []
    for det in alert.detections:
        rows.append(
            (
                alert.candid, alert.object_id, det.jd, det.fid, det.magpsf,
                det.sigmapsf, det.diffmaglim, "detection",
            )
        )
    for nd in alert.nondetections:
        rows.append(
            (
                alert.candid, alert.object_id, nd.jd, nd.fid, None, None,
                nd.diffmaglim, "nondetection",
            )
        )
    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO photometry
                (candid, object_id, jd, fid, magpsf, sigmapsf, diffmaglim, kind)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def save_prediction(conn: sqlite3.Connection, prediction) -> None:
    tab = _proba_columns(prediction.p_tab)
    img = _proba_columns(prediction.p_img)
    fused = _proba_columns(prediction.p_fused)
    provenance = prediction.feature_provenance
    conn.execute(
        """
        INSERT OR REPLACE INTO predictions (
            candid, status, status_reason, fusion_mode,
            p_tab_sn, p_tab_agn, p_tab_vs,
            p_img_sn, p_img_agn, p_img_vs,
            p_fused_sn, p_fused_agn, p_fused_vs,
            predicted_class, confidence, branch_disagree, fusion_flips,
            feature_provenance, n_features_present,
            t_feature_ms, t_stamp_ms, t_tab_ms, t_img_ms, t_fuse_ms,
            t_pipeline_ms, t_broker_to_classified_ms, t_emitted_to_classified_s,
            model_versions_json, trace_json, split_id, created_utc
        ) VALUES (?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?, ?,?,?,
                  ?,?,?,?)
        """,
        (
            prediction.candid, prediction.status, prediction.status_reason,
            prediction.fusion_mode,
            *tab, *img, *fused,
            prediction.predicted_class, prediction.confidence,
            int(prediction.branch_disagree), int(prediction.fusion_flips),
            provenance.source if provenance else None,
            provenance.n_present if provenance else None,
            prediction.tabular.elapsed_ms if prediction.tabular else None,
            prediction.t_stamp_ms,
            prediction.tabular.elapsed_ms if prediction.tabular else None,
            prediction.image.elapsed_ms if prediction.image else None,
            prediction.fusion.elapsed_ms if prediction.fusion else None,
            prediction.t_pipeline_ms,
            prediction.t_broker_to_classified_ms,
            prediction.t_emitted_to_classified_s,
            json.dumps(prediction.model_versions, default=str),
            json.dumps(prediction.trace, default=str),
            prediction.split_id,
            iso(prediction.created_utc),
        ),
    )


def save_health(
    conn: sqlite3.Connection,
    health: SourceHealth,
    queue_depth: int = 0,
    dropped_total: int = 0,
    decode_failures: int = 0,
    consumed_total: int = 0,
    alerce: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO stream_health (
            ts_utc, mode, connected, is_live_stream, topics_json, lag_json,
            committed_json, last_alert_utc, queue_depth, dropped_total,
            decode_failures, consumed_total, consumer_pid, error, alerce_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            iso(utcnow()), health.mode, int(health.connected),
            int(health.is_live_stream), json.dumps(list(health.topics)),
            json.dumps(health.lag_by_topic), json.dumps(health.committed_by_topic),
            iso(health.last_alert_utc), queue_depth, dropped_total,
            decode_failures, consumed_total, os.getpid(), health.error,
            json.dumps(alerce or {}),
        ),
    )
    # Keep the health table from growing without bound over a long session.
    conn.execute(
        "DELETE FROM stream_health WHERE ts_utc NOT IN "
        "(SELECT ts_utc FROM stream_health ORDER BY ts_utc DESC LIMIT 2000)"
    )


def upsert_known_labels(conn: sqlite3.Connection, rows: Iterable[tuple]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO known_labels
            (object_id, coarse, fine, plasticc_class, label_source,
             in_training_set, training_split)
        VALUES (?,?,?,?,?,?,?)
        """,
        rows,
    )
    return len(rows)


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
def latest_health(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM stream_health ORDER BY ts_utc DESC LIMIT 1"
    ).fetchone()


def health_series(conn: sqlite3.Connection, limit: int = 60) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT ts_utc, lag_json, queue_depth FROM stream_health "
        "ORDER BY ts_utc DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return list(reversed(rows))


def known_label(conn: sqlite3.Connection, object_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM known_labels WHERE object_id = ?", (object_id,)
    ).fetchone()


def load_alert(
    conn: sqlite3.Connection, candid: int, settings: Settings
) -> NormalisedAlert | None:
    """Rebuild a :class:`NormalisedAlert` from what was persisted.

    Everything the inference layer needs survives a round trip through the
    database: the scalars live in ``alerts``, the light curve in ``photometry``,
    and the cutout triplet in the ``.npy`` beside it. That is what makes
    re-classification possible without re-polling Kafka — the alert may be long
    gone from the broker's 4-day queue, but we still hold it.

    Used by ``scripts/backfill_features.py`` to upgrade image-only rows to full
    two-branch fusion once ALeRCE becomes reachable.
    """
    from demo.models import Detection, NonDetection, parse_iso

    row = conn.execute("SELECT * FROM alerts WHERE candid = ?", (candid,)).fetchone()
    if row is None:
        return None

    detections: list[Detection] = []
    nondetections: list[NonDetection] = []
    for p in conn.execute(
        "SELECT jd, fid, magpsf, sigmapsf, diffmaglim, kind FROM photometry "
        "WHERE candid = ? ORDER BY jd",
        (candid,),
    ):
        if p["kind"] == "detection" and p["magpsf"] is not None:
            detections.append(
                Detection(
                    jd=p["jd"], fid=p["fid"], magpsf=p["magpsf"],
                    sigmapsf=p["sigmapsf"], diffmaglim=p["diffmaglim"],
                )
            )
        elif p["diffmaglim"] is not None:
            nondetections.append(
                NonDetection(jd=p["jd"], fid=p["fid"], diffmaglim=p["diffmaglim"])
            )

    cutouts: dict[str, Any] = {}
    if row["stamp_path"]:
        path = settings.stamps_dir / row["stamp_path"]
        if path.exists():
            try:
                stack = np.load(path)
                from demo.config import CHANNEL_ORDER

                cutouts = {c: stack[i] for i, c in enumerate(CHANNEL_ORDER)}
            except Exception:  # pragma: no cover
                log.warning("could not load stamps for %s", candid)

    return NormalisedAlert(
        object_id=row["object_id"],
        candid=row["candid"],
        source=row["source"],
        topic=row["topic"],
        partition=row["kafka_partition"],
        offset=row["kafka_offset"],
        jd=row["jd"],
        kafka_ts_utc=parse_iso(row["kafka_ts_utc"]),
        broker_ingest_utc=parse_iso(row["broker_ingest_utc"]),
        received_utc=parse_iso(row["received_utc"]) or utcnow(),
        ra=row["ra"], dec=row["dec"], fid=row["fid"],
        magpsf=row["magpsf"], sigmapsf=row["sigmapsf"],
        diffmaglim=row["diffmaglim"], isdiffpos=row["isdiffpos"],
        distnr=row["distnr"], magnr=row["magnr"], sgscore1=row["sgscore1"],
        distpsnr1=row["distpsnr1"], neargaia=row["neargaia"],
        rb=row["rb"], drb=row["drb"], ndethist=row["ndethist"],
        detections=tuple(detections),
        nondetections=tuple(nondetections),
        cutouts=cutouts,
        cutout_status=row["cutout_status"],
        broker_meta=parse_json(row["broker_meta_json"], {}) or {},
        raw_packet_ref=row["raw_packet_ref"],
    )


def upgradeable_candids(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Alerts that would gain a tabular branch if features became available.

    These are exactly the rows classified without features: ``image_only``
    because the resolver could not reach ALeRCE, or unclassified outright.
    Alerts that legitimately have too few detections are excluded — no feature
    fetch would help those.
    """
    return [
        (r["candid"], r["object_id"])
        for r in conn.execute(
            """
            SELECT a.candid, a.object_id
            FROM predictions p
            JOIN alerts a ON a.candid = p.candid
            WHERE p.fusion_mode IN ('image_only', 'none')
              AND (p.feature_provenance = 'unavailable' OR p.feature_provenance IS NULL)
              AND a.n_det >= 1
            ORDER BY a.received_utc DESC
            """
        )
    ]


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else {k: row[k] for k in row.keys()}


def proba_dict(row: Any, prefix: str) -> dict[str, float] | None:
    """``p_fused_sn/agn/vs`` columns -> ``{"SN": .., "AGN": .., "VS": ..}``."""
    values = []
    for cls in CLASS_NAMES:
        key = f"{prefix}_{cls.lower()}"
        try:
            values.append(row[key])
        except (KeyError, IndexError):
            return None
    if any(v is None for v in values):
        return None
    return {cls: float(v) for cls, v in zip(CLASS_NAMES, values)}


def parse_json(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
