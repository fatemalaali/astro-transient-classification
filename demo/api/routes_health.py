"""Live-stream indicator, statistics and topic breakdown."""

from __future__ import annotations

import os
import sqlite3
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query

from demo.api.deps import get_db, settings as get_settings_dep
from demo.config import CLASS_NAMES, Settings
from demo.models import iso, parse_iso, utcnow
from demo.storage.db import health_series, latest_health, parse_json

router = APIRouter(prefix="/api", tags=["health"])

#: ZTF observes from Palomar Observatory. A quiet stream during a daytime viva
#: is normal, and the UI says so rather than looking broken.
#:
#: Resolved defensively: Windows has no system IANA database, so ``zoneinfo``
#: needs the ``tzdata`` package there. It is in requirements-demo.txt, but a
#: missing timezone must never take down the health endpoint — the demo can
#: perfectly well run without knowing what time it is at Palomar.
try:
    PALOMAR_TZ = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover - only without tzdata on Windows
    PALOMAR_TZ = timezone(timedelta(hours=-8))  # PST; DST is cosmetic here


def _column(row, name: str, default=None):
    """Read a column that may not exist on an un-migrated database."""
    if row is None:
        return default
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


#: A consumer heartbeats every ~2 s, including when the stream is idle, so
#: anything older than this means it is not running (or is wedged).
HEARTBEAT_STALE_S = 30.0


def _pid_alive(pid: int | None) -> bool | None:
    """Best effort process check. ``None`` means *unknown*, not dead.

    ``os.kill(pid, 0)`` is the POSIX idiom and does not work on Windows —
    signal 0 is not a valid Windows signal, so it raises
    ``OSError: [WinError 87] The parameter is incorrect`` for a perfectly
    healthy process. Reading that as "dead" marks every running consumer down.

    psutil handles it properly and ships as a fink-client dependency, so it is
    almost always available. Where neither works, return ``None`` so the caller
    falls back to the heartbeat, which is the signal that actually matters.
    """
    if not pid:
        return None
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as exc:
        if getattr(exc, "winerror", None) == 87:
            return None  # unsupported probe, not evidence of death
        return False
    except (ValueError, TypeError):
        return None


@router.get("/stream/health")
def stream_health(
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
) -> dict:
    row = latest_health(conn)
    now = utcnow()

    if row is None:
        return {
            "mode": config.mode,
            "connected": False,
            "is_live_stream": False,
            "badge": "NO CONSUMER",
            "message": "No consumer has reported yet. Start python -m demo.run_consumer.",
            "palomar_local_time": now.astimezone(PALOMAR_TZ).strftime("%H:%M"),
            "palomar_is_night": _is_night(now),
        }

    lag = parse_json(row["lag_json"], {}) or {}
    committed = parse_json(row["committed_json"], {}) or {}
    topics = parse_json(row["topics_json"], []) or []
    last_alert = parse_iso(row["last_alert_utc"])
    reported_at = parse_iso(row["ts_utc"])
    is_live = bool(row["is_live_stream"])
    connected = bool(row["connected"])
    pid_alive = _pid_alive(row["consumer_pid"])

    # Liveness is decided by the HEARTBEAT, not by the PID. The consumer writes
    # a health row every ~2 s whether or not alerts are flowing, so a recent row
    # is direct evidence it is running; a PID probe is indirect, platform-
    # dependent, and meaningless across process boundaries. The PID is used only
    # to distinguish "not running" from "running but wedged".
    stale = (
        reported_at is None
        or (now - reported_at).total_seconds() > HEARTBEAT_STALE_S
    )

    if stale:
        badge = "CONSUMER STALLED" if pid_alive else "CONSUMER DOWN"
    elif not connected:
        badge = "DISCONNECTED"
    elif row["mode"] == "live":
        badge = "LIVE (Kafka)"
    elif row["mode"] == "catchup":
        badge = "CATCHUP (Kafka)"
    elif row["mode"] == "replay":
        badge = "REPLAY (pinned offsets)"
    elif row["mode"] == "rest":
        badge = "REST fallback"
    else:
        badge = "OFFLINE replay"

    sparkline = [
        sum((parse_json(h["lag_json"], {}) or {}).values())
        for h in health_series(conn, limit=60)
    ]

    return {
        "mode": row["mode"],
        "badge": badge,
        "connected": connected and not stale,
        # Never green unless a genuine push stream is connected. A polled or
        # replayed source reports False here so the UI cannot fake liveness.
        "is_live_stream": is_live and connected and not stale,
        "consumer_pid": row["consumer_pid"],
        "consumer_alive": pid_alive,  # None = could not determine
        "heartbeat_age_s": (
            (now - reported_at).total_seconds() if reported_at else None
        ),
        "reported_at_utc": row["ts_utc"],
        "stale_report": stale,
        "topics": topics,
        "lag_by_topic": lag,
        "committed_by_topic": committed,
        "total_lag": sum(lag.values()) if lag else None,
        "lag_sparkline": sparkline,
        "last_alert_utc": row["last_alert_utc"],
        "seconds_since_last_alert": (
            (now - last_alert).total_seconds() if last_alert else None
        ),
        "queue_depth": row["queue_depth"],
        "dropped_total": row["dropped_total"],
        "decode_failures": row["decode_failures"],
        "consumed_total": row["consumed_total"],
        "error": row["error"],
        "palomar_local_time": now.astimezone(PALOMAR_TZ).strftime("%H:%M"),
        "palomar_is_night": _is_night(now),
        "palomar_note": (
            "ZTF observes at night from Palomar Observatory. Daytime gaps in the "
            "stream are expected, not a fault."
        ),
        "server_time_utc": iso(now),
    }


def _is_night(now) -> bool:
    hour = now.astimezone(PALOMAR_TZ).hour
    return hour >= 19 or hour < 6


@router.get("/alerce/status")
def alerce_status(
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
) -> dict:
    """Can the ingest process reach the ALeRCE feature service?

    This decides whether live alerts get a tabular branch at all, so it is
    reported as a first-class piece of demo state rather than buried in logs.
    The reading comes from the *consumer's* last health row, because the
    consumer is the process that actually fetches features.
    """
    row = latest_health(conn)
    # Read defensively: the API opens the database read-only, so a demo.db
    # created before this column existed will not have been migrated yet. An
    # older database must degrade to "unknown", never to a 500.
    raw = _column(row, "alerce_json")
    state = parse_json(raw, {}) if raw else {}
    pending = conn.execute(
        """
        SELECT COUNT(*) AS n FROM predictions p JOIN alerts a ON a.candid = p.candid
        WHERE p.fusion_mode IN ('image_only', 'none')
          AND (p.feature_provenance = 'unavailable' OR p.feature_provenance IS NULL)
          AND a.n_det >= 1
        """
    ).fetchone()["n"]
    return {
        "as_seen_by_consumer": state or None,
        "upgradeable_alerts": pending,
        "hint": (
            f"{pending} alert(s) would gain a tabular branch. On a network that "
            "can reach ALeRCE, run: python scripts/backfill_features.py"
            if pending
            else "Every stored alert already has a tabular branch."
        ),
        "endpoint": "https://api.alerce.online/ztf/v1",
    }


@router.post("/alerce/probe")
def alerce_probe(config: Settings = Depends(get_settings_dep)) -> dict:
    """Actively test ALeRCE reachability from *this* process, right now.

    Exists so you can switch to a phone hotspot mid-demo and confirm it worked
    from the dashboard, without restarting anything or reading a log.
    """
    from demo.inference.features import FeatureResolver

    resolver = FeatureResolver(config, ())
    reachable = resolver.probe()
    status = resolver.status()
    resolver.close()
    return {
        "reachable": reachable,
        "status": status,
        "next_step": (
            "Run: python scripts/backfill_features.py — stored image-only "
            "alerts will be upgraded to two-branch fusion."
            if reachable
            else "Still blocked. Try a phone hotspot, the campus network, or a "
                 "VPN. The replay path works regardless: "
                 "python -m demo.run_consumer --mode replay"
        ),
    }


@router.get("/stats")
def stats(
    conn: sqlite3.Connection = Depends(get_db),
    since: str | None = Query(None),
) -> dict:
    by_class = {
        r["predicted_class"]: r["n"]
        for r in conn.execute(
            "SELECT predicted_class, COUNT(*) AS n FROM predictions "
            "WHERE predicted_class IS NOT NULL GROUP BY predicted_class"
        )
    }
    by_mode = {
        r["fusion_mode"]: r["n"]
        for r in conn.execute(
            "SELECT fusion_mode, COUNT(*) AS n FROM predictions GROUP BY fusion_mode"
        )
    }
    by_topic = [
        {"topic": r["topic"], "n": r["n"]}
        for r in conn.execute(
            "SELECT topic, COUNT(*) AS n FROM alerts GROUP BY topic ORDER BY n DESC"
        )
    ]
    by_provenance = {
        r["feature_provenance"] or "none": r["n"]
        for r in conn.execute(
            "SELECT feature_provenance, COUNT(*) AS n FROM predictions "
            "GROUP BY feature_provenance"
        )
    }

    bins = [0.0] * 10
    for r in conn.execute(
        "SELECT confidence FROM predictions WHERE confidence IS NOT NULL"
    ):
        idx = min(int(float(r["confidence"]) * 10), 9)
        bins[idx] += 1

    latency = conn.execute(
        """
        SELECT COUNT(*) AS n,
               AVG(t_pipeline_ms) AS mean_pipeline_ms,
               AVG(t_broker_to_classified_ms) AS mean_broker_ms,
               AVG(t_emitted_to_classified_s) AS mean_emitted_s
        FROM predictions
        """
    ).fetchone()

    percentiles = {}
    for column in ("t_pipeline_ms", "t_broker_to_classified_ms"):
        values = [
            r[0]
            for r in conn.execute(
                f"SELECT {column} FROM predictions WHERE {column} IS NOT NULL "
                f"ORDER BY {column}"
            )
        ]
        if values:
            percentiles[column] = {
                "p50": values[len(values) // 2],
                "p95": values[min(int(len(values) * 0.95), len(values) - 1)],
                "max": values[-1],
            }

    totals = conn.execute(
        "SELECT COUNT(*) AS alerts, COUNT(DISTINCT object_id) AS objects FROM alerts"
    ).fetchone()
    disagree = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions WHERE branch_disagree = 1"
    ).fetchone()["n"]
    # "In the gold set" is not the same as "the model was fitted on it": the
    # test fold is gold-set but out-of-sample. Conflating the two would be as
    # misleading as not flagging leakage at all, so they are counted separately.
    fitted_on = conn.execute(
        "SELECT COUNT(*) AS n FROM alerts a JOIN known_labels k "
        "ON k.object_id = a.object_id "
        "WHERE k.in_training_set = 1 AND k.training_split IN ('train', 'val')"
    ).fetchone()["n"]
    held_out = conn.execute(
        "SELECT COUNT(*) AS n FROM alerts a JOIN known_labels k "
        "ON k.object_id = a.object_id "
        "WHERE k.in_training_set = 1 AND k.training_split = 'test'"
    ).fetchone()["n"]

    return {
        "totals": {
            "alerts": totals["alerts"],
            "objects": totals["objects"],
            "disagreements": disagree,
            "alerts_on_fitted_objects": fitted_on,
            "alerts_on_heldout_objects": held_out,
        },
        "by_class": {cls: by_class.get(cls, 0) for cls in CLASS_NAMES},
        "by_fusion_mode": by_mode,
        "by_topic": by_topic,
        "by_feature_provenance": by_provenance,
        "confidence_histogram": {
            "bins": [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)],
            "counts": bins,
        },
        "latency": {
            "n": latency["n"],
            "mean_pipeline_ms": latency["mean_pipeline_ms"],
            "mean_broker_to_classified_ms": latency["mean_broker_ms"],
            "mean_emitted_to_classified_s": latency["mean_emitted_s"],
            "percentiles": percentiles,
            "note": (
                "emitted_to_classified includes ZTF and Fink upstream processing "
                "(minutes) and is reported, not targeted. Only pipeline_ms is "
                "under this system's control."
            ),
        },
        "class_prior_caveat": (
            "The subscribed topics impose a selection prior that does not match "
            "the training prior (gold set: SN 7728 / VS 3517 / AGN 581). Live "
            "counts are not an accuracy estimate."
        ),
    }


@router.get("/topics")
def topics(
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
) -> dict:
    """Subscribed topics with descriptions and observed counts."""
    counts = {
        r["topic"]: r["n"]
        for r in conn.execute(
            "SELECT topic, COUNT(*) AS n FROM alerts GROUP BY topic"
        )
    }
    return {
        "configured": [
            {
                "topic": topic,
                "description": TOPIC_DESCRIPTIONS.get(topic, ""),
                "target_class": TOPIC_TARGET.get(topic),
                "count": counts.get(topic, 0),
            }
            for topic in config.topics
        ],
        "observed": [{"topic": k, "count": v} for k, v in sorted(counts.items())],
        "coverage_note": (
            "AGN and VS have no general-purpose Fink livestream topic. The only "
            "AGN-adjacent real-time filter is fink_blazar_ztf (SIMBAD "
            "Blazar/BLLac); for variable stars only magnetic CVs and YSOs are "
            "available. This limits live class coverage and is a property of "
            "Fink's filter catalogue, not of the classifier."
        ),
    }


#: From https://doc.ztf.fink-broker.org/broker/filters/ (verified 2026-08-05).
TOPIC_DESCRIPTIONS = {
    "fink_sn_candidates_ztf": "Alerts considered as SN candidates",
    "fink_early_sn_candidates_ztf": "Early SN-Ia candidates; pushed to TNS nightly",
    "fink_blazar_ztf": "Flagged Blazar / Blazar_Candidate / BLLac in SIMBAD",
    "fink_magnetic_cvs_ztf": "Match to a catalogue of Magnetic Cataclysmic Variables",
    "fink_yso_spicy_candidates_ztf": "Match in the SPICY catalogue at CDS",
    "fink_vra_ztf": "Not SIMBAD-matched and not likely asteroids (VRA feed)",
    "fink_tns_match": "Alerts with classified counterparts in TNS",
    "fink_kn_candidates_ztf": "Kilonova candidates (machine learning)",
    "fink_early_kn_candidates_ztf": "Kilonova candidates (crossmatch + cuts)",
    "fink_microlensing_candidates_ztf": "Microlensing candidates",
    "fink_sso_ztf_candidates_ztf": "Counterpart in the Minor Planet Center",
    "fink_sso_fink_candidates_ztf": "New Solar System Object candidates",
    "fink_new_hostless_ztf": "Newly appearing and hostless transients",
}

TOPIC_TARGET = {
    "fink_sn_candidates_ztf": "SN",
    "fink_early_sn_candidates_ztf": "SN",
    "fink_blazar_ztf": "AGN (narrow — blazars only)",
    "fink_magnetic_cvs_ztf": "VS (narrow — magnetic CVs only)",
    "fink_yso_spicy_candidates_ztf": "VS (narrow — YSOs only)",
    "fink_vra_ztf": "mixed / volume",
    "fink_tns_match": "any (carries a TNS spectroscopic label)",
}
