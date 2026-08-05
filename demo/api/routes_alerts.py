"""Alert-stream and object-detail endpoints."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from demo.api.deps import get_db
from demo.api.serializers import (
    alert_row,
    branch_comparison,
    photometry_rows,
)
from demo.config import CLASS_NAMES
from demo.models import parse_iso, parse_window, utcnow

router = APIRouter(prefix="/api", tags=["alerts"])

#: The joined projection every alert response is built from. Predictions are
#: LEFT JOINed so an alert that failed to classify still appears — a silent gap
#: in the table would be worse than a visible failure.
BASE_SELECT = """
SELECT a.*,
       p.status, p.status_reason, p.fusion_mode,
       p.p_tab_sn, p.p_tab_agn, p.p_tab_vs,
       p.p_img_sn, p.p_img_agn, p.p_img_vs,
       p.p_fused_sn, p.p_fused_agn, p.p_fused_vs,
       p.predicted_class, p.confidence, p.branch_disagree, p.fusion_flips,
       p.feature_provenance, p.n_features_present,
       p.t_tab_ms, p.t_img_ms, p.t_fuse_ms, p.t_pipeline_ms,
       p.t_broker_to_classified_ms, p.t_emitted_to_classified_s, p.split_id,
       k.coarse         AS known_coarse,
       k.fine           AS known_fine,
       k.plasticc_class AS known_plasticc,
       k.label_source   AS known_source,
       k.in_training_set AS known_in_training,
       k.training_split AS known_split
FROM alerts a
LEFT JOIN predictions p ON p.candid = a.candid
LEFT JOIN known_labels k ON k.object_id = a.object_id
"""


def _filters(
    predicted_class: str | None,
    min_confidence: float | None,
    since: str | None,
    topic: str | None,
    fusion_mode: str | None,
    disagree_only: bool,
    object_id: str | None,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if predicted_class:
        wanted = [
            c.strip().upper()
            for c in predicted_class.split(",")
            if c.strip().upper() in CLASS_NAMES
        ]
        if wanted:
            clauses.append(
                f"p.predicted_class IN ({','.join('?' * len(wanted))})"
            )
            params.extend(wanted)
    if min_confidence is not None:
        clauses.append("p.confidence >= ?")
        params.append(float(min_confidence))
    if topic:
        topics = [t.strip() for t in topic.split(",") if t.strip()]
        if topics:
            clauses.append(f"a.topic IN ({','.join('?' * len(topics))})")
            params.extend(topics)
    if fusion_mode:
        modes = [m.strip() for m in fusion_mode.split(",") if m.strip()]
        if modes:
            clauses.append(f"p.fusion_mode IN ({','.join('?' * len(modes))})")
            params.extend(modes)
    if disagree_only:
        clauses.append("p.branch_disagree = 1")
    if object_id:
        clauses.append("a.object_id LIKE ?")
        params.append(f"%{object_id.strip()}%")
    if since:
        window = parse_window(since)
        if window is not None:
            cutoff = utcnow() - window
        else:
            parsed = parse_iso(since)
            cutoff = parsed
        if cutoff is not None:
            clauses.append("a.received_utc >= ?")
            params.append(
                cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            )
    return clauses, params


@router.get("/alerts")
def list_alerts(
    conn: sqlite3.Connection = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, description="received_utc of the last row seen"),
    predicted_class: str | None = Query(None, alias="class"),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    since: str | None = Query(None, description="ISO timestamp or 1h / 24h / 7d"),
    topic: str | None = None,
    fusion_mode: str | None = None,
    disagree_only: bool = False,
    object_id: str | None = None,
    sort: str = Query("received", pattern="^(received|confidence|magpsf)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict:
    clauses, params = _filters(
        predicted_class, min_confidence, since, topic, fusion_mode,
        disagree_only, object_id,
    )

    total = conn.execute(
        "SELECT COUNT(*) AS n FROM alerts a "
        "LEFT JOIN predictions p ON p.candid = a.candid"
        + (" WHERE " + " AND ".join(clauses) if clauses else ""),
        params,
    ).fetchone()["n"]

    sort_column = {
        "received": "a.received_utc",
        "confidence": "p.confidence",
        "magpsf": "a.magpsf",
    }[sort]
    direction = "DESC" if order == "desc" else "ASC"

    page_clauses = list(clauses)
    page_params = list(params)
    # The cursor only applies to the default ordering; other sorts fall back to
    # offsetless first-page semantics, which is all the demo needs.
    if cursor and sort == "received":
        page_clauses.append(
            "a.received_utc < ?" if order == "desc" else "a.received_utc > ?"
        )
        page_params.append(cursor)

    sql = (
        BASE_SELECT
        + (" WHERE " + " AND ".join(page_clauses) if page_clauses else "")
        + f" ORDER BY {sort_column} {direction} NULLS LAST LIMIT ?"
    )
    page_params.append(limit)
    rows = conn.execute(sql, page_params).fetchall()

    items = [alert_row(r, include_broker=False) for r in rows]
    next_cursor = rows[-1]["received_utc"] if rows and sort == "received" else None
    newest = conn.execute(
        "SELECT MAX(received_utc) AS m FROM alerts"
    ).fetchone()["m"]
    return {
        "items": items,
        "next_cursor": next_cursor,
        "total_matching": total,
        "newest_received_utc": newest,
        "server_time_utc": utcnow().isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
    }


@router.get("/alerts/{candid}")
def get_alert(candid: int, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    row = conn.execute(
        BASE_SELECT + " WHERE a.candid = ?", (candid,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no alert with candid {candid}")
    payload = alert_row(row)
    payload["branch_comparison"] = branch_comparison(row)
    return payload


@router.get("/objects/{object_id}")
def get_object(object_id: str, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    rows = conn.execute(
        BASE_SELECT + " WHERE a.object_id = ? ORDER BY a.jd DESC", (object_id,)
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"no alerts for {object_id}")
    latest = rows[0]
    return {
        "object_id": object_id,
        "n_alerts": len(rows),
        "latest": alert_row(latest),
        "branch_comparison": branch_comparison(latest),
        "alerts": [alert_row(r, include_broker=False) for r in rows],
    }


@router.get("/objects/{object_id}/lightcurve")
def get_lightcurve(
    object_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> dict:
    rows = conn.execute(
        "SELECT jd, fid, magpsf, sigmapsf, diffmaglim, kind FROM photometry "
        "WHERE object_id = ? ORDER BY jd",
        (object_id,),
    ).fetchall()
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"no photometry stored for {object_id}"
        )
    payload = photometry_rows(rows)
    payload["object_id"] = object_id
    return payload
