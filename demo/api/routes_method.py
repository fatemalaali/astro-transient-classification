"""Methodology-showcase endpoints.

These are what the viva runs on: the end-to-end trace of a single alert, the
automatic disagreement surfacing, the provenance statement, and the held-out
evaluation table.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from demo.api.deps import get_db, settings as get_settings_dep
from demo.config import CLASS_NAMES, Settings
from demo.storage.db import parse_json, proba_dict

router = APIRouter(prefix="/api", tags=["methodology"])

#: Trace stages never shown in the UI. See get_trace().
HIDDEN_TRACE_STAGES = frozenset({"bogus"})


@router.get("/alerts/{candid}/trace")
def get_trace(candid: int, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    """The stage-by-stage trace recorded at classification time.

    This is a rendering of what actually happened, not a re-derivation: the
    stage list was written by the inference engine when the alert was
    classified, so the numbers shown are the numbers used.
    """
    row = conn.execute(
        "SELECT p.trace_json, p.split_id, a.object_id, a.topic "
        "FROM predictions p JOIN alerts a ON a.candid = p.candid "
        "WHERE p.candid = ?",
        (candid,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no prediction for {candid}")
    stages = parse_json(row["trace_json"], []) or []
    # Drop stages that represent nothing the system actually does. The
    # real/bogus hook is deliberately unimplemented, so an always-skipped card
    # in the trace is noise in a view whose job is to show the real pipeline.
    # Filtered here rather than by a migration, so alerts classified before this
    # change render the same as new ones.
    stages = [s for s in stages if s.get("id") not in HIDDEN_TRACE_STAGES]
    return {
        # String, not a number — see serializers.alert_row for why.
        "candid": str(candid),
        "object_id": row["object_id"],
        "topic": row["topic"],
        "split_id": row["split_id"],
        "stages": stages,
    }


@router.get("/alerts/{candid}/packet")
def get_packet(
    candid: int,
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
) -> dict:
    """The alert packet, split into instrument / broker-derived / system.

    ALeRCE's Explorer shows one flat AVRO table. Splitting it is where this
    demo makes the provenance boundary visible instead of asserting it in prose.
    """
    row = conn.execute("SELECT * FROM alerts WHERE candid = ?", (candid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no alert with candid {candid}")

    instrument = {
        "objectId": row["object_id"],
        "candid": str(row["candid"]),
        "jd": row["jd"],
        "mjd": row["mjd"],
        "ra": row["ra"],
        "dec": row["dec"],
        "fid": row["fid"],
        "magpsf": row["magpsf"],
        "sigmapsf": row["sigmapsf"],
        "diffmaglim": row["diffmaglim"],
        "isdiffpos": row["isdiffpos"],
        "distnr": row["distnr"],
        "magnr": row["magnr"],
        "sgscore1": row["sgscore1"],
        "distpsnr1": row["distpsnr1"],
        "neargaia": row["neargaia"],
        "rb": row["rb"],
        "drb": row["drb"],
        "ndethist": row["ndethist"],
        "n_detections": row["n_det"],
        "n_nondetections": row["n_nondet"],
    }
    system = {
        "source": row["source"],
        "topic": row["topic"],
        "partition": row["kafka_partition"],
        "offset": row["kafka_offset"],
        "kafka_ts_utc": row["kafka_ts_utc"],
        "received_utc": row["received_utc"],
        "emitted_utc": row["emitted_utc"],
        "cutout_status": row["cutout_status"],
        "raw_packet_ref": row["raw_packet_ref"],
    }
    broker = parse_json(row["broker_meta_json"], {}) or {}
    return {
        "candid": str(candid),
        "sections": {
            "instrument": {
                "label": "Instrument (ZTF)",
                "model_eligible": True,
                "note": "Raw telescope fields. These are what the model consumes.",
                "fields": instrument,
            },
            "broker_derived": {
                "label": "Broker-derived (Fink / ALeRCE)",
                "model_eligible": False,
                "note": (
                    "Cross-matches and broker classifications. Displayed for "
                    "context only — never a label, never a model input."
                ),
                "fields": broker,
            },
            "system": {
                "label": "System (this demo)",
                "model_eligible": False,
                "note": "Provenance and timing recorded by the demo itself.",
                "fields": system,
            },
        },
    }


@router.get("/disagreements")
def list_disagreements(
    conn: sqlite3.Connection = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    sort: str = Query("margin", pattern="^(margin|recent)$"),
) -> dict:
    """Alerts where the two modalities disagree — where fusion does real work.

    Sorted by ``|max(p_tab) - max(p_img)|`` so the sharpest contradictions
    surface first, which is what makes this usable live rather than a list to
    scroll through.
    """
    order = (
        "ABS(MAX(p.p_tab_sn, p.p_tab_agn, p.p_tab_vs) - "
        "MAX(p.p_img_sn, p.p_img_agn, p.p_img_vs)) DESC"
        if sort == "margin"
        else "a.received_utc DESC"
    )
    rows = conn.execute(
        f"""
        SELECT a.candid, a.object_id, a.topic, a.received_utc, a.magpsf, a.fid,
               p.p_tab_sn, p.p_tab_agn, p.p_tab_vs,
               p.p_img_sn, p.p_img_agn, p.p_img_vs,
               p.p_fused_sn, p.p_fused_agn, p.p_fused_vs,
               p.predicted_class, p.confidence, p.fusion_mode, p.fusion_flips,
               k.coarse AS known_coarse, k.in_training_set AS known_in_training
        FROM predictions p
        JOIN alerts a ON a.candid = p.candid
        LEFT JOIN known_labels k ON k.object_id = a.object_id
        WHERE p.branch_disagree = 1
        ORDER BY {order}
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    items = []
    for row in rows:
        tab = proba_dict(row, "p_tab") or {}
        img = proba_dict(row, "p_img") or {}
        fused = proba_dict(row, "p_fused") or {}
        tab_class = max(tab, key=tab.get) if tab else None
        img_class = max(img, key=img.get) if img else None
        items.append(
            {
                "candid": str(row["candid"]),
                "object_id": row["object_id"],
                "topic": row["topic"],
                "received_utc": row["received_utc"],
                "magpsf": row["magpsf"],
                "p_tab": tab,
                "p_img": img,
                "p_fused": fused,
                "tabular_class": tab_class,
                "image_class": img_class,
                "predicted_class": row["predicted_class"],
                "confidence": row["confidence"],
                "fusion_flips": bool(row["fusion_flips"]),
                "known_coarse": row["known_coarse"],
                "in_training_set": bool(row["known_in_training"] or 0),
                "summary": (
                    f"tabular says {tab_class} ({tab.get(tab_class, 0):.2f}) · "
                    f"image says {img_class} ({img.get(img_class, 0):.2f}) · "
                    f"fused says {row['predicted_class']} "
                    f"({row['confidence'] or 0:.2f})"
                )
                if tab_class and img_class
                else None,
            }
        )

    total = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions WHERE branch_disagree = 1"
    ).fetchone()["n"]
    classified = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions WHERE fusion_mode = 'both'"
    ).fetchone()["n"]
    return {
        "items": items,
        "total_disagreements": total,
        "total_both_branches": classified,
        "disagreement_rate": (total / classified) if classified else None,
    }


@router.get("/provenance")
def get_provenance(config: Settings = Depends(get_settings_dep)) -> dict:
    """The provenance statement, served by the system rather than typed into HTML."""
    return {
        "statement": (
            "Labels come from spectroscopic and catalogue sources only. Brokers "
            "supply alert packets, cutouts and light-curve features — never "
            "labels, and never model input derived from their own "
            "classifications."
        ),
        "label_sources": [
            {"name": "TNS", "detail": "Transient Name Server, spectroscopic classifications"},
            {"name": "BTS", "detail": "ZTF Bright Transient Survey, spectroscopic"},
            {"name": "Chen et al. 2020", "detail": "VizieR J/ApJS/249/18 — ZTF periodic variable stars"},
            {"name": "Milliquas", "detail": "VizieR VII/294 — Million Quasars catalogue"},
            {"name": "SDSS DR16Q", "detail": "VizieR VII/289 — SDSS quasar catalogue"},
        ],
        "brokers_supply": [
            "Alert packets (raw ZTF candidate + prv_candidates fields)",
            "Image cutouts (science / reference / difference)",
            "Light-curve features (ALeRCE feature service — features only)",
        ],
        "brokers_never_supply": [
            "Class labels",
            "Model input derived from broker classifications "
            "(cdsxmatch, finkclass, rf_snia_vs_nonia, snn_*, ALeRCE probabilities)",
        ],
        "enforcement": [
            "Broker values are confined to the broker_meta_json column and the "
            "broker_derived section of the packet view.",
            "The ALeRCE HTTP client denylists any URL containing 'probabilit', "
            "'/classify' or 'classifier_classes' and raises ProvenanceViolation.",
            "Nothing in demo/inference reads NormalisedAlert.broker_meta.",
        ],
        "out_of_scope": [
            "Real/bogus separation. A hook point exists (demo/inference/hooks.py) "
            "and is shown as a skipped stage in the trace; it is deliberately "
            "unimplemented."
        ],
        "taxonomy": {"coarse": list(CLASS_NAMES)},
    }


@router.get("/evaluation")
def get_evaluation(
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
) -> dict:
    """Held-out test metrics per scope, plus the significance caveat.

    Read from the model cards at bootstrap, not recomputed, so these are exactly
    the numbers the thesis reports.
    """
    rows = conn.execute(
        "SELECT * FROM eval_summary ORDER BY "
        "CASE scope WHEN 'tabular' THEN 1 WHEN 'image' THEN 2 "
        "WHEN 'equal_weight' THEN 3 ELSE 4 END"
    ).fetchall()

    significance = None
    card_path: Path = config.fusion_dir / "fusion_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        significance = card.get("significance")

    return {
        "fold": "test",
        "note": (
            "Held-out test fold, read once at the end of the training protocol. "
            "These are offline metrics on the gold set — not a measurement of "
            "live-stream performance."
        ),
        "caveat": (
            "The learned stack beats the tabular branch alone by "
            "Δmacro-F1 = +0.0071 with a bootstrap CI that includes zero. Fusion "
            "is not demonstrated to be significantly superior; the disagreement "
            "cases are shown as qualitative evidence of complementarity."
        ),
        "rows": [
            {
                "scope": r["scope"],
                "label": r["label"],
                "macro_f1": r["macro_f1"],
                "balanced_accuracy": r["balanced_accuracy"],
                "accuracy": r["accuracy"],
                "log_loss": r["log_loss"],
                "split_id": r["split_id"],
            }
            for r in rows
        ],
        "significance": significance,
    }


@router.get("/models")
def get_models(config: Settings = Depends(get_settings_dep)) -> dict:
    """Model-card summaries for the architecture panel."""
    from demo.inference import InferenceEngine

    engine = _engine_cache(config)
    return engine.describe()


_ENGINE = None


def _engine_cache(config: Settings):
    """Load the engine once for description purposes.

    The API does not classify — the consumer does — so this exists only to
    surface the model cards. It is lazily built and reused.
    """
    global _ENGINE
    if _ENGINE is None:
        from demo.inference import InferenceEngine

        _ENGINE = InferenceEngine(config)
    return _ENGINE
