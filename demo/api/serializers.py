"""Database rows -> API response dictionaries.

One rule runs through this module: broker-derived values are never mixed into
the same object as instrument values or predictions. They travel under
``broker_meta`` with a ``broker_derived: true`` marker, so the frontend cannot
render them without knowing what they are.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from demo.config import CLASS_NAMES
from demo.storage.db import parse_json, proba_dict

#: Fields inside broker_meta that are broker *classifications* rather than
#: cross-match facts. Rendered with the strongest warning in the UI.
BROKER_CLASSIFICATION_FIELDS = frozenset(
    {
        "finkclass", "tnsclass", "rf_snia_vs_nonia", "snn_snia_vs_nonia",
        "snn_sn_vs_all", "rf_kn_vs_nonkn", "mulens", "slsn_score",
        "anomaly_score", "t2", "query_class",
    }
)


def alert_row(row: sqlite3.Row, include_broker: bool = True) -> dict[str, Any]:
    """One joined alerts+predictions row -> the alert-stream table shape."""
    out: dict[str, Any] = {
        # Serialised as a STRING, deliberately. A ZTF candid is a 19-digit
        # integer (~3.5e18), well beyond JavaScript's Number.MAX_SAFE_INTEGER
        # (2^53 ~ 9.0e15), so a JSON number would be silently rounded by the
        # browser and every per-alert lookup would 404 on the wrong id.
        "candid": str(row["candid"]),
        "object_id": row["object_id"],
        "source": row["source"],
        "topic": row["topic"],
        "partition": row["kafka_partition"],
        "offset": row["kafka_offset"],
        "jd": row["jd"],
        "mjd": row["mjd"],
        "emitted_utc": row["emitted_utc"],
        "kafka_ts_utc": row["kafka_ts_utc"],
        "received_utc": row["received_utc"],
        "ra": row["ra"],
        "dec": row["dec"],
        "fid": row["fid"],
        "band": {1: "g", 2: "r", 3: "i"}.get(row["fid"], "?"),
        "magpsf": row["magpsf"],
        "sigmapsf": row["sigmapsf"],
        "diffmaglim": row["diffmaglim"],
        "n_det": row["n_det"],
        "n_nondet": row["n_nondet"],
        "cutout_status": row["cutout_status"],
        "has_stamps": bool(row["stamp_path"]),
    }

    out["status"] = _get(row, "status")
    out["status_reason"] = _get(row, "status_reason")
    out["predicted_class"] = _get(row, "predicted_class")
    out["confidence"] = _get(row, "confidence")
    out["fusion_mode"] = _get(row, "fusion_mode")
    out["branch_disagree"] = bool(_get(row, "branch_disagree") or 0)
    out["fusion_flips"] = bool(_get(row, "fusion_flips") or 0)
    out["p_fused"] = proba_dict(row, "p_fused")
    out["p_tab"] = proba_dict(row, "p_tab")
    out["p_img"] = proba_dict(row, "p_img")
    out["feature_provenance"] = _get(row, "feature_provenance")
    out["n_features_present"] = _get(row, "n_features_present")
    out["latency"] = {
        "pipeline_ms": _get(row, "t_pipeline_ms"),
        "broker_to_classified_ms": _get(row, "t_broker_to_classified_ms"),
        "emitted_to_classified_s": _get(row, "t_emitted_to_classified_s"),
        "tabular_ms": _get(row, "t_tab_ms"),
        "image_ms": _get(row, "t_img_ms"),
        "fusion_ms": _get(row, "t_fuse_ms"),
    }
    out["split_id"] = _get(row, "split_id")

    out["known_label"] = _known_label(row)

    if include_broker:
        meta = parse_json(row["broker_meta_json"], {}) or {}
        out["broker_meta"] = {
            "broker_derived": True,
            "note": (
                "Broker classifications are shown for context only. They are "
                "never used as labels or as model input."
            ),
            "classifications": {
                k: v for k, v in meta.items() if k in BROKER_CLASSIFICATION_FIELDS
            },
            "crossmatch": {
                k: v
                for k, v in meta.items()
                if k not in BROKER_CLASSIFICATION_FIELDS and not k.startswith("_")
            },
        }
    return out


def _known_label(row: sqlite3.Row) -> dict[str, Any]:
    """Ground truth from our own label sources, plus the training-leakage flag."""
    coarse = _get(row, "known_coarse")
    return {
        "coarse": coarse,
        "fine": _get(row, "known_fine"),
        "plasticc_class": _get(row, "known_plasticc"),
        "source": _get(row, "known_source"),
        "in_training_set": bool(_get(row, "known_in_training") or 0),
        "training_split": _get(row, "known_split"),
        "correct": _correct(coarse, _get(row, "predicted_class")),
    }


def _correct(truth: str | None, predicted: str | None) -> bool | None:
    if not truth or not predicted:
        return None
    return truth == predicted


def branch_comparison(row: sqlite3.Row) -> dict[str, Any]:
    """The three-way panel: tabular vs image vs fused."""
    tab = proba_dict(row, "p_tab")
    img = proba_dict(row, "p_img")
    fused = proba_dict(row, "p_fused")
    return {
        "class_names": list(CLASS_NAMES),
        "branches": [
            {
                "id": "tabular",
                "label": "Tabular (light curve)",
                "proba": tab,
                "argmax": _argmax(tab),
                "available": tab is not None,
            },
            {
                "id": "image",
                "label": "Image (stamps)",
                "proba": img,
                "argmax": _argmax(img),
                "available": img is not None,
            },
            {
                "id": "fused",
                "label": "Late fusion",
                "proba": fused,
                "argmax": _argmax(fused),
                "available": fused is not None,
            },
        ],
        "fusion_mode": _get(row, "fusion_mode"),
        "branch_disagree": bool(_get(row, "branch_disagree") or 0),
        "fusion_flips": bool(_get(row, "fusion_flips") or 0),
    }


def _argmax(proba: dict | None) -> str | None:
    if not proba:
        return None
    return max(proba.items(), key=lambda kv: kv[1])[0]


def photometry_rows(rows: list[sqlite3.Row]) -> dict[str, Any]:
    """Light-curve points grouped by filter, ready for the plot."""
    series: dict[str, dict[str, list]] = {}
    for row in rows:
        band = {1: "g", 2: "r", 3: "i"}.get(row["fid"], "?")
        bucket = series.setdefault(
            band,
            {
                "detections": [],
                "nondetections": [],
            },
        )
        if row["kind"] == "detection":
            bucket["detections"].append(
                {
                    "jd": row["jd"],
                    "mjd": row["jd"] - 2400000.5,
                    "magpsf": row["magpsf"],
                    "sigmapsf": row["sigmapsf"],
                }
            )
        else:
            bucket["nondetections"].append(
                {
                    "jd": row["jd"],
                    "mjd": row["jd"] - 2400000.5,
                    "diffmaglim": row["diffmaglim"],
                }
            )
    for bucket in series.values():
        bucket["detections"].sort(key=lambda d: d["jd"])
        bucket["nondetections"].sort(key=lambda d: d["jd"])
    return {
        "bands": series,
        "note": (
            "Difference-image PSF magnitudes. Downward triangles are 5-sigma "
            "upper limits (epochs observed with no detection)."
        ),
    }


def _get(row: sqlite3.Row, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None
