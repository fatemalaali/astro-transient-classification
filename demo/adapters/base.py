"""Shared normalisation helpers used by every adapter.

The Avro packet, the REST row and the archived file all describe the same
physical event with different field names and nesting. Everything that turns
one of those into a :class:`~demo.models.NormalisedAlert` lives here, so the
per-source adapters stay thin.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable

from demo.models import Detection, NonDetection

log = logging.getLogger("demo.adapters")

#: Fink science-module and cross-match fields. Copied to ``broker_meta`` for
#: display and excluded from everything else. Sourced from
#: https://doc.ztf.fink-broker.org/broker/science_modules/ (verified 2026-08-05).
BROKER_FIELDS: tuple[str, ...] = (
    # cross-match
    "cdsxmatch", "gcvs", "vsx", "Plx", "e_Plx", "DR3Name", "gaiaVarFlag",
    "gaiaClass", "x4lac", "x3hsp", "mangrove", "spicy_id", "spicy_class", "tns",
    # machine learning
    "rf_snia_vs_nonia", "snn_snia_vs_nonia", "snn_sn_vs_all", "mulens",
    "rf_kn_vs_nonkn", "slsn_score", "anomaly_score", "anomaly_score_beta",
    "anomaly_score_anais", "anomaly_score_emille", "anomaly_score_julien",
    "anomaly_score_maria", "anomaly_score_emille_30days", "anomaly_score_varvara",
    "t2",
    # Fink's own light-curve features. NOT the feature set our tabular model was
    # trained on (that is ALeRCE's, 242 columns) — see docs/demo-plan.md 6.3.
    "lc_features_g", "lc_features_r",
    # standard / derived
    "roid", "nalerthist", "jd_first_real_det", "jdstarthist_dt", "mag_rate",
    "sigma_rate", "lower_rate", "upper_rate", "delta_time", "from_upper",
    "blazar_stats", "is_transient", "tracklet", "kstest_static",
    # derived labels + versions
    "finkclass", "tnsclass", "fink_broker_version", "fink_science_version",
    # Broker processing timestamps. Observed on live schemavsn 4.02 (2026-08-05)
    # but absent from the schema bundled with fink-client 11.0, which is a
    # useful reminder that the live schema moves ahead of the pinned one.
    # They are display-only like everything else here, but they are the reason
    # the trace panel can break the end-to-end latency into ZTF -> Fink ingest,
    # Fink processing, and Fink -> us.
    "brokerIngestTimestamp", "brokerStartProcessTimestamp",
    "brokerEndProcessTimestamp",
    # Per-author anomaly scores; new ones appear without notice.
    "anomaly_score_alexanta",
)

#: Raw ZTF ``candidate`` fields lifted onto the normalised record.
CANDIDATE_FIELDS: tuple[str, ...] = (
    "ra", "dec", "fid", "magpsf", "sigmapsf", "diffmaglim", "isdiffpos",
    "distnr", "magnr", "sigmagnr", "chinr", "sharpnr", "sgscore1", "distpsnr1",
    "neargaia", "rb", "drb", "ndethist", "ncovhist", "jdstarthist", "jdendhist",
)


def num(value: Any) -> float | None:
    """Coerce to float, mapping ZTF/Fink null sentinels to ``None``.

    ZTF uses -999.0 for "not applicable" in several catalogue columns, and the
    REST API returns the string "nan" for missing cross-match values.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "nan", "null", "none", "-"):
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    if out == -999.0:
        return None
    return out


def integer(value: Any) -> int | None:
    """Small integers (fid, ndethist, …). Goes via float, so not for candids."""
    out = num(value)
    return None if out is None else int(out)


def big_int(value: Any) -> int | None:
    """Exact integer parse for values too large for float64.

    A ZTF ``candid`` is a 19-digit integer (~3.5e18), well beyond float64's
    53-bit exact-integer range: ``int(float("3502194355615015028"))`` yields
    ``3502194355615015424``. Routing a candid through :func:`num` therefore
    corrupts it silently, and every downstream lookup keyed on it — cutout
    fetches, database rows, permalinks — targets an alert that does not exist.

    Parses the digits directly instead, never touching a float.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("nan", "null", "none", "-"):
        return None
    # Tolerate a trailing ".0" from a JSON serialiser that used a float type,
    # but refuse anything with real fractional digits — that would mean the
    # value has already lost precision upstream and must not be trusted.
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return int(text)
    except ValueError:
        return None


def is_detection(row: dict) -> bool:
    """A ``prv_candidates`` entry is a detection iff it carries a magnitude.

    Entries with a null ``magpsf`` but a set ``diffmaglim`` are upper limits:
    the field was observed and nothing was found. This is the ZTF convention
    and it is what drives the light-curve plot's downward triangles.
    """
    return num(row.get("magpsf")) is not None


def build_lightcurve(
    candidate: dict, prv_candidates: Iterable[dict] | None
) -> tuple[tuple[Detection, ...], tuple[NonDetection, ...]]:
    """Split ``candidate`` + ``prv_candidates`` into detections and upper limits.

    The current alert's ``candidate`` block is itself a detection and is
    included, so the light curve always contains the epoch being classified.
    Duplicates (same jd, same filter) are collapsed — Fink occasionally repeats
    an epoch across the history window.
    """
    detections: dict[tuple[float, int], Detection] = {}
    nondetections: dict[tuple[float, int], NonDetection] = {}

    def add(row: dict) -> None:
        jd = num(row.get("jd"))
        fid = integer(row.get("fid"))
        if jd is None or fid is None:
            return
        key = (round(jd, 6), fid)
        if is_detection(row):
            detections[key] = Detection(
                jd=jd,
                fid=fid,
                magpsf=float(num(row.get("magpsf"))),
                sigmapsf=num(row.get("sigmapsf")),
                diffmaglim=num(row.get("diffmaglim")),
                isdiffpos=(
                    str(row["isdiffpos"]) if row.get("isdiffpos") is not None else None
                ),
            )
        else:
            limit = num(row.get("diffmaglim"))
            if limit is not None:
                nondetections[key] = NonDetection(jd=jd, fid=fid, diffmaglim=limit)

    for row in prv_candidates or ():
        if isinstance(row, dict):
            add(row)
    if isinstance(candidate, dict):
        add(candidate)

    dets = tuple(sorted(detections.values(), key=lambda d: d.jd))
    nondets = tuple(sorted(nondetections.values(), key=lambda d: d.jd))
    return dets, nondets


def extract_candidate_fields(candidate: dict) -> dict[str, Any]:
    """Pull the instrument-origin scalars we keep, coercing sentinels to None."""
    out: dict[str, Any] = {}
    for name in CANDIDATE_FIELDS:
        raw = candidate.get(name)
        if name == "isdiffpos":
            out[name] = str(raw) if raw is not None else None
        elif name in ("fid", "ndethist", "ncovhist"):
            out[name] = integer(raw)
        else:
            out[name] = num(raw)
    out["fid"] = out.get("fid") or 0
    return out


#: Top-level keys that are raw ZTF or transport, not broker value-added.
_NON_BROKER_TOP_LEVEL = frozenset({
    "schemavsn", "publisher", "objectId", "candid", "candidate",
    "prv_candidates", "prv_forced_sources", "fp_hists",
    "cutoutScience", "cutoutTemplate", "cutoutDifference", "timestamp", "topic",
})


def extract_broker_meta(alert: dict) -> dict[str, Any]:
    """Collect broker-derived values for display. Never read by inference.

    Values are coerced to JSON-safe primitives here so the whole blob can go
    into a single TEXT column and straight out to the browser.

    Anything top-level that is not raw ZTF and not in :data:`BROKER_FIELDS` is
    swept up too. Fink adds value-added fields faster than the pinned schema
    tracks them — live schemavsn 4.02 carries 57 top-level keys against the 25
    in the schema bundled with fink-client 11.0 — and the safe default for an
    unrecognised broker field is to treat it as broker-derived, so a new Fink
    column can never quietly reach a model input.
    """
    meta: dict[str, Any] = {}
    for name in set(BROKER_FIELDS) | (set(alert) - _NON_BROKER_TOP_LEVEL):
        if name not in alert:
            continue
        value = alert[name]
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[name] = value
        elif isinstance(value, dict):
            meta[name] = {str(k): _jsonable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            meta[name] = [_jsonable(v) for v in value]
        else:
            meta[name] = str(value)
    return meta


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
