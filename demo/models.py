"""The normalised alert record — the one type every ingestion source produces.

Field origins are marked in the docstrings and mirrored in ``FIELD_ORIGIN``:

``instrument``  raw ZTF/telescope fields. **Model-eligible.**
``broker``      broker-derived values (Fink science modules, ALeRCE
                classifications). **Display only — never model input.**
``system``      produced by this demo (timings, offsets, status).

The split is not documentation, it is enforced: :func:`partition_packet` is what
the API serves to the packet modal, and the inference layer only ever reads
``instrument``-origin fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Literal, Protocol

import numpy as np

#: JD of 1858-11-17T00:00:00Z, the MJD epoch.
_MJD_OFFSET = 2400000.5
#: Unix epoch expressed as a Julian Date.
_JD_UNIX_EPOCH = 2440587.5

SourceKind = Literal["fink_kafka", "fink_rest", "alerce_rest", "replay"]
CutoutStatus = Literal["ok", "partial", "missing", "decode_error"]
FusionMode = Literal["both", "tabular_only", "image_only", "none"]


def jd_to_utc(jd: float) -> datetime:
    """Julian Date -> timezone-aware UTC datetime.

    ZTF ships ``candidate.jd`` as a UTC-based Julian Date of the exposure
    mid-point, so no TT/TAI correction is applied (doing so would introduce a
    ~69 s offset that is meaningless at our latency resolution).
    """
    return datetime.fromtimestamp(
        (float(jd) - _JD_UNIX_EPOCH) * 86400.0, tz=timezone.utc
    )


def utc_to_jd(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0 + _JD_UNIX_EPOCH


def jd_to_mjd(jd: float) -> float:
    return float(jd) - _MJD_OFFSET


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    """ISO-8601 with a trailing Z — the only timestamp format crossing our wire."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_window(value: str | None) -> timedelta | None:
    """``"1h"``, ``"30m"``, ``"7d"``, ``"90s"`` -> timedelta. ``None`` if unparseable."""
    if not value:
        return None
    text = str(value).strip().lower()
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    if text and text[-1] in units and text[:-1].replace(".", "", 1).isdigit():
        return timedelta(**{units[text[-1]]: float(text[:-1])})
    return None


@dataclass(frozen=True, slots=True)
class Detection:
    """One photometric detection. All magnitudes are difference-image PSF mags."""

    jd: float
    fid: int  # 1=g 2=r 3=i
    magpsf: float
    sigmapsf: float | None = None
    diffmaglim: float | None = None
    isdiffpos: str | None = None  # 't'/'1' positive subtraction, 'f'/'0' negative


@dataclass(frozen=True, slots=True)
class NonDetection:
    """An upper limit: the epoch was observed, nothing was detected."""

    jd: float
    fid: int
    diffmaglim: float


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """What an :class:`AlertSource` reports about itself for the live indicator."""

    connected: bool
    mode: str
    topics: tuple[str, ...] = ()
    lag_by_topic: dict[str, int] = field(default_factory=dict)
    committed_by_topic: dict[str, int] = field(default_factory=dict)
    last_alert_utc: datetime | None = None
    error: str | None = None
    #: True only for a genuine push stream. REST/offline modes set this False so
    #: the UI never displays a fabricated "live" badge or a fake lag number.
    is_live_stream: bool = False


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """Where the 242 tabular features came from, and how complete they are."""

    source: Literal["gold_cache", "disk_cache", "alerce_live", "unavailable"]
    n_present: int
    n_expected: int
    fetched_utc: datetime | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.source != "unavailable"


@dataclass
class NormalisedAlert:
    """One alert, source-agnostic. See docs/demo-plan.md section 6.2."""

    # --- identity ------------------------------------------------- instrument
    object_id: str
    candid: int

    # --- provenance ---------------------------------------------------- system
    source: SourceKind
    topic: str | None = None
    partition: int | None = None
    offset: int | None = None

    # --- timing -------------------------------------------------------------
    jd: float = 0.0  # instrument: exposure mid-point, Julian Date (UTC-based)
    kafka_ts_utc: datetime | None = None  # system: Kafka broker timestamp
    broker_ingest_utc: datetime | None = None  # broker: packet 'timestamp' field
    received_utc: datetime = field(default_factory=utcnow)  # system: our clock

    # --- astrometry ------------------------------------------------ instrument
    ra: float = float("nan")  # degrees, ICRS
    dec: float = float("nan")  # degrees, ICRS

    # --- photometry, this alert ------------------------------------ instrument
    fid: int = 0
    magpsf: float | None = None
    sigmapsf: float | None = None
    diffmaglim: float | None = None
    isdiffpos: str | None = None

    # --- reference-image / catalogue context ----------------------- instrument
    distnr: float | None = None  # arcsec to nearest source in the reference image
    magnr: float | None = None
    sigmagnr: float | None = None
    chinr: float | None = None
    sharpnr: float | None = None
    sgscore1: float | None = None  # PS1 star-galaxy score, 0-1 (1 = star-like)
    distpsnr1: float | None = None  # arcsec
    neargaia: float | None = None  # arcsec
    #: ZTF real/bogus scores. Carried ONLY so the (unimplemented) bogus hook has
    #: its inputs; never a model feature — bogus is out of scope for this thesis.
    rb: float | None = None
    drb: float | None = None

    # --- history counters ------------------------------------------ instrument
    ndethist: int | None = None
    ncovhist: int | None = None
    jdstarthist: float | None = None
    jdendhist: float | None = None

    # --- light curve ----------------------------------------------- instrument
    detections: tuple[Detection, ...] = ()
    nondetections: tuple[NonDetection, ...] = ()

    # --- imaging ---------------------------------------------------- instrument
    #: keys "science" / "reference" / "difference" -> float32 (H, W), native
    #: scale. Sentinels (|v| > 1e30) and NaNs are LEFT INTACT here; repairing
    #: them is the image branch's job, so that serving matches training exactly.
    cutouts: dict[str, np.ndarray | None] = field(default_factory=dict)
    cutout_status: CutoutStatus = "missing"

    # --- broker metadata ------------------------------------------------ broker
    #: cdsxmatch, finkclass, tns, rf_snia_vs_nonia, snn_*, roid, anomaly_score,
    #: lc_features_* … DISPLAY ONLY. Nothing in demo/inference reads this.
    broker_meta: dict[str, Any] = field(default_factory=dict)

    # --- misc ------------------------------------------------------------ system
    raw_packet_ref: str | None = None

    # ------------------------------------------------------------------ #
    @property
    def mjd(self) -> float:
        return jd_to_mjd(self.jd)

    @property
    def emitted_utc(self) -> datetime:
        return jd_to_utc(self.jd)

    @property
    def band(self) -> str:
        from demo.config import FID_TO_BAND

        return FID_TO_BAND.get(int(self.fid or 0), "?")

    @property
    def n_det(self) -> int:
        return len(self.detections)

    @property
    def n_nondet(self) -> int:
        return len(self.nondetections)

    @property
    def n_broker_fields(self) -> int:
        """How many broker-derived fields arrived — a count, never the values.

        The trace panel shows this so an examiner can see that broker metadata
        was present and deliberately unused. Exposing it as a count is what lets
        ``demo/inference`` avoid touching ``broker_meta`` at all, which is the
        structural form of the provenance guarantee (tests/test_provenance.py).
        """
        return len(self.broker_meta)

    @property
    def has_cutouts(self) -> bool:
        from demo.config import CHANNEL_ORDER

        return all(
            isinstance(self.cutouts.get(c), np.ndarray) for c in CHANNEL_ORDER
        )

    def stamp_stack(self) -> np.ndarray | None:
        """(3, H, W) float32 in canonical channel order, or None if incomplete.

        Channel order is science, reference, difference — the order the CNN was
        trained on. Getting this wrong is silent, so it is centralised here.
        """
        from demo.config import CHANNEL_ORDER

        if not self.has_cutouts:
            return None
        planes = [np.asarray(self.cutouts[c], dtype=np.float32) for c in CHANNEL_ORDER]
        shapes = {p.shape for p in planes}
        if len(shapes) != 1:
            return None
        return np.stack(planes, axis=0)

    def stamp_key(self) -> str:
        return f"{self.object_id}_{self.candid}"


@dataclass(frozen=True, slots=True)
class BranchResult:
    """One branch's calibrated output plus the numbers the trace panel shows."""

    name: str  # "tabular" | "image"
    proba: np.ndarray | None  # (3,) calibrated, or None if the branch could not run
    raw: np.ndarray | None = None  # pre-calibration probabilities or logits
    temperature: float = 1.0
    elapsed_ms: float = 0.0
    ok: bool = True
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FusionResult:
    """The fused prediction and the mode that produced it."""

    proba: np.ndarray | None  # (3,)
    mode: FusionMode
    stack_input: np.ndarray | None = None  # the 6-vector fed to the stack
    elapsed_ms: float = 0.0
    reason: str | None = None

    @property
    def predicted_class(self) -> str | None:
        from demo.config import CLASS_NAMES

        if self.proba is None:
            return None
        return CLASS_NAMES[int(np.argmax(self.proba))]

    @property
    def confidence(self) -> float | None:
        return None if self.proba is None else float(np.max(self.proba))


class AlertSource(Protocol):
    """Every ingestion path implements exactly this."""

    name: str

    def stream(self) -> Iterator[NormalisedAlert]:
        """Yield alerts until closed. May block; must honour the shutdown event."""
        ...

    def health(self) -> SourceHealth: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# provenance partitioning — backs the packet modal and the UI badges
# --------------------------------------------------------------------------- #

#: Every instrument-origin field on NormalisedAlert. The inference layer is only
#: permitted to read from this set (asserted by tests/test_provenance.py).
INSTRUMENT_FIELDS = frozenset(
    {
        "object_id", "candid", "jd", "ra", "dec", "fid", "magpsf", "sigmapsf",
        "diffmaglim", "isdiffpos", "distnr", "magnr", "sigmagnr", "chinr",
        "sharpnr", "sgscore1", "distpsnr1", "neargaia", "rb", "drb", "ndethist",
        "ncovhist", "jdstarthist", "jdendhist", "detections", "nondetections",
        "cutouts",
    }
)

SYSTEM_FIELDS = frozenset(
    {
        "source", "topic", "partition", "offset", "kafka_ts_utc", "received_utc",
        "cutout_status", "raw_packet_ref",
    }
)


def partition_packet(alert: NormalisedAlert) -> dict[str, dict[str, Any]]:
    """Split an alert into instrument / broker-derived / system sections.

    This is what ``GET /api/alerts/{candid}/packet`` serves. ALeRCE's Explorer
    shows one flat AVRO table; splitting it is where this demo makes the
    provenance argument visible rather than merely asserting it in prose.
    """
    instrument = {
        "objectId": alert.object_id,
        "candid": alert.candid,
        "jd": alert.jd,
        "mjd": alert.mjd,
        "ra": alert.ra,
        "dec": alert.dec,
        "fid": alert.fid,
        "band": alert.band,
        "magpsf": alert.magpsf,
        "sigmapsf": alert.sigmapsf,
        "diffmaglim": alert.diffmaglim,
        "isdiffpos": alert.isdiffpos,
        "distnr": alert.distnr,
        "magnr": alert.magnr,
        "sigmagnr": alert.sigmagnr,
        "chinr": alert.chinr,
        "sharpnr": alert.sharpnr,
        "sgscore1": alert.sgscore1,
        "distpsnr1": alert.distpsnr1,
        "neargaia": alert.neargaia,
        "rb": alert.rb,
        "drb": alert.drb,
        "ndethist": alert.ndethist,
        "ncovhist": alert.ncovhist,
        "jdstarthist": alert.jdstarthist,
        "jdendhist": alert.jdendhist,
        "n_detections": alert.n_det,
        "n_nondetections": alert.n_nondet,
    }
    system = {
        "source": alert.source,
        "topic": alert.topic,
        "partition": alert.partition,
        "offset": alert.offset,
        "kafka_ts_utc": iso(alert.kafka_ts_utc),
        "received_utc": iso(alert.received_utc),
        "emitted_utc": iso(alert.emitted_utc),
        "cutout_status": alert.cutout_status,
        "raw_packet_ref": alert.raw_packet_ref,
    }
    broker = dict(alert.broker_meta)
    if alert.broker_ingest_utc is not None:
        broker.setdefault("timestamp", iso(alert.broker_ingest_utc))
    return {
        "instrument": instrument,
        "broker_derived": broker,
        "system": system,
    }
