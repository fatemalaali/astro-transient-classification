"""RestPollAdapter — the fallback ingestion path.

Secondary to Kafka, and marked as such everywhere it surfaces. It exists for
two reasons: it needs no credentials, so the whole system can be built while
Fink credentials are pending; and it is the demo-day insurance if the Kafka
cluster is unreachable.

Endpoints and field names verified live against
``https://api.ztf.fink-portal.org`` (Fink/ZTF object API 3.5.0) on 2026-08-05:

* ``POST /api/v1/latests``  ``{class, n, startdate, stopdate, columns, ...}``
  -> one row per alert, columns prefixed ``i:`` (instrument), ``d:`` (Fink
  derived) and ``b:`` (binary).
* ``POST /api/v1/objects``  ``{objectId, withupperlim, withcutouts, ...}``
  -> one row per epoch; ``d:tag`` in ``{valid, upperlim, badquality}``.
* ``POST /api/v1/cutouts``  ``{objectId, candid, kind: "All",
  output-format: "array"}`` -> ``{b:cutoutScience_stampData: 63x63, ...}``.

The ``i:`` / ``d:`` / ``b:`` prefix convention is Fink's own, and it happens to
encode exactly the provenance split this thesis argues for — so the adapter
keeps it rather than flattening it away.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Iterator

import requests

from demo.adapters.base import big_int, build_lightcurve, extract_broker_meta, num
from demo.adapters.cutouts import decode_rest_row
from demo.config import Settings
from demo.models import NormalisedAlert, SourceHealth, utcnow

log = logging.getLogger("demo.fink_rest")

API_BASE = "https://api.ztf.fink-portal.org"

#: Fink classes polled by default, chosen to span SN / AGN / VS.
#:
#: These are used ONLY to decide which alerts to fetch — exactly as
#: build_dataset.ipynb uses ALeRCE only to fetch data for objects whose labels
#: were already fixed by TNS/BTS/VizieR. They are never a label and never a
#: model input. See docs/demo-plan.md section 3.1.
DEFAULT_CLASSES: tuple[str, ...] = (
    "SN candidate",
    "Early SN Ia candidate",
    "(SIMBAD) AGN",
    "(SIMBAD) QSO",
    "(SIMBAD) Blazar",
    "(SIMBAD) RRLyr",
    "(SIMBAD) EB*",
    "(SIMBAD) CataclyV*",
)


def _strip_prefix(row: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in row.items() if k.startswith(prefix)}


class RestPollAdapter:
    """Polls the Fink REST API on a fixed interval and emits new alerts."""

    name = "fink_rest"

    def __init__(
        self,
        settings: Settings,
        classes: tuple[str, ...] = DEFAULT_CLASSES,
        interval_s: float = 60.0,
        per_class: int = 5,
        shutdown: threading.Event | None = None,
        limit: int | None = None,
        with_cutouts: bool = True,
    ) -> None:
        self.settings = settings
        self.classes = classes
        self.interval_s = interval_s
        self.per_class = per_class
        self.shutdown = shutdown or threading.Event()
        self.limit = limit
        self.with_cutouts = with_cutouts
        self._seen: set[int] = set()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "astro-transient-demo/1.0 (MSc thesis)"
        self._last_alert_utc: datetime | None = None
        self._error: str | None = None
        self._emitted = 0
        self.on_idle = None  # heartbeat between polls; see FinkKafkaAdapter

    # ------------------------------------------------------------------ #
    def _post(self, endpoint: str, payload: dict) -> Any:
        url = f"{API_BASE}{endpoint}"
        response = self._session.post(
            url, json=payload, timeout=self.settings.alerce_timeout_s
        )
        response.raise_for_status()
        return response.json()

    def fetch_latests(self, fink_class: str, n: int) -> list[dict]:
        try:
            rows = self._post("/api/v1/latests", {"class": fink_class, "n": str(n)})
            return rows if isinstance(rows, list) else []
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            log.warning("latests(%s) failed: %s", fink_class, exc)
            return []

    def fetch_object_rows(self, object_id: str) -> list[dict]:
        """All epochs for an object, including upper limits (``d:tag``)."""
        try:
            rows = self._post(
                "/api/v1/objects", {"objectId": object_id, "withupperlim": True}
            )
            return rows if isinstance(rows, list) else []
        except Exception as exc:
            log.warning("objects(%s) failed: %s", object_id, exc)
            return []

    def fetch_cutouts(self, object_id: str, candid: int) -> tuple[dict, str]:
        """The triplet as 63x63 arrays for one alert."""
        if not self.with_cutouts:
            return {}, "missing"
        try:
            payload = self._post(
                "/api/v1/cutouts",
                {
                    "objectId": object_id,
                    "candid": str(candid),
                    "kind": "All",
                    "output-format": "array",
                },
            )
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
            return decode_rest_row(payload or {})
        except Exception as exc:
            log.warning("cutouts(%s, %s) failed: %s", object_id, candid, exc)
            return {}, "decode_error"

    # ------------------------------------------------------------------ #
    def normalise(self, row: dict, fink_class: str) -> NormalisedAlert:
        """One ``/api/v1/latests`` row (plus follow-up fetches) -> NormalisedAlert."""
        instrument = _strip_prefix(row, "i:")
        derived = _strip_prefix(row, "d:")
        object_id = str(instrument.get("objectId"))
        # big_int, never num(): a 19-digit candid does not survive float64.
        candid = big_int(instrument.get("candid")) or 0

        detections: tuple = ()
        nondetections: tuple = ()
        epochs = self.fetch_object_rows(object_id)
        if epochs:
            det_rows, nondet_rows = [], []
            for epoch in epochs:
                fields = _strip_prefix(epoch, "i:")
                tag = epoch.get("d:tag", "valid")
                if tag == "upperlim":
                    # Upper limits carry no magpsf; build_lightcurve keys off that.
                    nondet_rows.append(
                        {
                            "jd": fields.get("jd"),
                            "fid": fields.get("fid"),
                            "diffmaglim": fields.get("diffmaglim"),
                        }
                    )
                elif tag == "valid":
                    det_rows.append(fields)
            detections, nondetections = build_lightcurve(
                {}, det_rows + nondet_rows
            )

        cutouts, cutout_status = self.fetch_cutouts(object_id, candid)

        broker_meta = extract_broker_meta(derived)
        # The class used to *select* this alert, recorded as broker metadata so
        # the UI can show it as such. Never a label, never a model input.
        broker_meta["query_class"] = fink_class
        broker_meta["_provenance_note"] = (
            "Fink class used as a fetch filter only — not a label, not model input"
        )

        return NormalisedAlert(
            object_id=object_id,
            candid=candid,
            source="fink_rest",
            topic=f"rest:{fink_class}",
            jd=float(num(instrument.get("jd")) or 0.0),
            received_utc=utcnow(),
            ra=float(num(instrument.get("ra")) or float("nan")),
            dec=float(num(instrument.get("dec")) or float("nan")),
            fid=int(num(instrument.get("fid")) or 0),
            magpsf=num(instrument.get("magpsf")),
            sigmapsf=num(instrument.get("sigmapsf")),
            diffmaglim=num(instrument.get("diffmaglim")),
            isdiffpos=instrument.get("isdiffpos"),
            distnr=num(instrument.get("distnr")),
            magnr=num(instrument.get("magnr")),
            sigmagnr=num(instrument.get("sigmagnr")),
            chinr=num(instrument.get("chinr")),
            sharpnr=num(instrument.get("sharpnr")),
            sgscore1=num(instrument.get("sgscore1")),
            distpsnr1=num(instrument.get("distpsnr1")),
            neargaia=num(instrument.get("neargaia")),
            rb=num(instrument.get("rb")),
            drb=num(instrument.get("drb")),
            ndethist=int(num(instrument.get("ndethist")) or 0) or None,
            ncovhist=int(num(instrument.get("ncovhist")) or 0) or None,
            jdstarthist=num(instrument.get("jdstarthist")),
            jdendhist=num(instrument.get("jdendhist")),
            detections=detections,
            nondetections=nondetections,
            cutouts=cutouts,
            cutout_status=cutout_status,
            broker_meta=broker_meta,
        )

    # ------------------------------------------------------------------ #
    def poll_once(self) -> list[NormalisedAlert]:
        """One sweep across all configured classes. New alerts only."""
        found: list[NormalisedAlert] = []
        for fink_class in self.classes:
            if self.shutdown.is_set():
                break
            for row in self.fetch_latests(fink_class, self.per_class):
                candid = big_int(row.get("i:candid")) or 0
                if candid == 0 or candid in self._seen:
                    continue
                self._seen.add(candid)
                try:
                    found.append(self.normalise(row, fink_class))
                except Exception:
                    log.exception("failed to normalise REST row %s", candid)
        if found:
            self._last_alert_utc = utcnow()
            self._error = None
        return found

    def stream(self) -> Iterator[NormalisedAlert]:
        while not self.shutdown.is_set():
            if self.limit is not None and self._emitted >= self.limit:
                return
            started = time.monotonic()
            for record in self.poll_once():
                self._emitted += 1
                yield record
                if self.limit is not None and self._emitted >= self.limit:
                    return
            if self.on_idle:
                self.on_idle()
            elapsed = time.monotonic() - started
            self.shutdown.wait(max(1.0, self.interval_s - elapsed))

    def commit(self, record: NormalisedAlert) -> None:
        """No-op: REST has no offsets to commit."""

    def health(self) -> SourceHealth:
        return SourceHealth(
            connected=self._error is None,
            mode="rest",
            topics=tuple(f"rest:{c}" for c in self.classes),
            lag_by_topic={},  # deliberately empty: REST has no lag to report
            committed_by_topic={},
            last_alert_utc=self._last_alert_utc,
            error=self._error,
            is_live_stream=False,  # never claim "live" on a polled source
        )

    def close(self) -> None:
        self._session.close()
