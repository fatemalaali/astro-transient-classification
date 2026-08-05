"""The inference engine: one alert in, one classified prediction out.

Nothing in this package reads ``NormalisedAlert.broker_meta``. Broker
classifications (``cdsxmatch``, ``finkclass``, ``rf_snia_vs_nonia``, ``snn_*``,
ALeRCE probabilities) travel alongside the prediction for display and have no
path into any model input. That absence is the thesis' central claim, and
``tests/test_provenance.py`` asserts it mechanically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from demo.config import CLASS_NAMES, Settings
from demo.inference.features import FeatureResolver
from demo.inference.fusion import LogisticStackFusion
from demo.inference.hooks import bogus_filter
from demo.models import (
    BranchResult,
    FeatureProvenance,
    FusionResult,
    NormalisedAlert,
    iso,
    utcnow,
)

log = logging.getLogger("demo.inference")


@dataclass
class Prediction:
    """Everything one alert's classification produces."""

    candid: int
    object_id: str
    status: str = "ok"  # ok | unclassified | error
    status_reason: str | None = None

    tabular: BranchResult | None = None
    image: BranchResult | None = None
    fusion: FusionResult | None = None
    feature_provenance: FeatureProvenance | None = None

    t_stamp_ms: float = 0.0
    t_pipeline_ms: float = 0.0
    t_broker_to_classified_ms: float | None = None
    t_emitted_to_classified_s: float | None = None

    model_versions: dict[str, Any] = field(default_factory=dict)
    split_id: str | None = None
    created_utc: datetime = field(default_factory=utcnow)
    trace: list[dict] = field(default_factory=list)

    # --- convenience for the storage layer ----------------------------- #
    @property
    def p_tab(self) -> np.ndarray | None:
        return self.tabular.proba if self.tabular else None

    @property
    def p_img(self) -> np.ndarray | None:
        return self.image.proba if self.image else None

    @property
    def p_fused(self) -> np.ndarray | None:
        return self.fusion.proba if self.fusion else None

    @property
    def fusion_mode(self) -> str:
        return self.fusion.mode if self.fusion else "none"

    @property
    def predicted_class(self) -> str | None:
        return self.fusion.predicted_class if self.fusion else None

    @property
    def confidence(self) -> float | None:
        return self.fusion.confidence if self.fusion else None

    @property
    def branch_disagree(self) -> bool:
        """The two modalities pick different classes — where fusion does real work."""
        if self.p_tab is None or self.p_img is None:
            return False
        return int(np.argmax(self.p_tab)) != int(np.argmax(self.p_img))

    @property
    def fusion_flips(self) -> bool:
        """Fusion chose a class neither branch picked on its own."""
        if self.p_tab is None or self.p_img is None or self.p_fused is None:
            return False
        fused = int(np.argmax(self.p_fused))
        return fused not in (int(np.argmax(self.p_tab)), int(np.argmax(self.p_img)))


class InferenceEngine:
    """Loads the branches once, then classifies alerts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.use_stubs = settings.use_stubs

        if self.use_stubs:
            from demo.inference.stubs import StubImageBranch, StubTabularBranch

            self.tabular: Any = StubTabularBranch()
            self.image: Any = StubImageBranch()
            log.warning("DEMO_USE_STUBS=1 — predictions are seeded stubs, not a model")
        else:
            from demo.inference.image import ImageBranch
            from demo.inference.tabular import TabularBranch

            self.tabular = TabularBranch(settings.tabular_dir)
            self.image = ImageBranch(settings.image_dir)

        self.fusion = LogisticStackFusion(settings.fusion_dir)
        self.resolver = FeatureResolver(
            settings, getattr(self.tabular, "feature_names", ())
        )
        self._check_split_identity()

    def _check_split_identity(self) -> None:
        """Warn loudly if the branches were not trained on the same partition.

        The offline equivalent is ``protocol.assert_same_split``. Here it is a
        warning rather than an assertion: a mismatched demo is still worth
        showing, but nobody should quote its numbers.
        """
        ids = {
            name: getattr(component, "split_id", None)
            for name, component in (
                ("tabular", self.tabular),
                ("image", self.image),
                ("fusion", self.fusion),
            )
        }
        present = {k: v for k, v in ids.items() if v}
        if len(set(present.values())) > 1:
            log.error(
                "SPLIT IDENTITY MISMATCH across components: %s — "
                "these models were not trained on the same partition",
                present,
            )
        self.split_id = next(iter(present.values()), None) if present else None

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return bool(
            getattr(self.tabular, "available", False)
            or getattr(self.image, "available", False)
        )

    def describe(self) -> dict:
        return {
            "tabular": self.tabular.describe(),
            "image": self.image.describe(),
            "fusion": self.fusion.describe(),
            "split_id": self.split_id,
            "class_names": list(CLASS_NAMES),
            "using_stubs": self.use_stubs,
        }

    def model_versions(self) -> dict:
        return {
            "tabular": getattr(self.tabular, "algorithm", "stub"),
            "image": getattr(self.image, "architecture", "stub"),
            "fusion": self.fusion._card.get("fusion_type", "unavailable"),
            "stubs": self.use_stubs,
        }

    # ------------------------------------------------------------------ #
    def classify(self, alert: NormalisedAlert) -> Prediction:
        """Run both branches and fuse. Never raises — failures become status rows."""
        started = time.perf_counter()
        prediction = Prediction(
            candid=alert.candid,
            object_id=alert.object_id,
            split_id=self.split_id,
            model_versions=self.model_versions(),
        )
        trace: list[dict] = []

        # --- stage 1-3: transport, decode, normalise (already done upstream)
        trace.append(
            {
                "id": "kafka" if alert.source == "fink_kafka" else "ingest",
                "label": "Kafka topic" if alert.source == "fink_kafka" else "Ingest",
                "ok": True,
                "detail": {
                    "source": alert.source,
                    "topic": alert.topic,
                    "partition": alert.partition,
                    "offset": alert.offset,
                    "kafka_ts": iso(alert.kafka_ts_utc),
                },
            }
        )
        trace.append(
            {
                "id": "avro",
                "label": "Alert packet",
                "ok": True,
                "detail": {
                    "objectId": alert.object_id,
                    # String: this is rendered in the browser, where a 19-digit
                    # JSON number would be rounded to the nearest 2^k and shown
                    # wrong. Display-only here, but wrong on screen is wrong.
                    "candid": str(alert.candid),
                    "n_detections": alert.n_det,
                    "n_nondetections": alert.n_nondet,
                    "cutouts_present": sum(
                        1 for v in alert.cutouts.values() if v is not None
                    ),
                    # A count, not the values: nothing in this package reads
                    # broker-derived metadata. See NormalisedAlert.n_broker_fields.
                    "broker_fields_present_and_unused": alert.n_broker_fields,
                },
            }
        )
        trace.append(
            {
                "id": "normalise",
                "label": "Normalised record",
                "ok": True,
                "detail": {
                    "ra": alert.ra,
                    "dec": alert.dec,
                    "fid": alert.fid,
                    "band": alert.band,
                    "magpsf": alert.magpsf,
                    "diffmaglim": alert.diffmaglim,
                    "n_detections": alert.n_det,
                    "n_nondetections": alert.n_nondet,
                    "emitted_utc": iso(alert.emitted_utc),
                    "cutout_status": alert.cutout_status,
                },
            }
        )

        # The real/bogus hook is still called so its position in the pipeline is
        # unambiguous, but it is NOT shown in the trace: an always-skipped stage
        # is noise in a demo whose job is to show what the system actually does.
        bogus_filter(alert)

        # --- stage 5: features
        prediction.tabular = self._run_tabular(alert, prediction, trace)

        # --- stage 6: image
        prediction.image = self._run_image(alert, prediction, trace)

        # --- stage 7: fusion
        fusion = self.fusion.fuse(
            prediction.tabular.proba if prediction.tabular else None,
            prediction.image.proba if prediction.image else None,
        )
        prediction.fusion = fusion
        trace.append(
            {
                "id": "fusion",
                "label": "Late fusion",
                "ok": fusion.proba is not None,
                "detail": {
                    "type": self.fusion._card.get(
                        "fusion_type", "unavailable"
                    ),
                    "input_space": self.fusion._card.get("input_space"),
                    "input_columns": list(self.fusion.input_columns),
                    "z": _tolist(fusion.stack_input),
                    "fused": _tolist(fusion.proba),
                    "mode": fusion.mode,
                    "reason": fusion.reason,
                    "elapsed_ms": round(fusion.elapsed_ms, 3),
                },
            }
        )

        # --- timings
        prediction.t_pipeline_ms = (time.perf_counter() - started) * 1e3
        # For a live stream the classification instant is simply "now", and the
        # gap between receipt and classification (queue wait + inference) is
        # real latency worth counting. For a replayed or seeded alert it is not:
        # the packet may be hours old by wall clock, and reporting that as
        # system latency would be plainly false. Anchor on the recorded receipt
        # instead, so a replay reports the processing time it actually took.
        if alert.source == "replay":
            classified_at = alert.received_utc + timedelta(
                milliseconds=prediction.t_pipeline_ms
            )
        else:
            classified_at = utcnow()
        reference = alert.kafka_ts_utc or alert.broker_ingest_utc
        if reference is not None:
            prediction.t_broker_to_classified_ms = (
                classified_at - reference
            ).total_seconds() * 1e3
        if alert.jd:
            prediction.t_emitted_to_classified_s = (
                classified_at - alert.emitted_utc
            ).total_seconds()

        if fusion.proba is None:
            prediction.status = "unclassified"
            prediction.status_reason = fusion.reason
        trace.append(
            {
                "id": "result",
                "label": "Final class",
                "ok": fusion.proba is not None,
                "detail": {
                    "class": prediction.predicted_class,
                    "confidence": prediction.confidence,
                    "fusion_mode": fusion.mode,
                    "branch_disagree": prediction.branch_disagree,
                    "fusion_flips": prediction.fusion_flips,
                    "t_pipeline_ms": round(prediction.t_pipeline_ms, 2),
                    "t_broker_to_classified_ms": _round(
                        prediction.t_broker_to_classified_ms, 1
                    ),
                    "t_emitted_to_classified_s": _round(
                        prediction.t_emitted_to_classified_s, 1
                    ),
                },
            }
        )
        prediction.trace = trace
        return prediction

    # ------------------------------------------------------------------ #
    def _run_tabular(
        self, alert: NormalisedAlert, prediction: Prediction, trace: list[dict]
    ) -> BranchResult:
        started = time.perf_counter()
        features: dict | None = None
        provenance: FeatureProvenance

        if self.use_stubs:
            features = {"_oid": alert.object_id}
            provenance = FeatureProvenance(
                source="disk_cache", n_present=0, n_expected=0
            )
        else:
            features, provenance = self.resolver.resolve(alert.object_id)
        prediction.feature_provenance = provenance
        feature_ms = (time.perf_counter() - started) * 1e3

        trace.append(
            {
                "id": "features",
                "label": "Feature resolve",
                "ok": provenance.ok,
                "detail": {
                    "provenance": provenance.source,
                    "n_present": provenance.n_present,
                    "n_expected": provenance.n_expected,
                    "fetched_utc": iso(provenance.fetched_utc),
                    "error": provenance.error,
                    "elapsed_ms": round(feature_ms, 2),
                },
            }
        )

        # Too few detections: the same rule build_dataset.ipynb used to decide an
        # object was image-branch-only (Config.min_detections = 5).
        if alert.n_det and alert.n_det < self.settings.min_detections:
            reason = (
                f"only {alert.n_det} detections (< {self.settings.min_detections}) — "
                "light-curve features are not reliable"
            )
            trace.append(_branch_stage("tabular", None, None, 0.0, False, reason))
            return BranchResult(
                "tabular", None, elapsed_ms=feature_ms, ok=False, reason=reason
            )

        if not getattr(self.tabular, "available", False):
            reason = "tabular model artefact not loaded"
            trace.append(_branch_stage("tabular", None, None, 0.0, False, reason))
            return BranchResult(
                "tabular", None, elapsed_ms=feature_ms, ok=False, reason=reason
            )

        if features is None:
            reason = provenance.error or "features unavailable"
            trace.append(_branch_stage("tabular", None, None, 0.0, False, reason))
            return BranchResult(
                "tabular", None, elapsed_ms=feature_ms, ok=False, reason=reason
            )

        try:
            proba, raw, elapsed = self.tabular.predict_one(features)
        except Exception as exc:
            log.exception("tabular branch failed for %s", alert.object_id)
            reason = f"{type(exc).__name__}: {exc}"
            trace.append(_branch_stage("tabular", None, None, 0.0, False, reason))
            return BranchResult(
                "tabular", None, elapsed_ms=feature_ms, ok=False, reason=reason
            )

        trace.append(
            _branch_stage(
                "tabular",
                proba,
                raw,
                elapsed,
                True,
                None,
                temperature=getattr(self.tabular, "temperature", 1.0),
                model=getattr(self.tabular, "algorithm", "stub"),
            )
        )
        return BranchResult(
            "tabular",
            proba,
            raw=raw,
            temperature=getattr(self.tabular, "temperature", 1.0),
            elapsed_ms=elapsed,
        )

    def _run_image(
        self, alert: NormalisedAlert, prediction: Prediction, trace: list[dict]
    ) -> BranchResult:
        started = time.perf_counter()
        stack = alert.stamp_stack()
        prediction.t_stamp_ms = (time.perf_counter() - started) * 1e3

        if stack is None:
            reason = f"cutouts {alert.cutout_status}"
            trace.append(_branch_stage("image", None, None, 0.0, False, reason))
            return BranchResult("image", None, ok=False, reason=reason)

        if not getattr(self.image, "available", False):
            reason = "image model artefact not loaded"
            trace.append(_branch_stage("image", None, None, 0.0, False, reason))
            return BranchResult("image", None, ok=False, reason=reason)

        try:
            proba, raw, elapsed = self.image.predict_one(stack)
        except Exception as exc:
            log.exception("image branch failed for %s", alert.object_id)
            reason = f"{type(exc).__name__}: {exc}"
            trace.append(_branch_stage("image", None, None, 0.0, False, reason))
            return BranchResult("image", None, ok=False, reason=reason)

        trace.append(
            _branch_stage(
                "image",
                proba,
                raw,
                elapsed,
                True,
                None,
                temperature=getattr(self.image, "temperature", 1.0),
                model=getattr(self.image, "architecture", "stub"),
                extra={
                    "input": f"3x{stack.shape[1]}x{stack.shape[2]} -> "
                    f"3x{getattr(self.image, 'input_size', 160)}x"
                    f"{getattr(self.image, 'input_size', 160)} bilinear",
                    "channel_order": list(
                        getattr(self.image, "channel_order", CLASS_NAMES)
                    ),
                },
            )
        )
        return BranchResult(
            "image",
            proba,
            raw=raw,
            temperature=getattr(self.image, "temperature", 1.0),
            elapsed_ms=elapsed,
        )

    def close(self) -> None:
        self.resolver.close()


# --------------------------------------------------------------------------- #
def _tolist(arr: np.ndarray | None) -> list | None:
    return None if arr is None else [float(v) for v in np.asarray(arr).ravel()]


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _branch_stage(
    branch: str,
    proba: np.ndarray | None,
    raw: np.ndarray | None,
    elapsed_ms: float,
    ok: bool,
    reason: str | None,
    temperature: float = 1.0,
    model: str | None = None,
    extra: dict | None = None,
) -> dict:
    detail: dict[str, Any] = {
        "model": model,
        "raw": _tolist(raw),
        "T": temperature,
        "calibrated": _tolist(proba),
        "elapsed_ms": round(elapsed_ms, 3),
        "reason": reason,
    }
    if extra:
        detail.update(extra)
    return {
        "id": branch,
        "label": f"{branch.capitalize()} branch",
        "ok": ok,
        "detail": detail,
    }
