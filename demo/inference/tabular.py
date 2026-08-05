"""Tabular branch — the trained LightGBM light-curve classifier.

Loads ``models/lc/ztf/lightgbm/model.txt`` with ``lightgbm.Booster`` rather than
the joblib pickle: the native text format carries no scikit-learn version
coupling, so serving survives an environment that does not exactly match the
training one.

Feature names and the calibration temperature are read from ``model_card.json``
and never hardcoded — if the branch is retrained with a different feature set,
this module follows automatically.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from demo.inference.contracts import calibrate

log = logging.getLogger("demo.tabular")


class TabularBranch:
    """LightGBM over 242 ALeRCE light-curve features."""

    name = "tabular"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.available = False
        self.class_names: tuple[str, ...] = ("SN", "AGN", "VS")
        self.feature_names: tuple[str, ...] = ()
        self.temperature = 1.0
        self.algorithm = "lightgbm"
        self.split_id: str | None = None
        self._booster = None
        self._card: dict = {}
        self._load()

    def _load(self) -> None:
        card_path = self.model_dir / "model_card.json"
        if not card_path.exists():
            log.warning("tabular model card not found at %s", card_path)
            return
        self._card = json.loads(card_path.read_text(encoding="utf-8"))
        self.class_names = tuple(self._card.get("class_names", self.class_names))
        self.feature_names = tuple(self._card.get("feature_names", ()))
        self.temperature = float(self._card.get("temperature", 1.0))
        self.algorithm = self._card.get("algorithm", "lightgbm")
        self.split_id = self._card.get("split_id")

        native = self.model_dir / self._card.get("native_model", "model.txt")
        try:
            import lightgbm as lgb

            if native.exists():
                self._booster = lgb.Booster(model_file=str(native))
            else:  # pragma: no cover - fallback for non-LightGBM winners
                import joblib

                self._booster = joblib.load(self.model_dir / "model.joblib")
        except Exception:
            log.exception("could not load the tabular model from %s", self.model_dir)
            return

        # The booster stores its own feature order; if it disagrees with the
        # card, trust the booster — it is what actually scores.
        booster_names = getattr(self._booster, "feature_name", lambda: None)()
        if booster_names and tuple(booster_names) != self.feature_names:
            log.warning(
                "model card feature order differs from the booster; using the booster's"
            )
            self.feature_names = tuple(booster_names)

        self.available = True
        log.info(
            "tabular branch ready: %s, %d features, T=%.4f",
            self.algorithm,
            len(self.feature_names),
            self.temperature,
        )

    # ------------------------------------------------------------------ #
    def build_row(self, features: dict | None) -> np.ndarray:
        """A feature dict -> a (1, n_features) array in the trained column order.

        Missing features become NaN. LightGBM consumes NaN natively
        (``consumes_nan_natively: true`` in the model card), so no imputation is
        applied — imputing here would differ from how the model was trained.
        """
        row = np.full((1, len(self.feature_names)), np.nan, dtype=np.float64)
        if not features:
            return row
        for i, name in enumerate(self.feature_names):
            value = features.get(name)
            if value is None:
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isnan(candidate) and not np.isinf(candidate):
                row[0, i] = candidate
        return row

    def n_present(self, features: dict | None) -> int:
        if not features:
            return 0
        return int(np.count_nonzero(~np.isnan(self.build_row(features)[0])))

    def predict_proba(self, batch) -> np.ndarray:
        """(n, n_features) -> (n, 3) calibrated probabilities."""
        if not self.available:
            raise RuntimeError("tabular branch is not available")
        X = np.asarray(batch, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self._booster.predict(X)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:  # binary boosters return a 1-D vector
            raw = np.column_stack([1.0 - raw, raw])
        return calibrate(raw, self.temperature)

    def predict_one(self, features: dict | None) -> tuple[np.ndarray, np.ndarray, float]:
        """Convenience: ``(calibrated, raw, elapsed_ms)`` for a single alert."""
        started = time.perf_counter()
        X = self.build_row(features)
        raw = np.asarray(self._booster.predict(X), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        proba = calibrate(raw, self.temperature)
        return proba[0], raw[0], (time.perf_counter() - started) * 1e3

    def describe(self) -> dict:
        return {
            "branch": "tabular",
            "available": self.available,
            "algorithm": self.algorithm,
            "n_features": len(self.feature_names),
            "class_names": list(self.class_names),
            "temperature": self.temperature,
            "split_id": self.split_id,
            "train_date": self._card.get("train_date"),
            "test_metrics": self._card.get("test_metrics"),
            "oof_provenance": self._card.get("oof_provenance"),
            "base_provenance": self._card.get("base_provenance"),
            "model_dir": str(self.model_dir),
        }
