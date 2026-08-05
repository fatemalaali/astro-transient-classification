"""Stub branches — so the dashboard is buildable before training finishes.

Set ``DEMO_USE_STUBS=1`` and the whole pipeline runs with no ``.pt``, ``.txt``
or ``.joblib`` on disk: adapters, storage, API and frontend are all exercised
end to end.

Outputs are a seeded Dirichlet draw keyed on ``object_id``, so they are stable
across restarts (the same object always gets the same numbers — screenshots
stay valid) and non-degenerate (they are not all 1/3, so the UI renders real
bars and disagreements actually occur). The two branches use different seed
salts, so they disagree at a realistic rate and the disagreement panel has
content.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from demo.config import CLASS_NAMES

#: Roughly the gold-set class prior (SN 7728 / VS 3517 / AGN 581), so a stubbed
#: stream looks plausible rather than uniform.
_PRIOR = np.array([6.0, 0.8, 3.0])


def _seed(text: str, salt: str) -> int:
    digest = hashlib.sha1(f"{salt}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class _StubBranch:
    available = True
    class_names = CLASS_NAMES
    temperature = 1.0
    _salt = "stub"
    _concentration = 1.0

    def __init__(self, name: str) -> None:
        self.name = name

    def _draw(self, key: str) -> np.ndarray:
        rng = np.random.default_rng(_seed(key, self._salt))
        return rng.dirichlet(_PRIOR * self._concentration)

    def predict_one(self, payload) -> tuple[np.ndarray, np.ndarray, float]:
        started = time.perf_counter()
        key = str(payload) if isinstance(payload, str) else repr(type(payload))
        proba = self._draw(key)
        return proba, proba, (time.perf_counter() - started) * 1e3

    def describe(self) -> dict:
        return {
            "branch": self.name,
            "available": True,
            "stub": True,
            "class_names": list(self.class_names),
            "temperature": self.temperature,
            "note": "seeded stub — no trained artefact loaded (DEMO_USE_STUBS=1)",
        }


class StubTabularBranch(_StubBranch):
    _salt = "tabular"
    _concentration = 2.5  # sharper: the tabular branch is the stronger one
    feature_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        super().__init__("tabular")

    def build_row(self, features):  # pragma: no cover - shape parity only
        return np.zeros((1, 0))

    def n_present(self, features) -> int:
        return len(features or {})

    def predict_one(self, features):
        started = time.perf_counter()
        key = (features or {}).get("_oid", "unknown")
        proba = self._draw(str(key))
        return proba, proba, (time.perf_counter() - started) * 1e3


class StubImageBranch(_StubBranch):
    _salt = "image"
    _concentration = 1.2  # flatter: the image branch is the weaker one
    input_size = 160
    channel_order = ("science", "reference", "difference")

    def __init__(self) -> None:
        super().__init__("image")

    def predict_one(self, stamp):
        started = time.perf_counter()
        arr = np.asarray(stamp, dtype=np.float64)
        # Key on the pixel content so the same stamp always scores the same.
        key = f"{arr.shape}:{float(np.nansum(arr)):.4f}"
        proba = self._draw(key)
        return proba, np.log(np.clip(proba, 1e-12, 1.0)), (
            time.perf_counter() - started
        ) * 1e3
