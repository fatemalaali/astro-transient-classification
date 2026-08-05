"""The interfaces the trained models must satisfy.

Defined separately from the implementations so the dashboard can be built and
demoed while training is still in progress: ``demo/inference/stubs.py`` provides
conforming implementations that need no artefacts on disk.

Probability convention, fixed everywhere: a ``(3,)`` float64 array over
``("SN", "AGN", "VS")`` summing to 1, already temperature-calibrated. Each
branch owns its own calibration, using the temperature it fitted on its
forward-chained OOF predictions and stored in its model card.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

_LOG_EPS = 1e-12


@runtime_checkable
class Branch(Protocol):
    """One modality's classifier."""

    name: str
    class_names: tuple[str, ...]
    temperature: float
    available: bool

    def predict_proba(self, batch) -> np.ndarray:
        """(n, 3) calibrated probabilities."""
        ...

    def describe(self) -> dict:
        """Model-card summary for the UI and the per-alert log."""
        ...


@runtime_checkable
class FusionHead(Protocol):
    """Combines branch outputs into one decision."""

    class_names: tuple[str, ...]
    input_columns: tuple[str, ...]
    available: bool

    def fuse(self, p_tab: np.ndarray | None, p_img: np.ndarray | None): ...

    def describe(self) -> dict: ...


# --------------------------------------------------------------------------- #
# probability helpers — same maths as protocol.py, duplicated here only so the
# serving path has no import-time dependency on the notebook-side module.
# --------------------------------------------------------------------------- #
def as_logits(proba) -> np.ndarray:
    """Log-probabilities, usable as logits.

    Temperature scaling on ``log p`` is identical to scaling raw logits, because
    softmax is invariant to per-row additive constants. That equivalence is what
    lets one temperature serve both the probabilistic (LightGBM) and the logit
    (CNN) branch. Mirrors ``protocol.as_logits``.
    """
    return np.log(np.clip(np.asarray(proba, dtype=np.float64), _LOG_EPS, 1.0))


def softmax(z, T: float = 1.0) -> np.ndarray:
    """Numerically stable row-wise softmax with temperature. Mirrors ``protocol.softmax``."""
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def calibrate(proba, temperature: float) -> np.ndarray:
    """Apply temperature scaling to a probability vector.

    Exactly ``cal()`` from fusion_ztf.ipynb: ``softmax(as_logits(p), T)``.
    """
    return softmax(as_logits(proba), T=temperature)
