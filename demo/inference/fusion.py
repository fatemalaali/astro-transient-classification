"""Late fusion — the trained multinomial logistic stack.

The deployed head is read from ``models/fusion/logreg_stack/fusion_card.json``
as plain text (``W`` 3x6, ``b`` 3) rather than from ``meta_learner.pt``. Two
reasons: a 21-parameter linear map does not justify a torch dependency in the
serving path, and weights that live in a readable JSON file are auditable by an
examiner without running anything.

Scoring, matching fusion_ztf.ipynb exactly::

    z        = [log p_tab_cal ; log p_img_cal]        # 6-vector, log space
    p_fused  = softmax(z @ W.T + b)

Both inputs are already temperature-calibrated by their own branch.

**Missing modalities are handled by falling back to the surviving branch, never
by imputing.** Feeding a uniform [1/3, 1/3, 1/3] into the stack would be
off-manifold: ``W`` was fitted only on OOF rows where both branches were
present, so it would apply a systematic, uncalibrated shift to a vector it has
never seen. Returning the surviving branch's own calibrated output is the
honest degradation, and the mode is recorded on every row.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from demo.inference.contracts import as_logits, softmax
from demo.models import FusionResult

log = logging.getLogger("demo.fusion")


class LogisticStackFusion:
    """The deployed learned stack, with an equal-weight baseline for comparison."""

    name = "logreg_stack"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.available = False
        self.class_names: tuple[str, ...] = ("SN", "AGN", "VS")
        self.input_columns: tuple[str, ...] = (
            "log_p_tab_SN", "log_p_tab_AGN", "log_p_tab_VS",
            "log_p_img_SN", "log_p_img_AGN", "log_p_img_VS",
        )
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.split_id: str | None = None
        self._card: dict = {}
        self._load()

    def _load(self) -> None:
        card_path = self.model_dir / "fusion_card.json"
        if not card_path.exists():
            log.warning("fusion card not found at %s", card_path)
            return
        self._card = json.loads(card_path.read_text(encoding="utf-8"))
        self.class_names = tuple(self._card.get("class_names", self.class_names))
        self.input_columns = tuple(
            self._card.get("input_columns", self.input_columns)
        )
        self.split_id = self._card.get("split_id")
        try:
            self.W = np.asarray(self._card["W"], dtype=np.float64)
            self.b = np.asarray(self._card["b"], dtype=np.float64)
        except KeyError:
            log.error("fusion card at %s carries no W/b", card_path)
            return
        n_classes = len(self.class_names)
        if self.W.shape != (n_classes, 2 * n_classes) or self.b.shape != (n_classes,):
            log.error(
                "unexpected fusion weight shapes W=%s b=%s for %d classes",
                self.W.shape,
                self.b.shape,
                n_classes,
            )
            return
        self.available = True
        log.info(
            "fusion head ready: %s, W%s, effective_w=%s",
            self._card.get("fusion_type"),
            self.W.shape,
            self._card.get("blend", {}).get("learned_effective_w"),
        )

    # ------------------------------------------------------------------ #
    def stack_input(self, p_tab: np.ndarray, p_img: np.ndarray) -> np.ndarray:
        """``z = [log p_tab ; log p_img]`` — the 6-vector the stack consumes."""
        return np.concatenate([as_logits(p_tab), as_logits(p_img)])

    def fuse(
        self, p_tab: np.ndarray | None, p_img: np.ndarray | None
    ) -> FusionResult:
        started = time.perf_counter()

        if p_tab is None and p_img is None:
            return FusionResult(
                proba=None,
                mode="none",
                reason="neither branch produced a prediction",
                elapsed_ms=(time.perf_counter() - started) * 1e3,
            )
        if p_img is None:
            return FusionResult(
                proba=np.asarray(p_tab, dtype=np.float64),
                mode="tabular_only",
                reason="image branch unavailable — no imputation applied",
                elapsed_ms=(time.perf_counter() - started) * 1e3,
            )
        if p_tab is None:
            return FusionResult(
                proba=np.asarray(p_img, dtype=np.float64),
                mode="image_only",
                reason="tabular branch unavailable — no imputation applied",
                elapsed_ms=(time.perf_counter() - started) * 1e3,
            )
        if not self.available:
            proba = self.equal_weight(p_tab, p_img)
            return FusionResult(
                proba=proba,
                mode="both",
                reason="learned stack unavailable — equal-weight geometric mean used",
                elapsed_ms=(time.perf_counter() - started) * 1e3,
            )

        z = self.stack_input(np.asarray(p_tab), np.asarray(p_img))
        proba = softmax(z @ self.W.T + self.b)
        return FusionResult(
            proba=proba,
            mode="both",
            stack_input=z,
            elapsed_ms=(time.perf_counter() - started) * 1e3,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def equal_weight(
        p_tab: np.ndarray, p_img: np.ndarray, w: float = 0.5
    ) -> np.ndarray:
        """Weighted geometric mean of the two calibrated vectors.

        This is baseline (b) from fusion_ztf.ipynb — ``geometric_mean``, not an
        arithmetic average. With log-probability inputs it is exactly the point
        the learned stack's penalty is centred on, which is what makes the two
        comparable.

        **Weight selection.** If a fixed blend is ever preferred over the
        learned head, ``w`` must be chosen by grid search minimising log-loss on
        the forward-chained OOF predictions. The card records
        ``blend.oof_optimal_w`` for exactly that purpose. It also records
        ``test_optimal_w``; selecting on that would be a protocol violation
        (see ``protocol.select_winner``) and it is reported only to show how far
        the test optimum drifts from the OOF one.
        """
        return softmax(w * as_logits(p_tab) + (1.0 - w) * as_logits(p_img))

    def describe(self) -> dict:
        blend = self._card.get("blend", {})
        return {
            "component": "fusion",
            "available": self.available,
            "fusion_type": self._card.get("fusion_type"),
            "input_space": self._card.get("input_space"),
            "input_columns": list(self.input_columns),
            "class_names": list(self.class_names),
            "W": self.W.tolist() if self.W is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
            "penalty": self._card.get("penalty"),
            "meta_fit_fold": self._card.get("meta_fit_fold"),
            "split_id": self.split_id,
            "train_date": self._card.get("train_date"),
            "test_metrics": self._card.get("test_metrics"),
            "baselines": self._card.get("baselines"),
            "blend": blend,
            "significance": self._card.get("significance"),
            "model_dir": str(self.model_dir),
        }
