"""Image branch — the trained EfficientNet-B0 stamp classifier.

Loaded from ``model_scripted.pt`` (TorchScript), so serving needs neither
``timm`` nor ``torchvision``.

Preprocessing is reproduced **exactly** as in ``stamp_classifier_ztf.ipynb``,
because any divergence here is silent and would degrade accuracy without
raising anything:

1. ``np.nan_to_num(nan=0, posinf=0, neginf=0)`` — applied at *fetch* time by
   ``_fetch_stamp`` in build_dataset.ipynb, so gold stamps never contained NaN.
   Kafka cutouts can, so it must be applied here to match.
2. ``robust_normalise``: sentinel ``|v| > 1e30 -> 0``; clip to the per-image,
   per-channel 1st-99th percentile; z-score by ``(median, std + 1e-6)``.
3. Bilinear interpolation 63 -> 160 with ``align_corners=False``.
   Note the order: normalisation happens at 63x63, *before* upsampling.
4. Forward pass -> logits -> ``softmax(logits / T)`` with ``T = 3.0176``.

``softmax(logits/T)`` is identical to the notebook's
``softmax(as_logits(softmax(logits)), T)`` because softmax is invariant to
per-row additive constants — one fewer round trip, same numbers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("demo.image")


def robust_normalise(arr: np.ndarray) -> np.ndarray:
    """Sentinel repair + per-image, per-channel percentile clip + z-score.

    Verbatim behaviour from ``robust_normalise`` in stamp_classifier_ztf.ipynb.
    Input ``(n, 3, H, W)`` float32, output the same shape.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where(np.abs(arr) > 1e30, 0.0, arr).astype(np.float32)
    lo = np.percentile(arr, 1, axis=(2, 3), keepdims=True)
    hi = np.percentile(arr, 99, axis=(2, 3), keepdims=True)
    arr = np.clip(arr, lo, hi)
    med = np.median(arr, axis=(2, 3), keepdims=True)
    std = arr.std(axis=(2, 3), keepdims=True)
    return (arr - med) / (std + 1e-6)


class ImageBranch:
    """EfficientNet-B0 over science/reference/difference cutouts."""

    name = "image"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.available = False
        self.class_names: tuple[str, ...] = ("SN", "AGN", "VS")
        self.channel_order: tuple[str, ...] = ("science", "reference", "difference")
        self.input_size = 160
        self.temperature = 1.0
        self.architecture = "effnet_b0"
        self.split_id: str | None = None
        self._model = None
        self._torch = None
        self._card: dict = {}
        self._load()

    def _load(self) -> None:
        card_path = self.model_dir / "model_card.json"
        if not card_path.exists():
            log.warning("image model card not found at %s", card_path)
            return
        self._card = json.loads(card_path.read_text(encoding="utf-8"))
        self.class_names = tuple(self._card.get("class_names", self.class_names))
        self.channel_order = tuple(
            self._card.get("channel_order", self.channel_order)
        )
        self.input_size = int(self._card.get("input_size", 160))
        self.temperature = float(self._card.get("temperature", 1.0))
        self.architecture = self._card.get("architecture", "effnet_b0")
        self.split_id = self._card.get("split_id")

        # preprocess.json is the authority on the serving transform; assert it
        # still describes what this module implements.
        pre_path = self.model_dir / "preprocess.json"
        if pre_path.exists():
            pre = json.loads(pre_path.read_text(encoding="utf-8"))
            self.input_size = int(pre.get("input_size", self.input_size))
            self.channel_order = tuple(pre.get("channel_order", self.channel_order))
            if "1e30" not in pre.get("normalisation", ""):
                log.warning(
                    "preprocess.json describes an unexpected normalisation (%r); "
                    "demo/inference/image.py may no longer match training",
                    pre.get("normalisation"),
                )

        scripted = self.model_dir / "model_scripted.pt"
        if not scripted.exists():
            log.warning("TorchScript model not found at %s", scripted)
            return
        try:
            import torch

            self._torch = torch
            self._model = torch.jit.load(str(scripted), map_location="cpu")
            self._model.eval()
        except Exception:
            log.exception("could not load the image model from %s", scripted)
            return

        self.available = True
        log.info(
            "image branch ready: %s, input %d, channels %s, T=%.4f",
            self.architecture,
            self.input_size,
            "/".join(self.channel_order),
            self.temperature,
        )

    # ------------------------------------------------------------------ #
    def preprocess(self, stamps: np.ndarray):
        """(n, 3, 63, 63) raw -> torch tensor (n, 3, input_size, input_size)."""
        torch = self._torch
        import torch.nn.functional as F

        arr = np.asarray(stamps, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        normalised = robust_normalise(arr)
        tensor = torch.from_numpy(np.ascontiguousarray(normalised)).float()
        if tensor.shape[-1] != self.input_size:
            tensor = F.interpolate(
                tensor,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def predict_proba(self, batch) -> np.ndarray:
        """(n, 3, 63, 63) -> (n, 3) calibrated probabilities."""
        proba, _logits, _ms = self._forward(batch)
        return proba

    def _forward(self, batch) -> tuple[np.ndarray, np.ndarray, float]:
        if not self.available:
            raise RuntimeError("image branch is not available")
        torch = self._torch
        started = time.perf_counter()
        tensor = self.preprocess(batch)
        with torch.no_grad():
            logits = self._model(tensor).cpu().numpy().astype(np.float64)
        # softmax(logits / T) — the temperature-scaled probabilities.
        z = logits / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        proba = e / e.sum(axis=1, keepdims=True)
        return proba, logits, (time.perf_counter() - started) * 1e3

    def predict_one(self, stamp: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Convenience: ``(calibrated, raw_logits, elapsed_ms)`` for one stamp."""
        proba, logits, elapsed = self._forward(np.asarray(stamp)[None, ...])
        return proba[0], logits[0], elapsed

    def describe(self) -> dict:
        return {
            "branch": "image",
            "available": self.available,
            "architecture": self.architecture,
            "input_size": self.input_size,
            "channel_order": list(self.channel_order),
            "class_names": list(self.class_names),
            "temperature": self.temperature,
            "split_id": self.split_id,
            "train_date": self._card.get("train_date"),
            "test_metrics": self._card.get("test_metrics"),
            "oof_provenance": self._card.get("oof_provenance"),
            "base_provenance": self._card.get("base_provenance"),
            "model_dir": str(self.model_dir),
        }
