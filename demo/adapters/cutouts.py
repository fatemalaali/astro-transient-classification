"""Cutout decoding — the one place transport formats differ per source.

Fink Kafka ships cutouts as gzipped FITS bytes inside
``alert["cutoutScience"]["stampData"]``; the Fink REST API ships them as a
63x63 nested JSON array; ALeRCE ships FITS. All three land here and leave as
``float32 (63, 63)``.

Sentinel values (|v| > 1e30) and NaNs are deliberately **not** repaired here.
``build_dataset.ipynb`` stored raw stamps and let ``robust_normalise`` handle
sentinels at training time, so repairing early would make serving differ from
training. The image branch does it, and only it.
"""

from __future__ import annotations

import gzip
import io
import logging
from typing import Any

import numpy as np

from demo.config import CHANNEL_ORDER, FINK_CUTOUT_KEYS

log = logging.getLogger("demo.cutouts")

#: ZTF native stamp size. Matches Config.cutout_size in build_dataset.ipynb.
CUTOUT_SIZE = 63

_GZIP_MAGIC = b"\x1f\x8b"


class CutoutDecodeError(Exception):
    """Raised when a cutout payload is present but cannot be turned into pixels."""


def fit_size(img: np.ndarray, s: int = CUTOUT_SIZE) -> np.ndarray:
    """Centre-crop or zero-pad a 2-D array to (s, s).

    Copied verbatim in behaviour from ``_fit_size`` in ``build_dataset.ipynb``
    so that a stamp served here is padded/cropped exactly as the training
    stamps were. ZTF cutouts are occasionally 61x63 or 63x61 near chip edges,
    which is precisely the case this handles.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise CutoutDecodeError(f"expected a 2-D stamp, got shape {img.shape}")
    h, w = img.shape
    out = np.zeros((s, s), dtype=np.float32)
    y0 = max((h - s) // 2, 0)
    x0 = max((w - s) // 2, 0)
    cropped = img[y0 : y0 + min(h, s), x0 : x0 + min(w, s)]
    hh, ww = cropped.shape
    yo = (s - hh) // 2
    xo = (s - ww) // 2
    out[yo : yo + hh, xo : xo + ww] = cropped
    return out


def decode_fits_bytes(payload: bytes, gzipped: bool | None = None) -> np.ndarray:
    """gzipped FITS bytes -> float32 2-D array.

    Mirrors ``readstamp()`` in ``fink_client/visualisation.py``, with two
    deliberate differences: no row reversal (that is a *display* convention —
    see ``show_stamps``, which does ``[::-1]`` for plotting only), and gzip
    detection by magic number rather than by survey, so LSST-style uncompressed
    FITS also works.

    Orientation is load-bearing: ``build_dataset.ipynb`` took ``hdul[i].data``
    straight from ALeRCE with no flip, so we must not flip either. See
    ``scripts/compare_stamp_orientation.py``.
    """
    if not payload:
        raise CutoutDecodeError("empty cutout payload")
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - astropy is a soft dependency
        raise CutoutDecodeError(
            "astropy is required to decode FITS cutouts (pip install astropy)"
        ) from exc

    data = bytes(payload)
    if gzipped is None:
        gzipped = data[:2] == _GZIP_MAGIC
    try:
        if gzipped:
            data = gzip.decompress(data)
        with fits.open(io.BytesIO(data), memmap=False) as hdul:
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    return np.asarray(hdu.data, dtype=np.float32)
    except CutoutDecodeError:
        raise
    except Exception as exc:
        raise CutoutDecodeError(f"FITS decode failed: {exc}") from exc
    raise CutoutDecodeError("no HDU in the cutout carried image data")


def decode_array(payload: Any) -> np.ndarray:
    """A nested list / ndarray (the Fink REST ``b:cutout*_stampData`` shape) -> float32."""
    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim != 2:
        raise CutoutDecodeError(f"expected a 2-D array, got shape {arr.shape}")
    return arr


def decode_fink_packet(alert: dict) -> tuple[dict[str, np.ndarray | None], str]:
    """Extract the triplet from a Fink Avro alert dict.

    Returns ``({channel: array|None}, status)`` where status is one of
    ``ok`` / ``partial`` / ``missing`` / ``decode_error``.

    The three cutout fields are nullable in the Fink distribution schema, so
    "declared in the schema" is not "present in this alert" — hence the
    per-channel status rather than a bare exception.
    """
    out: dict[str, np.ndarray | None] = {c: None for c in CHANNEL_ORDER}
    present = 0
    failed = 0
    for fink_key, channel in FINK_CUTOUT_KEYS.items():
        block = alert.get(fink_key)
        if not isinstance(block, dict):
            continue
        payload = block.get("stampData")
        if payload is None:
            continue
        present += 1
        try:
            out[channel] = fit_size(decode_fits_bytes(payload))
        except CutoutDecodeError as exc:
            failed += 1
            log.warning(
                "cutout decode failed for %s/%s: %s",
                alert.get("objectId"),
                fink_key,
                exc,
            )
    if present == 0:
        return out, "missing"
    if failed:
        return out, "decode_error"
    if all(out[c] is not None for c in CHANNEL_ORDER):
        return out, "ok"
    return out, "partial"


def decode_rest_row(row: dict) -> tuple[dict[str, np.ndarray | None], str]:
    """Extract the triplet from a Fink REST ``/api/v1/objects`` row.

    Verified live 2026-08-05: ``withcutouts=true`` returns
    ``b:cutoutScience_stampData`` as a 63x63 nested JSON array, so no
    gzip/FITS step is needed on this path.
    """
    out: dict[str, np.ndarray | None] = {c: None for c in CHANNEL_ORDER}
    present = 0
    failed = 0
    for fink_key, channel in FINK_CUTOUT_KEYS.items():
        payload = row.get(f"b:{fink_key}_stampData")
        if payload is None:
            continue
        present += 1
        try:
            out[channel] = fit_size(decode_array(payload))
        except CutoutDecodeError as exc:
            failed += 1
            log.warning("REST cutout decode failed for %s: %s", fink_key, exc)
    if present == 0:
        return out, "missing"
    if failed:
        return out, "decode_error"
    if all(out[c] is not None for c in CHANNEL_ORDER):
        return out, "ok"
    return out, "partial"
