"""Stamp rendering.

Cutouts are rendered to PNG server-side and cached on disk, which keeps the
frontend to plain ``<img>`` tags with no client-side image processing and no
extra JavaScript dependency.

The default stretch is ``sigmoid``, matching the Fink REST cutout service's own
default, so a stamp here looks like the same stamp on the Fink portal.

Display orientation uses ``origin="lower"`` semantics (row 0 at the bottom),
matching the ``imshow(..., origin="lower")`` used in build_dataset.ipynb's
verification cell. The array served by ``/stamps.npy`` is **unflipped** — it is
exactly what the model consumed.
"""

from __future__ import annotations

import io
import sqlite3

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from demo.api.deps import get_db, settings as get_settings_dep
from demo.config import CHANNEL_ORDER, Settings

router = APIRouter(prefix="/api", tags=["stamps"])

STRETCHES = ("sigmoid", "linear", "sqrt", "log")


def _load_stack(config: Settings, stamp_path: str) -> np.ndarray:
    path = config.stamps_dir / stamp_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"stamp file missing: {stamp_path}")
    return np.load(path)


def _stretch(plane: np.ndarray, mode: str, pmin: float, pmax: float) -> np.ndarray:
    """Map raw pixel values to 0-255 for display only. Never feeds the model."""
    arr = np.nan_to_num(np.asarray(plane, dtype=np.float64), nan=0.0,
                        posinf=0.0, neginf=0.0)
    arr = np.where(np.abs(arr) > 1e30, 0.0, arr)
    lo, hi = np.percentile(arr, [pmin, pmax])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    if mode == "sqrt":
        scaled = np.sqrt(scaled)
    elif mode == "log":
        scaled = np.log1p(9.0 * scaled) / np.log(10.0)
    elif mode == "sigmoid":
        # Same family as Fink's default: a soft S-curve about the midpoint.
        scaled = 1.0 / (1.0 + np.exp(-10.0 * (scaled - 0.5)))
    return (scaled * 255.0).astype(np.uint8)


@router.get("/alerts/{candid}/stamp/{kind}.png")
def stamp_png(
    candid: int,
    kind: str,
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
    stretch: str = Query("sigmoid"),
    invert: bool = Query(False, description="white background, for printing"),
) -> Response:
    if kind not in CHANNEL_ORDER:
        raise HTTPException(
            status_code=400, detail=f"kind must be one of {CHANNEL_ORDER}"
        )
    if stretch not in STRETCHES:
        raise HTTPException(status_code=400, detail=f"stretch must be one of {STRETCHES}")

    row = conn.execute(
        "SELECT stamp_path FROM alerts WHERE candid = ?", (candid,)
    ).fetchone()
    if row is None or not row["stamp_path"]:
        raise HTTPException(status_code=404, detail="no stamps stored for this alert")

    cache = config.png_cache_dir / f"{candid}_{kind}_{stretch}{'_inv' if invert else ''}.png"
    if cache.exists():
        return FileResponse(cache, media_type="image/png")

    stack = _load_stack(config, row["stamp_path"])
    plane = stack[CHANNEL_ORDER.index(kind)]
    pixels = _stretch(plane, stretch, 0.5, 99.5)
    # Row 0 at the bottom, matching imshow(origin="lower").
    pixels = pixels[::-1]
    if invert:
        pixels = 255 - pixels

    try:
        from PIL import Image

        image = Image.fromarray(pixels, mode="L").resize(
            (252, 252), Image.NEAREST
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
    except ImportError:
        raise HTTPException(
            status_code=503, detail="Pillow is required to render stamps"
        )

    try:
        config.png_cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    except OSError:
        pass
    return Response(content=payload, media_type="image/png")


@router.get("/alerts/{candid}/stamps.npy")
def stamps_npy(
    candid: int,
    conn: sqlite3.Connection = Depends(get_db),
    config: Settings = Depends(get_settings_dep),
) -> Response:
    """The raw (3, 63, 63) float32 array the model consumed — for reproducibility."""
    row = conn.execute(
        "SELECT stamp_path FROM alerts WHERE candid = ?", (candid,)
    ).fetchone()
    if row is None or not row["stamp_path"]:
        raise HTTPException(status_code=404, detail="no stamps stored for this alert")
    path = config.stamps_dir / row["stamp_path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="stamp file missing")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{candid}_stamps.npy",
    )
