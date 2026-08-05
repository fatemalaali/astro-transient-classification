"""One-time database seeding from the gold layer and the model cards.

Two jobs:

1. **``known_labels``** — every object in ``data/gold`` with its coarse/fine
   label, its label source, and (critically) whether it was in the training set
   and in which fold. A live alert for an object the model trained on must be
   badged as such: quoting live accuracy over objects the model has already seen
   is exactly the mistake an examiner will look for.

   ``plasticc_class`` is copied through verbatim and never recomputed, so the
   separate LSST-readiness study (RQ3) is unaffected.

2. **``eval_summary``** — held-out test metrics for tabular-only, image-only,
   equal-weight and learned fusion, read from the model cards rather than
   recomputed, so the dashboard reports exactly the numbers the thesis reports.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from demo.config import Settings
from demo.storage.db import upsert_known_labels

log = logging.getLogger("demo.bootstrap")


def load_known_labels(conn: sqlite3.Connection, settings: Settings) -> int:
    """Populate ``known_labels`` from gold_labels + gold_splits + gold_metadata."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        log.warning("pandas unavailable — skipping known_labels bootstrap")
        return 0

    labels_path = settings.gold_dir / "gold_labels.parquet"
    if not labels_path.exists():
        log.warning("no gold labels at %s — known_labels left empty", labels_path)
        return 0

    labels = pd.read_parquet(labels_path)
    frame = labels.copy()

    splits_path = settings.gold_dir / "gold_splits.parquet"
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)[["oid", "split"]]
        frame = frame.merge(splits, on="oid", how="left")
    else:
        frame["split"] = None

    meta_path = settings.gold_dir / "gold_metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)[["oid", "source"]]
        frame = frame.merge(meta, on="oid", how="left")
    else:
        frame["source"] = "gold"

    rows = [
        (
            str(r.oid),
            _text(getattr(r, "coarse", None)),
            _text(getattr(r, "fine", None)),
            _text(getattr(r, "plasticc_class", None)),
            _text(getattr(r, "source", None)) or "gold",
            1,  # every gold object was part of the modelling data
            _text(getattr(r, "split", None)),
        )
        for r in frame.itertuples(index=False)
    ]
    written = upsert_known_labels(conn, rows)
    conn.commit()
    log.info("known_labels: %d objects from the gold layer", written)
    return written


def load_eval_summary(conn: sqlite3.Connection, settings: Settings) -> int:
    """Populate ``eval_summary`` from the branch and fusion model cards."""
    entries: list[tuple] = []

    def card(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    tab = card(settings.tabular_dir / "model_card.json")
    img = card(settings.image_dir / "model_card.json")
    fus = card(settings.fusion_dir / "fusion_card.json")

    def add(scope: str, label: str, metrics: dict | None, split_id: str | None) -> None:
        if not metrics:
            return
        entries.append(
            (
                scope,
                label,
                metrics.get("macro_f1"),
                metrics.get("balanced_accuracy"),
                metrics.get("accuracy"),
                metrics.get("log_loss"),
                "test",
                split_id,
            )
        )

    add(
        "tabular",
        f"Tabular only ({tab.get('algorithm', 'lightgbm')})",
        tab.get("test_metrics"),
        tab.get("split_id"),
    )
    add(
        "image",
        f"Image only ({img.get('architecture', 'CNN')})",
        img.get("test_metrics"),
        img.get("split_id"),
    )
    baselines = fus.get("baselines") or {}
    add(
        "equal_weight",
        "Equal-weight geometric mean",
        (baselines.get("equal_weight") or {}).get("test_metrics"),
        fus.get("split_id"),
    )
    add(
        "fused",
        "Learned late fusion",
        fus.get("test_metrics"),
        fus.get("split_id"),
    )

    if entries:
        conn.executemany(
            """
            INSERT OR REPLACE INTO eval_summary
                (scope, label, macro_f1, balanced_accuracy, accuracy, log_loss,
                 fold, split_id)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            entries,
        )
        conn.commit()
    log.info("eval_summary: %d scopes from the model cards", len(entries))
    return len(entries)


def bootstrap(conn: sqlite3.Connection, settings: Settings) -> dict:
    return {
        "known_labels": load_known_labels(conn, settings),
        "eval_summary": load_eval_summary(conn, settings),
    }


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "None", "NaT"):
        return None
    return text
