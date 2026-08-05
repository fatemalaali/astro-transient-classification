"""The image-only -> full-fusion upgrade path.

``scripts/backfill_features.py`` is the operational answer to ALeRCE being
edge-blocked on some networks: ingest whenever, then upgrade the stored alerts
from a network that can reach it. That only works if an alert survives a round
trip through the database intact — scalars, light curve *and* cutout triplet —
and re-classifies identically.

These tests exercise the whole mechanism with the **gold feature cache** as the
feature source, so they prove the upgrade works without needing ALeRCE. The one
thing they cannot cover is the network gate itself, which is checked at runtime
by ``resolver.probe()``.

    python tests/test_backfill.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from demo.config import CHANNEL_ORDER, Settings, get_settings  # noqa: E402
from demo.models import Detection, NonDetection, NormalisedAlert, utcnow  # noqa: E402
from demo.storage import db as store  # noqa: E402


class Skip(Exception):
    """Raised when the gold layer or models are not present."""


def _gold_object() -> tuple[str, np.ndarray]:
    """An object that exists in both the stamp archive and the feature cache."""
    settings = get_settings()
    npz_path = settings.gold_dir / "gold_stamps.npz"
    if not npz_path.exists():
        raise Skip("gold stamps not present")
    npz = np.load(npz_path, allow_pickle=True)
    oids = npz["oids"].astype(str)
    for i, oid in enumerate(oids[:400]):
        if (settings.gold_feature_cache_dir / f"{oid}.json").exists():
            return str(oid), npz["stamps"][i].astype(np.float32)
    raise Skip("no gold object with cached features")


def _temp_settings(tmp: Path) -> Settings:
    settings = get_settings()
    scratch = Settings.from_env()
    scratch.data_dir = tmp
    return scratch


def _make_alert(oid: str, stack: np.ndarray) -> NormalisedAlert:
    return NormalisedAlert(
        object_id=oid,
        candid=3502194355615015028,  # a real 19-digit candid
        source="fink_kafka",
        topic="fink_sn_candidates_ztf",
        partition=9,
        offset=12345,
        jd=2461256.6943519,
        received_utc=utcnow(),
        ra=280.4295164, dec=-6.5621545, fid=1,
        magpsf=18.822468, sigmapsf=0.0899, diffmaglim=20.465876, isdiffpos="t",
        sgscore1=0.727083, distpsnr1=2.217, rb=0.633, drb=0.992, ndethist=16,
        detections=tuple(
            Detection(jd=2461250.0 + i, fid=1 + i % 2, magpsf=19.0 - 0.05 * i,
                      sigmapsf=0.09, diffmaglim=20.4, isdiffpos="t")
            for i in range(8)
        ),
        nondetections=(NonDetection(jd=2461240.0, fid=1, diffmaglim=20.3),),
        cutouts={c: stack[i] for i, c in enumerate(CHANNEL_ORDER)},
        cutout_status="ok",
        broker_meta={"cdsxmatch": "Unknown", "rf_snia_vs_nonia": 0.0},
    )


def test_alert_survives_a_database_round_trip():
    """Everything inference needs must come back out of the database.

    If it does not, the upgrade path is impossible: Fink's queue only holds
    alerts for a few days, so a re-poll is not an option.
    """
    oid, stack = _gold_object()
    with tempfile.TemporaryDirectory() as tmp:
        settings = _temp_settings(Path(tmp))
        settings.ensure_dirs()
        conn = store.init_db(settings)
        alert = _make_alert(oid, stack)

        name = f"{alert.stamp_key()}.npy"
        np.save(settings.stamps_dir / name, alert.stamp_stack())
        store.save_alert(conn, alert, stamp_path=name)
        store.save_photometry(conn, alert)
        conn.commit()

        back = store.load_alert(conn, alert.candid, settings)
        assert back is not None
        assert back.object_id == alert.object_id
        assert back.candid == alert.candid, "candid must round-trip exactly"
        assert back.topic == alert.topic
        assert back.partition == alert.partition and back.offset == alert.offset
        assert abs(back.jd - alert.jd) < 1e-9
        assert back.n_det == alert.n_det
        assert back.n_nondet == alert.n_nondet
        assert back.cutout_status == "ok"
        assert back.has_cutouts, "the cutout triplet must survive"
        assert np.allclose(back.stamp_stack(), alert.stamp_stack()), (
            "stamp pixels must be bit-identical, or the CNN would score "
            "differently after a backfill"
        )
        assert back.broker_meta.get("cdsxmatch") == "Unknown"
        conn.close()


def test_backfill_upgrades_image_only_to_both():
    """With features available, a re-classified alert gains the tabular branch."""
    from demo.inference import InferenceEngine

    oid, stack = _gold_object()
    with tempfile.TemporaryDirectory() as tmp:
        settings = _temp_settings(Path(tmp))
        settings.ensure_dirs()
        conn = store.init_db(settings)
        alert = _make_alert(oid, stack)
        name = f"{alert.stamp_key()}.npy"
        np.save(settings.stamps_dir / name, alert.stamp_stack())
        store.save_alert(conn, alert, stamp_path=name)
        store.save_photometry(conn, alert)

        # --- simulate ingest on a network where ALeRCE is blocked ---------
        settings.alerce_enabled = False
        blocked = InferenceEngine(settings)
        blocked.resolver.settings = settings
        # Deny the gold cache too, so this really is "no features at all".
        blocked.resolver._from_gold_cache = lambda _oid: None
        first = blocked.classify(alert)
        store.save_prediction(conn, first)
        conn.commit()
        blocked.close()

        if not first.image or first.image.proba is None:
            raise Skip("image branch unavailable — models not present")
        assert first.fusion_mode == "image_only"
        assert first.feature_provenance.source == "unavailable"

        pending = store.upgradeable_candids(conn)
        assert (alert.candid, oid) in pending, (
            "the row must be listed as upgradeable, or backfill would skip it"
        )

        # --- now re-run with features reachable (gold cache stands in) ----
        settings.alerce_enabled = True
        engine = InferenceEngine(settings)
        reloaded = store.load_alert(conn, alert.candid, settings)
        second = engine.classify(reloaded)
        store.save_prediction(conn, second)
        conn.commit()

        assert second.feature_provenance.source == "gold_cache"
        assert second.feature_provenance.n_present > 0
        assert second.fusion_mode == "both", "the upgrade must produce two branches"
        assert second.p_tab is not None and second.p_img is not None
        assert second.p_fused is not None

        # The image branch must be unchanged — only the tabular one is new.
        assert np.allclose(first.p_img, second.p_img, atol=1e-6), (
            "re-classification changed the image branch; the stamps did not "
            "round-trip faithfully"
        )

        # And the row must no longer be listed as upgradeable.
        assert (alert.candid, oid) not in store.upgradeable_candids(conn)

        engine.close()
        conn.close()


def test_upgradeable_excludes_alerts_features_cannot_help():
    """Alerts with no detections are not listed — no fetch would fix them."""
    oid, stack = _gold_object()
    with tempfile.TemporaryDirectory() as tmp:
        settings = _temp_settings(Path(tmp))
        settings.ensure_dirs()
        conn = store.init_db(settings)
        alert = _make_alert(oid, stack)
        alert.detections = ()
        alert.nondetections = ()
        store.save_alert(conn, alert, stamp_path=None)
        conn.execute(
            "INSERT INTO predictions (candid, status, fusion_mode, "
            "feature_provenance, created_utc) VALUES (?,?,?,?,?)",
            (alert.candid, "ok", "image_only", "unavailable", "2026-08-05T00:00:00Z"),
        )
        conn.commit()
        assert not store.upgradeable_candids(conn)
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as exc:
            print(f"SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print()
    print("backfill path holds" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
