"""Populate demo.db for offline development and demo-day insurance.

Two sources:

``--from gold`` (default)
    Replays objects out of the gold layer as if they had just arrived: real
    63x63 stamp triplets from ``gold_stamps.npz``, real ALeRCE features from
    ``_cache_features/``, real spectroscopic labels from ``gold_labels.parquet``.
    Needs no network and no credentials, and produces a fully populated
    dashboard in about a minute.

    By default it seeds from the **test** split, so the predictions on screen
    are on objects the deployed models never trained on — every row is badged
    with its training split either way, but seeding from test keeps the default
    view honest.

``--from avro``
    Replays real archived alert packets from ``data/demo/raw_alerts``.

Usage
-----
    python scripts/seed_demo_db.py                    # 200 test-split objects
    python scripts/seed_demo_db.py --n 500 --split all
    python scripts/seed_demo_db.py --from avro
    python scripts/seed_demo_db.py --reset            # wipe first
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from demo.config import get_settings  # noqa: E402
from demo.inference import InferenceEngine  # noqa: E402
from demo.models import (  # noqa: E402
    Detection,
    NonDetection,
    NormalisedAlert,
    SourceHealth,
    utc_to_jd,
    utcnow,
)
from demo.storage import db as store  # noqa: E402
from demo.storage.bootstrap import bootstrap  # noqa: E402

log = logging.getLogger("seed")

#: Which synthetic topic a seeded object is attributed to. Chosen so the topic
#: filter has realistic content and so the topic->class selection bias the
#: thesis warns about is visible in the dashboard.
TOPIC_BY_CLASS = {
    "SN": "fink_sn_candidates_ztf",
    "AGN": "fink_blazar_ztf",
    "VS": "fink_magnetic_cvs_ztf",
}


def seed_from_gold(settings, n: int, split: str, spread_minutes: int) -> int:
    import pandas as pd

    gold = settings.gold_dir
    stamps_path = gold / "gold_stamps.npz"
    if not stamps_path.exists():
        log.error("no gold stamps at %s", stamps_path)
        return 0

    npz = np.load(stamps_path, allow_pickle=True)
    oids = npz["oids"].astype(str)
    stamps = npz["stamps"]
    channels = [str(c) for c in npz["channels"]] if "channels" in npz else [
        "science", "reference", "difference"
    ]

    labels = pd.read_parquet(gold / "gold_labels.parquet").set_index("oid")
    meta = pd.read_parquet(gold / "gold_metadata.parquet").set_index("oid")
    splits = pd.read_parquet(gold / "gold_splits.parquet").set_index("oid")

    candidates = list(oids)
    if split != "all":
        wanted = set(splits.index[splits["split"] == split])
        candidates = [o for o in candidates if o in wanted]
        log.info("%d objects in the '%s' split", len(candidates), split)
    if not candidates:
        log.error("no objects matched split=%s", split)
        return 0

    rng = np.random.default_rng(42)
    chosen = list(
        rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    )
    position = {o: i for i, o in enumerate(oids)}

    engine = InferenceEngine(settings)
    conn = store.init_db(settings)
    bootstrap(conn, settings)

    now = utcnow()
    written = 0
    for i, oid in enumerate(chosen):
        row_meta = meta.loc[oid] if oid in meta.index else None
        row_label = labels.loc[oid] if oid in labels.index else None
        coarse = str(row_label["coarse"]) if row_label is not None else "SN"

        # Spread arrivals across the recent past so the time filters and the
        # "seconds since last alert" indicator have something to work with.
        received = now - timedelta(
            seconds=spread_minutes * 60 * (len(chosen) - i) / max(len(chosen), 1)
        )
        jd = utc_to_jd(received - timedelta(minutes=7))  # emitted before ingest

        stack = stamps[position[oid]].astype(np.float32)
        cutouts = {ch: stack[j] for j, ch in enumerate(channels)}

        ndet = int(row_meta["ndet"]) if row_meta is not None and not pd.isna(
            row_meta.get("ndet")
        ) else 12
        detections, nondetections = _synth_lightcurve(jd, ndet, rng)

        alert = NormalisedAlert(
            object_id=str(oid),
            # Deterministic synthetic candid: high bit set so it can never
            # collide with a real ZTF candid if both end up in one database.
            candid=(1 << 62) + abs(hash(oid)) % (10**15),
            source="replay",
            topic=TOPIC_BY_CLASS.get(coarse, "fink_vra_ztf"),
            partition=i % 5,
            offset=1000 + i,
            jd=jd,
            kafka_ts_utc=received - timedelta(seconds=1.3),
            received_utc=received,
            ra=float(row_meta["ra"]) if row_meta is not None else float("nan"),
            dec=float(row_meta["dec"]) if row_meta is not None else float("nan"),
            fid=1,
            magpsf=float(detections[-1].magpsf) if detections else 19.0,
            sigmapsf=0.08,
            diffmaglim=20.4,
            isdiffpos="t",
            ndethist=ndet,
            detections=detections,
            nondetections=nondetections,
            cutouts=cutouts,
            cutout_status="ok",
            broker_meta={
                "_seeded": True,
                "_note": (
                    "Seeded from the gold layer for offline development. Stamps, "
                    "features and labels are real; the alert envelope is synthetic."
                ),
            },
        )

        prediction = engine.classify(alert)
        stamp_name = f"{alert.stamp_key()}.npy"
        settings.stamps_dir.mkdir(parents=True, exist_ok=True)
        np.save(settings.stamps_dir / stamp_name, stack)

        store.save_alert(conn, alert, stamp_path=stamp_name)
        store.save_photometry(conn, alert)
        store.save_prediction(conn, prediction)
        written += 1
        if written % 25 == 0:
            conn.commit()
            log.info("seeded %d/%d", written, len(chosen))

    store.save_health(
        conn,
        SourceHealth(
            connected=True,
            mode="offline",
            topics=tuple(sorted({TOPIC_BY_CLASS.get(c, "fink_vra_ztf")
                                 for c in ("SN", "AGN", "VS")})),
            last_alert_utc=now,
            is_live_stream=False,
        ),
        consumed_total=written,
    )
    conn.commit()
    conn.close()
    engine.close()
    return written


def _synth_lightcurve(jd_now: float, ndet: int, rng) -> tuple[tuple, tuple]:
    """A plausible g/r light curve ending at the alert epoch.

    The photometry is synthetic — only the stamps, features and labels are real.
    The dashboard labels seeded rows accordingly (``broker_meta._seeded``), so
    nothing here can be mistaken for a measurement.
    """
    ndet = int(max(3, min(ndet, 40)))
    detections = []
    for k in range(ndet):
        age = (ndet - k) * 2.0
        fid = 1 if k % 2 == 0 else 2
        mag = 19.5 - 1.5 * np.exp(-((age - 12) ** 2) / 120.0) + rng.normal(0, 0.05)
        detections.append(
            Detection(
                jd=jd_now - age,
                fid=fid,
                magpsf=float(mag),
                sigmapsf=float(abs(rng.normal(0.08, 0.02))),
                diffmaglim=20.5,
                isdiffpos="t",
            )
        )
    nondetections = tuple(
        NonDetection(jd=jd_now - (ndet * 2.0) - 3 * j, fid=1 + j % 2,
                     diffmaglim=float(20.3 + rng.normal(0, 0.2)))
        for j in range(1, 4)
    )
    return tuple(detections), nondetections


def seed_from_avro(settings, n: int) -> int:
    from demo.adapters.replay import ReplayAdapter

    engine = InferenceEngine(settings)
    conn = store.init_db(settings)
    bootstrap(conn, settings)

    adapter = ReplayAdapter(settings, limit=n)
    written = 0
    for alert in adapter.stream():
        prediction = engine.classify(alert)
        stamp_name = None
        stack = alert.stamp_stack()
        if stack is not None:
            stamp_name = f"{alert.stamp_key()}.npy"
            settings.stamps_dir.mkdir(parents=True, exist_ok=True)
            np.save(settings.stamps_dir / stamp_name, stack)
        store.save_alert(conn, alert, stamp_path=stamp_name)
        store.save_photometry(conn, alert)
        store.save_prediction(conn, prediction)
        written += 1
        if written % 25 == 0:
            conn.commit()
    conn.commit()
    conn.close()
    engine.close()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="source", choices=("gold", "avro"),
                        default="gold")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="gold split to seed from (default: test — unseen by the models)",
    )
    parser.add_argument(
        "--spread-minutes",
        type=int,
        default=180,
        help="spread synthetic arrival times over this many minutes",
    )
    parser.add_argument("--reset", action="store_true", help="delete demo.db first")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    settings = get_settings(refresh=True)
    settings.ensure_dirs()

    if args.reset and settings.db_path.exists():
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(settings.db_path) + suffix)
            path.unlink(missing_ok=True)
        log.info("removed %s", settings.db_path)

    if args.source == "gold":
        written = seed_from_gold(settings, args.n, args.split, args.spread_minutes)
    else:
        written = seed_from_avro(settings, args.n)

    log.info("seeded %d alert(s) into %s", written, settings.db_path)
    if written:
        print()
        print("Now start the dashboard:")
        print("  python -m demo.run_api --open")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
