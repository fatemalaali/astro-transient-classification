"""Upgrade image-only alerts to full two-branch fusion, once ALeRCE is reachable.

This is the operational half of "reach ALeRCE from a different network". You
cannot always classify with both branches at ingest time — on a network where
ALeRCE is edge-blocked (HTTP 403 from ``awselb``), every new object falls back
to ``image_only``. But nothing is lost: the alert, its light curve and its cutout
triplet are all persisted, so the tabular branch can be run later and the
prediction re-fused.

So the workflow becomes:

    # 1. ingest whenever you like, on whatever network you have
    python -m demo.run_consumer --mode live

    # 2. later, on a network that can reach ALeRCE (hotspot / campus / VPN)
    python scripts/backfill_features.py

    # 3. the dashboard now shows two-branch fusion for those alerts

Alerts stay in Fink's queue for only a few days, but this reads from *our*
database, so it works on alerts of any age.

Options
-------
    --dry-run     report what would change, touch nothing
    --limit N     cap the number of objects fetched (default: all)
    --workers N   parallel feature fetches (default 4; be polite)
    --recheck     also retry objects previously marked unavailable in the cache

Exit codes: 0 = something was upgraded or nothing needed doing, 1 = ALeRCE is
unreachable from this network (the whole point of the script), 2 = setup error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.config import get_settings  # noqa: E402
from demo.inference import InferenceEngine  # noqa: E402
from demo.storage import db as store  # noqa: E402

log = logging.getLogger("backfill")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--recheck", action="store_true",
        help="retry objects ALeRCE previously had no features for",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    settings = get_settings(refresh=True)
    if not settings.db_path.exists():
        log.error("no database at %s — run the consumer first", settings.db_path)
        return 2
    if not settings.alerce_enabled:
        log.error(
            "DEMO_ALERCE_ENABLED=0 — this script exists to use ALeRCE. "
            "Unset it (or set it to 1) and re-run."
        )
        return 2

    conn = store.init_db(settings)
    todo = store.upgradeable_candids(conn)
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        log.info("nothing to upgrade — every alert already has a tabular branch")
        conn.close()
        return 0

    objects = sorted({oid for _, oid in todo})
    log.info(
        "%d alert(s) across %d object(s) are missing the tabular branch",
        len(todo), len(objects),
    )

    engine = InferenceEngine(settings)
    resolver = engine.resolver

    # --- the gate: is ALeRCE actually reachable from this network? --------
    log.info("probing %s ...", resolver.status()["endpoint"])
    if not resolver.probe(objects[0]):
        status = resolver.status()
        log.error("")
        log.error("ALeRCE is NOT reachable from this network (%s).", status["reason"])
        log.error("")
        log.error("This script only helps once you are on a network that can reach it.")
        log.error("Try: phone hotspot, the Polytechnic network, or a VPN, then re-run.")
        log.error("Verify with:  python scripts/check_connectivity.py")
        log.error("")
        log.error("Meanwhile the demo still works — the replay path is unaffected:")
        log.error("  python -m demo.run_consumer --mode replay")
        engine.close()
        conn.close()
        return 1
    log.info("ALeRCE is reachable — fetching features")

    # --- fetch features per object, in parallel ---------------------------
    resolved: dict[str, bool] = {}

    def fetch(oid: str) -> tuple[str, bool]:
        if args.recheck:
            cache = settings.feature_cache_dir / f"{oid}.json"
            cache.unlink(missing_ok=True)
        features, provenance = resolver.resolve(oid)
        return oid, bool(features) and provenance.n_present > 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch, oid): oid for oid in objects}
        for i, future in enumerate(as_completed(futures), 1):
            oid, ok = future.result()
            resolved[oid] = ok
            if i % 20 == 0 or i == len(objects):
                log.info("  fetched %d/%d objects", i, len(objects))

    got = sum(resolved.values())
    log.info(
        "features obtained for %d/%d object(s) (%d have none in ALeRCE yet)",
        got, len(objects), len(objects) - got,
    )
    if not got:
        log.warning(
            "ALeRCE is reachable but has no features for any of these objects — "
            "typical for very new transients it has not featurised yet."
        )
        engine.close()
        conn.close()
        return 0

    if args.dry_run:
        log.info("--dry-run: would re-classify %d alert(s); nothing written",
                 sum(1 for _, oid in todo if resolved.get(oid)))
        engine.close()
        conn.close()
        return 0

    # --- re-classify the affected alerts ----------------------------------
    upgraded = failed = skipped = 0
    for candid, oid in todo:
        if not resolved.get(oid):
            skipped += 1
            continue
        alert = store.load_alert(conn, candid, settings)
        if alert is None:
            failed += 1
            continue
        before = conn.execute(
            "SELECT predicted_class, confidence FROM predictions WHERE candid = ?",
            (candid,),
        ).fetchone()
        prediction = engine.classify(alert)
        store.save_prediction(conn, prediction)
        upgraded += 1
        if prediction.fusion_mode == "both":
            changed = (
                before
                and before["predicted_class"] != prediction.predicted_class
            )
            log.info(
                "  %s %s -> %s (%.2f) %s",
                oid,
                (before["predicted_class"] if before else "?"),
                prediction.predicted_class,
                prediction.confidence or 0.0,
                "** CLASS CHANGED **" if changed else "",
            )
        if upgraded % 25 == 0:
            conn.commit()
    conn.commit()

    # --- report -----------------------------------------------------------
    modes = dict(
        conn.execute("SELECT fusion_mode, COUNT(*) FROM predictions GROUP BY 1")
    )
    log.info("")
    log.info("re-classified %d alert(s); %d skipped (no ALeRCE features), %d failed",
             upgraded, skipped, failed)
    log.info("fusion modes now: %s", modes)
    conn.close()
    engine.close()

    print()
    print("Refresh the dashboard — those alerts now carry a tabular branch,")
    print("a fused prediction, and appear in the disagreement gallery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
