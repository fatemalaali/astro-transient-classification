"""PHASE 1 — is a served stamp oriented the same way as a training stamp?

This is the highest-value silent-failure check in the build. ``build_dataset.ipynb``
took ``hdul[i].data`` straight from ALeRCE with no flip, whereas fink-client's
own viewer reverses rows (``[::-1]``) for *display*. If the serving path picked
up that flip, the CNN would receive vertically mirrored inputs, accuracy would
drop, and **nothing would raise**.

The test compares the *reference* (template) channel, because that is the one
image that does not change between alerts: the science and difference frames
belong to a specific exposure, but the reference is a per-field co-add, so the
same object should yield near-identical reference pixels regardless of which
alert delivered them.

Sources compared, best available first:

1. a saved ``.avro`` alert (Kafka path) for an object present in the gold set;
2. the Fink REST cutout service, which needs no credentials and was verified
   to return 63x63 arrays.

Correlation is computed for the identity, the vertical flip, the horizontal
flip and the transpose. The identity must win.

    python scripts/compare_stamp_orientation.py
    python scripts/compare_stamp_orientation.py --n 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from demo.adapters.cutouts import decode_fink_packet, decode_rest_row  # noqa: E402
from demo.adapters.replay import read_avro  # noqa: E402
from demo.config import CHANNEL_ORDER, get_settings  # noqa: E402

log = logging.getLogger("orientation")

TRANSFORMS = {
    "identity": lambda a: a,
    "flip_vertical": lambda a: a[::-1, :],
    "flip_horizontal": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.T,
}


def normalise(arr: np.ndarray) -> np.ndarray:
    """Rank-normalise so the comparison is insensitive to scale and outliers."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    a = np.where(np.abs(a) > 1e30, 0.0, a)
    flat = a.ravel()
    order = flat.argsort().argsort().astype(np.float64)
    return (order / max(order.size - 1, 1)).reshape(a.shape)


def correlations(served: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    ref = normalise(reference)
    out = {}
    for name, fn in TRANSFORMS.items():
        try:
            candidate = normalise(fn(served))
        except Exception:
            continue
        if candidate.shape != ref.shape:
            continue
        out[name] = float(
            np.corrcoef(candidate.ravel(), ref.ravel())[0, 1]
        )
    return out


def gold_stamps(settings):
    path = settings.gold_dir / "gold_stamps.npz"
    if not path.exists():
        return None, None
    npz = np.load(path, allow_pickle=True)
    return npz["oids"].astype(str), npz["stamps"]


def from_avro(settings, gold_oids: set[str]) -> list[tuple[str, dict]]:
    found = []
    for path in sorted(settings.raw_alerts_dir.glob("*.avro")):
        for alert in read_avro(path):
            oid = str(alert.get("objectId"))
            if oid in gold_oids:
                cutouts, status = decode_fink_packet(alert)
                if status in ("ok", "partial"):
                    found.append((oid, cutouts))
    return found


def from_rest(oids: list[str], timeout: float) -> list[tuple[str, dict]]:
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "astro-transient-demo/1.0 (orientation check)"
    found = []
    for oid in oids:
        try:
            response = session.post(
                "https://api.ztf.fink-portal.org/api/v1/cutouts",
                json={"objectId": oid, "kind": "All", "output-format": "array"},
                timeout=timeout,
            )
            if not response.ok:
                log.debug("REST cutouts for %s: HTTP %s", oid, response.status_code)
                continue
            payload = response.json()
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
            cutouts, status = decode_rest_row(payload or {})
            if status in ("ok", "partial"):
                found.append((oid, cutouts))
        except Exception as exc:
            log.debug("REST cutouts for %s failed: %s", oid, exc)
    session.close()
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=3, help="objects to compare")
    parser.add_argument(
        "--channel", default="reference", choices=list(CHANNEL_ORDER),
        help="channel to compare (default: reference — stable across alerts)",
    )
    parser.add_argument("--no-rest", action="store_true", help="skip the REST fallback")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout
    )
    settings = get_settings(refresh=True)

    oids, stamps = gold_stamps(settings)
    if oids is None:
        log.error("no gold stamps at %s", settings.gold_dir / "gold_stamps.npz")
        return 2
    index = {o: i for i, o in enumerate(oids)}
    channel_idx = CHANNEL_ORDER.index(args.channel)

    pairs = from_avro(settings, set(index))
    origin = "Kafka .avro"
    if not pairs and not args.no_rest:
        log.info(
            "no saved .avro alerts matched a gold object — falling back to the "
            "Fink REST cutout service (no credentials needed)"
        )
        rng = np.random.default_rng(7)
        sample = [str(o) for o in rng.choice(oids, size=min(args.n * 4, len(oids)),
                                             replace=False)]
        pairs = from_rest(sample, settings.alerce_timeout_s)
        origin = "Fink REST"

    if not pairs:
        log.error(
            "could not obtain a served cutout for any gold object. Save an alert "
            "with `fink_consumer --display --save -outdir data/demo/raw_alerts "
            "-limit 1`, or check network access to api.ztf.fink-portal.org."
        )
        return 2

    print(f"\nComparing the '{args.channel}' channel, served via {origin}, "
          f"against gold_stamps.npz\n")
    verdicts = []
    for oid, cutouts in pairs[: args.n]:
        served = cutouts.get(args.channel)
        if served is None:
            continue
        reference = stamps[index[oid]][channel_idx]
        scores = correlations(np.asarray(served), np.asarray(reference))
        if not scores:
            continue
        best = max(scores, key=scores.get)
        verdicts.append(best)
        print(f"  {oid}")
        for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
            marker = " <-- best" if name == best else ""
            print(f"    {name:<16} r = {value:+.4f}{marker}")
        print()

    if not verdicts:
        log.error("no comparable stamps were produced")
        return 2

    identity_wins = sum(1 for v in verdicts if v == "identity")
    print("=" * 70)
    if identity_wins == len(verdicts):
        print(f"PASS — identity wins on all {len(verdicts)} object(s).")
        print("The serving path preserves the training orientation. Proceed to Phase 2.")
        return 0

    print(f"FAIL — identity won on only {identity_wins}/{len(verdicts)} object(s).")
    print(f"Most common best transform: {max(set(verdicts), key=verdicts.count)}")
    print()
    print("The serving decode is mirrored relative to the training stamps. Fix it in")
    print("demo/adapters/cutouts.py (decode_fits_bytes / decode_array) — NOT in the")
    print("image branch, so that every source is corrected in one place. Note that")
    print("fink_client.visualisation.show_stamps reverses rows for display only;")
    print("that flip must not reach the model.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
