"""PHASE 0 — verify that Livestream alerts really carry decodable cutouts.

This is the first task in the build order because it is the single highest-risk
unknown: without the science/reference/difference triplet the CNN branch cannot
run off Kafka at all.

Evidence already gathered (docs/demo-plan.md 2.3) says cutouts *are* present:
the Fink distribution schema declares them, the sample alerts in the fink-client
repo carry three gzip streams each, and the client ships a decoder and a viewer
for them. This script converts "almost certainly" into "verified on my own
credentials".

Usage
-----
    # 1. save one alert from your subscribed topics
    mkdir -p data/demo/raw_alerts
    fink_consumer --display --save -outdir data/demo/raw_alerts -limit 1

    # 2. check it
    python scripts/verify_cutouts.py

    # or point at a specific file
    python scripts/verify_cutouts.py data/demo/raw_alerts/ZTF....avro

Exit code 0 = the Kafka stamp path is viable. Non-zero = fall back to the Fink
REST cutout path (verified working: /api/v1/objects?withcutouts=true returns a
63x63 array).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from demo.adapters.cutouts import CUTOUT_SIZE, decode_fink_packet  # noqa: E402
from demo.adapters.replay import read_avro  # noqa: E402
from demo.config import CHANNEL_ORDER, FINK_CUTOUT_KEYS, get_settings  # noqa: E402


def check_alert(alert: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True

    notes.append(f"objectId     : {alert.get('objectId')}")
    notes.append(f"candid       : {alert.get('candid')}")
    notes.append(f"schemavsn    : {alert.get('schemavsn')}")
    notes.append(f"top-level keys: {len(alert)}")

    for key in FINK_CUTOUT_KEYS:
        block = alert.get(key)
        if not isinstance(block, dict):
            notes.append(f"  {key:<18} MISSING from the packet")
            ok = False
            continue
        payload = block.get("stampData")
        if payload is None:
            notes.append(f"  {key:<18} present but stampData is NULL")
            ok = False
            continue
        notes.append(
            f"  {key:<18} {len(payload):>7,} bytes  fileName={block.get('fileName')}"
        )

    cutouts, status = decode_fink_packet(alert)
    notes.append(f"decode status: {status}")
    for channel in CHANNEL_ORDER:
        arr = cutouts.get(channel)
        if arr is None:
            notes.append(f"  {channel:<12} DECODE FAILED")
            ok = False
            continue
        finite = np.isfinite(arr)
        notes.append(
            f"  {channel:<12} shape={arr.shape} dtype={arr.dtype} "
            f"finite={finite.mean():.1%} "
            f"range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]"
        )
        if arr.shape != (CUTOUT_SIZE, CUTOUT_SIZE):
            notes.append(
                f"    ! expected ({CUTOUT_SIZE}, {CUTOUT_SIZE}) — check fit_size()"
            )
            ok = False

    stack = None
    if all(cutouts.get(c) is not None for c in CHANNEL_ORDER):
        stack = np.stack([cutouts[c] for c in CHANNEL_ORDER])
        notes.append(f"stacked      : {stack.shape} — CNN input shape satisfied")

    # Fink value-added fields are informational here: they confirm we are on a
    # Fink stream rather than a raw ZTF one, and they are display-only.
    broker = [
        k
        for k in ("cdsxmatch", "rf_snia_vs_nonia", "snn_snia_vs_nonia", "finkclass")
        if k in alert
    ]
    notes.append(f"Fink added values present: {broker or 'none'}")
    return ok, notes


def main(argv: list[str]) -> int:
    settings = get_settings()
    target = Path(argv[0]) if argv else settings.raw_alerts_dir

    if target.is_dir():
        files = sorted(target.glob("*.avro"))
    elif target.exists():
        files = [target]
    else:
        files = []

    if not files:
        print(f"No .avro files found at {target}.")
        print()
        print("Save one first:")
        print("  mkdir -p data/demo/raw_alerts")
        print(
            "  fink_consumer --display --save -outdir data/demo/raw_alerts -limit 1"
        )
        return 2

    print(f"Checking {len(files)} file(s) under {target}\n")
    all_ok = True
    for path in files[:5]:
        alerts = read_avro(path)
        if not alerts:
            print(f"{path.name}: could not read any alert")
            all_ok = False
            continue
        ok, notes = check_alert(alerts[0])
        print(f"=== {path.name}")
        for line in notes:
            print("  " + line)
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}\n")
        all_ok = all_ok and ok

    print("=" * 70)
    if all_ok:
        print("PASS — Livestream packets carry decodable 63x63 cutout triplets.")
        print("The Kafka stamp path is viable; proceed with Phase 1.")
        return 0
    print("FAIL — cutouts are absent, null, or undecodable on this stream.")
    print()
    print("Fallback (verified working 2026-08-05): fetch cutouts over REST instead.")
    print("  POST https://api.ztf.fink-portal.org/api/v1/cutouts")
    print('       {"objectId": ..., "candid": ..., "kind": "All",')
    print('        "output-format": "array"}  ->  63x63 arrays')
    print("Wire it in demo/adapters/fink_kafka.py by calling")
    print("RestPollAdapter.fetch_cutouts() when decode_fink_packet returns 'missing'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
