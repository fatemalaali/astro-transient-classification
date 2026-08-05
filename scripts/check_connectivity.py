"""PHASE 0 — what is reachable from THIS machine?

Written because the planning environment could not settle one question: every
ALeRCE host returned HTTP 403 on 2026-08-05 (api.alerce.online,
api-lsst.alerce.online, avro.alerce.online, and even alerce.online's own
server-rendered object routes), while alerce.online's static app loaded fine.
A uniform 403 across unrelated hosts looks like egress filtering rather than an
outage — but that has to be checked from where the demo will actually run.

The answer decides the tabular branch's fate: the 242 trained features come
from ALeRCE's feature service, and without it every alert falls back to
image-only (test macro-F1 0.759 versus 0.953 fused).

    python scripts/check_connectivity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402

from demo.config import get_settings  # noqa: E402
from demo.inference.features import ALERCE_ZTF_BASE  # noqa: E402

TIMEOUT = 20
UA = {"User-Agent": "astro-transient-demo/1.0 (MSc thesis connectivity check)"}

# A ZTF object known to exist; used only to exercise the endpoints.
PROBE_OID = "ZTF21aaxtctv"


def probe(label: str, method: str, url: str, payload: dict | None = None) -> dict:
    started = time.perf_counter()
    try:
        if method == "GET":
            response = requests.get(url, timeout=TIMEOUT, headers=UA)
        else:
            response = requests.post(
                url, json=payload, timeout=TIMEOUT, headers=UA
            )
        elapsed = (time.perf_counter() - started) * 1e3
        body = response.content[:200]
        return {
            "label": label,
            "url": url,
            "status": response.status_code,
            "ok": response.ok,
            "ms": round(elapsed, 1),
            "preview": body.decode("utf-8", "replace").replace("\n", " ")[:120],
        }
    except Exception as exc:
        return {
            "label": label,
            "url": url,
            "status": None,
            "ok": False,
            "ms": round((time.perf_counter() - started) * 1e3, 1),
            "preview": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    settings = get_settings()
    results: list[dict] = []

    print("Fink REST (fallback ingestion path)")
    results.append(
        probe(
            "fink:swagger",
            "GET",
            "https://api.ztf.fink-portal.org/swagger.json",
        )
    )
    results.append(
        probe(
            "fink:latests",
            "POST",
            "https://api.ztf.fink-portal.org/api/v1/latests",
            {"class": "SN candidate", "n": "1"},
        )
    )
    results.append(
        probe(
            "fink:cutouts",
            "POST",
            "https://api.ztf.fink-portal.org/api/v1/cutouts",
            {
                "objectId": PROBE_OID,
                "kind": "All",
                "output-format": "array",
            },
        )
    )

    print("ALeRCE (feature resolution for the tabular branch)")
    results.append(probe("alerce:classifiers", "GET", f"{ALERCE_ZTF_BASE}/classifiers"))
    results.append(
        probe("alerce:features", "GET", f"{ALERCE_ZTF_BASE}/objects/{PROBE_OID}/features")
    )

    print("Fink Kafka (primary ingestion path)")
    kafka = _check_kafka(settings)

    print()
    print(f"{'check':<22} {'status':>7} {'ms':>8}  detail")
    print("-" * 100)
    for r in results:
        mark = "OK " if r["ok"] else "FAIL"
        print(
            f"{r['label']:<22} {str(r['status'] or mark):>7} {r['ms']:>8}  "
            f"{r['preview'][:60]}"
        )
    print(f"{'kafka:credentials':<22} {'OK' if kafka['creds'] else 'FAIL':>7} "
          f"{'-':>8}  {kafka['detail']}")

    fink_ok = all(r["ok"] for r in results if r["label"].startswith("fink:"))
    alerce_ok = any(r["ok"] for r in results if r["label"].startswith("alerce:"))

    print()
    print("=" * 100)
    print(f"Fink REST fallback : {'AVAILABLE' if fink_ok else 'UNAVAILABLE'}")
    print(f"ALeRCE features    : {'AVAILABLE' if alerce_ok else 'UNAVAILABLE'}")
    print(f"Fink credentials   : {'REGISTERED' if kafka['creds'] else 'NOT REGISTERED'}")
    print()

    if not alerce_ok:
        print("ALeRCE is unreachable from this machine. Consequences:")
        print("  * Objects in data/gold/_cache_features still resolve offline")
        print(f"    ({_gold_cache_size(settings)} cached feature vectors available),")
        print("    so replay demos over gold objects keep full two-branch fusion.")
        print("  * NEW objects will fall back to image-only classification.")
        print("  * Set DEMO_ALERCE_ENABLED=0 to skip the failing calls entirely and")
        print("    avoid paying the timeout on every alert.")
    if not kafka["creds"]:
        print("No Fink credentials yet. Build and demo with:")
        print("  python -m demo.run_consumer --mode rest     (needs no credentials)")
        print("  python -m demo.run_consumer --mode offline  (needs no network)")

    out = REPO_ROOT / "data" / "demo" / "connectivity.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"results": results, "kafka": kafka}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\nWritten to {out}")
    except OSError:
        pass

    return 0 if fink_ok else 1


def _check_kafka(settings) -> dict:
    path = Path.home() / ".finkclient" / "ztf_credentials.yml"
    legacy = Path.home() / ".finkclient" / "credentials.yml"
    if path.exists():
        return {"creds": True, "detail": f"{path} (v10+ layout)"}
    if legacy.exists():
        return {
            "creds": True,
            "detail": (
                f"{legacy} — pre-v10 filename. fink-client 10+ expects "
                "ztf_credentials.yml; re-run fink_client_register -survey ztf"
            ),
        }
    return {
        "creds": False,
        "detail": (
            "no ~/.finkclient/ztf_credentials.yml — request credentials at "
            "https://forms.gle/2td4jysT4e9pkf889 then run fink_client_register"
        ),
    }


def _gold_cache_size(settings) -> int:
    try:
        return len(list(settings.gold_feature_cache_dir.glob("*.json")))
    except OSError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
