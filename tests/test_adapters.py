"""Adapter-level invariants that are silent when broken.

Every test here corresponds to a bug that was actually hit during the build.
None of them raise anything visible in normal operation, which is exactly why
they are worth pinning down.

    python tests/test_adapters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from demo.adapters.base import big_int, build_lightcurve, extract_broker_meta, num  # noqa: E402
from demo.adapters.cutouts import CUTOUT_SIZE, fit_size  # noqa: E402
from demo.models import NormalisedAlert, jd_to_utc, utc_to_jd  # noqa: E402

#: A real ZTF candid, from a live /api/v1/latests response.
REAL_CANDID = 3502194355615015028


def test_candid_survives_parsing():
    """A 19-digit candid must not be routed through float64.

    ``int(float("3502194355615015028"))`` is ``3502194355615015424``. That
    corrupted id then produces HTTP 500s from the cutout service and rows keyed
    on an alert that does not exist — with no exception anywhere.
    """
    assert big_int(str(REAL_CANDID)) == REAL_CANDID
    assert big_int(REAL_CANDID) == REAL_CANDID
    assert big_int(f"{REAL_CANDID}.0") == REAL_CANDID

    # Demonstrate the failure mode the helper exists to avoid.
    assert int(num(str(REAL_CANDID))) != REAL_CANDID

    assert big_int(None) is None
    assert big_int("nan") is None
    assert big_int("") is None
    assert big_int("not-a-number") is None
    assert big_int(True) is None, "bools must not be read as ids"


def test_candid_is_json_safe_as_a_string():
    """The API serialises candids as strings; that must round-trip exactly."""
    text = str(REAL_CANDID)
    assert int(text) == REAL_CANDID
    # And confirm why: the value is past JavaScript's exact-integer range.
    assert REAL_CANDID > 2**53


def test_num_maps_ztf_sentinels_to_none():
    """-999.0 is ZTF's 'not applicable'; treating it as a measurement is wrong."""
    assert num(-999.0) is None
    assert num("nan") is None
    assert num("") is None
    assert num(float("inf")) is None
    assert num("18.82") == 18.82
    assert num(0.0) == 0.0, "zero is a real value, not a sentinel"


def test_upper_limits_are_not_detections():
    """A prv_candidates row with no magpsf is an upper limit, not a data point."""
    candidate = {"jd": 2461256.0, "fid": 1, "magpsf": 18.8, "sigmapsf": 0.09}
    history = [
        {"jd": 2461250.0, "fid": 1, "magpsf": 19.2, "sigmapsf": 0.11},
        {"jd": 2461248.0, "fid": 2, "magpsf": None, "diffmaglim": 20.4},
        {"jd": 2461246.0, "fid": 1, "diffmaglim": 20.1},
    ]
    detections, nondetections = build_lightcurve(candidate, history)
    assert len(detections) == 2
    assert len(nondetections) == 2
    assert all(d.magpsf is not None for d in detections)
    assert all(n.diffmaglim is not None for n in nondetections)
    # The alert being classified must appear in its own light curve.
    assert any(abs(d.jd - 2461256.0) < 1e-6 for d in detections)
    # And the result must be time-ordered for plotting.
    assert list(d.jd for d in detections) == sorted(d.jd for d in detections)


def test_duplicate_epochs_collapse():
    """Fink repeats epochs across the history window; they must not double-plot."""
    candidate = {"jd": 2461256.0, "fid": 1, "magpsf": 18.8}
    history = [
        {"jd": 2461256.0, "fid": 1, "magpsf": 18.8},
        {"jd": 2461256.0, "fid": 1, "magpsf": 18.8},
    ]
    detections, _ = build_lightcurve(candidate, history)
    assert len(detections) == 1


def test_fit_size_crops_and_pads_to_the_native_stamp():
    """Edge-of-chip cutouts arrive smaller or larger; the CNN needs exactly 63x63."""
    for shape in [(63, 63), (61, 63), (63, 61), (65, 65), (31, 31)]:
        out = fit_size(np.ones(shape, dtype=np.float32))
        assert out.shape == (CUTOUT_SIZE, CUTOUT_SIZE), shape
        assert out.dtype == np.float32
    # Padding must be zeros, and the original content must be centred.
    padded = fit_size(np.ones((31, 31), dtype=np.float32))
    assert padded[0, 0] == 0.0
    assert padded[CUTOUT_SIZE // 2, CUTOUT_SIZE // 2] == 1.0


def test_broker_metadata_is_collected_but_kept_separate():
    """Known broker fields are carried; instrument data never is."""
    alert = {
        "objectId": "ZTF26abecjxu",
        "candid": REAL_CANDID,
        "candidate": {"jd": 2461256.0, "magpsf": 18.8},
        "prv_candidates": [{"jd": 2461250.0}],
        "cutoutScience": {"stampData": b"..."},
        "cdsxmatch": "Unknown",
        "rf_snia_vs_nonia": 0.0,
        "snn_snia_vs_nonia": 0.756,
    }
    meta = extract_broker_meta(alert)
    assert "cdsxmatch" in meta and "snn_snia_vs_nonia" in meta
    for instrument_key in (
        "objectId", "candid", "candidate", "prv_candidates", "cutoutScience"
    ):
        assert instrument_key not in meta, (
            f"instrument field {instrument_key!r} leaked into broker_meta"
        )


def test_unknown_top_level_fields_default_to_broker_derived():
    """A Fink field we have never seen must be treated as broker-derived.

    Live schemavsn 4.02 carries 57 top-level fields against the 25 in the schema
    bundled with fink-client 11.0, and new value-added columns appear without
    notice. Failing open — treating an unknown field as instrument data — would
    let a broker classification reach a model input. Failing closed costs
    nothing, since broker_meta is display-only.
    """
    alert = {
        "objectId": "ZTF26abecjxu",
        "candidate": {"jd": 2461256.0},
        "some_future_fink_score": 0.42,
        "anomaly_score_alexanta": -0.02,   # real, observed 2026-08-05
        "brokerIngestTimestamp": "2026-07-28 05:30:43.189795",
    }
    meta = extract_broker_meta(alert)
    assert meta["some_future_fink_score"] == 0.42
    assert "anomaly_score_alexanta" in meta
    assert "brokerIngestTimestamp" in meta
    assert "candidate" not in meta and "objectId" not in meta


def test_stamp_stack_enforces_channel_order():
    """science, reference, difference — the order the CNN was trained on."""
    planes = {
        "science": np.full((63, 63), 1.0, dtype=np.float32),
        "reference": np.full((63, 63), 2.0, dtype=np.float32),
        "difference": np.full((63, 63), 3.0, dtype=np.float32),
    }
    alert = NormalisedAlert(
        object_id="ZTF00aaaaaaa", candid=1, source="replay", cutouts=planes
    )
    stack = alert.stamp_stack()
    assert stack.shape == (3, 63, 63)
    assert stack[0, 0, 0] == 1.0 and stack[1, 0, 0] == 2.0 and stack[2, 0, 0] == 3.0

    # An incomplete triplet must yield None, not a partial tensor.
    alert.cutouts = {"science": planes["science"], "reference": None,
                     "difference": planes["difference"]}
    assert alert.stamp_stack() is None

    # Mismatched shapes must also refuse rather than raise inside numpy.
    alert.cutouts = dict(planes, reference=np.ones((31, 31), dtype=np.float32))
    assert alert.stamp_stack() is None


def test_julian_date_round_trip():
    from datetime import datetime, timezone

    when = datetime(2026, 8, 5, 21, 14, 7, tzinfo=timezone.utc)
    jd = utc_to_jd(when)
    assert abs((jd_to_utc(jd) - when).total_seconds()) < 1e-3
    # Sanity: a 2026 date should sit near JD 2461250.
    assert 2461000 < jd < 2462000


def test_broker_metadata_is_never_a_model_input_surface():
    """``n_broker_fields`` exposes a count so inference need not read the values."""
    alert = NormalisedAlert(
        object_id="ZTF00aaaaaaa", candid=1, source="replay",
        broker_meta={"cdsxmatch": "AGN", "finkclass": "SN candidate"},
    )
    assert alert.n_broker_fields == 2


if __name__ == "__main__":  # pragma: no cover
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print()
    print("adapter invariants hold" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
