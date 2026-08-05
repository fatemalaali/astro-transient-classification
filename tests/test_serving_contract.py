"""The serving path must reproduce the trained numbers exactly.

A demo that quietly disagrees with the thesis' reported metrics is worse than
no demo. These tests re-derive the fusion card's held-out test metrics through
the *serving* code — the same loaders, calibration and stack the consumer uses —
and assert they match the card.

They need the trained artefacts under ``models/``; they skip cleanly without
them, so the suite still runs on a fresh checkout.

    python -m pytest tests/test_serving_contract.py -v
    python tests/test_serving_contract.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from demo.config import CLASS_NAMES, get_settings  # noqa: E402
from demo.inference.contracts import calibrate, softmax  # noqa: E402
from demo.inference.fusion import LogisticStackFusion  # noqa: E402


class Skip(Exception):
    """Raised when the trained artefacts are absent."""


def _load_branch(directory: Path) -> dict:
    needed = ["test_proba.npy", "test_oids.npy", "temperature.json"]
    if not all((directory / name).exists() for name in needed):
        raise Skip(f"missing artefacts in {directory}")
    return {
        "proba": np.load(directory / "test_proba.npy"),
        "oids": np.load(directory / "test_oids.npy", allow_pickle=True).astype(str),
        "T": float(
            json.loads((directory / "temperature.json").read_text())["temperature"]
        ),
    }


def _aligned():
    settings = get_settings()
    tab = _load_branch(settings.tabular_dir)
    img = _load_branch(settings.image_dir)
    common = np.array(sorted(set(tab["oids"]) & set(img["oids"])))
    pa = {o: i for i, o in enumerate(tab["oids"])}
    pb = {o: i for i, o in enumerate(img["oids"])}
    p_tab = calibrate(tab["proba"][[pa[o] for o in common]], tab["T"])
    p_img = calibrate(img["proba"][[pb[o] for o in common]], img["T"])
    return settings, common, p_tab, p_img


def _truth(oids):
    import pandas as pd

    settings = get_settings()
    path = settings.gold_dir / "gold_labels.parquet"
    if not path.exists():
        raise Skip("gold labels not present")
    coarse = pd.read_parquet(path).set_index("oid")["coarse"]
    return np.array([CLASS_NAMES.index(coarse.loc[o]) for o in oids])


def _macro_f1(y, proba) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(y, np.asarray(proba).argmax(1), average="macro",
                 labels=list(range(len(CLASS_NAMES))), zero_division=0)
    )


def test_learned_stack_reproduces_the_card_test_metrics():
    """Serving fusion must match ``fusion_card.json`` to 4 decimal places."""
    settings, oids, p_tab, p_img = _aligned()
    fusion = LogisticStackFusion(settings.fusion_dir)
    if not fusion.available:
        raise Skip("fusion card not present")

    fused = np.vstack(
        [fusion.fuse(p_tab[i], p_img[i]).proba for i in range(len(oids))]
    )
    y = _truth(oids)
    card = json.loads(
        (settings.fusion_dir / "fusion_card.json").read_text(encoding="utf-8")
    )["test_metrics"]

    got = _macro_f1(y, fused)
    assert abs(got - card["macro_f1"]) < 1e-4, (
        f"serving macro-F1 {got:.4f} != card {card['macro_f1']:.4f} — the "
        "serving path has drifted from the trained model"
    )

    from sklearn.metrics import log_loss

    got_ll = log_loss(y, fused, labels=list(range(len(CLASS_NAMES))))
    assert abs(got_ll - card["log_loss"]) < 1e-4


def test_equal_weight_baseline_reproduces_the_card():
    """The fallback blend must match the card's equal-weight baseline."""
    settings, oids, p_tab, p_img = _aligned()
    fusion = LogisticStackFusion(settings.fusion_dir)
    if not fusion.available:
        raise Skip("fusion card not present")

    blended = np.vstack(
        [fusion.equal_weight(p_tab[i], p_img[i]) for i in range(len(oids))]
    )
    card = json.loads(
        (settings.fusion_dir / "fusion_card.json").read_text(encoding="utf-8")
    )
    expected = card["baselines"]["equal_weight"]["test_metrics"]["macro_f1"]
    got = _macro_f1(_truth(oids), blended)
    assert abs(got - expected) < 1e-4, f"{got:.4f} != {expected:.4f}"


def test_missing_modality_falls_back_without_imputing():
    """A missing branch must yield the survivor unchanged, not an imputed stack."""
    settings = get_settings()
    fusion = LogisticStackFusion(settings.fusion_dir)
    p_tab = np.array([0.7, 0.2, 0.1])
    p_img = np.array([0.1, 0.1, 0.8])

    only_tab = fusion.fuse(p_tab, None)
    assert only_tab.mode == "tabular_only"
    assert np.allclose(only_tab.proba, p_tab), "tabular output must pass through"

    only_img = fusion.fuse(None, p_img)
    assert only_img.mode == "image_only"
    assert np.allclose(only_img.proba, p_img), "image output must pass through"

    neither = fusion.fuse(None, None)
    assert neither.mode == "none" and neither.proba is None

    # And the both-present path must NOT equal either branch, or the stack is
    # not actually combining anything.
    if fusion.available:
        both = fusion.fuse(p_tab, p_img)
        assert both.mode == "both"
        assert not np.allclose(both.proba, p_tab)
        assert both.stack_input is not None and both.stack_input.shape == (6,)


def test_probabilities_are_normalised():
    settings, oids, p_tab, p_img = _aligned()
    for matrix, name in ((p_tab, "tabular"), (p_img, "image")):
        sums = matrix.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9), f"{name} rows do not sum to 1"


def test_calibration_matches_the_notebook_formula():
    """``calibrate`` must equal ``softmax(log p / T)`` — the fusion notebook's ``cal``."""
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=50)
    for temperature in (0.5, 1.0, 1.6718156695642157, 3.017607741977585):
        expected = softmax(np.log(np.clip(proba, 1e-12, 1.0)), T=temperature)
        assert np.allclose(calibrate(proba, temperature), expected)


def test_image_preprocessing_matches_training():
    """``robust_normalise`` must reproduce the notebook's transform bit for bit."""
    from demo.inference.image import robust_normalise

    rng = np.random.default_rng(1)
    arr = rng.normal(0, 100, (4, 3, 63, 63)).astype(np.float32)
    arr[0, 0, 0, 0] = 1e31        # sentinel
    arr[1, 1, 5, 5] = np.nan      # NaN, as Kafka cutouts can carry

    expected = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    expected = np.where(np.abs(expected) > 1e30, 0.0, expected).astype(np.float32)
    lo = np.percentile(expected, 1, axis=(2, 3), keepdims=True)
    hi = np.percentile(expected, 99, axis=(2, 3), keepdims=True)
    expected = np.clip(expected, lo, hi)
    med = np.median(expected, axis=(2, 3), keepdims=True)
    std = expected.std(axis=(2, 3), keepdims=True)
    expected = (expected - med) / (std + 1e-6)

    got = robust_normalise(arr)
    assert np.allclose(got, expected, atol=1e-6)
    assert np.isfinite(got).all(), "normalisation must not emit non-finite values"


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
    print("serving contract holds" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
