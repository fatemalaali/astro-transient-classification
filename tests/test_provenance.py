"""Provenance is enforced, not merely asserted.

The thesis' central methodological claim is that brokers supply alert packets,
cutouts and features but never labels and never model input derived from their
own classifications. Prose in a dissertation cannot demonstrate that. These
tests can, and an examiner can run them.

    python -m pytest tests/test_provenance.py -v
    python tests/test_provenance.py          # also works without pytest
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:  # pytest is optional — the __main__ block below runs these without it
    import pytest
except ImportError:  # pragma: no cover
    class _Pytest:
        """Minimal stand-in providing only what these tests use."""

        class _Raises:
            def __init__(self, expected):
                self.expected = expected

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:
                    raise AssertionError(f"expected {self.expected.__name__}")
                return issubclass(exc_type, self.expected)

        def raises(self, expected):
            return self._Raises(expected)

        class Skipped(Exception):
            pass

        def skip(self, reason):
            raise self.Skipped(reason)

    pytest = _Pytest()  # type: ignore[assignment]

from demo.adapters.base import BROKER_FIELDS  # noqa: E402
from demo.config import BROKER_URL_DENYLIST  # noqa: E402
from demo.inference.features import ProvenanceViolation, assert_allowed_url  # noqa: E402

INFERENCE_DIR = REPO_ROOT / "demo" / "inference"


def test_denylist_blocks_probability_endpoints():
    """The ALeRCE probabilities endpoint must be unreachable by construction."""
    banned = [
        "https://api.alerce.online/ztf/v1/objects/ZTF21aaxtctv/probabilities",
        "https://api-lsst.alerce.online/probability_api/probability",
        "https://api.alerce.online/ztf/v1/classifiers/lc_classifier/1.0/classes",
    ]
    for url in banned:
        with pytest.raises(ProvenanceViolation):
            assert_allowed_url(url)


def test_denylist_allows_feature_endpoints():
    """Feature and light-curve endpoints must stay reachable."""
    allowed = [
        "https://api.alerce.online/ztf/v1/objects/ZTF21aaxtctv/features",
        "https://api.alerce.online/ztf/v1/objects/ZTF21aaxtctv/lightcurve",
        "https://api.ztf.fink-portal.org/api/v1/cutouts",
    ]
    for url in allowed:
        assert_allowed_url(url)  # must not raise


def test_denylist_is_not_empty():
    assert BROKER_URL_DENYLIST, "an empty denylist would silently permit everything"


def _docstrings(tree: ast.AST) -> set[int]:
    """ids() of the Constant nodes that are docstrings, so they can be excluded."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _module_reads_broker_meta(path: Path) -> bool:
    """True if the module actually accesses ``.broker_meta`` in executable code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Attribute) and node.attr == "broker_meta"
        for node in ast.walk(tree)
    )


def _module_mentions_in_code(path: Path, needle: str) -> bool:
    """True if ``needle`` appears in executable code — not in a docstring.

    Comments never reach the AST at all, and docstrings are excluded
    explicitly, so a module is free to *document* a field it must not consume.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip and needle in node.value:
                return True
        elif isinstance(node, ast.Name) and needle in node.id:
            return True
        elif isinstance(node, ast.Attribute) and needle in node.attr:
            return True
    return False


def test_inference_never_reads_broker_metadata():
    """No module under demo/inference may touch ``NormalisedAlert.broker_meta``.

    This is the structural version of the provenance claim: broker
    classifications have no path into a model input because the code that
    builds model inputs cannot see them.
    """
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in INFERENCE_DIR.rglob("*.py")
        if _module_reads_broker_meta(path)
    ]
    assert not offenders, (
        "these inference modules read broker-derived metadata: " + ", ".join(offenders)
    )


def test_broker_fields_are_excluded_from_the_tabular_feature_space():
    """No Fink value-added field may collide with a trained feature name.

    A collision would let a broker-derived value be picked up by
    ``TabularBranch.build_row`` under a matching key. Verified against the real
    model card when it is present.
    """
    from demo.config import get_settings

    card = get_settings().tabular_dir / "model_card.json"
    if not card.exists():
        pytest.skip("tabular model card not present")
    import json

    names = set(json.loads(card.read_text(encoding="utf-8"))["feature_names"])
    collisions = names & set(BROKER_FIELDS)
    assert not collisions, f"broker fields collide with trained features: {collisions}"


def test_fink_lc_features_are_not_consumed():
    """Fink computes its own light-curve features; they must stay display-only.

    They are *features*, so admitting them would not breach the letter of the
    rule — but they are a different library from the 242 the model was trained
    on, and mixing them would both blur the provenance line and introduce
    train/serve skew.
    """
    assert "lc_features_g" in BROKER_FIELDS
    assert "lc_features_r" in BROKER_FIELDS
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in INFERENCE_DIR.rglob("*.py")
        if _module_mentions_in_code(path, "lc_features")
    ]
    assert not offenders, (
        "these inference modules reference Fink's own lc_features in executable "
        "code: " + ", ".join(offenders)
    )


def test_known_labels_come_only_from_our_own_sources():
    """The label bootstrap must read the gold layer, never a broker endpoint."""
    source = (REPO_ROOT / "demo" / "storage" / "bootstrap.py").read_text(
        encoding="utf-8"
    )
    for banned in ("alerce.online", "fink-portal.org", "requests.get", "requests.post"):
        assert banned not in source, (
            f"bootstrap.py must not reach a broker ({banned!r} found)"
        )
    assert "gold_labels.parquet" in source


if __name__ == "__main__":  # pragma: no cover
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except getattr(pytest, "Skipped", ()) as exc:  # type: ignore[misc]
            print(f"SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print()
    print("all provenance checks passed" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
