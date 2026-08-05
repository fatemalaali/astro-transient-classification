"""FeatureResolver — supplying the 242 features the tabular branch expects.

**The mismatch this module exists to bridge.** ``models/lc/ztf/lightgbm``
declares 242 feature names (``SPM_*``, ``MHPS_*``, ``GP_DRW_*``,
``Harmonics_*``, ``Multiband_period_12``, ``W1-W2``, …). Those are *ALeRCE's*
light-curve feature set, obtained by ``build_dataset.ipynb`` from
``alerce.query_features(oid)``. A Fink Kafka packet contains none of them: it
carries 103 raw ``candidate`` fields, 57 fields per ``prv_candidates`` epoch,
and Fink's own 26-feature ``lc_features_g/r`` — a different library with
different definitions. There is no meaningful renaming table between them.

Rather than recompute (alert history is truncated to ~30 days, and catalogue
features such as ``W1-W2`` are not in the packet at all) or retrain on a shared
subset (which would invalidate every number in the fusion card), the resolver
fetches the same features from the same endpoint the training set used. That
gives byte-identical train/serve provenance.

**Cost, stated honestly:** the tabular branch's freshness is bounded by ALeRCE's
own update cadence, so end-to-end "real time" holds for the image branch and for
fusion-given-features, not for the tabular branch in isolation. The provenance
and fetch timestamp travel with every prediction so this stays visible rather
than implied.

Three tiers, cheapest first:

1. ``gold_cache``  ``data/gold/_cache_features/<oid>.json`` — the 11 826 vectors
   build_dataset.ipynb already fetched. Read-only, offline, deterministic.
2. ``disk_cache``  ``data/demo/_cache_features/<oid>.json`` — anything fetched
   live during a demo, in the identical layout.
3. ``alerce_live`` the ALeRCE ZTF features endpoint.

**Only feature endpoints are ever called.** ``/probabilities`` is denylisted and
the check runs on every outbound URL.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

from demo.config import BROKER_URL_DENYLIST, Settings
from demo.models import FeatureProvenance, iso, utcnow

log = logging.getLogger("demo.features")

#: From the pinned ALeRCE client config (alerce/default_config.json).
#: NOTE: could not be verified live from the planning environment — every ALeRCE
#: host returned HTTP 403 on 2026-08-05. Run scripts/check_connectivity.py.
ALERCE_ZTF_BASE = "https://api.alerce.online/ztf/v1"


class ProvenanceViolation(RuntimeError):
    """Raised when code attempts to call a broker classification endpoint.

    Brokers supply alert packets, features and stamps. Never labels, never
    predictions. This is the enforcement point for that rule.
    """


def assert_allowed_url(url: str) -> None:
    lowered = url.lower()
    for banned in BROKER_URL_DENYLIST:
        if banned in lowered:
            raise ProvenanceViolation(
                f"refusing to call {url!r}: broker classifications are never used "
                f"as labels or model input (matched {banned!r})"
            )


class FeatureResolver:
    """Resolves an object id to the tabular branch's feature dictionary."""

    #: How long to wait before retrying ALeRCE after it looks blocked. Short
    #: enough that plugging into a phone hotspot mid-demo is picked up without
    #: a restart, long enough that a genuinely blocked network is not hammered.
    RETRY_BLOCKED_AFTER_S = 120.0

    def __init__(self, settings: Settings, feature_names: tuple[str, ...]) -> None:
        self.settings = settings
        self.feature_names = tuple(feature_names)
        self.n_expected = len(self.feature_names)
        self._memory: dict[str, tuple[float, dict | None]] = {}
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "astro-transient-demo/1.0 (MSc thesis; features only)"
        )
        self.settings.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        # --- reachability circuit breaker -------------------------------- #
        # ALeRCE is edge-blocked (HTTP 403 from awselb) on some networks. Paying
        # a full request per alert to rediscover that is wasteful, and giving up
        # permanently means a mid-session network change goes unnoticed. So:
        # trip the breaker on failure, retry one probe every RETRY_BLOCKED_AFTER_S,
        # and close it again the moment a call succeeds.
        self._alerce_state = "unknown"  # unknown | reachable | blocked
        self._alerce_reason: str | None = None
        self._blocked_since = 0.0
        self._alerce_calls = 0
        self._alerce_failures = 0
        self._last_success_utc = None

    # ------------------------------------------------------------------ #
    # cache tiers
    # ------------------------------------------------------------------ #
    def _from_memory(self, oid: str) -> dict | None:
        with self._lock:
            entry = self._memory.get(oid)
        if entry is None:
            return None
        stored_at, payload = entry
        if time.monotonic() - stored_at > self.settings.feature_cache_ttl_s:
            return None
        return payload

    def _remember(self, oid: str, payload: dict | None) -> None:
        with self._lock:
            self._memory[oid] = (time.monotonic(), payload)
            if len(self._memory) > 4096:  # keep the LRU honest without a dependency
                oldest = sorted(self._memory.items(), key=lambda kv: kv[1][0])[:512]
                for key, _ in oldest:
                    self._memory.pop(key, None)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            log.debug("unreadable feature cache entry %s", path)
            return None

    def _from_gold_cache(self, oid: str) -> dict | None:
        return self._read_json(self.settings.gold_feature_cache_dir / f"{oid}.json")

    def _from_disk_cache(self, oid: str) -> dict | None:
        return self._read_json(self.settings.feature_cache_dir / f"{oid}.json")

    def _write_disk_cache(self, oid: str, payload: dict) -> None:
        try:
            (self.settings.feature_cache_dir / f"{oid}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        except Exception:  # pragma: no cover
            log.debug("could not write the feature cache for %s", oid)

    # ------------------------------------------------------------------ #
    # network
    # ------------------------------------------------------------------ #
    @staticmethod
    def pivot(rows: Any) -> dict:
        """ALeRCE's long feature table -> the wide ``<name>_<fid>`` dictionary.

        Identical key convention to ``_fetch_features`` in build_dataset.ipynb:
        a feature with a filter id becomes ``"<name>_<fid>"``, one without keeps
        its bare name. Getting this wrong would silently produce an all-NaN row.
        """
        out: dict[str, float] = {}
        if isinstance(rows, dict):
            rows = rows.get("items") or rows.get("features") or []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if name is None:
                continue
            value = row.get("value")
            fid = row.get("fid")
            key = str(name)
            if fid is not None:
                try:
                    key = f"{name}_{int(fid)}"
                except (TypeError, ValueError):
                    key = str(name)
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------ #
    # reachability
    # ------------------------------------------------------------------ #
    def _breaker_open(self) -> bool:
        """True when we should skip the network call entirely."""
        if self._alerce_state != "blocked":
            return False
        if (time.monotonic() - self._blocked_since) >= self.RETRY_BLOCKED_AFTER_S:
            log.info("retrying ALeRCE — the network may have changed")
            return False  # allow one probe through
        return True

    def _mark_reachable(self) -> None:
        if self._alerce_state != "reachable":
            log.info("ALeRCE is reachable — the tabular branch is live")
        self._alerce_state = "reachable"
        self._alerce_reason = None
        self._last_success_utc = utcnow()

    def _mark_blocked(self, reason: str) -> None:
        if self._alerce_state != "blocked":
            log.warning(
                "ALeRCE unreachable (%s) — falling back to image-only for objects "
                "outside the feature cache. Retrying every %.0fs, so switching "
                "network will be picked up without a restart.",
                reason,
                self.RETRY_BLOCKED_AFTER_S,
            )
        self._alerce_state = "blocked"
        self._alerce_reason = reason
        self._blocked_since = time.monotonic()

    def status(self) -> dict:
        """Reachability summary for the API and the live indicator."""
        return {
            "enabled": self.settings.alerce_enabled,
            "state": "disabled" if not self.settings.alerce_enabled else self._alerce_state,
            "reason": self._alerce_reason,
            "calls": self._alerce_calls,
            "failures": self._alerce_failures,
            "last_success_utc": iso(self._last_success_utc),
            "retry_after_s": self.RETRY_BLOCKED_AFTER_S,
            "endpoint": ALERCE_ZTF_BASE,
        }

    def probe(self, oid: str = "ZTF21aaxtctv") -> bool:
        """Force a reachability check. Used at startup and by scripts."""
        if not self.settings.alerce_enabled:
            return False
        self._alerce_state = "unknown"
        self._blocked_since = 0.0
        return self._from_alerce(oid, force=True) is not None

    def _from_alerce(self, oid: str, force: bool = False) -> dict | None:
        if not self.settings.alerce_enabled:
            return None
        if not force and self._breaker_open():
            return None
        url = f"{ALERCE_ZTF_BASE}/objects/{oid}/features"
        assert_allowed_url(url)  # provenance guard, on every call
        self._alerce_calls += 1
        try:
            response = self._session.get(
                url, timeout=self.settings.alerce_timeout_s
            )
            if response.status_code == 404:
                # Reachable, but ALeRCE has not featurised this object yet —
                # common for a brand-new transient, and NOT a network problem.
                self._mark_reachable()
                return {}
            if response.status_code in (401, 403, 407, 451):
                # Edge/WAF block: the network, not the object.
                self._alerce_failures += 1
                self._mark_blocked(f"HTTP {response.status_code}")
                return None
            response.raise_for_status()
            self._mark_reachable()
            return self.pivot(response.json())
        except ProvenanceViolation:
            raise
        except Exception as exc:
            self._alerce_failures += 1
            self._mark_blocked(f"{type(exc).__name__}: {exc}")
            log.debug("ALeRCE feature fetch failed for %s: %s", oid, exc)
            return None

    # ------------------------------------------------------------------ #
    def resolve(self, object_id: str) -> tuple[dict | None, FeatureProvenance]:
        """Return ``(features, provenance)`` for one object.

        ``features`` is a dict keyed by the trained feature names; anything
        absent stays absent and becomes NaN downstream, which LightGBM consumes
        natively.
        """
        oid = str(object_id)

        cached = self._from_memory(oid)
        if cached is not None:
            return cached, FeatureProvenance(
                source="disk_cache",
                n_present=self._count(cached),
                n_expected=self.n_expected,
                fetched_utc=utcnow(),
            )

        payload = self._from_gold_cache(oid)
        if payload:
            self._remember(oid, payload)
            return payload, FeatureProvenance(
                source="gold_cache",
                n_present=self._count(payload),
                n_expected=self.n_expected,
            )

        payload = self._from_disk_cache(oid)
        if payload:
            self._remember(oid, payload)
            return payload, FeatureProvenance(
                source="disk_cache",
                n_present=self._count(payload),
                n_expected=self.n_expected,
            )

        payload = self._from_alerce(oid)
        if payload:
            self._write_disk_cache(oid, payload)
            self._remember(oid, payload)
            return payload, FeatureProvenance(
                source="alerce_live",
                n_present=self._count(payload),
                n_expected=self.n_expected,
                fetched_utc=utcnow(),
            )

        if not self.settings.alerce_enabled:
            reason = "ALeRCE disabled (DEMO_ALERCE_ENABLED=0)"
        elif self._alerce_state == "blocked":
            reason = (
                f"ALeRCE unreachable ({self._alerce_reason}) — network-level "
                "block, not a property of this object. Re-run "
                "scripts/backfill_features.py from a network that can reach it "
                "to upgrade this alert to two-branch fusion."
            )
        else:
            reason = "ALeRCE reachable but has no features for this object yet"
        return None, FeatureProvenance(
            source="unavailable",
            n_present=0,
            n_expected=self.n_expected,
            error=reason,
        )

    def _count(self, payload: dict) -> int:
        """How many of the *trained* feature names this payload actually fills."""
        return sum(
            1
            for name in self.feature_names
            if payload.get(name) is not None
        )

    def close(self) -> None:
        self._session.close()
