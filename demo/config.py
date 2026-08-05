"""Demo configuration — one settings object, read from the environment.

Credentials are never stored here: Fink credentials live in
``~/.finkclient/ztf_credentials.yml`` (written by ``fink_client_register``) and
TNS credentials in ``.env``. This module only carries *behaviour* settings.

Paths are resolved against the repository root, so the demo runs identically
from any working directory — which matters on Windows, where launching from the
Start menu and from Git Bash give different CWDs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# demo/config.py -> demo/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES: tuple[str, str, str] = ("SN", "AGN", "VS")
N_CLASSES = 3

#: Fink filter-id -> human band name. ZTF ships 1=g, 2=r, 3=i.
FID_TO_BAND = {1: "g", 2: "r", 3: "i"}
BAND_COLOURS = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b"}

#: Canonical channel order. Load-bearing: the CNN was trained on exactly this.
CHANNEL_ORDER: tuple[str, str, str] = ("science", "reference", "difference")

#: Fink cutout key -> our channel name.
FINK_CUTOUT_KEYS = {
    "cutoutScience": "science",
    "cutoutTemplate": "reference",
    "cutoutDifference": "difference",
}

#: Ingestion modes. See config/demo.env.example.
MODES = ("live", "replay", "catchup", "rest", "offline")

#: Substrings that must never appear in an outbound broker URL. Brokers supply
#: alert packets, features and stamps — never classifications. Enforced in
#: demo/inference/features.py and asserted by tests.
BROKER_URL_DENYLIST = (
    "probabilit",      # /objects/{oid}/probabilities, probability_api/probability
    "/classify",
    "/classifiers",    # /classifiers/{name}/{version}/classes
    "classifier_classes",
)


def _load_dotenv() -> None:
    """Load .env then config/demo.env, if python-dotenv is available.

    Both are optional. Real environment variables always win.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a soft dependency
        return
    for name in (".env", "config/demo.env"):
        path = REPO_ROOT / name
        if path.exists():
            load_dotenv(path, override=False)


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return default if value is None or value == "" else value


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env_str(key, "1" if default else "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass
class Settings:
    """Everything the demo reads from the environment, resolved once."""

    # --- ingestion ---
    mode: str = "live"
    topics: tuple[str, ...] = ()
    poll_timeout_s: float = 10.0
    queue_maxsize: int = 256
    max_backlog: int = 500
    backlog_confirm_threshold: int = 5000

    # --- inference ---
    use_stubs: bool = False
    min_detections: int = 5
    alerce_enabled: bool = True
    alerce_timeout_s: float = 20.0
    feature_cache_ttl_s: float = 300.0

    # --- serving ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- paths ---
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "demo")
    gold_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "gold")
    models_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        raw_topics = _env_str(
            "DEMO_TOPICS",
            "fink_sn_candidates_ztf,fink_early_sn_candidates_ztf,fink_blazar_ztf,"
            "fink_magnetic_cvs_ztf,fink_vra_ztf,fink_tns_match_ztf",
        )
        topics = tuple(
            t.strip() for t in raw_topics.replace(" ", ",").split(",") if t.strip()
        )
        mode = _env_str("DEMO_MODE", "live").strip().lower()
        if mode not in MODES:
            raise ValueError(f"DEMO_MODE={mode!r} not in {MODES}")
        return cls(
            mode=mode,
            topics=topics,
            poll_timeout_s=_env_float("DEMO_POLL_TIMEOUT_S", 10.0),
            queue_maxsize=_env_int("DEMO_QUEUE_MAXSIZE", 256),
            max_backlog=_env_int("DEMO_MAX_BACKLOG", 500),
            backlog_confirm_threshold=_env_int("DEMO_BACKLOG_CONFIRM_THRESHOLD", 5000),
            use_stubs=_env_bool("DEMO_USE_STUBS", False),
            min_detections=_env_int("DEMO_MIN_DETECTIONS", 5),
            alerce_enabled=_env_bool("DEMO_ALERCE_ENABLED", True),
            alerce_timeout_s=_env_float("DEMO_ALERCE_TIMEOUT_S", 20.0),
            feature_cache_ttl_s=_env_float("DEMO_FEATURE_CACHE_TTL_S", 300.0),
            api_host=_env_str("DEMO_API_HOST", "127.0.0.1"),
            api_port=_env_int("DEMO_API_PORT", 8000),
            data_dir=_resolve(_env_str("DEMO_DATA_DIR", "data/demo")),
            gold_dir=_resolve(_env_str("DEMO_GOLD_DIR", "data/gold")),
            models_dir=_resolve(_env_str("DEMO_MODELS_DIR", "models")),
        )

    # --- derived paths -------------------------------------------------- #
    @property
    def db_path(self) -> Path:
        return self.data_dir / "demo.db"

    @property
    def stamps_dir(self) -> Path:
        return self.data_dir / "stamps"

    @property
    def png_cache_dir(self) -> Path:
        return self.data_dir / "stamps" / "png"

    @property
    def raw_alerts_dir(self) -> Path:
        return self.data_dir / "raw_alerts"

    @property
    def bad_alerts_dir(self) -> Path:
        return self.data_dir / "bad_alerts"

    @property
    def feature_cache_dir(self) -> Path:
        return self.data_dir / "_cache_features"

    @property
    def gold_feature_cache_dir(self) -> Path:
        """The 11 826 feature vectors already fetched by build_dataset.ipynb.

        Read-only. Reusing it means every training object is served offline and
        replay runs are deterministic.
        """
        return self.gold_dir / "_cache_features"

    @property
    def inference_log(self) -> Path:
        return self.data_dir / "inference.log"

    @property
    def replay_manifest(self) -> Path:
        return REPO_ROOT / "config" / "replay_manifest.json"

    # --- model artefacts ------------------------------------------------ #
    @property
    def tabular_dir(self) -> Path:
        return self.models_dir / "lc" / "ztf" / "lightgbm"

    @property
    def image_dir(self) -> Path:
        return self.models_dir / "stamp" / "effnet_b0"

    @property
    def fusion_dir(self) -> Path:
        return self.models_dir / "fusion" / "logreg_stack"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.stamps_dir,
            self.png_cache_dir,
            self.raw_alerts_dir,
            self.bad_alerts_dir,
            self.feature_cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings.from_env()
    return _settings
