"""core/cache.py — Content-addressed output cache for the EDA pipeline.

Cache is dormant (is_enabled() → False) unless EDA_MODE == "final".

Key recipe (SHA-256):
    file_bytes
    + json.dumps(canonical_params, sort_keys=True).encode()
    + f"|{EDA_MODE}|{MODEL_NAME}|{PIPELINE_VERSION}|{PROMPT_VERSION}".encode()

enable_openlit is intentionally excluded from canonical_params.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS: int = 7
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "outputs" / ".cache"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model_name() -> str:
    """Read the active model name from config (at call time, not import time)."""
    from config import LLM_CONFIG  # noqa: PLC0415

    return LLM_CONFIG["config_list"][0]["model"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """True iff EDA_MODE == 'final'.  The cache is dormant in dev mode."""
    return os.getenv("EDA_MODE") == "final"


def compute_key(
    file_path: Path,
    params: dict,
    *,
    prompt_version: str,
    pipeline_version: str,
) -> str:
    """Return a deterministic 64-char hex cache key (SHA-256).

    Sensitive to: file content, params, EDA_MODE, MODEL_NAME,
                  PIPELINE_VERSION, PROMPT_VERSION.
    NOT sensitive to: file path, file mtime, enable_openlit.
    """
    file_bytes = file_path.read_bytes()
    eda_mode = os.getenv("EDA_MODE", "dev")
    model_name = _get_model_name()
    suffix = f"|{eda_mode}|{model_name}|{pipeline_version}|{prompt_version}".encode()

    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(suffix)
    return h.hexdigest()


def lookup(key: str) -> Path | None:
    """Return the cache dir for *key* if it exists, else None."""
    target = CACHE_DIR / key
    if target.is_dir():
        return target
    return None


def store(key: str, run_dir: Path) -> None:
    """Atomically copy *run_dir* into CACHE_DIR / *key*.

    Writes a manifest.json with versioning metadata before the rename.
    If the copy fails mid-way the target entry is never created (atomicity).
    """
    import importlib  # lazy — avoids circular import at module level

    _pipeline = importlib.import_module("pipeline")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = CACHE_DIR / f"{key}.tmp"
    target_dir = CACHE_DIR / key

    # Remove any incomplete attempt from a previous run
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Copy into .tmp (may raise; if so, .tmp is cleaned up on next call)
    shutil.copytree(str(run_dir), str(tmp_dir))

    manifest = {
        "key": key,
        "pipeline_version": _pipeline.PIPELINE_VERSION,
        "prompt_version": _pipeline.PROMPT_VERSION,
        "eda_mode": os.getenv("EDA_MODE", "dev"),
        "model": _get_model_name(),
        "stored_at_iso": datetime.now(UTC).isoformat(),
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Atomic rename (single rename(2) syscall on POSIX, same filesystem)
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    os.replace(str(tmp_dir), str(target_dir))

    logger.info("Cache: stored entry key=%s...", key[:8])


def cleanup(ttl_days: int = CACHE_TTL_DAYS) -> None:
    """Delete cache entries whose manifest reports an age > *ttl_days*.

    Intended to run once per process startup (see Step 5 for the guard).
    """
    if not CACHE_DIR.exists():
        return

    cutoff_ts = (datetime.now(UTC).timestamp()) - ttl_days * 86400

    for entry in CACHE_DIR.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.endswith(".tmp"):
            continue
        try:
            manifest_path = entry / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                stored_at_str = manifest.get("stored_at_iso", "")
                stored_dt = datetime.fromisoformat(stored_at_str)
                if stored_dt.tzinfo is None:
                    stored_dt = stored_dt.replace(tzinfo=UTC)
                if stored_dt.timestamp() < cutoff_ts:
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.info("Cache cleanup: removed entry %s...", entry.name[:8])
            else:
                # Fallback: use directory mtime when manifest is absent
                if entry.stat().st_mtime < cutoff_ts:
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.info("Cache cleanup: removed entry (no manifest) %s...", entry.name[:8])
        except Exception:  # noqa: BLE001
            logger.warning("Cache cleanup: could not process entry %s...", entry.name[:8])
