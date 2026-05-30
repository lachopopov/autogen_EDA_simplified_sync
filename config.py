"""
config.py — LLM configuration, environment loading, EDA_MODE switch.

Architecture Reference: architecture.md § 9
AG2 Version: 0.10.3

Models:
  - gpt-5-mini  ($0.25 / $2.00 per 1M tokens) — dev & iteration
  - gpt-5-mini  ($0.25 / $2.00 per 1M tokens) and gpt-5  ($1.25 / $10.00 per 1M tokens) for FindingsGeneratorAgent only — final validation only

Usage:
  EDA_MODE=dev                          → dev mode: gpt-5-mini, app cache off, AG2 cache off
  EDA_MODE=final                        → production: gpt-5-mini + gpt-5, app cache on, AG2 cache off
  EDA_MODE=final AG2_CACHE_SEED=42      → validation: both caches on (cost-saving reproduction)
"""

import datetime
import logging
import os
import shutil
import threading
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (same directory as this file)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# --- Base configuration shared across all models ---
# Note: temperature is NOT set here. gpt-5-nano and gpt-5-mini only support
# the default temperature (1). Setting temperature=0.0 causes a 400 error.
# AG2 LLM cache: opt-in via AG2_CACHE_SEED env var (e.g. AG2_CACHE_SEED=42).
# Decoupled from EDA_MODE: true production uses EDA_MODE=final but no AG2 cache.
# Validation runs use EDA_MODE=final AG2_CACHE_SEED=42 (both caches on).
_BASE: dict = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "cache_seed": int(os.getenv("AG2_CACHE_SEED")) if os.getenv("AG2_CACHE_SEED") else None,
}

# --- Model-specific configurations ---
LLM_CONFIG_DEV: dict = {
    "config_list": [{
        **_BASE,
        "model": "gpt-5-mini",
        "price": [0.00025, 0.002],  # $0.25/$2.00 per 1M tokens
    }],
}

#sets the config of FindingsGeneratorAgent to use gpt-5 instead of gpt-5-mini, while keeping the rest of the agents on gpt-5-mini.
LLM_CONFIG_FINAL: dict = {
    "config_list": [{
        **_BASE,
        "model": "gpt-5",
        "price": [0.00125, 0.01],  # $1.25/$10.00 per 1M tokens
    }],
}
# For cost control during final validation, only the FindingsGeneratorAgent uses gpt-5; all other agents remain on gpt-5-mini. This allows us to validate the critical findings generation step with the more powerful model while keeping overall costs manageable.

LLM_CONFIG_FINAL_REST: dict = {
    "config_list": [{
        **_BASE,
        "model": "gpt-5-mini",
        "price": [0.00025, 0.002],  # $0.25/$2.00 per 1M tokens
    }],
}
# --- Active configuration (selected via EDA_MODE environment variable) ---
LLM_CONFIG: dict = (
    LLM_CONFIG_FINAL_REST if os.getenv("EDA_MODE") == "final" else LLM_CONFIG_DEV
)

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
GLOBAL_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = GLOBAL_OUTPUTS_DIR / "runs"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Once-per-process cleanup guard
# ---------------------------------------------------------------------------
_cleanup_done: bool = False
_cleanup_lock: threading.Lock = threading.Lock()

def get_outputs_dir(session_id: str | None = None) -> Path:
    """Return the output directory for a given session, or the global outputs dir.

    Falls back to the cache directory when *session_id* corresponds to a
    cached run (i.e. the value returned by run_pipeline() on a cache hit).
    """
    if session_id:
        run_dir = RUNS_DIR / session_id
        if run_dir.exists():
            return run_dir
        # Fall back to cache dir for cached session_ids (Step 3 contract).
        cache_dir = GLOBAL_OUTPUTS_DIR / ".cache" / session_id
        if cache_dir.exists():
            return cache_dir
        return run_dir  # caller will mkdir
    return GLOBAL_OUTPUTS_DIR

def get_plots_dir(session_id: str | None = None) -> Path:
    """Return the plots directory for a given session."""
    return get_outputs_dir(session_id) / "plots"

def cleanup_old_runs(hours: int = 24) -> None:
    """Remove run directories older than the specified number of hours."""
    if not RUNS_DIR.exists():
        return
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(hours=hours)

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            folder_time = datetime.datetime.fromtimestamp(run_dir.stat().st_mtime)
            if folder_time < cutoff:
                shutil.rmtree(run_dir, ignore_errors=True)
                logger.info(f"Cleaned up old run directory: {run_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {run_dir}: {e}")

def ensure_run_dirs(session_id: str) -> None:
    """Ensure the necessary output directories exist for the session.

    Cleanup (old runs + cache) is performed at most once per process using a
    double-checked lock, so repeated calls within the same process are cheap.
    """
    global _cleanup_done
    if not _cleanup_done:
        with _cleanup_lock:
            if not _cleanup_done:  # double-checked locking pattern
                cleanup_old_runs(hours=24)
                try:
                    from core import cache as _cache  # lazy — avoids import cycle
                    _cache.cleanup()
                except Exception:  # noqa: BLE001
                    logger.warning("core.cache.cleanup() failed; continuing without cache cleanup")
                _cleanup_done = True
    out_dir = get_outputs_dir(session_id)
    plots_dir = get_plots_dir(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

# --- Feature toggles ---
IPYNB_EXPORT: bool = os.getenv("IPYNB_EXPORT", "false").lower() == "true"

# --- OpenLIT observability ---
# Enable via OPENLIT_ENABLE=true or --openlit CLI flag.
# If OPENLIT_ENDPOINT is set, traces go there; otherwise they print to console.
OPENLIT_ENABLE: bool = os.getenv("OPENLIT_ENABLE", "false").lower() == "true"
OPENLIT_ENDPOINT: str | None = os.getenv("OPENLIT_ENDPOINT")
OPENLIT_EVAL_MODEL: str = os.getenv("OPENLIT_EVAL_MODEL", "gpt-5")

# --- Critic config ---
MAX_CRITIC_ITERATIONS: int = int(os.getenv("MAX_CRITIC_ITERATIONS", "2"))

# --- Encoded-categorical detection ---
# Model used for the single pre-pipeline LLM call that identifies numerically
# encoded categoricals.  Defaults to gpt-5-mini (cheap, sufficient for this task).
RECLASSIFY_MODEL: str = os.getenv("RECLASSIFY_MODEL", "gpt-5-mini")

# --- GroupChat config ---
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "70"))
# Raised from 50 → 70 to accommodate:
#   - W4 (analyze_categoricals = 5th EDA tool, +2 rounds worst-case sequential)
#   - Planned W7 (feature importance = 6th EDA tool, +2 more rounds)
#   - 2 critic-loop retries (each retry = +10 rounds FindingsGenerator + Critic)
#   - Total worst-case sequential path peaks ~55 rounds; 70 is a safe ceiling.

# --- CSV / Excel missing-value sentinel tokens ---
# These tokens are treated as NaN at load time (in addition to pandas defaults).
# Override via CSV_NA_TOKENS env var (comma-separated) for datasets where
# any of these strings is a legitimate value rather than a missing sentinel.
_env_na = os.getenv("CSV_NA_TOKENS")
NA_TOKENS: list[str] = (
    [t.strip() for t in _env_na.split(",")]
    if _env_na
    else [
        "?", "??",                                      # UCI / survey sentinel
        "NA", "N/A", "n/a", "na", "N\\A",             # standard abbreviations
        "NULL", "null", "None", "none",                # programming defaults
        "NaN", "nan", "<NA>", "<missing>",             # typed representations
        "missing", "MISSING", "Missing",
        "Unknown", "unknown", "UNK", "unk",
        "Refused", "refused", "No answer",
        "Not applicable", "Not available",
    ]
)
