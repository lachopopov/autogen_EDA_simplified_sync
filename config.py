"""
config.py — LLM configuration, environment loading, EDA_MODE switch.

Architecture Reference: architecture.md § 9
AG2 Version: 0.10.3

Models:
  - gpt-5-nano  ($0.05 / $0.40 per 1M tokens) — dev & iteration
  - gpt-5-mini  ($0.25 / $2.00 per 1M tokens) — final validation only

Usage:
  EDA_MODE=dev   → gpt-5-nano  (default)
  EDA_MODE=final → gpt-5-mini
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (same directory as this file)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# --- Base configuration shared across all models ---
# Note: temperature is NOT set here. gpt-5-nano and gpt-5-mini only support
# the default temperature (1). Setting temperature=0.0 causes a 400 error.
_BASE: dict = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "cache_seed": None,  # ephemeral cache — no stale outputs across runs
}

# --- Model-specific configurations ---
LLM_CONFIG_DEV: dict = {
    "config_list": [{
        **_BASE,
        "model": "gpt-5-nano",
        "price": [0.00005, 0.0004],  # $0.05/$0.40 per 1M tokens
    }],
}

LLM_CONFIG_FINAL: dict = {
    "config_list": [{
        **_BASE,
        "model": "gpt-5",
        "price": [0.0025, 0.015],  # $2.5/$15.00 per 1M tokens
    }],
}

# --- Active configuration (selected via EDA_MODE environment variable) ---
LLM_CONFIG: dict = (
    LLM_CONFIG_FINAL if os.getenv("EDA_MODE") == "final" else LLM_CONFIG_DEV
)

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

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
