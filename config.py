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
    "config_list": [{**_BASE, "model": "gpt-5-nano"}],
}

LLM_CONFIG_FINAL: dict = {
    "config_list": [{**_BASE, "model": "gpt-5-mini"}],
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

# --- Critic config ---
MAX_CRITIC_ITERATIONS: int = int(os.getenv("MAX_CRITIC_ITERATIONS", "2"))

# --- GroupChat config ---
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "50"))
