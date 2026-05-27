"""
conftest.py — shared fixtures for all EDA tool tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so `import config` / `import eda_state` work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch):
    """Guarantee OPENAI_API_KEY exists for config.py import (uses a dummy in tests)."""
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")

@pytest.fixture(autouse=True)
def reset_pipeline_state_contextvars():
    """Reset ContextVars after each test to prevent state bleeding between tests."""
    from tools._pipeline_state import _session_ctx

    # Store initial token
    token = _session_ctx.set(None)

    yield

    # Reset back after test (also force to None to be extra safe)
    _session_ctx.reset(token)
    _session_ctx.set(None)

