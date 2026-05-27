"""
tools/_pipeline_state.py — Disk-backed artifact store with reference passing.

Problem:
    In AG2 GroupChat, each agent reads tool outputs from conversation history.
    With small LLMs (gpt-5-nano), agents fail to reliably copy large JSON blobs
    (e.g., 15KB DataFrame JSON) from conversation to tool parameters — they
    truncate, mangle, or fabricate the JSON.

Solution — Artifact Store + Reference Passing + Three-Tier Fallback:
    1. Each tool saves its output to a well-known file on disk (fixed key).
    2. Tools return a short reference string ``STATE_REF:<key>`` when a
       session is active (production), or full JSON when no session (tests).
    3. Downstream tools use ``resolve()`` to dereference:
       - Tier 1: ``STATE_REF:<key>`` prefix → load from disk
       - Tier 2: Valid JSON string → use directly (test compat)
       - Tier 3: Garbage → fallback to known-key file (corruption guard)

Architecture note:
    This module is pure-Python (no AG2 imports).  It lives in tools/ and
    is imported by other tool modules.  The Hard Boundary Rule is preserved.

    State is infrastructure, not LLM reasoning.  The LLM never sees, manages,
    or reasons about sessions, keys, or artifact files.

Scaling note:
    ``_session_id`` is currently a module-level global — fine for single-process
    CLI usage.  For FastAPI / multi-worker / async concurrency, replace with
    ``contextvars.ContextVar[str | None]``.  The public API surface stays
    identical; only the storage mechanism changes.

State directory: ``outputs/.pipeline_state/<session_id>/``
"""

from __future__ import annotations

import contextvars
import json
import logging
import shutil
import threading
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolved relative to project root (one level above tools/)
_BASE_STATE_DIR: Path = Path(__file__).resolve().parent.parent / "outputs" / ".pipeline_state"

# Prefix used in reference strings returned to the LLM
STATE_REF_PREFIX: str = "STATE_REF:"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PipelineStateError(RuntimeError):
    """Raised when an artifact cannot be resolved — upstream tool may not have run."""


# ---------------------------------------------------------------------------
# Context-local session state (thread/async safe for Streamlit)
# ---------------------------------------------------------------------------

_session_ctx = contextvars.ContextVar[str | None]('_session_ctx', default=None)

# Process-wide registry of all currently active session IDs.
_active_sessions: set[str] = set()
_active_lock: threading.Lock = threading.Lock()


def get_active_sessions() -> frozenset[str]:
    """Return a snapshot of all session IDs currently active in this process."""
    with _active_lock:
        return frozenset(_active_sessions)


def get_session_id() -> str | None:
    """Return the currently active session ID, or None."""
    return _session_ctx.get()

def _session_dir() -> Path:
    """Return the current session's artifact directory.  Raises if no session."""
    session_id = get_session_id()
    if session_id is None:
        raise PipelineStateError("No active pipeline session — call init_session() first")
    return _BASE_STATE_DIR / session_id


# ---------------------------------------------------------------------------
# Lifecycle API (called from main.py only)
# ---------------------------------------------------------------------------


def init_session(session_id: str | None = None) -> str:
    """
    Create a fresh session directory with the given ID (or generate a UUID).

    Sets the context variable ``_session_ctx`` so all subsequent calls in this
    context target this session.

    Returns the session ID.
    """
    # Deregister the previous session in this context (if any) before replacing it.
    old_id = get_session_id()
    if old_id is not None:
        with _active_lock:
            _active_sessions.discard(old_id)

    if session_id is None:
        import datetime
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{now_str}_{uuid4().hex[:6]}"

    _session_ctx.set(session_id)
    _session_dir().mkdir(parents=True, exist_ok=True)
    with _active_lock:
        _active_sessions.add(session_id)
    logger.info("Pipeline session initialized: %s", session_id)
    return session_id


def clear_session() -> None:
    """
    Remove the current session directory and reset ``_session_ctx``.

    Idempotent — safe to call even if no session is active or the
    directory was already removed.
    """
    session_id = get_session_id()
    if session_id is not None:
        path = _BASE_STATE_DIR / session_id
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        with _active_lock:
            _active_sessions.discard(session_id)
        logger.info("Pipeline session cleared: %s", session_id)
        _session_ctx.set(None)


def is_active() -> bool:
    """Return ``True`` if a pipeline session is currently active."""
    return get_session_id() is not None


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def save_state(key: str, data: str) -> None:
    """
    Persist a JSON string under the fixed ``key`` in the current session.

    Silently skips on I/O errors — non-critical (pipeline can still work
    via LLM-passed params or tier-2 resolve).
    """
    if not is_active():
        return
    try:
        path = _session_dir() / f"{key}.json"
        path.write_text(data, encoding="utf-8")
        logger.debug("Artifact saved: %s (%d bytes)", key, len(data))
    except OSError:
        logger.warning("Could not save artifact for key=%s", key)


def load_state(key: str) -> str | None:
    """
    Load a previously saved JSON string by fixed ``key``.

    Returns ``None`` if the file doesn't exist or no session is active.
    """
    if not is_active():
        return None
    path = _session_dir() / f"{key}.json"
    if path.exists():
        try:
            data = path.read_text(encoding="utf-8")
            logger.debug("Artifact loaded: %s (%d bytes)", key, len(data))
            return data
        except OSError:
            logger.warning("Could not read artifact for key=%s", key)
    return None


# ---------------------------------------------------------------------------
# Three-tier resolve — the corruption-tolerance engine
# ---------------------------------------------------------------------------


def resolve(param: str, fallback_key: str) -> str:
    """
    Resolve a tool parameter to valid data using three tiers:

    1. **Reference**: ``param`` starts with ``STATE_REF:`` → extract key,
       load artifact from disk.
    2. **Raw JSON**: ``param`` is a valid JSON string → return as-is
       (backwards compat with tests and correct LLM output).
    3. **Fallback**: ``param`` is garbage → load artifact by
       ``fallback_key`` from disk (LLM corruption guard).

    Raises
    ------
    PipelineStateError
        If all three tiers fail — the upstream tool likely did not run.
        Never returns ``None`` silently.
    """
    # --- Tier 1: STATE_REF:<key> ---
    if isinstance(param, str) and param.startswith(STATE_REF_PREFIX):
        ref_key = param[len(STATE_REF_PREFIX):]
        data = load_state(ref_key)
        if data is not None:
            logger.debug("Resolve tier-1 (ref): key=%s", ref_key)
            return data
        # Reference key not found — fall through to tier 3

    # --- Tier 2: valid JSON ---
    if isinstance(param, str) and param.strip():
        try:
            json.loads(param)
            logger.debug("Resolve tier-2 (raw JSON): len=%d", len(param))
            return param
        except (json.JSONDecodeError, ValueError):
            pass  # Not valid JSON — fall through

    # --- Tier 3: fallback to known key ---
    data = load_state(fallback_key)
    if data is not None:
        logger.info("Resolve tier-3 (fallback): key=%s", fallback_key)
        return data

    # All tiers exhausted
    raise PipelineStateError(
        f"Cannot resolve artifact '{fallback_key}'. "
        f"Upstream tool may not have executed. "
        f"LLM param was: {param!r:.200}"
    )
