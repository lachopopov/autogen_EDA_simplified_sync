"""core/metrics.py — Lightweight phase-timing spans for the EDA pipeline.

JSONL schema per record:
    {"ts": "ISO8601ms", "session_id": "...", "phase": "...",
     "duration_ms": float, "extra": {}}

Two entry points
----------------
``span(name)``
    Context-manager variant.  Use when the timed block can be wrapped:
    ``with span("file_load"): ...``

``record_span(name, duration_ms)``
    Direct write variant.  Use when start and end events fire in separate
    call frames — e.g. the AG2 router, where one callback fires on agent
    entry and the next fires on agent exit.  The caller computes the
    duration from ``time.perf_counter()`` deltas.

Behaviour:
  * Both functions are true no-ops (zero file I/O) when no session is active.
  * Exceptions inside ``with span(...)`` still propagate; the record is
    always written with the elapsed duration.
  * Writes are append-mode to ``outputs/runs/<session_id>/timings.jsonl``.

Design note — pre-session spans are intentional no-ops
------------------------------------------------------
Spans called before ``init_session()`` (e.g. ``file_load``, ``target_resolve``,
``encoded_categorical_resolve``) will always be silent no-ops.  This is by
design: those phases must complete *before* the session directory is created
because their outputs (file hash, target column) determine the cache key.  Moving
them inside the session would break the cache-first logic.  Those phases are
also cheap (<100 ms) and are not the bottleneck.

Per-agent timing
----------------
The dominant cost is ``initiate_chat`` (the full AG2 groupchat).  Per-agent
breakdown is captured by the deterministic router in ``orchestrator.py`` via
``record_span()``.  The router records a ``agent.<AgentName>`` span for each
AssistantAgent stage, covering all LLM calls and tool roundtrips for that
agent.  OpenLIT (when enabled) provides the same data plus call-level detail
in its dashboard — the two are complementary: local ``timings.jsonl`` works
without any external infrastructure; OpenLIT adds hallucination/bias/toxicity
evaluation on top.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from config import get_outputs_dir  # module-level so tests can patch core.metrics.get_outputs_dir

logger = logging.getLogger(__name__)


@contextmanager
def span(name: str, extra: dict | None = None) -> Iterator[None]:
    """Time a phase; append a JSONL record to the session's timings.jsonl.

    A true no-op when ``tools._pipeline_state.is_active()`` is False.
    """
    from tools._pipeline_state import get_session_id, is_active  # lazy — avoids circular at import

    if not is_active():
        yield
        return

    session_id = get_session_id()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        record = {
            "ts": ts,
            "session_id": session_id,
            "phase": name,
            "duration_ms": duration_ms,
            "extra": extra or {},
        }
        try:
            timings_path = get_outputs_dir(session_id) / "timings.jsonl"
            with timings_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            logger.warning("metrics.span: could not write timings for phase=%s", name)


def record_span(name: str, duration_ms: float, extra: dict | None = None) -> None:
    """Write a single timing record directly (non-context-manager variant).

    Use when the timed block cannot be wrapped in a ``with span()`` context —
    e.g. the AG2 router in ``orchestrator.py``, where start and end events
    fire in separate calls to ``state_flow_transition``.

    The caller is responsible for computing ``duration_ms`` from
    ``time.perf_counter()`` deltas::

        start = time.perf_counter()
        # ... work happens across multiple router calls ...
        duration_ms = (time.perf_counter() - start) * 1000
        record_span("agent.DataPrepAgent", duration_ms)

    A true no-op when no session is active.
    """
    from tools._pipeline_state import get_session_id, is_active  # lazy — avoids circular at import

    if not is_active():
        return

    session_id = get_session_id()
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    record = {
        "ts": ts,
        "session_id": session_id,
        "phase": name,
        "duration_ms": round(duration_ms, 3),
        "extra": extra or {},
    }
    try:
        timings_path = get_outputs_dir(session_id) / "timings.jsonl"
        with timings_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("metrics.record_span: could not write timings for phase=%s", name)
