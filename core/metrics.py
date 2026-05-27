"""core/metrics.py — Lightweight phase-timing spans for the EDA pipeline.

JSONL schema per record:
    {"ts": "ISO8601ms", "session_id": "...", "phase": "...",
     "duration_ms": float, "extra": {}}

Behaviour:
  * span() is a true no-op (zero file I/O) when no session is active.
  * Exceptions inside ``with span(...)`` still propagate; the record is
    always written with the elapsed duration.
  * Writes are append-mode to ``outputs/runs/<session_id>/timings.jsonl``.
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
