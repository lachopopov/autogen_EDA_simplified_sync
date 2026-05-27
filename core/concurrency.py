"""core/concurrency.py — Singleton semaphore guard for the EDA pipeline.

Ensures at most one pipeline run executes at a time within a process.
A non-blocking acquire raises SystemBusy immediately rather than queuing.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class SystemBusy(RuntimeError):
    """Raised when the pipeline semaphore cannot be acquired non-blockingly."""


_PIPELINE_SEM: threading.BoundedSemaphore = threading.BoundedSemaphore(1)


@contextmanager
def pipeline_guard() -> Iterator[None]:
    """Acquire the singleton semaphore; raise :class:`SystemBusy` if already held.

    The acquire is non-blocking so callers receive an immediate error
    instead of waiting.  The semaphore is always released in the finally
    block, even when the body raises an exception.
    """
    if not _PIPELINE_SEM.acquire(blocking=False):
        raise SystemBusy("Another pipeline run is already in progress.")
    try:
        yield
    finally:
        _PIPELINE_SEM.release()
