"""
tests/test_concurrency.py — Unit tests for core/concurrency.py

Tests (per HLD Step 4):
  * Single caller acquires and releases cleanly.
  * Two concurrent threads: exactly one succeeds, the other raises SystemBusy.
  * An exception inside the guard still releases the semaphore.
  * BoundedSemaphore prevents over-release (regression guard).
  * SystemBusy is a subclass of RuntimeError.
"""
from __future__ import annotations

import threading
import time

import pytest

import core.concurrency as conc_mod
from core.concurrency import SystemBusy, pipeline_guard

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_semaphore():
    """Recreate the module-level semaphore fresh before and after each test."""
    conc_mod._PIPELINE_SEM = threading.BoundedSemaphore(1)
    yield
    conc_mod._PIPELINE_SEM = threading.BoundedSemaphore(1)


# ---------------------------------------------------------------------------
# Basic guard behaviour
# ---------------------------------------------------------------------------


class TestPipelineGuard:
    def test_single_caller_succeeds(self):
        with pipeline_guard():
            pass  # must not raise

    def test_releases_semaphore_on_success(self):
        with pipeline_guard():
            pass
        # Should be acquirable again immediately.
        with pipeline_guard():
            pass

    def test_releases_semaphore_on_exception(self):
        with pytest.raises(ValueError, match="boom"):
            with pipeline_guard():
                raise ValueError("boom")
        # Semaphore must have been released despite the exception.
        with pipeline_guard():
            pass

    def test_yields_to_body(self):
        ran = []
        with pipeline_guard():
            ran.append(True)
        assert ran == [True]


# ---------------------------------------------------------------------------
# Concurrency: second caller raises SystemBusy
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    def test_second_caller_raises_system_busy(self):
        acquired = threading.Event()
        results: list[str] = []

        def hold():
            with pipeline_guard():
                acquired.set()     # signal: guard is held
                time.sleep(0.2)    # keep holding
                results.append("success")

        def attempt():
            acquired.wait()        # wait until hold() has the guard
            try:
                with pipeline_guard():
                    results.append("success")
            except SystemBusy:
                results.append("busy")

        t1 = threading.Thread(target=hold)
        t2 = threading.Thread(target=attempt)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert results.count("success") == 1
        assert results.count("busy") == 1


# ---------------------------------------------------------------------------
# Error hierarchy and semaphore guards
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_system_busy_is_runtime_error(self):
        assert issubclass(SystemBusy, RuntimeError)

    def test_system_busy_message(self):
        # Manually exhaust the semaphore to trigger SystemBusy.
        conc_mod._PIPELINE_SEM.acquire(blocking=False)
        try:
            with pytest.raises(SystemBusy):
                with pipeline_guard():
                    pass
        finally:
            conc_mod._PIPELINE_SEM.release()


class TestBoundedSemaphore:
    def test_over_release_raises_value_error(self):
        """BoundedSemaphore must catch accidental double-release (regression guard)."""
        # The semaphore starts at 1 (initial value = bound).
        # Releasing again without acquiring would exceed the bound → ValueError.
        with pytest.raises(ValueError):
            conc_mod._PIPELINE_SEM.release()
