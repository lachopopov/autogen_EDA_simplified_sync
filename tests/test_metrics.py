"""
tests/test_metrics.py — Unit tests for core/metrics.py

Tests:
  * span() is a true no-op (no file I/O) when no session is active.
  * span() writes a correct JSONL record when a session IS active.
  * Duration is at least as long as a known sleep.
  * An exception inside span() still produces a JSONL record.
  * Multiple spans append to the same file.
  * Timestamp format matches expected ISO-8601 millisecond precision.
"""
from __future__ import annotations

import json
import time

import pytest

from core import metrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def active_session(tmp_path, monkeypatch):
    """Start a pipeline session and redirect timings writes to tmp_path."""
    from tools._pipeline_state import clear_session, init_session  # noqa: PLC0415

    session_id = init_session()
    monkeypatch.setattr("core.metrics.get_outputs_dir", lambda sid: tmp_path)
    yield session_id, tmp_path
    clear_session()


# ---------------------------------------------------------------------------
# No-session behaviour
# ---------------------------------------------------------------------------


class TestSpanNoSession:
    """span() must be a complete no-op when no session is active."""

    def test_no_file_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.metrics.get_outputs_dir", lambda sid: tmp_path)
        # conftest resets _session_ctx before each test, so no session is active.
        with metrics.span("noop_phase"):
            pass
        assert not (tmp_path / "timings.jsonl").exists()

    def test_yield_still_executes_body(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.metrics.get_outputs_dir", lambda sid: tmp_path)
        ran = []
        with metrics.span("noop"):
            ran.append(1)
        assert ran == [1]

    def test_exception_propagates_without_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.metrics.get_outputs_dir", lambda sid: tmp_path)
        with pytest.raises(ValueError, match="expected_error"):
            with metrics.span("failing_noop"):
                raise ValueError("expected_error")
        assert not (tmp_path / "timings.jsonl").exists()


# ---------------------------------------------------------------------------
# Active-session behaviour
# ---------------------------------------------------------------------------


class TestSpanWithSession:
    """span() writes a JSONL record when a session is active."""

    def test_jsonl_written(self, active_session):
        session_id, tmp_path = active_session
        with metrics.span("load"):
            pass

        timings = tmp_path / "timings.jsonl"
        assert timings.exists()
        record = json.loads(timings.read_text().strip())
        assert record["phase"] == "load"
        assert record["session_id"] == session_id
        assert isinstance(record["duration_ms"], float)
        assert record["duration_ms"] >= 0.0

    def test_ts_is_iso8601_milliseconds(self, active_session):
        _, tmp_path = active_session
        with metrics.span("ts_check"):
            pass

        record = json.loads((tmp_path / "timings.jsonl").read_text().strip())
        ts = record["ts"]
        # Must end with 'Z' and have exactly 3 fractional-second digits.
        assert ts.endswith("Z"), f"ts does not end with Z: {ts!r}"
        fractional = ts.split(".")[-1][:-1]  # strip trailing 'Z'
        assert len(fractional) == 3, f"Expected 3 ms digits, got: {fractional!r}"

    def test_duration_at_least_sleep_ms(self, active_session):
        _, tmp_path = active_session
        with metrics.span("slow_phase"):
            time.sleep(0.05)  # 50 ms

        record = json.loads((tmp_path / "timings.jsonl").read_text().strip())
        assert record["duration_ms"] >= 40.0  # generous lower bound

    def test_exception_inside_span_still_records(self, active_session):
        _, tmp_path = active_session
        with pytest.raises(RuntimeError, match="oops"):
            with metrics.span("failing"):
                raise RuntimeError("oops")

        timings = tmp_path / "timings.jsonl"
        assert timings.exists()
        record = json.loads(timings.read_text().strip())
        assert record["phase"] == "failing"
        assert isinstance(record["duration_ms"], float)

    def test_extra_dict_included_in_record(self, active_session):
        _, tmp_path = active_session
        with metrics.span("with_extra", extra={"cache_hit": True, "key": "abc"}):
            pass

        record = json.loads((tmp_path / "timings.jsonl").read_text().strip())
        assert record["extra"] == {"cache_hit": True, "key": "abc"}

    def test_empty_extra_defaults_to_empty_dict(self, active_session):
        _, tmp_path = active_session
        with metrics.span("no_extra"):
            pass

        record = json.loads((tmp_path / "timings.jsonl").read_text().strip())
        assert record["extra"] == {}

    def test_multiple_spans_appended_in_order(self, active_session):
        _, tmp_path = active_session
        with metrics.span("phase_a"):
            pass
        with metrics.span("phase_b"):
            pass

        lines = (tmp_path / "timings.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        phases = [json.loads(line)["phase"] for line in lines]
        assert phases == ["phase_a", "phase_b"]

    def test_io_error_handled_silently(self, active_session, monkeypatch):
        """An OSError when writing must not surface to the caller."""
        session_id, tmp_path = active_session

        def bad_outputs_dir(sid):
            # Return a path whose parent doesn't exist so open() raises OSError.
            return tmp_path / "nonexistent" / "subdir"

        monkeypatch.setattr("core.metrics.get_outputs_dir", bad_outputs_dir)
        # Must not raise, even though the write fails.
        with metrics.span("io_error_test"):
            pass


# ---------------------------------------------------------------------------
# record_span() — direct write variant
# ---------------------------------------------------------------------------


class TestRecordSpanNoSession:
    """record_span() must be a complete no-op when no session is active."""

    def test_no_file_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.metrics.get_outputs_dir", lambda sid: tmp_path)
        metrics.record_span("noop_agent", 123.4)
        assert not (tmp_path / "timings.jsonl").exists()


class TestRecordSpanWithSession:
    """record_span() writes a JSONL record when a session is active."""

    def test_jsonl_written(self, active_session):
        session_id, tmp_path = active_session
        metrics.record_span("agent.DataPrepAgent", 12340.5)

        timings = tmp_path / "timings.jsonl"
        assert timings.exists()
        record = json.loads(timings.read_text().strip())
        assert record["phase"] == "agent.DataPrepAgent"
        assert record["session_id"] == session_id
        assert record["duration_ms"] == 12340.5

    def test_duration_rounded_to_3dp(self, active_session):
        _, tmp_path = active_session
        metrics.record_span("agent.X", 99.99999)
        record = json.loads((tmp_path / "timings.jsonl").read_text().strip())
        assert record["duration_ms"] == 100.0

    def test_extra_dict_included(self, active_session):
        _, tmp_path = active_session
        metrics.record_span("agent.Y", 50.0, extra={"iteration": 2})
        record = json.loads((tmp_path / "timings.jsonl").read_text().strip())
        assert record["extra"] == {"iteration": 2}

    def test_appends_after_span(self, active_session):
        _, tmp_path = active_session
        with metrics.span("phase_a"):
            pass
        metrics.record_span("agent.Z", 999.0)

        lines = (tmp_path / "timings.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        phases = [json.loads(line)["phase"] for line in lines]
        assert phases == ["phase_a", "agent.Z"]

    def test_io_error_handled_silently(self, active_session, monkeypatch):
        _, tmp_path = active_session

        def bad_outputs_dir(sid):
            return tmp_path / "nonexistent" / "subdir"

        monkeypatch.setattr("core.metrics.get_outputs_dir", bad_outputs_dir)
        metrics.record_span("agent.W", 1.0)  # must not raise
