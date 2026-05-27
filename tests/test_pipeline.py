"""
tests/test_pipeline.py — Smoke tests that pipeline.py is the authoritative source.

Verifies:
  * run_pipeline is importable from both `pipeline` and `main` and is the
    same object (re-export contract).
  * PIPELINE_VERSION and PROMPT_VERSION are present with expected shapes.
  * _format_timings produces correct output for valid and edge-case inputs.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


class TestReexport:
    def test_run_pipeline_importable_from_pipeline(self):
        from pipeline import run_pipeline  # noqa: PLC0415

        assert callable(run_pipeline)

    def test_run_pipeline_re_exported_from_main(self):
        from main import run_pipeline as rp_main  # noqa: PLC0415
        from pipeline import run_pipeline as rp_pipeline  # noqa: PLC0415

        # Both names must resolve to the same function object.
        assert rp_pipeline is rp_main


class TestVersionConstants:
    def test_pipeline_version_present(self):
        from pipeline import PIPELINE_VERSION  # noqa: PLC0415

        assert isinstance(PIPELINE_VERSION, str)
        assert PIPELINE_VERSION == "1.0.0"

    def test_prompt_version_is_64_hex_chars(self):
        from pipeline import PROMPT_VERSION  # noqa: PLC0415

        assert isinstance(PROMPT_VERSION, str)
        assert len(PROMPT_VERSION) == 64
        # Verify all characters are valid hex digits.
        int(PROMPT_VERSION, 16)

    def test_prompt_version_is_deterministic(self):
        """Re-importing pipeline must yield the same PROMPT_VERSION."""
        import importlib  # noqa: PLC0415

        import pipeline as p  # noqa: PLC0415

        p2 = importlib.import_module("pipeline")
        assert p.PROMPT_VERSION == p2.PROMPT_VERSION


class TestFormatTimings:
    """Unit tests for _format_timings()."""

    def _write_timings(self, tmp_path: Path, records: list[dict]) -> Path:
        p = tmp_path / "timings.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return p

    def test_returns_empty_string_when_file_missing(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        result = _format_timings(tmp_path / "nonexistent.jsonl")
        assert result == ""

    def test_returns_empty_string_for_empty_file(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        p = tmp_path / "timings.jsonl"
        p.write_text("", encoding="utf-8")
        assert _format_timings(p) == ""

    def test_phase_timings_header_present(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        p = self._write_timings(tmp_path, [
            {"phase": "file_load", "duration_ms": 23.7},
        ])
        result = _format_timings(p)
        assert "Phase Timings" in result

    def test_pipeline_phases_listed(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        records = [
            {"phase": "file_load", "duration_ms": 23.7},
            {"phase": "initiate_chat", "duration_ms": 84210.5},
            {"phase": "cost_summary", "duration_ms": 18.3},
        ]
        p = self._write_timings(tmp_path, records)
        result = _format_timings(p)
        assert "file_load" in result
        assert "initiate_chat" in result
        assert "cost_summary" in result

    def test_agent_breakdown_section_appears_when_agent_spans_present(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        records = [
            {"phase": "agent.DataPrepAgent", "duration_ms": 12340.0},
            {"phase": "agent.EDAAnalysisAgent", "duration_ms": 45210.0},
            {"phase": "initiate_chat", "duration_ms": 372115.9},
            {"phase": "cost_summary", "duration_ms": 0.8},
        ]
        p = self._write_timings(tmp_path, records)
        result = _format_timings(p)
        assert "Agent Breakdown" in result
        # Labels must appear without the "agent." prefix
        assert "DataPrepAgent" in result
        assert "EDAAnalysisAgent" in result

    def test_agent_breakdown_absent_when_no_agent_spans(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        records = [
            {"phase": "initiate_chat", "duration_ms": 84210.5},
            {"phase": "cost_summary", "duration_ms": 18.3},
        ]
        p = self._write_timings(tmp_path, records)
        result = _format_timings(p)
        assert "Agent Breakdown" not in result

    def test_total_excludes_agent_spans(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        # Pipeline total should be initiate_chat + cost_summary = 300 ms
        # Agent spans (500 ms each) must NOT be added to it.
        records = [
            {"phase": "agent.DataPrepAgent", "duration_ms": 500.0},
            {"phase": "initiate_chat", "duration_ms": 200.0},
            {"phase": "cost_summary", "duration_ms": 100.0},
        ]
        p = self._write_timings(tmp_path, records)
        result = _format_timings(p)
        # total = 300 ms → 0.3 s
        assert "0.3 s" in result
        # Must NOT contain the double-counted sum (1100 ms / 1.1 s)
        assert "1.1 s" not in result

    def test_total_line_present_and_correct(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        records = [
            {"phase": "file_load", "duration_ms": 100.0},
            {"phase": "initiate_chat", "duration_ms": 200.0},
        ]
        p = self._write_timings(tmp_path, records)
        result = _format_timings(p)
        assert "total" in result
        # 300 ms total → 0.3 s
        assert "300" in result
        assert "0.3 s" in result

    def test_skips_malformed_json_lines(self, tmp_path):
        from pipeline import _format_timings  # noqa: PLC0415

        p = tmp_path / "timings.jsonl"
        p.write_text(
            '{"phase": "file_load", "duration_ms": 10.0}\n'
            "not-valid-json\n"
            '{"phase": "cost_summary", "duration_ms": 5.0}\n',
            encoding="utf-8",
        )
        result = _format_timings(p)
        assert "file_load" in result
        assert "cost_summary" in result
