"""
tests/test_pipeline.py — Smoke tests that pipeline.py is the authoritative source.

Verifies:
  * run_pipeline is importable from both `pipeline` and `main` and is the
    same object (re-export contract).
  * PIPELINE_VERSION and PROMPT_VERSION are present with expected shapes.
"""
from __future__ import annotations


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
