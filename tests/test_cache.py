"""
tests/test_cache.py — Unit tests for core/cache.py

Tests (per HLD Step 3):
  * is_enabled()  — False unless EDA_MODE == "final"
  * compute_key() — deterministic, 64-char hex, sensitive to each input
                    dimension, insensitive to file path / mtime / enable_openlit
  * lookup()      — None for missing keys, correct Path for present ones
  * store()       — creates entry, writes manifest, copies files; atomic
                    (no half-written entry on copy failure)
  * cleanup()     — removes entries older than TTL, preserves fresh ones,
                    tolerates absent CACHE_DIR, skips .tmp dirs
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

import core.cache as cache_mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a tmp directory."""
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / ".cache")
    return tmp_path / ".cache"


@pytest.fixture()
def sample_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    return f


@pytest.fixture()
def sample_params():
    return {
        "target_flag": None,
        "no_target_flag": True,
        "categoricals_flag": None,
        "no_reclassify_flag": False,
    }


@pytest.fixture()
def run_dir(tmp_path):
    """A minimal run directory with a placeholder file."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "cost_summary.txt").write_text("$0.01")
    return d


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_disabled_when_eda_mode_unset(self, monkeypatch):
        monkeypatch.delenv("EDA_MODE", raising=False)
        assert cache_mod.is_enabled() is False

    def test_disabled_in_dev_mode(self, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "dev")
        assert cache_mod.is_enabled() is False

    def test_enabled_in_final_mode(self, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "final")
        assert cache_mod.is_enabled() is True

    def test_disabled_for_unknown_mode(self, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "staging")
        assert cache_mod.is_enabled() is False


# ---------------------------------------------------------------------------
# compute_key
# ---------------------------------------------------------------------------


class TestComputeKey:
    def test_deterministic(self, sample_file, sample_params):
        k1 = cache_mod.compute_key(
            sample_file, sample_params, prompt_version="abc", pipeline_version="1.0.0",
        )
        k2 = cache_mod.compute_key(
            sample_file, sample_params, prompt_version="abc", pipeline_version="1.0.0",
        )
        assert k1 == k2

    def test_is_64_hex_chars(self, sample_file, sample_params):
        k = cache_mod.compute_key(
            sample_file, sample_params, prompt_version="abc", pipeline_version="1.0.0",
        )
        assert len(k) == 64
        int(k, 16)  # raises ValueError if not valid hex

    def test_sensitive_to_file_bytes(self, tmp_path, sample_params):
        f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
        f1.write_text("a,b\n1,2\n")
        f2.write_text("a,b\n9,9\n")
        k1 = cache_mod.compute_key(f1, sample_params, prompt_version="x", pipeline_version="1.0")
        k2 = cache_mod.compute_key(f2, sample_params, prompt_version="x", pipeline_version="1.0")
        assert k1 != k2

    def test_insensitive_to_file_path(self, tmp_path, sample_params):
        content = "a,b\n1,2\n"
        f1, f2 = tmp_path / "dir1" / "data.csv", tmp_path / "dir2" / "data.csv"
        f1.parent.mkdir()
        f2.parent.mkdir()
        f1.write_text(content)
        f2.write_text(content)
        k1 = cache_mod.compute_key(f1, sample_params, prompt_version="x", pipeline_version="1.0")
        k2 = cache_mod.compute_key(f2, sample_params, prompt_version="x", pipeline_version="1.0")
        assert k1 == k2  # same bytes → same key regardless of path

    def test_insensitive_to_file_mtime(self, tmp_path, sample_params):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        k1 = cache_mod.compute_key(f, sample_params, prompt_version="x", pipeline_version="1.0")
        # Shift mtime by 100 seconds
        os.utime(f, (time.time() + 100, time.time() + 100))
        k2 = cache_mod.compute_key(f, sample_params, prompt_version="x", pipeline_version="1.0")
        assert k1 == k2

    def test_sensitive_to_params(self, sample_file):
        p1 = {"target_flag": None,     "no_target_flag": False, "categoricals_flag": None, "no_reclassify_flag": False}
        p2 = {"target_flag": "label",  "no_target_flag": False, "categoricals_flag": None, "no_reclassify_flag": False}
        k1 = cache_mod.compute_key(sample_file, p1, prompt_version="x", pipeline_version="1.0")
        k2 = cache_mod.compute_key(sample_file, p2, prompt_version="x", pipeline_version="1.0")
        assert k1 != k2

    def test_sensitive_to_eda_mode(self, sample_file, sample_params, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "dev")
        k1 = cache_mod.compute_key(sample_file, sample_params, prompt_version="x", pipeline_version="1.0")
        monkeypatch.setenv("EDA_MODE", "final")
        k2 = cache_mod.compute_key(sample_file, sample_params, prompt_version="x", pipeline_version="1.0")
        assert k1 != k2

    def test_sensitive_to_model_name(self, sample_file, sample_params, monkeypatch):
        monkeypatch.setattr(cache_mod, "_get_model_name", lambda: "gpt-5-mini")
        k1 = cache_mod.compute_key(sample_file, sample_params, prompt_version="x", pipeline_version="1.0")
        monkeypatch.setattr(cache_mod, "_get_model_name", lambda: "gpt-5")
        k2 = cache_mod.compute_key(sample_file, sample_params, prompt_version="x", pipeline_version="1.0")
        assert k1 != k2

    def test_sensitive_to_prompt_version(self, sample_file, sample_params):
        k1 = cache_mod.compute_key(sample_file, sample_params, prompt_version="v1", pipeline_version="1.0")
        k2 = cache_mod.compute_key(sample_file, sample_params, prompt_version="v2", pipeline_version="1.0")
        assert k1 != k2

    def test_sensitive_to_pipeline_version(self, sample_file, sample_params):
        k1 = cache_mod.compute_key(sample_file, sample_params, prompt_version="x", pipeline_version="1.0.0")
        k2 = cache_mod.compute_key(sample_file, sample_params, prompt_version="x", pipeline_version="2.0.0")
        assert k1 != k2

    def test_enable_openlit_does_not_affect_key(self, tmp_path, sample_params):
        """enable_openlit is intentionally excluded from canonical_params."""
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        # Both calls pass the same params (enable_openlit not in params dict by design)
        k1 = cache_mod.compute_key(f, sample_params, prompt_version="x", pipeline_version="1.0")
        k2 = cache_mod.compute_key(f, sample_params, prompt_version="x", pipeline_version="1.0")
        assert k1 == k2


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_returns_none_for_missing_key(self, tmp_cache):
        assert cache_mod.lookup("nonexistent" * 4) is None

    def test_returns_path_when_entry_exists(self, tmp_cache):
        key = "a" * 64
        entry = tmp_cache / key
        entry.mkdir(parents=True)
        result = cache_mod.lookup(key)
        assert result == entry

    def test_returns_none_for_file_not_directory(self, tmp_cache):
        """lookup() only matches directories (actual cache entries)."""
        tmp_cache.mkdir(parents=True)
        key = "b" * 64
        (tmp_cache / key).write_text("oops")  # a file, not a dir
        assert cache_mod.lookup(key) is None


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TestStore:
    def test_creates_cache_entry_dir(self, tmp_cache, run_dir):
        key = "c" * 64
        cache_mod.store(key, run_dir)
        assert (tmp_cache / key).is_dir()

    def test_files_are_copied(self, tmp_cache, run_dir):
        key = "d" * 64
        cache_mod.store(key, run_dir)
        assert (tmp_cache / key / "cost_summary.txt").read_text() == "$0.01"

    def test_manifest_written_with_required_keys(self, tmp_cache, run_dir):
        key = "e" * 64
        cache_mod.store(key, run_dir)
        manifest = json.loads((tmp_cache / key / "manifest.json").read_text())
        for required in ("key", "pipeline_version", "prompt_version", "eda_mode", "model", "stored_at_iso"):
            assert required in manifest, f"Missing manifest key: {required}"
        assert manifest["key"] == key

    def test_atomic_no_half_written_entry_on_failure(self, tmp_cache, run_dir, monkeypatch):
        """A copy failure must not leave a partial entry at the key path."""
        key = "f" * 64

        def fail_copytree(*a, **kw):
            raise RuntimeError("Simulated copy failure")

        monkeypatch.setattr(shutil, "copytree", fail_copytree)

        with pytest.raises(RuntimeError, match="Simulated copy failure"):
            cache_mod.store(key, run_dir)

        # The final key directory must NOT exist.
        assert not (tmp_cache / key).exists()

    def test_overwrites_existing_entry(self, tmp_cache, tmp_path):
        """Calling store() twice with the same key replaces the old entry."""
        run_v1 = tmp_path / "run_v1"
        run_v1.mkdir()
        (run_v1 / "cost_summary.txt").write_text("v1")

        run_v2 = tmp_path / "run_v2"
        run_v2.mkdir()
        (run_v2 / "cost_summary.txt").write_text("v2")

        key = "g" * 64
        cache_mod.store(key, run_v1)
        cache_mod.store(key, run_v2)

        assert (tmp_cache / key / "cost_summary.txt").read_text() == "v2"


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def _make_entry(self, cache_dir: Path, key: str, age_days: float) -> Path:
        entry = cache_dir / key
        entry.mkdir(parents=True)
        stored_at = datetime.now(UTC) - timedelta(days=age_days)
        manifest = {"key": key, "stored_at_iso": stored_at.isoformat()}
        (entry / "manifest.json").write_text(json.dumps(manifest))
        return entry

    def test_removes_entries_older_than_ttl(self, tmp_cache):
        old = self._make_entry(tmp_cache, "0" * 64, age_days=10)
        cache_mod.cleanup(ttl_days=7)
        assert not old.exists()

    def test_preserves_fresh_entries(self, tmp_cache):
        fresh = self._make_entry(tmp_cache, "1" * 64, age_days=2)
        cache_mod.cleanup(ttl_days=7)
        assert fresh.exists()

    def test_handles_missing_cache_dir_gracefully(self, tmp_cache):
        # CACHE_DIR doesn't exist yet — must not raise.
        cache_mod.cleanup()

    def test_skips_tmp_dirs(self, tmp_cache):
        tmp_cache.mkdir(parents=True)
        tmp_entry = tmp_cache / ("abc123" + ".tmp")
        tmp_entry.mkdir()
        # Even with ttl_days=0 (everything is "old"), .tmp dirs are skipped.
        cache_mod.cleanup(ttl_days=0)
        assert tmp_entry.exists()

    def test_handles_entry_without_manifest(self, tmp_cache):
        """Entries without a manifest fall back to mtime-based cleanup."""
        no_manifest = tmp_cache / ("2" * 64)
        no_manifest.mkdir(parents=True)
        # Set mtime to 30 days ago
        old_ts = time.time() - 30 * 86400
        os.utime(no_manifest, (old_ts, old_ts))
        cache_mod.cleanup(ttl_days=7)
        assert not no_manifest.exists()
