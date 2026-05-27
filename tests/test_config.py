"""
test_config.py — Unit tests for config.py

Tests LLM configuration loading, environment variable handling,
project paths, and feature toggles.
"""

class TestLLMConfigDev:
    """Default (dev) mode uses gpt-5-mini."""

    def test_dev_model_name(self):
        import config

        model = config.LLM_CONFIG_DEV["config_list"][0]["model"]
        assert model == "gpt-5-mini"

    def test_dev_no_temperature_key(self):
        import config

        # gpt-5-nano/mini only support default temperature (1);
        # temperature must NOT be set in config to avoid 400 errors.
        assert "temperature" not in config.LLM_CONFIG_DEV["config_list"][0]

    def test_dev_cache_seed_none(self):
        import config

        seed = config.LLM_CONFIG_DEV["config_list"][0]["cache_seed"]
        assert seed is None

    def test_dev_has_api_key(self):
        import config

        key = config.LLM_CONFIG_DEV["config_list"][0]["api_key"]
        assert isinstance(key, str)
        assert len(key) > 0

    def test_dev_has_price_field(self):
        import config

        price = config.LLM_CONFIG_DEV["config_list"][0]["price"]
        assert isinstance(price, list)
        assert len(price) == 2

    def test_dev_price_values(self):
        import config

        price = config.LLM_CONFIG_DEV["config_list"][0]["price"]
        # gpt-5-mini: $0.25/1M prompt, $2.00/1M completion → per 1K
        assert price[0] == 0.00025
        assert price[1] == 0.002


class TestLLMConfigFinal:
    """Final mode uses gpt-5."""

    def test_final_model_name(self):
        import config

        model = config.LLM_CONFIG_FINAL["config_list"][0]["model"]
        assert model == "gpt-5"

    def test_final_no_temperature_key(self):
        import config

        # gpt-5-nano/mini only support default temperature (1);
        # temperature must NOT be set in config to avoid 400 errors.
        assert "temperature" not in config.LLM_CONFIG_FINAL["config_list"][0]

    def test_final_cache_seed_none(self):
        import config

        seed = config.LLM_CONFIG_FINAL["config_list"][0]["cache_seed"]
        assert seed is None

    def test_final_has_price_field(self):
        import config

        price = config.LLM_CONFIG_FINAL["config_list"][0]["price"]
        assert isinstance(price, list)
        assert len(price) == 2

    def test_final_price_values(self):
        import config

        price = config.LLM_CONFIG_FINAL["config_list"][0]["price"]
        # gpt-5: $1.25/1M prompt, $10.00/1M completion → per 1K
        assert price[0] == 0.00125
        assert price[1] == 0.01


class TestLLMConfigFinalRest:
    """Final-rest config uses gpt-5-mini (non-FindingsGenerator agents in final mode)."""

    def test_final_rest_model_name(self):
        import config

        model = config.LLM_CONFIG_FINAL_REST["config_list"][0]["model"]
        assert model == "gpt-5-mini"

    def test_final_rest_price_matches_gpt5_mini(self):
        import config

        price = config.LLM_CONFIG_FINAL_REST["config_list"][0]["price"]
        # Must match gpt-5-mini rates ($0.25/$2.00 per 1M), not gpt-5
        assert price[0] == 0.00025
        assert price[1] == 0.002


class TestEDAModeSwitch:
    """EDA_MODE env var selects active LLM config."""

    def test_default_mode_is_dev(self, monkeypatch):
        monkeypatch.delenv("EDA_MODE", raising=False)
        # Force re-evaluation by reloading
        import importlib

        import config

        importlib.reload(config)
        assert config.LLM_CONFIG["config_list"][0]["model"] == "gpt-5-mini"

    def test_final_mode_selects_mini(self, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "final")
        import importlib

        import config

        importlib.reload(config)
        assert config.LLM_CONFIG["config_list"][0]["model"] == "gpt-5-mini"

    def test_unknown_mode_falls_back_to_dev(self, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "unknown_value")
        import importlib

        import config

        importlib.reload(config)
        assert config.LLM_CONFIG["config_list"][0]["model"] == "gpt-5-mini"

    def test_dev_mode_cache_seed_none(self, monkeypatch):
        monkeypatch.delenv("EDA_MODE", raising=False)
        import importlib

        import config

        importlib.reload(config)
        assert config.LLM_CONFIG["config_list"][0]["cache_seed"] is None

    def test_final_mode_cache_seed_42(self, monkeypatch):
        monkeypatch.setenv("EDA_MODE", "final")
        import importlib

        import config

        importlib.reload(config)
        assert config.LLM_CONFIG["config_list"][0]["cache_seed"] == 42


class TestProjectPaths:
    """Project path constants resolve correctly."""

    def test_project_root_exists(self):
        import config

        assert config.PROJECT_ROOT.exists()

    def test_outputs_dir_under_project_root(self):
        import config

        assert config.GLOBAL_OUTPUTS_DIR.parent == config.PROJECT_ROOT

    def test_plots_dir_under_outputs(self):
        import config

        out_dir = config.get_outputs_dir("test")
        plots_dir = config.get_plots_dir("test")
        assert plots_dir.parent == out_dir


class TestFeatureToggles:
    """Feature flags and numeric config from env vars."""

    def test_ipynb_export_default_false(self, monkeypatch):
        monkeypatch.delenv("IPYNB_EXPORT", raising=False)
        import importlib

        import config

        importlib.reload(config)
        assert config.IPYNB_EXPORT is False

    def test_ipynb_export_true(self, monkeypatch):
        monkeypatch.setenv("IPYNB_EXPORT", "true")
        import importlib

        import config

        importlib.reload(config)
        assert config.IPYNB_EXPORT is True

    def test_ipynb_export_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("IPYNB_EXPORT", "True")
        import importlib

        import config

        importlib.reload(config)
        assert config.IPYNB_EXPORT is True

    def test_max_critic_iterations_default(self, monkeypatch):
        monkeypatch.delenv("MAX_CRITIC_ITERATIONS", raising=False)
        import importlib

        import config

        importlib.reload(config)
        assert config.MAX_CRITIC_ITERATIONS == 2

    def test_max_critic_iterations_override(self, monkeypatch):
        monkeypatch.setenv("MAX_CRITIC_ITERATIONS", "5")
        import importlib

        import config

        importlib.reload(config)
        assert config.MAX_CRITIC_ITERATIONS == 5

    def test_max_rounds_default(self, monkeypatch):
        monkeypatch.delenv("MAX_ROUNDS", raising=False)
        import importlib

        import config

        importlib.reload(config)
        assert config.MAX_ROUNDS == 70

    def test_max_rounds_override(self, monkeypatch):
        monkeypatch.setenv("MAX_ROUNDS", "30")
        import importlib

        import config

        importlib.reload(config)
        assert config.MAX_ROUNDS == 30


# ---------------------------------------------------------------------------
# get_outputs_dir / get_plots_dir
# ---------------------------------------------------------------------------


class TestGetOutputsDir:
    """get_outputs_dir returns correct paths and falls back gracefully."""

    def test_no_session_returns_global_outputs(self):
        import config

        result = config.get_outputs_dir(None)
        assert result == config.GLOBAL_OUTPUTS_DIR

    def test_existing_run_dir_returned(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        run_dir = tmp_path / "runs" / "mysession"
        run_dir.mkdir(parents=True)
        result = config.get_outputs_dir("mysession")
        assert result == run_dir

    def test_missing_run_dir_falls_back_to_cache(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        cache_dir = tmp_path / ".cache" / "mysession"
        cache_dir.mkdir(parents=True)
        result = config.get_outputs_dir("mysession")
        assert result == cache_dir

    def test_neither_run_nor_cache_returns_run_path(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        result = config.get_outputs_dir("newsession")
        # Falls through to caller-will-mkdir path: runs/newsession
        assert result == tmp_path / "runs" / "newsession"

    def test_empty_session_id_treated_as_falsy(self):
        import config

        result = config.get_outputs_dir("")
        assert result == config.GLOBAL_OUTPUTS_DIR

    def test_plots_dir_is_child_of_outputs_dir(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        out = config.get_outputs_dir("s1")
        plots = config.get_plots_dir("s1")
        assert plots.parent == out

    def test_returns_path_object(self):
        import config

        result = config.get_outputs_dir("somesession")
        assert hasattr(result, "exists")  # Path duck-type


# ---------------------------------------------------------------------------
# cleanup_old_runs
# ---------------------------------------------------------------------------


class TestCleanupOldRuns:
    """cleanup_old_runs removes stale directories and leaves fresh ones."""

    def test_removes_old_directory(self, tmp_path, monkeypatch):
        import datetime
        import os

        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        old_dir = tmp_path / "old_run"
        old_dir.mkdir()
        old_mtime = (datetime.datetime.now() - datetime.timedelta(hours=25)).timestamp()
        os.utime(old_dir, (old_mtime, old_mtime))
        config.cleanup_old_runs(hours=24)
        assert not old_dir.exists()

    def test_keeps_fresh_directory(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        fresh_dir = tmp_path / "fresh_run"
        fresh_dir.mkdir()
        config.cleanup_old_runs(hours=24)
        assert fresh_dir.exists()

    def test_no_error_when_runs_dir_missing(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "nonexistent")
        config.cleanup_old_runs(hours=24)  # Must not raise


# ---------------------------------------------------------------------------
# ensure_run_dirs — once-per-process cleanup guard
# ---------------------------------------------------------------------------


class TestEnsureRunDirs:
    """ensure_run_dirs creates dirs and fires cleanup exactly once per process."""

    def _reset(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "_cleanup_done", False)

    def test_creates_output_dir(self, tmp_path, monkeypatch):
        import config

        self._reset(monkeypatch)
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        config.ensure_run_dirs("sess1")
        assert (tmp_path / "runs" / "sess1").is_dir()

    def test_creates_plots_subdir(self, tmp_path, monkeypatch):
        import config

        self._reset(monkeypatch)
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        config.ensure_run_dirs("sess1")
        assert (tmp_path / "runs" / "sess1" / "plots").is_dir()

    def test_cleanup_called_once(self, tmp_path, monkeypatch):
        import config

        self._reset(monkeypatch)
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        call_count = {"n": 0}
        original = config.cleanup_old_runs

        def counting_cleanup(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(config, "cleanup_old_runs", counting_cleanup)
        config.ensure_run_dirs("sess1")
        config.ensure_run_dirs("sess2")
        config.ensure_run_dirs("sess3")
        assert call_count["n"] == 1  # fired exactly once

    def test_cleanup_done_flag_set_after_first_call(self, tmp_path, monkeypatch):
        import config

        self._reset(monkeypatch)
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        assert config._cleanup_done is False
        config.ensure_run_dirs("sess1")
        assert config._cleanup_done is True

    def test_cleanup_skipped_when_flag_already_set(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setattr(config, "_cleanup_done", True)
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(config, "GLOBAL_OUTPUTS_DIR", tmp_path)
        call_count = {"n": 0}
        monkeypatch.setattr(config, "cleanup_old_runs", lambda *a, **kw: call_count.update({"n": call_count["n"] + 1}))
        config.ensure_run_dirs("sess1")
        assert call_count["n"] == 0  # cleanup skipped
