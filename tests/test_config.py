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
        # gpt-5: $2.50/1M prompt, $15.00/1M completion → per 1K
        assert price[0] == 0.0025
        assert price[1] == 0.015


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
