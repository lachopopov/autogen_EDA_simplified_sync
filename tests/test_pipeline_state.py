"""
tests/test_pipeline_state.py — Unit tests for tools/_pipeline_state.py

Tests the disk-backed artifact store: init/clear lifecycle, save/load
round-trips, three-tier resolve, PipelineStateError, session isolation,
and edge cases.

No LLM calls — pure infrastructure tests.
"""

import json

import pytest

import tools._pipeline_state as ps_module
from tools._pipeline_state import (
    _BASE_STATE_DIR,
    STATE_REF_PREFIX,
    PipelineStateError,
    clear_session,
    get_active_sessions,
    init_session,
    is_active,
    load_state,
    resolve,
    save_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_session():
    """Ensure no session is active before/after each test."""
    clear_session()
    yield
    clear_session()


@pytest.fixture()
def active_session():
    """Start a session and return its ID.  Cleaned up by _clean_session."""
    return init_session()


# ---------------------------------------------------------------------------
# Lifecycle: init_session / clear_session / is_active
# ---------------------------------------------------------------------------


class TestInitSession:
    def test_returns_hybrid_id(self):
        sid = init_session()
        # Hybrid id is format YYYYMMDD_HHMMSS_xxxxxx (15 + 1 + 6 = 22)
        assert len(sid) == 22
        assert "_" in sid

    def test_creates_directory(self):
        sid = init_session()
        session_dir = _BASE_STATE_DIR / sid
        assert session_dir.is_dir()

    def test_sets_module_global(self):
        sid = init_session()
        assert ps_module.get_session_id() == sid

    def test_activates_session(self):
        assert not is_active()
        init_session()
        assert is_active()

    def test_second_init_overwrites(self):
        sid1 = init_session()
        sid2 = init_session()
        assert sid1 != sid2
        assert ps_module.get_session_id() == sid2
        # Both directories still exist (clear wasn't called)
        assert (_BASE_STATE_DIR / sid1).is_dir()
        assert (_BASE_STATE_DIR / sid2).is_dir()


class TestClearSession:
    def test_removes_directory(self, active_session):
        session_dir = _BASE_STATE_DIR / active_session
        assert session_dir.is_dir()
        clear_session()
        assert not session_dir.exists()

    def test_resets_to_none(self, active_session):
        clear_session()
        assert ps_module.get_session_id() is None

    def test_deactivates(self, active_session):
        assert is_active()
        clear_session()
        assert not is_active()

    def test_idempotent_no_session(self):
        """Calling clear_session when no session is active does not raise."""
        assert not is_active()
        clear_session()  # Should not raise
        assert not is_active()

    def test_idempotent_double_clear(self, active_session):
        clear_session()
        clear_session()  # Second call should not raise
        assert not is_active()

    def test_idempotent_dir_already_removed(self, active_session):
        """Edge case: directory was already removed externally."""
        session_dir = _BASE_STATE_DIR / active_session
        session_dir.rmdir()  # remove manually
        clear_session()  # Should not raise
        assert not is_active()


class TestIsActive:
    def test_false_initially(self):
        assert not is_active()

    def test_true_after_init(self, active_session):
        assert is_active()

    def test_false_after_clear(self, active_session):
        clear_session()
        assert not is_active()


# ---------------------------------------------------------------------------
# Persistence: save_state / load_state
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trip(self, active_session):
        data = json.dumps({"col": [1, 2, 3]})
        save_state("data_json", data)
        loaded = load_state("data_json")
        assert loaded == data

    def test_json_fidelity(self, active_session):
        """Saved JSON can be re-parsed identically."""
        obj = {"describe": {"a": {"mean": 3.14}}, "nulls": None}
        data = json.dumps(obj)
        save_state("describe_stats", data)
        loaded = load_state("describe_stats")
        assert json.loads(loaded) == obj

    def test_multiple_keys(self, active_session):
        save_state("k1", '"hello"')
        save_state("k2", '"world"')
        assert load_state("k1") == '"hello"'
        assert load_state("k2") == '"world"'

    def test_overwrite_key(self, active_session):
        save_state("k", '"v1"')
        save_state("k", '"v2"')
        assert load_state("k") == '"v2"'

    def test_file_created_on_disk(self, active_session):
        save_state("mykey", '{"x":1}')
        path = _BASE_STATE_DIR / active_session / "mykey.json"
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == '{"x":1}'


class TestLoadStateMissing:
    def test_missing_key_returns_none(self, active_session):
        assert load_state("nonexistent") is None

    def test_no_session_returns_none(self):
        assert not is_active()
        assert load_state("data_json") is None


class TestSaveStateInactive:
    def test_save_when_inactive_is_noop(self):
        """save_state silently skips if no session is active."""
        assert not is_active()
        save_state("data_json", '{"x": 1}')  # Should not raise
        # Nothing persisted
        assert load_state("data_json") is None


# ---------------------------------------------------------------------------
# Three-tier resolve
# ---------------------------------------------------------------------------


class TestResolveTier1:
    """Tier 1: STATE_REF:<key> prefix → load from disk."""

    def test_valid_ref(self, active_session):
        save_state("data_json", '[1,2,3]')
        result = resolve(f"{STATE_REF_PREFIX}data_json", "data_json")
        assert result == '[1,2,3]'

    def test_ref_different_key(self, active_session):
        """Ref key and fallback key can differ (ref key wins)."""
        save_state("describe_stats", '{"a":1}')
        save_state("data_json", '{"b":2}')
        result = resolve(f"{STATE_REF_PREFIX}describe_stats", "data_json")
        assert result == '{"a":1}'

    def test_ref_key_not_found_falls_to_tier3(self, active_session):
        """If ref key doesn't exist, resolve falls through to tier 3."""
        save_state("data_json", '{"fallback": true}')
        result = resolve(f"{STATE_REF_PREFIX}does_not_exist", "data_json")
        assert result == '{"fallback": true}'


class TestResolveTier2:
    """Tier 2: valid JSON string → pass-through."""

    def test_json_object(self, active_session):
        raw = '{"col": [1, 2, 3]}'
        result = resolve(raw, "data_json")
        assert result == raw

    def test_json_array(self, active_session):
        raw = '[1, 2, 3]'
        result = resolve(raw, "data_json")
        assert result == raw

    def test_json_string(self, active_session):
        raw = '"hello"'
        result = resolve(raw, "data_json")
        assert result == raw

    def test_json_number(self, active_session):
        raw = '42'
        result = resolve(raw, "data_json")
        assert result == raw

    def test_json_null(self, active_session):
        raw = 'null'
        result = resolve(raw, "data_json")
        assert result == raw

    def test_works_without_active_session(self):
        """Tier 2 works even without a session (test compatibility)."""
        raw = '{"col": [1, 2, 3]}'
        result = resolve(raw, "data_json")
        assert result == raw


class TestResolveTier3:
    """Tier 3: garbage input → fallback to known key on disk."""

    def test_garbage_falls_back(self, active_session):
        save_state("data_json", '{"saved": true}')
        result = resolve("LLM hallucinated this garbage!", "data_json")
        assert result == '{"saved": true}'

    def test_truncated_json_falls_back(self, active_session):
        save_state("data_json", '{"complete": true}')
        result = resolve('{"incomplete": tru', "data_json")
        assert result == '{"complete": true}'

    def test_empty_string_falls_back(self, active_session):
        save_state("data_json", '{"empty_input": true}')
        result = resolve("", "data_json")
        assert result == '{"empty_input": true}'

    def test_corrupted_prefix_falls_back(self, active_session):
        """Typo in prefix (STATE_REFF:) → not tier 1 → not valid JSON → tier 3."""
        save_state("data_json", '{"saved": true}')
        result = resolve("STATE_REFF:data_json", "data_json")
        assert result == '{"saved": true}'


class TestResolveError:
    """PipelineStateError when all tiers fail."""

    def test_no_session_garbage_raises(self):
        """No session + garbage → tier 2 fails, tier 3 fails → error."""
        with pytest.raises(PipelineStateError, match="Cannot resolve artifact"):
            resolve("garbage", "data_json")

    def test_active_session_no_artifact_raises(self, active_session):
        """Active session but no artifact saved → all tiers fail."""
        with pytest.raises(PipelineStateError, match="Cannot resolve artifact"):
            resolve("garbage", "missing_key")

    def test_error_mentions_fallback_key(self, active_session):
        with pytest.raises(PipelineStateError, match="missing_analysis"):
            resolve("bad", "missing_analysis")

    def test_error_includes_param_snippet(self, active_session):
        with pytest.raises(PipelineStateError, match="LLM param was"):
            resolve("some garbage text", "data_json")

    def test_broken_ref_no_fallback_raises(self, active_session):
        """STATE_REF to missing key + no fallback artifact → error."""
        with pytest.raises(PipelineStateError, match="Cannot resolve artifact"):
            resolve(f"{STATE_REF_PREFIX}no_such_key", "also_missing")


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    def test_two_sessions_independent(self):
        """Artifacts from session A are invisible in session B."""
        sid_a = init_session()
        save_state("data_json", '{"session": "A"}')
        clear_session()

        sid_b = init_session()
        assert sid_a != sid_b
        assert load_state("data_json") is None  # B has no data from A

    def test_artifacts_survive_within_session(self, active_session):
        """Save at the start, load at the end — same session."""
        save_state("step1", '{"done": true}')
        save_state("step2", '{"done": true}')
        assert load_state("step1") == '{"done": true}'
        assert load_state("step2") == '{"done": true}'


# ---------------------------------------------------------------------------
# Constants & error class
# ---------------------------------------------------------------------------


class TestConstants:
    def test_ref_prefix_value(self):
        assert STATE_REF_PREFIX == "STATE_REF:"

    def test_pipeline_state_error_is_runtime_error(self):
        assert issubclass(PipelineStateError, RuntimeError)

    def test_pipeline_state_error_message(self):
        err = PipelineStateError("test message")
        assert str(err) == "test message"

    def test_base_state_dir_under_outputs(self):
        assert ".pipeline_state" in str(_BASE_STATE_DIR)
        assert "outputs" in str(_BASE_STATE_DIR)


# ---------------------------------------------------------------------------
# Dual-mode integration smoke tests
# ---------------------------------------------------------------------------


class TestDualModeSmoke:
    """Verify tool-level dual-mode behavior: ref when active, JSON when not."""

    def test_ref_prefix_format(self, active_session):
        """Reference strings have expected format."""
        ref = f"{STATE_REF_PREFIX}data_json"
        assert ref == "STATE_REF:data_json"
        assert ref.startswith(STATE_REF_PREFIX)

    def test_resolve_round_trip_with_ref(self, active_session):
        """Full workflow: save → build ref → resolve → get original data."""
        original = '{"records": [1, 2, 3]}'
        key = "data_json"
        save_state(key, original)
        ref = f"{STATE_REF_PREFIX}{key}"
        resolved = resolve(ref, key)
        assert resolved == original
        assert json.loads(resolved) == json.loads(original)

    def test_large_payload_survives(self, active_session):
        """Artifact store handles payloads larger than LLM context."""
        # Simulate a 50KB DataFrame JSON
        big_data = json.dumps({"col": list(range(5000))})
        assert len(big_data) > 10_000
        save_state("data_json", big_data)
        ref = f"{STATE_REF_PREFIX}data_json"
        resolved = resolve(ref, "data_json")
        assert resolved == big_data
        assert json.loads(resolved)["col"][-1] == 4999


# ---------------------------------------------------------------------------
# Edge cases & corruption resilience
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_whitespace_only_param_falls_to_tier3(self, active_session):
        """Whitespace-only param is not valid JSON → tier 3."""
        save_state("data_json", '{"ws": true}')
        result = resolve("   \n\t  ", "data_json")
        assert result == '{"ws": true}'

    def test_partial_ref_prefix(self, active_session):
        """Partial prefix 'STATE_' is not tier 1 → falls through."""
        save_state("data_json", '{"partial": true}')
        result = resolve("STATE_data_json", "data_json")
        assert result == '{"partial": true}'

    def test_resolve_prefers_tier1_over_tier2(self, active_session):
        """If param is a valid STATE_REF AND the ref resolves, tier 1 wins."""
        save_state("mykey", '{"from_disk": true}')
        # STATE_REF:mykey is NOT valid JSON (no quotes), so tier 2 wouldn't match anyway
        result = resolve(f"{STATE_REF_PREFIX}mykey", "mykey")
        assert json.loads(result) == {"from_disk": True}

    def test_tier2_beats_tier3(self, active_session):
        """If param is valid JSON, tier 2 returns it even if tier 3 has data."""
        save_state("data_json", '{"from_disk": true}')
        passed_json = '{"from_llm": true}'
        result = resolve(passed_json, "data_json")
        assert json.loads(result) == {"from_llm": True}


# ---------------------------------------------------------------------------
# Active sessions registry
# ---------------------------------------------------------------------------


class TestActiveSessions:
    def test_empty_initially(self):
        """No sessions active at test start (autouse fixture clears state)."""
        assert get_active_sessions() == frozenset()

    def test_registered_after_init(self):
        sid = init_session()
        assert sid in get_active_sessions()

    def test_deregistered_after_clear(self):
        sid = init_session()
        assert sid in get_active_sessions()
        clear_session()
        assert sid not in get_active_sessions()

    def test_empty_after_clear(self):
        init_session()
        clear_session()
        assert get_active_sessions() == frozenset()

    def test_returns_frozenset(self):
        init_session()
        result = get_active_sessions()
        assert isinstance(result, frozenset)

    def test_snapshot_is_independent(self):
        """Modifying the returned frozenset does not affect the registry."""
        sid = init_session()
        snap = get_active_sessions()
        assert sid in snap
        # frozensets are immutable — this verifies the return type is correct
        with pytest.raises(AttributeError):
            snap.add("phantom")  # type: ignore[attr-defined]
