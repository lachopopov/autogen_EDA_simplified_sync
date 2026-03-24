"""
tests/test_main.py — Tests for main.py (CLI entry point).

Covers:
  - CLI argument parsing (valid, missing, extra)
  - File validation (exists, not-a-file)
  - Output directory creation
  - Pipeline invocation (mocked — no LLM calls)
  - main() error handling and exit codes
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import ensure_output_dirs, main, parse_args, run_pipeline, _resolve_target, _build_target_info, _init_openlit, _shutdown_openlit, _confirm_reclassify_interactive, _resolve_reclassification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def csv_file(tmp_path):
    """Create a minimal CSV file in a temp directory."""
    f = tmp_path / "sample.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    return f


@pytest.fixture()
def non_existent_path(tmp_path):
    """Return a path that does not exist."""
    return tmp_path / "does_not_exist.csv"


@pytest.fixture()
def output_root(tmp_path):
    """Provide isolated output directories via monkeypatch."""
    return tmp_path / "outputs"


# ===================================================================
# TestParseArgs — CLI argument parsing
# ===================================================================


class TestParseArgs:
    """Tests for parse_args()."""

    def test_single_file_argument(self):
        args = parse_args(["data.csv"])
        assert args.file_path == Path("data.csv")

    def test_absolute_path(self):
        args = parse_args(["/tmp/some/data.parquet"])
        assert args.file_path == Path("/tmp/some/data.parquet")

    def test_returns_path_type(self):
        args = parse_args(["input.xlsx"])
        assert isinstance(args.file_path, Path)

    def test_missing_argument_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args([])
        assert exc_info.value.code == 2  # argparse convention

    def test_too_many_arguments_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["file1.csv", "file2.csv"])
        assert exc_info.value.code == 2

    def test_help_flag_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_path_with_spaces(self):
        args = parse_args(["path with spaces/data.csv"])
        assert args.file_path == Path("path with spaces/data.csv")

    def test_relative_path_preserved(self):
        args = parse_args(["../data/input.csv"])
        assert str(args.file_path) == "../data/input.csv"

    def test_target_flag(self):
        args = parse_args(["--target", "species", "data.csv"])
        assert args.target == "species"
        assert args.no_target is False

    def test_no_target_flag(self):
        args = parse_args(["--no-target", "data.csv"])
        assert args.no_target is True
        assert args.target is None

    def test_target_and_no_target_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--target", "col", "--no-target", "data.csv"])
        assert exc_info.value.code == 2

    def test_default_no_target_flags(self):
        args = parse_args(["data.csv"])
        assert args.target is None
        assert args.no_target is False


# ===================================================================
# TestEnsureOutputDirs — directory creation
# ===================================================================


class TestEnsureOutputDirs:
    """Tests for ensure_output_dirs()."""

    def test_creates_outputs_dir(self, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        plots = out / "plots"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", plots)

        ensure_output_dirs()

        assert out.is_dir()

    def test_creates_plots_dir(self, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        plots = out / "plots"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", plots)

        ensure_output_dirs()

        assert plots.is_dir()

    def test_idempotent(self, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        plots = out / "plots"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", plots)

        ensure_output_dirs()
        ensure_output_dirs()  # second call should not raise

        assert out.is_dir()
        assert plots.is_dir()

    def test_handles_existing_dirs(self, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        plots = out / "plots"
        plots.mkdir(parents=True)
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", plots)

        ensure_output_dirs()  # should not raise

        assert out.is_dir()
        assert plots.is_dir()

    def test_nested_parent_creation(self, monkeypatch, tmp_path):
        out = tmp_path / "deep" / "nested" / "outputs"
        plots = out / "plots"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", plots)

        ensure_output_dirs()

        assert out.is_dir()
        assert plots.is_dir()


# ===================================================================
# TestRunPipeline — file validation + orchestrator wiring
# ===================================================================


class TestRunPipelineValidation:
    """Tests for run_pipeline() input validation (no LLM calls)."""

    def test_file_not_found_raises(self, non_existent_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            run_pipeline(non_existent_path)

    def test_directory_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not a file"):
            run_pipeline(tmp_path)  # tmp_path is a directory

    def test_file_not_found_includes_path(self, non_existent_path):
        with pytest.raises(FileNotFoundError, match=str(non_existent_path)):
            run_pipeline(non_existent_path)

    def test_symlink_to_nonexistent_raises(self, tmp_path):
        link = tmp_path / "broken_link.csv"
        link.symlink_to(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            run_pipeline(link)


class TestRunPipelineExecution:
    """Tests for run_pipeline() — mocked GroupChat (no real LLM calls)."""

    @patch("orchestrator.build_group_chat")
    def test_calls_build_group_chat(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        mock_build.assert_called_once()

    @patch("orchestrator.build_group_chat")
    def test_calls_initiate_chat(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        proxy.initiate_chat.assert_called_once()

    @patch("orchestrator.build_group_chat")
    def test_initiate_chat_receives_manager(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        args, kwargs = proxy.initiate_chat.call_args
        assert args[0] is manager

    @patch("orchestrator.build_group_chat")
    def test_message_contains_file_path(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        _, kwargs = proxy.initiate_chat.call_args
        message = kwargs.get("message", "")
        assert str(csv_file.resolve()) in message

    @patch("orchestrator.build_group_chat")
    def test_message_contains_eda_instruction(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        _, kwargs = proxy.initiate_chat.call_args
        message = kwargs.get("message", "")
        assert "EDA" in message or "pipeline" in message.lower()

    @patch("orchestrator.build_group_chat")
    def test_creates_output_dirs_before_chat(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        plots = out / "plots"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", plots)
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        assert out.is_dir()
        assert plots.is_dir()

    @patch("orchestrator.build_group_chat")
    def test_resolves_relative_path(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file, no_target_flag=True)

        _, kwargs = proxy.initiate_chat.call_args
        message = kwargs.get("message", "")
        # The message should contain the resolved (absolute) path
        assert str(csv_file.resolve()) in message


# ===================================================================
# TestMain — top-level main() function
# ===================================================================


class TestMain:
    """Tests for main() entry point."""

    @patch("main.run_pipeline")
    def test_calls_run_pipeline(self, mock_run, csv_file):
        main([str(csv_file)])
        mock_run.assert_called_once()

    @patch("main.run_pipeline")
    def test_passes_parsed_path(self, mock_run, csv_file):
        main([str(csv_file)])
        called_path = mock_run.call_args[0][0]
        assert called_path == csv_file

    @patch("main.run_pipeline")
    def test_file_not_found_exits_1(self, mock_run, non_existent_path):
        mock_run.side_effect = FileNotFoundError("not found")
        with pytest.raises(SystemExit) as exc_info:
            main([str(non_existent_path)])
        assert exc_info.value.code == 1

    @patch("main.run_pipeline")
    def test_value_error_exits_1(self, mock_run, tmp_path):
        mock_run.side_effect = ValueError("not a file")
        with pytest.raises(SystemExit) as exc_info:
            main([str(tmp_path)])
        assert exc_info.value.code == 1

    @patch("main.run_pipeline")
    def test_no_args_exits_2(self, mock_run):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    @patch("main.run_pipeline")
    def test_unexpected_exception_propagates(self, mock_run, csv_file):
        mock_run.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError, match="unexpected"):
            main([str(csv_file)])


# ===================================================================
# TestIntegration — lightweight end-to-end (still mocked LLM)
# ===================================================================


class TestIntegration:
    """Lightweight integration: main.py → orchestrator wiring (mocked LLM)."""

    @patch("orchestrator.build_group_chat")
    def test_full_flow_no_exception(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        # Should not raise
        main(["--no-target", str(csv_file)])

    @patch("orchestrator.build_group_chat")
    def test_full_flow_initiate_chat_called(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {}, [])

        main(["--no-target", str(csv_file)])

        proxy.initiate_chat.assert_called_once_with(
            manager,
            message=proxy.initiate_chat.call_args[1]["message"],
        )


# ===================================================================
# TestCostTracking — cost_summary.txt output (Option C)
# ===================================================================


class TestCostTracking:
    """Tests for post-pipeline cost_summary.txt generation."""

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_cost_summary_file_created(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])
        mock_gather.return_value = {"usage_including_cached_inference": {}}

        run_pipeline(csv_file, no_target_flag=True)

        cost_file = out / "cost_summary.txt"
        assert cost_file.exists()

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_cost_summary_is_human_readable(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])
        mock_gather.return_value = {
            "usage_including_cached_inference": {
                "total_cost": 0.04,
                "gpt-5-mini": {"cost": 0.04, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            },
            "usage_excluding_cached_inference": {
                "total_cost": 0.04,
                "gpt-5-mini": {"cost": 0.04, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            },
        }

        run_pipeline(csv_file, no_target_flag=True)

        cost_file = out / "cost_summary.txt"
        content = cost_file.read_text(encoding="utf-8")
        assert "EDA Pipeline" in content
        assert "Per-Agent Breakdown" in content
        assert "Grand Totals" in content
        assert "Pipeline total:" in content
        assert "$0.04" in content

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_per_agent_breakdown_included(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        """Per-agent rows appear when agents have usage."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()

        agent_a = MagicMock()
        agent_a.name = "DataPrepAgent"
        agent_a.get_total_usage.return_value = {
            "gpt-5-mini": {"cost": 0.001, "prompt_tokens": 500, "completion_tokens": 100},
        }
        agent_b = MagicMock()
        agent_b.name = "FindingsGeneratorAgent"
        agent_b.get_total_usage.return_value = {
            "gpt-5-mini": {"cost": 0.03, "prompt_tokens": 5000, "completion_tokens": 2000},
        }

        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [agent_a, agent_b])
        mock_gather.return_value = {
            "usage_including_cached_inference": {"total_cost": 0.031},
        }

        run_pipeline(csv_file, no_target_flag=True)

        content = (out / "cost_summary.txt").read_text(encoding="utf-8")
        assert "DataPrepAgent" in content
        assert "FindingsGeneratorAgent" in content
        assert "gpt-5-mini" in content
        assert "gpt-5-mini" in content

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_gather_called_with_agents_list(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        agents_list = [MagicMock(), MagicMock()]
        # Set names and usage for the formatter
        agents_list[0].name = "AgentA"
        agents_list[0].get_total_usage.return_value = None
        agents_list[1].name = "AgentB"
        agents_list[1].get_total_usage.return_value = None
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, agents_list)
        mock_gather.return_value = {}

        run_pipeline(csv_file, no_target_flag=True)

        mock_gather.assert_called_once_with(agents_list)

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_cost_summary_written_after_pipeline(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        """Cost file created even when gather_usage_summary returns empty dict."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])
        mock_gather.return_value = {}

        run_pipeline(csv_file, no_target_flag=True)

        cost_file = out / "cost_summary.txt"
        assert cost_file.exists()
        assert cost_file.stat().st_size > 0

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_cost_summary_not_in_artifact_store(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        """Cost data is written directly to outputs/, not the artifact store."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])
        mock_gather.return_value = {"total_cost": 0.04}

        run_pipeline(csv_file, no_target_flag=True)

        # File should be directly in outputs, not in .pipeline_state
        assert (out / "cost_summary.txt").exists()
        assert not (out / ".pipeline_state").exists() or not list(
            (out / ".pipeline_state").rglob("cost_*")
        )

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_zero_usage_agents_excluded(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        """Agents with no LLM usage (get_total_usage returns None) are excluded."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()

        agent_with_usage = MagicMock()
        agent_with_usage.name = "EDAAnalysisAgent"
        agent_with_usage.get_total_usage.return_value = {
            "gpt-5-mini": {"cost": 0.002, "prompt_tokens": 1000, "completion_tokens": 300},
        }
        agent_no_usage = MagicMock()
        agent_no_usage.name = "SkippedAgent"
        agent_no_usage.get_total_usage.return_value = None

        mock_build.return_value = (
            MagicMock(), MagicMock(), proxy, {}, {},
            [agent_with_usage, agent_no_usage],
        )
        mock_gather.return_value = {"usage_including_cached_inference": {"total_cost": 0.002}}

        run_pipeline(csv_file, no_target_flag=True)

        content = (out / "cost_summary.txt").read_text(encoding="utf-8")
        assert "EDAAnalysisAgent" in content
        assert "SkippedAgent" not in content

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_cost_summary_includes_eval_cost(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path
    ):
        """Eval cost from _eval_cost_info appears in the cost summary."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()

        agent_a = MagicMock()
        agent_a.name = "DataPrepAgent"
        agent_a.get_total_usage.return_value = {
            "gpt-5-mini": {"cost": 0.001, "prompt_tokens": 500, "completion_tokens": 100},
        }
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [agent_a])
        mock_gather.return_value = {
            "usage_including_cached_inference": {"total_cost": 0.001},
        }

        # Populate eval cost info before running
        import tools.findings_tools as ft
        ft._eval_cost_info.update({
            "model": "gpt-5-2025-08-07",
            "prompt_tokens": 3000,
            "completion_tokens": 200,
            "cost": 0.0058,
        })

        try:
            run_pipeline(csv_file, no_target_flag=True)
        finally:
            ft._eval_cost_info.clear()

        content = (out / "cost_summary.txt").read_text(encoding="utf-8")
        assert "HallucinationEval" in content
        assert "gpt-5-2025-08-07" in content
        assert "$0.0058" in content
        # Grand total should include eval cost
        assert "Pipeline total:" in content

    def test_format_cost_summary_with_eval_cost(self):
        """_format_cost_summary includes eval_cost in breakdown and totals."""
        from main import _format_cost_summary

        agent_a = MagicMock()
        agent_a.name = "TestAgent"
        agent_a.get_total_usage.return_value = {
            "gpt-5-mini": {"cost": 0.01, "prompt_tokens": 1000, "completion_tokens": 200},
        }
        usage_dict = {
            "usage_including_cached_inference": {
                "total_cost": 0.01,
                "gpt-5-mini": {"cost": 0.01, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            },
        }
        eval_cost = {
            "model": "gpt-5-2025-08-07",
            "prompt_tokens": 2000,
            "completion_tokens": 100,
            "cost": 0.005,
        }

        text = _format_cost_summary([agent_a], usage_dict, eval_cost=eval_cost)
        assert "HallucinationEval" in text
        assert "gpt-5-2025-08-07" in text
        # Grand total = agent 0.01 + eval 0.005 = 0.015
        assert "$0.0150" in text

    def test_format_cost_summary_without_eval_cost(self):
        """_format_cost_summary works without eval_cost (backward compatible)."""
        from main import _format_cost_summary

        agent_a = MagicMock()
        agent_a.name = "TestAgent"
        agent_a.get_total_usage.return_value = None
        usage_dict = {
            "usage_including_cached_inference": {"total_cost": 0.02},
        }

        text = _format_cost_summary([agent_a], usage_dict)
        assert "HallucinationEval" not in text
        assert "$0.0200" in text


# ===================================================================
# TestResolveTarget — target detection helpers
# ===================================================================


class TestResolveTarget:
    """Tests for _resolve_target(), _build_target_info()."""

    @pytest.fixture()
    def sample_df(self):
        import pandas as pd
        return pd.DataFrame({
            "feat_a": [1, 2, 3, 4, 5],
            "feat_b": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        })

    def test_no_target_flag_returns_unsupervised(self, sample_df):
        result = _resolve_target(sample_df, target_flag=None, no_target_flag=True)
        assert result.column is None
        assert result.problem_type == "unsupervised"
        assert result.detection_method == "none"

    def test_target_flag_sets_column(self, sample_df):
        result = _resolve_target(sample_df, target_flag="target", no_target_flag=False)
        assert result.column == "target"
        assert result.detection_method == "user_specified"

    def test_target_flag_invalid_column_exits(self, sample_df):
        with pytest.raises(SystemExit):
            _resolve_target(sample_df, target_flag="nonexistent", no_target_flag=False)

    def test_non_tty_auto_accepts_heuristic(self, sample_df, monkeypatch):
        """Non-TTY mode auto-accepts the heuristic candidate."""
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
        result = _resolve_target(sample_df, target_flag=None, no_target_flag=False)
        # Heuristic should detect 'target' column (low-cardinality)
        assert result.column is not None
        assert result.problem_type in ("classification", "regression")

    def test_build_target_info_classification(self, sample_df):
        info = _build_target_info(sample_df, "target")
        assert info.column == "target"
        assert info.detection_method == "user_specified"
        assert info.problem_type == "classification"

    def test_build_target_info_regression(self):
        import pandas as pd
        df = pd.DataFrame({
            "feat": list(range(50)),
            "price": [float(x) for x in range(50)],
        })
        info = _build_target_info(df, "price")
        assert info.column == "price"
        assert info.problem_type == "regression"


class TestRunPipelineWithTarget:
    """Tests for run_pipeline() with target flags (mocked GroupChat)."""

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_message_includes_target_context(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path,
    ):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        agent = MagicMock()
        agent.name = "A"
        agent.get_total_usage.return_value = None
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [agent])
        mock_gather.return_value = {"usage_including_cached_inference": {"total_cost": 0}}

        run_pipeline(csv_file, no_target_flag=True)

        _, kwargs = proxy.initiate_chat.call_args
        message = kwargs.get("message", "")
        assert "unsupervised" in message.lower() or "No target" in message

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_target_flag_detected_in_message(
        self, mock_build, mock_gather, monkeypatch, tmp_path,
    ):
        # Create CSV with a 'target' column
        csv = tmp_path / "data.csv"
        csv.write_text("feat,target\n1,0\n2,1\n3,0\n")
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        agent = MagicMock()
        agent.name = "A"
        agent.get_total_usage.return_value = None
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [agent])
        mock_gather.return_value = {"usage_including_cached_inference": {"total_cost": 0}}

        run_pipeline(csv, target_flag="target")

        _, kwargs = proxy.initiate_chat.call_args
        message = kwargs.get("message", "")
        assert "target" in message.lower()


# ===================================================================
# TestParseArgsOpenlit — CLI --openlit / --no-openlit flags
# ===================================================================


class TestParseArgsOpenlit:
    """Tests for --openlit / --no-openlit CLI flags."""

    def test_openlit_flag(self):
        args = parse_args(["--openlit", "data.csv"])
        assert args.openlit is True
        assert args.no_openlit is False

    def test_no_openlit_flag(self):
        args = parse_args(["--no-openlit", "data.csv"])
        assert args.no_openlit is True

    def test_default_openlit_is_none(self):
        args = parse_args(["data.csv"])
        assert args.openlit is None
        assert args.no_openlit is False


# ===================================================================
# TestInitOpenlit — _init_openlit() helper
# ===================================================================


class TestInitOpenlit:
    """Tests for _init_openlit() — OpenLIT observability initialisation."""

    @patch("main.openlit", create=True)
    def test_init_calls_openlit_init(self, mock_module, monkeypatch):
        """openlit.init() is called when the package is available."""
        monkeypatch.setattr("main.OPENLIT_ENDPOINT", None)
        # Mock the import inside _init_openlit
        with patch.dict("sys.modules", {"openlit": mock_module}):
            _init_openlit()
        mock_module.init.assert_called_once()
        call_kwargs = mock_module.init.call_args[1]
        assert call_kwargs["disabled_instrumentors"] == ["agno"]

    @patch("main.openlit", create=True)
    def test_init_passes_endpoint(self, mock_module, monkeypatch):
        """OPENLIT_ENDPOINT is forwarded as otlp_endpoint kwarg."""
        monkeypatch.setattr("main.OPENLIT_ENDPOINT", "http://localhost:4318")
        with patch.dict("sys.modules", {"openlit": mock_module}):
            _init_openlit()
        mock_module.init.assert_called_once()
        call_kwargs = mock_module.init.call_args[1]
        assert call_kwargs["otlp_endpoint"] == "http://localhost:4318"
        assert call_kwargs["disabled_instrumentors"] == ["agno"]

    def test_init_graceful_when_not_installed(self, monkeypatch):
        """No crash when openlit is not installed — logs a warning instead."""
        monkeypatch.setattr("main.OPENLIT_ENDPOINT", None)
        with patch.dict("sys.modules", {"openlit": None}):
            # importlib raises ImportError when module is None in sys.modules
            _init_openlit()  # should not raise

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_openlit_enabled_calls_init(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path,
    ):
        """run_pipeline(enable_openlit=True) calls _init_openlit."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        agent = MagicMock()
        agent.name = "A"
        agent.get_total_usage.return_value = None
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [agent])
        mock_gather.return_value = {"usage_including_cached_inference": {"total_cost": 0}}

        with patch("main._init_openlit") as mock_init, \
             patch("main._shutdown_openlit") as mock_shutdown:
            run_pipeline(csv_file, no_target_flag=True, enable_openlit=True)
            mock_init.assert_called_once()
            mock_shutdown.assert_called_once()

    @patch("autogen.gather_usage_summary")
    @patch("orchestrator.build_group_chat")
    def test_openlit_disabled_skips_init(
        self, mock_build, mock_gather, csv_file, monkeypatch, tmp_path,
    ):
        """run_pipeline(enable_openlit=False) does NOT call _init_openlit."""
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        agent = MagicMock()
        agent.name = "A"
        agent.get_total_usage.return_value = None
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [agent])
        mock_gather.return_value = {"usage_including_cached_inference": {"total_cost": 0}}

        with patch("main._init_openlit") as mock_init, \
             patch("main._shutdown_openlit") as mock_shutdown:
            run_pipeline(csv_file, no_target_flag=True, enable_openlit=False)
            mock_init.assert_not_called()
            mock_shutdown.assert_not_called()

    @patch("main.run_pipeline")
    def test_main_openlit_flag_enables(self, mock_run, csv_file):
        """main(["--openlit", ...]) passes enable_openlit=True."""
        main(["--openlit", "--no-target", str(csv_file)])
        _, kwargs = mock_run.call_args
        assert kwargs["enable_openlit"] is True

    @patch("main.run_pipeline")
    def test_main_no_openlit_flag_disables(self, mock_run, csv_file):
        """main(["--no-openlit", ...]) passes enable_openlit=False."""
        main(["--no-openlit", "--no-target", str(csv_file)])
        _, kwargs = mock_run.call_args
        assert kwargs["enable_openlit"] is False

    @patch("main.run_pipeline")
    def test_main_env_var_fallback(self, mock_run, csv_file, monkeypatch):
        """When no CLI flag, OPENLIT_ENABLE env var is used."""
        monkeypatch.setattr("main.OPENLIT_ENABLE", True)
        main(["--no-target", str(csv_file)])
        _, kwargs = mock_run.call_args
        assert kwargs["enable_openlit"] is True

    @patch("main.run_pipeline")
    def test_no_openlit_overrides_env(self, mock_run, csv_file, monkeypatch):
        """--no-openlit overrides OPENLIT_ENABLE=true."""
        monkeypatch.setattr("main.OPENLIT_ENABLE", True)
        main(["--no-openlit", "--no-target", str(csv_file)])
        _, kwargs = mock_run.call_args
        assert kwargs["enable_openlit"] is False


# ===================================================================
# TestShutdownOpenlit — _shutdown_openlit() helper
# ===================================================================


class TestShutdownOpenlit:
    """Tests for _shutdown_openlit() — OTel tracer + meter flush/shutdown."""

    def test_flush_and_shutdown_called(self):
        """force_flush and shutdown are called on the tracer provider."""
        mock_provider = MagicMock()
        mock_provider.force_flush.return_value = True
        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            _shutdown_openlit()
        mock_provider.force_flush.assert_called_once_with(timeout_millis=10_000)
        mock_provider.shutdown.assert_called_once()

    def test_meter_provider_flushed(self):
        """force_flush and shutdown are called on the meter provider."""
        mock_trace_provider = MagicMock()
        mock_meter_provider = MagicMock()
        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_trace_provider), \
             patch("opentelemetry.metrics.get_meter_provider", return_value=mock_meter_provider):
            _shutdown_openlit()
        mock_meter_provider.force_flush.assert_called_once_with(timeout_millis=10_000)
        mock_meter_provider.shutdown.assert_called_once()

    def test_no_crash_when_otel_not_available(self):
        """Graceful no-op when opentelemetry is not importable."""
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None, "opentelemetry.metrics": None}):
            _shutdown_openlit()  # should not raise

    def test_no_crash_when_provider_has_no_flush(self):
        """Handles providers that lack force_flush/shutdown methods."""
        mock_provider = MagicMock(spec=[])  # no attributes at all
        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider):
            _shutdown_openlit()  # should not raise

    def test_meter_flush_independent_of_trace_failure(self):
        """Meter flush runs even if trace flush raises."""
        mock_trace_provider = MagicMock()
        mock_trace_provider.force_flush.side_effect = RuntimeError("trace error")
        mock_meter_provider = MagicMock()
        with patch("opentelemetry.trace.get_tracer_provider", return_value=mock_trace_provider), \
             patch("opentelemetry.metrics.get_meter_provider", return_value=mock_meter_provider):
            _shutdown_openlit()  # should not raise
        # Meter provider should still be flushed even though trace provider failed
        mock_meter_provider.force_flush.assert_called_once_with(timeout_millis=10_000)


# ===================================================================
# TestParseArgs — Encoded categorical CLI flags
# ===================================================================


class TestParseArgsCategoricals:
    """Tests for --categoricals and --no-reclassify CLI flags."""

    def test_categoricals_flag(self):
        args = parse_args(["--categoricals", "SEX,EDUCATION", "data.csv"])
        assert args.categoricals == "SEX,EDUCATION"
        assert args.no_reclassify is False

    def test_no_reclassify_flag(self):
        args = parse_args(["--no-reclassify", "data.csv"])
        assert args.no_reclassify is True
        assert args.categoricals is None

    def test_categoricals_and_no_reclassify_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--categoricals", "SEX", "--no-reclassify", "data.csv"])
        assert exc_info.value.code == 2

    def test_default_no_categorical_flags(self):
        args = parse_args(["data.csv"])
        assert args.categoricals is None
        assert args.no_reclassify is False


# ===================================================================
# TestConfirmReclassifyInteractive
# ===================================================================


class TestConfirmReclassifyInteractive:
    """Tests for the interactive per-column reclassification prompt."""

    @pytest.fixture()
    def suspects(self):
        from eda_state import EncodedCategoricalSuspect
        return [
            EncodedCategoricalSuspect(
                column="SEX", nunique=2, sample_values=[1, 2],
                min_val=1.0, max_val=2.0, is_all_integer=True,
                reason="Binary gender code", subtype="nominal",
            ),
            EncodedCategoricalSuspect(
                column="EDUCATION", nunique=4, sample_values=[1, 2, 3, 4],
                min_val=1.0, max_val=4.0, is_all_integer=True,
                reason="Education level codes 1-4", subtype="ordinal",
            ),
        ]

    def test_accept_all(self, suspects):
        # Empty string = accept
        with patch("builtins.input", side_effect=["", ""]):
            result = _confirm_reclassify_interactive(suspects)
        assert result == ["SEX", "EDUCATION"]

    def test_reject_all(self, suspects):
        with patch("builtins.input", side_effect=["n", "n"]):
            result = _confirm_reclassify_interactive(suspects)
        assert result == []

    def test_partial_accept(self, suspects):
        with patch("builtins.input", side_effect=["", "n"]):
            result = _confirm_reclassify_interactive(suspects)
        assert result == ["SEX"]

    def test_empty_suspects(self):
        result = _confirm_reclassify_interactive([])
        assert result == []


# ===================================================================
# TestResolveReclassification
# ===================================================================


class TestResolveReclassification:
    """Tests for _resolve_reclassification() — various code paths."""

    @pytest.fixture()
    def df(self):
        import pandas as pd
        return pd.DataFrame({
            "SEX": [1, 2, 1, 2],
            "EDUCATION": [1, 2, 3, 4],
            "AGE": [25, 30, 35, 40],
        })

    def test_no_reclassify_flag(self, df):
        cols, subtypes = _resolve_reclassification(
            df, target_column=None,
            categoricals_flag=None, no_reclassify_flag=True,
        )
        assert cols == []
        assert subtypes == {}

    def test_explicit_categoricals_flag(self, df):
        cols, subtypes = _resolve_reclassification(
            df, target_column=None,
            categoricals_flag="SEX,EDUCATION", no_reclassify_flag=False,
        )
        assert set(cols) == {"SEX", "EDUCATION"}
        assert subtypes == {}  # --categoricals has no LLM subtypes

    def test_explicit_categoricals_invalid_columns_filtered(self, df):
        cols, subtypes = _resolve_reclassification(
            df, target_column=None,
            categoricals_flag="SEX,NONEXISTENT", no_reclassify_flag=False,
        )
        assert cols == ["SEX"]
        assert subtypes == {}

    def test_llm_detection_non_tty_auto_accept(self, df):
        """Non-TTY mode auto-accepts all LLM suspects."""
        from eda_state import EncodedCategoricalSuspect
        suspects = [
            EncodedCategoricalSuspect(column="SEX", nunique=2, reason="code"),
        ]
        with patch("tools.data_loader.detect_encoded_categoricals", return_value=suspects), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            cols, subtypes = _resolve_reclassification(
                df, target_column=None,
                categoricals_flag=None, no_reclassify_flag=False,
            )
        assert cols == ["SEX"]
        assert subtypes == {"SEX": "nominal"}

    def test_llm_detection_no_suspects(self, df):
        with patch("tools.data_loader.detect_encoded_categoricals", return_value=[]):
            cols, subtypes = _resolve_reclassification(
                df, target_column=None,
                categoricals_flag=None, no_reclassify_flag=False,
            )
        assert cols == []
        assert subtypes == {}
