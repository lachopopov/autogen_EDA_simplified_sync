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

from main import ensure_output_dirs, main, parse_args, run_pipeline


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

        run_pipeline(csv_file)

        mock_build.assert_called_once()

    @patch("orchestrator.build_group_chat")
    def test_calls_initiate_chat(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {}, [])

        run_pipeline(csv_file)

        proxy.initiate_chat.assert_called_once()

    @patch("orchestrator.build_group_chat")
    def test_initiate_chat_receives_manager(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {}, [])

        run_pipeline(csv_file)

        args, kwargs = proxy.initiate_chat.call_args
        assert args[0] is manager

    @patch("orchestrator.build_group_chat")
    def test_message_contains_file_path(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file)

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

        run_pipeline(csv_file)

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

        run_pipeline(csv_file)

        assert out.is_dir()
        assert plots.is_dir()

    @patch("orchestrator.build_group_chat")
    def test_resolves_relative_path(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {}, [])

        run_pipeline(csv_file)

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
        main([str(csv_file)])

    @patch("orchestrator.build_group_chat")
    def test_full_flow_initiate_chat_called(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {}, [])

        main([str(csv_file)])

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

        run_pipeline(csv_file)

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
                "gpt-5-nano": {"cost": 0.04, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            },
            "usage_excluding_cached_inference": {
                "total_cost": 0.04,
                "gpt-5-nano": {"cost": 0.04, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            },
        }

        run_pipeline(csv_file)

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
            "gpt-5-nano": {"cost": 0.001, "prompt_tokens": 500, "completion_tokens": 100},
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

        run_pipeline(csv_file)

        content = (out / "cost_summary.txt").read_text(encoding="utf-8")
        assert "DataPrepAgent" in content
        assert "FindingsGeneratorAgent" in content
        assert "gpt-5-nano" in content
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

        run_pipeline(csv_file)

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

        run_pipeline(csv_file)

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

        run_pipeline(csv_file)

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
            "gpt-5-nano": {"cost": 0.002, "prompt_tokens": 1000, "completion_tokens": 300},
        }
        agent_no_usage = MagicMock()
        agent_no_usage.name = "SkippedAgent"
        agent_no_usage.get_total_usage.return_value = None

        mock_build.return_value = (
            MagicMock(), MagicMock(), proxy, {}, {},
            [agent_with_usage, agent_no_usage],
        )
        mock_gather.return_value = {"usage_including_cached_inference": {"total_cost": 0.002}}

        run_pipeline(csv_file)

        content = (out / "cost_summary.txt").read_text(encoding="utf-8")
        assert "EDAAnalysisAgent" in content
        assert "SkippedAgent" not in content
