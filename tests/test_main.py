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
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {})

        run_pipeline(csv_file)

        mock_build.assert_called_once()

    @patch("orchestrator.build_group_chat")
    def test_calls_initiate_chat(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {})

        run_pipeline(csv_file)

        proxy.initiate_chat.assert_called_once()

    @patch("orchestrator.build_group_chat")
    def test_initiate_chat_receives_manager(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {})

        run_pipeline(csv_file)

        args, kwargs = proxy.initiate_chat.call_args
        assert args[0] is manager

    @patch("orchestrator.build_group_chat")
    def test_message_contains_file_path(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {})

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
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {})

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
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {})

        run_pipeline(csv_file)

        assert out.is_dir()
        assert plots.is_dir()

    @patch("orchestrator.build_group_chat")
    def test_resolves_relative_path(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {})

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
        mock_build.return_value = (MagicMock(), MagicMock(), proxy, {}, {})

        # Should not raise
        main([str(csv_file)])

    @patch("orchestrator.build_group_chat")
    def test_full_flow_initiate_chat_called(self, mock_build, csv_file, monkeypatch, tmp_path):
        out = tmp_path / "outputs"
        monkeypatch.setattr("main.OUTPUTS_DIR", out)
        monkeypatch.setattr("main.PLOTS_DIR", out / "plots")
        proxy = MagicMock()
        manager = MagicMock()
        mock_build.return_value = (MagicMock(), manager, proxy, {}, {})

        main([str(csv_file)])

        proxy.initiate_chat.assert_called_once_with(
            manager,
            message=proxy.initiate_chat.call_args[1]["message"],
        )
