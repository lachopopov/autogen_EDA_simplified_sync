"""
main.py — CLI entry point for the EDA Multi-Agent pipeline.

Usage:
    python main.py <path-to-data-file>

Supported formats: CSV, Parquet, XLSX (handled by DataPrepAgent + data_loader).

Architecture Reference: architecture.md § 3, § 10

Flow:
  1. Parse CLI argument (file path)
  2. Validate that the file exists
  3. Ensure output directories exist (outputs/, outputs/plots/)
  4. Build GroupChat via orchestrator.build_group_chat()
  5. Kick off pipeline: user_proxy.initiate_chat(manager, message=...)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import OUTPUTS_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:] when None).

    Returns
    -------
    argparse.Namespace
        Parsed args with ``file_path`` attribute (a Path).
    """
    parser = argparse.ArgumentParser(
        prog="eda-pipeline",
        description="Run EDA multi-agent pipeline on a data file.",
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to the input data file (CSV, Parquet, or XLSX).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    """Create ``outputs/`` and ``outputs/plots/`` directories if they don't exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output dirs ready: %s, %s", OUTPUTS_DIR, PLOTS_DIR)


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------


def run_pipeline(file_path: Path) -> None:
    """
    Build the GroupChat and start the EDA pipeline.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the input data file.
        Must exist and be a regular file.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If ``file_path`` is not a file (e.g., a directory).
    """
    resolved = file_path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    # Ensure output directories exist before pipeline starts
    ensure_output_dirs()

    # Initialize artifact store session (disk-backed, UUID-scoped)
    from tools._pipeline_state import init_session, clear_session
    session_id = init_session()
    logger.info("Pipeline session: %s", session_id)

    # Lazy import to avoid heavy AG2 imports on --help / parse-only usage
    from orchestrator import build_group_chat

    _groupchat, manager, user_proxy, _agents, _executors = build_group_chat()

    # Kick off the pipeline — user_proxy submits the file path as the
    # initial message, which DataPrepAgent will receive.
    initial_message = (
        f"Please run the full EDA pipeline on the following data file:\n"
        f"{resolved}"
    )

    logger.info("Starting EDA pipeline for: %s", resolved)

    try:
        user_proxy.initiate_chat(manager, message=initial_message)
    finally:
        clear_session()

    logger.info("EDA pipeline completed.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """
    Entry point: parse args, validate, and run the pipeline.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:] when None).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)

    try:
        run_pipeline(args.file_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
