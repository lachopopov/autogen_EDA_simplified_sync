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

# OpenLIT defaults from config — kept here so tests can patch main.OPENLIT_ENDPOINT etc.
from config import (  # noqa: F401
    OPENLIT_ENABLE,
    OPENLIT_ENDPOINT,
    ensure_run_dirs,
    get_outputs_dir,
    get_plots_dir,
)

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

    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target",
        type=str,
        default=None,
        metavar="COLUMN",
        help="Specify the target variable column name (skip heuristic detection).",
    )
    target_group.add_argument(
        "--no-target",
        action="store_true",
        default=False,
        help="Skip target detection entirely (unsupervised mode).",
    )

    parser.add_argument(
        "--openlit",
        action="store_true",
        default=None,
        help="Enable OpenLIT observability (LLM tracing, token tracking).",
    )
    parser.add_argument(
        "--no-openlit",
        action="store_true",
        default=False,
        help="Disable OpenLIT observability even if OPENLIT_ENABLE=true.",
    )

    cat_group = parser.add_mutually_exclusive_group()
    cat_group.add_argument(
        "--categoricals",
        type=str,
        default=None,
        metavar="COL1,COL2,...",
        help="Comma-separated list of numeric columns to reclassify as categorical "
             "(skip LLM detection).",
    )
    cat_group.add_argument(
        "--no-reclassify",
        action="store_true",
        default=False,
        help="Skip encoded-categorical detection entirely.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------


def ensure_output_dirs(session_id: str) -> None:
    """Create output directories for the specific run and perform cleanup."""
    ensure_run_dirs(session_id)
    out_dir = get_outputs_dir(session_id)
    plots_dir = get_plots_dir(session_id)
    logger.info("Output dirs ready: %s, %s", out_dir, plots_dir)


# The heavy pipeline implementation and related private helpers have been moved
# into `pipeline.py` as part of the refactor.  Re-export the symbols here so
# existing imports (tests and external callers) continue to work.

from pipeline import (  # noqa: E402, F401
    _build_target_info,
    _confirm_reclassify_interactive,
    _format_cost_summary,
    _init_openlit,
    _resolve_reclassification,
    _resolve_target,
    _shutdown_openlit,
    run_pipeline,
)

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

    # Resolve OpenLIT: --openlit wins, --no-openlit disables, else env var
    if args.no_openlit:
        use_openlit = False
    elif args.openlit:
        use_openlit = True
    else:
        use_openlit = OPENLIT_ENABLE

    try:
        run_pipeline(
            args.file_path,
            target_flag=args.target,
            no_target_flag=args.no_target,
            enable_openlit=use_openlit,
            categoricals_flag=args.categoricals,
            no_reclassify_flag=args.no_reclassify,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
