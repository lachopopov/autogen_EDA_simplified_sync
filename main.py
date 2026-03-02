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
from datetime import datetime, timezone
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
# Cost Summary Formatter
# ---------------------------------------------------------------------------


def _format_cost_summary(agents_list: list, usage_dict: dict) -> str:
    """
    Build a human-readable cost summary with per-agent breakdown.

    Uses ``agent.get_total_usage()`` for per-agent rows and the aggregate
    ``usage_dict`` (from ``gather_usage_summary``) for grand totals.

    Only agents with usage > 0 are included.
    """
    lines: list[str] = []
    lines.append("EDA Pipeline — Cost Summary")
    lines.append("=" * 40)
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # --- Per-Agent Breakdown ---
    lines.append("Per-Agent Breakdown")
    lines.append("-" * 40)

    max_name = max((len(a.name) for a in agents_list), default=20)
    col_fmt = f"{{:<{max_name}}}  {{:<28}}  ${{:<10.4f}}  {{:>7,}} prompt / {{:>7,}} completion"

    for agent in agents_list:
        usage = agent.get_total_usage()
        if usage is None:
            continue
        for model, stats in usage.items():
            if model == "total_cost":
                continue
            cost = stats.get("cost", 0.0)
            prompt = stats.get("prompt_tokens", 0)
            completion = stats.get("completion_tokens", 0)
            if cost == 0 and prompt == 0 and completion == 0:
                continue
            lines.append(col_fmt.format(
                agent.name, model, cost, prompt, completion,
            ))

    lines.append("")

    # --- Grand Totals ---
    totals = usage_dict.get("usage_including_cached_inference", {})
    lines.append("Grand Totals")
    lines.append("-" * 40)

    grand_total = 0.0
    for model, stats in totals.items():
        if model == "total_cost":
            grand_total = stats if isinstance(stats, (int, float)) else 0.0
            continue
        cost = stats.get("cost", 0.0)
        prompt = stats.get("prompt_tokens", 0)
        completion = stats.get("completion_tokens", 0)
        total_tok = stats.get("total_tokens", prompt + completion)
        lines.append(
            f"  {model:<28}  ${cost:<10.4f}  {prompt:>7,} prompt / "
            f"{completion:>7,} completion ({total_tok:>7,} total)"
        )

    lines.append(f"  {'':28}  ----------")
    lines.append(f"  {'Pipeline total:':<28}  ${grand_total:<10.4f}")
    lines.append("")

    return "\n".join(lines) + "\n"


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

    _groupchat, manager, user_proxy, _agents, _executors, agents_list = build_group_chat()

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

    # --- Cost tracking (Option C: standalone file, post-pipeline) ---
    # gather_usage_summary is the canonical AG2 cost API (grand totals).
    # agent.get_total_usage() gives per-agent breakdowns.
    # Imported lazily to preserve fast --help / parse-only startup.
    from autogen import gather_usage_summary  # noqa: E402

    usage_dict = gather_usage_summary(agents_list)
    logger.info(
        "Cost tracking: gathered usage summary from %d agents",
        len(agents_list),
    )

    cost_text = _format_cost_summary(agents_list, usage_dict)
    cost_path = OUTPUTS_DIR / "cost_summary.txt"
    cost_path.write_text(cost_text, encoding="utf-8")
    logger.info("Cost summary written to %s", cost_path)

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
