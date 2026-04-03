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
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import get_outputs_dir, get_plots_dir, ensure_run_dirs
from eda_state import TargetInfo

# OpenLIT defaults from config (can be overridden by CLI --openlit flag)
from config import OPENLIT_ENABLE, OPENLIT_ENDPOINT

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


# ---------------------------------------------------------------------------
# Cost Summary Formatter
# ---------------------------------------------------------------------------


def _format_cost_summary(
    agents_list: list,
    usage_dict: dict,
    eval_cost: dict | None = None,
) -> str:
    """
    Build a human-readable cost summary with per-agent breakdown.

    Uses ``agent.get_total_usage()`` for per-agent rows and the aggregate
    ``usage_dict`` (from ``gather_usage_summary``) for grand totals.

    Parameters
    ----------
    eval_cost : dict | None
        Optional dict with keys {model, prompt_tokens, completion_tokens, cost}
        for the hallucination evaluator LLM call (tracked separately from
        AG2 agents).

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

    # --- Evaluator Cost (hallucination judge, outside AG2 agent framework) ---
    if eval_cost and eval_cost.get("cost", 0) > 0:
        lines.append(col_fmt.format(
            "HallucinationEval",
            eval_cost.get("model", "unknown"),
            eval_cost["cost"],
            eval_cost.get("prompt_tokens", 0),
            eval_cost.get("completion_tokens", 0),
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

    # Add eval cost to grand total
    eval_total = eval_cost.get("cost", 0.0) if eval_cost else 0.0
    grand_total += eval_total
    if eval_cost and eval_total > 0:
        em = eval_cost.get("model", "unknown")
        ep = eval_cost.get("prompt_tokens", 0)
        ec = eval_cost.get("completion_tokens", 0)
        lines.append(
            f"  {em + ' (eval)':<28}  ${eval_total:<10.4f}  {ep:>7,} prompt / "
            f"{ec:>7,} completion ({ep + ec:>7,} total)"
        )

    lines.append(f"  {'':28}  ----------")
    lines.append(f"  {'Pipeline total:':<28}  ${grand_total:<10.4f}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Target Detection Helpers
# ---------------------------------------------------------------------------


def _confirm_target_interactive(candidate: TargetInfo, df) -> TargetInfo:
    """
    Display the heuristic detection result and ask the user to confirm,
    decline, or override.

    Interactive prompt (TTY only):
      [Enter] Accept  |  [n] No target  |  Type column name to override
    """

    print("\n\u2501\u2501\u2501 Target Variable Detection \u2501\u2501\u2501")
    if candidate.column:
        print(f"Detected: '{candidate.column}' ({candidate.problem_type}")
        if candidate.problem_type == "classification":
            print(f"  Classes ({candidate.n_classes}): ", end="")
            parts = [f"{k} ({v})" for k, v in candidate.class_counts.items()]
            print(", ".join(parts))
            print(f"  Imbalance ratio: {candidate.imbalance_ratio:.1f}")
        elif candidate.problem_type == "regression":
            print(f"  Continuous target (>{df[candidate.column].nunique()} unique values)")
        method_label = (
            "keyword match" if candidate.detection_method == "name_heuristic"
            else "last low-cardinality column (fallback)"
        )
        print(f"  Method: {candidate.detection_method} ({method_label})")
    else:
        print("No target candidate detected.")

    if candidate.has_datetime_index:
        print("  Note: datetime column detected — possible time series data.")

    print()
    prompt = "  [Enter] Accept  |  [n] No target  |  Type column name to override: "
    response = input(prompt).strip()

    if response == "":
        # Accept the candidate as-is
        if candidate.column is None:
            candidate.detection_method = "none"
        return candidate

    if response.lower() == "n":
        return TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
            has_datetime_index=candidate.has_datetime_index,
        )

    # User typed a column name
    if response not in df.columns:
        print(f"  Column '{response}' not found. Available: {list(df.columns)}")
        print("  Falling back to unsupervised mode.")
        return TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
            has_datetime_index=candidate.has_datetime_index,
        )

    from tools.data_loader import _classify_target
    info = _classify_target(df, response)
    info.detection_method = "user_specified"
    print(f"  Using '{response}' as target ({info.problem_type}).")
    return info


def _build_target_info(
    df, column: str, *, has_datetime: bool = False,
) -> TargetInfo:
    """Build TargetInfo for a user-specified column via --target flag."""
    from tools.data_loader import _classify_target
    info = _classify_target(df, column)
    info.detection_method = "user_specified"
    info.has_datetime_index = has_datetime
    return info


def _resolve_target(
    df, *, target_flag: str | None, no_target_flag: bool,
) -> TargetInfo:
    """
    Resolve the target variable using CLI flags or interactive prompt.

    Raises SystemExit in non-TTY mode when neither --target nor --no-target
    is provided.
    """
    from tools.data_loader import detect_target, _has_datetime_column

    if no_target_flag:
        has_dt = _has_datetime_column(df)
        return TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
            has_datetime_index=has_dt,
        )

    if target_flag:
        if target_flag not in df.columns:
            logger.error(
                "--target column '%s' not found. Available: %s",
                target_flag, list(df.columns),
            )
            sys.exit(1)
        return _build_target_info(df, target_flag)

    # Heuristic detection + interactive confirmation
    data_json = df.to_json(orient="records")
    candidate_json = detect_target(data_json)
    candidate = TargetInfo.model_validate_json(candidate_json)

    if not sys.stdin.isatty():
        if candidate.column:
            logger.info(
                "Non-interactive mode: auto-accepting heuristic target '%s' "
                "(override with --target COLUMN or --no-target).",
                candidate.column,
            )
            return candidate
        # No candidate found → unsupervised
        logger.info(
            "Non-interactive mode: no target candidate detected, "
            "running unsupervised (override with --target COLUMN).",
        )
        has_dt = _has_datetime_column(df)
        return TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
            has_datetime_index=has_dt,
        )

    return _confirm_target_interactive(candidate, df)


# ---------------------------------------------------------------------------
# Encoded-Categorical Detection Helpers
# ---------------------------------------------------------------------------


def _confirm_reclassify_interactive(
    suspects: list,
) -> list[str]:
    """Display LLM-suggested encoded categoricals and ask user to confirm each.

    Interactive prompt (TTY only) — mirrors _confirm_target_interactive pattern.

    Args:
        suspects: List of EncodedCategoricalSuspect from detect_encoded_categoricals().

    Returns:
        List of confirmed column names to reclassify.
    """
    print("\n\u2501\u2501\u2501 Encoded Categorical Detection \u2501\u2501\u2501")
    print("The following numeric columns may be encoded categoricals:\n")

    accepted: list[str] = []
    for i, s in enumerate(suspects, 1):
        vals_str = str(s.sample_values[:10])
        if len(s.sample_values) > 10:
            vals_str = vals_str[:-1] + ", ...]"
        print(f"  {i}. {s.column}  (nunique={s.nunique}, values: {vals_str})")
        print(f"     {s.reason}  [{s.subtype}]")
        response = input("     [Enter] Accept  |  [n] Reject: ").strip().lower()
        if response in ("", "y", "yes"):
            accepted.append(s.column)
            print("     \u2713 Accepted")
        else:
            print("     \u2717 Rejected")
        print()

    if accepted:
        print(f"Reclassified as categorical: {', '.join(accepted)}")
    else:
        print("No columns reclassified.")
    return accepted


def _resolve_reclassification(
    df,
    *,
    target_column: str | None,
    categoricals_flag: str | None,
    no_reclassify_flag: bool,
) -> tuple[list[str], dict[str, str]]:
    """Resolve encoded-categorical reclassification using CLI flags or interactive prompt.

    Mirrors _resolve_target() in structure.

    Args:
        df: The loaded DataFrame.
        target_column: Target column name (excluded from candidates).
        categoricals_flag: Explicit comma-separated column list from --categoricals.
        no_reclassify_flag: If True, skip detection entirely.

    Returns:
        Tuple of (confirmed column names, subtype mapping {col: "nominal"|"ordinal"}).
        Subtypes may be empty when using --categoricals (no LLM detection).
    """
    if no_reclassify_flag:
        logger.info("Encoded-categorical detection skipped (--no-reclassify)")
        return [], {}

    if categoricals_flag:
        cols = [c.strip() for c in categoricals_flag.split(",") if c.strip()]
        valid = [c for c in cols if c in df.columns]
        invalid = [c for c in cols if c not in df.columns]
        if invalid:
            logger.warning(
                "--categoricals: columns not found (ignored): %s", invalid,
            )
        logger.info("Explicit reclassification via --categoricals: %s", valid)
        return valid, {}

    # LLM detection + interactive confirmation
    from tools.data_loader import detect_encoded_categoricals

    suspects = detect_encoded_categoricals(df, target_column=target_column)
    if not suspects:
        logger.info("No encoded-categorical suspects detected")
        return [], {}

    # Build subtype mapping from LLM detection results
    _subtypes = {s.column: s.subtype or "nominal" for s in suspects}

    if not sys.stdin.isatty():
        accepted = [s.column for s in suspects]
        logger.info(
            "Non-interactive mode: auto-accepting %d encoded-categorical suspects "
            "(override with --categoricals COL1,COL2 or --no-reclassify): %s",
            len(accepted), accepted,
        )
        return accepted, {c: _subtypes[c] for c in accepted}

    confirmed = _confirm_reclassify_interactive(suspects)
    return confirmed, {c: _subtypes[c] for c in confirmed if c in _subtypes}


# ---------------------------------------------------------------------------
# OpenLIT Initialisation
# ---------------------------------------------------------------------------


def _init_openlit() -> None:
    """Initialise OpenLIT observability tracing.

    Follows the AG2 recommended pattern:
      https://docs.ag2.ai/latest/docs/use-cases/notebooks/notebooks/agentchat_openlit/

    When ``OPENLIT_ENDPOINT`` is set, traces are sent to that OTLP endpoint.
    Otherwise traces are printed to the console (useful during development).
    """
    try:
        import openlit  # noqa: F811
    except ImportError:
        logger.warning(
            "OpenLIT requested (--openlit) but 'openlit' package is not installed. "
            "Install with: pip install openlit"
        )
        return

    kwargs: dict = {}
    if OPENLIT_ENDPOINT:
        kwargs["otlp_endpoint"] = OPENLIT_ENDPOINT
        logger.info("OpenLIT: sending traces to %s", OPENLIT_ENDPOINT)
    else:
        logger.info("OpenLIT: tracing to console (no OPENLIT_ENDPOINT set)")

    # Disable the agno instrumentor — openlit 1.36.x has a SyntaxError in
    # async_agno.py (return-with-value inside async generator). See Lesson #29.
    kwargs["disabled_instrumentors"] = ["agno"]

    # Custom pricing for gpt-5-nano/gpt-5-mini (not in openlit's default pricing.json)
    pricing_path = Path(__file__).resolve().parent / "openlit_pricing.json"
    if pricing_path.exists():
        kwargs["pricing_json"] = str(pricing_path)
        logger.info("OpenLIT: using custom pricing from %s", pricing_path)

    openlit.init(**kwargs)
    logger.info("OpenLIT initialised — auto-tracking LLM calls, tokens, costs.")


def _shutdown_openlit() -> None:
    """Flush and shut down the OpenTelemetry tracer and meter providers.

    The openlit SDK configures a ``BatchSpanProcessor`` that flushes on a
    timer (default 5 s).  If the Python process exits before the next flush
    cycle, pending spans are silently lost.  Calling ``force_flush`` followed
    by ``shutdown`` guarantees every span is exported to the OTLP collector
    before the process terminates.

    The ``MeterProvider`` is also flushed so that evaluation counters
    (e.g. hallucination eval via ``collect_metrics=True``) are exported
    before exit — the default ``PeriodicExportingMetricReader`` interval
    is 60 s, far longer than a typical pipeline run.
    """
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=10_000)
            logger.info("OpenLIT: tracer provider flushed.")
        if hasattr(provider, "shutdown"):
            provider.shutdown()
            logger.info("OpenLIT: tracer provider shut down.")
    except Exception:  # noqa: BLE001
        logger.debug("OpenLIT shutdown: no-op (tracing not initialised).")

    try:
        from opentelemetry import metrics

        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "force_flush"):
            meter_provider.force_flush(timeout_millis=10_000)
            logger.info("OpenLIT: meter provider flushed.")
        if hasattr(meter_provider, "shutdown"):
            meter_provider.shutdown()
            logger.info("OpenLIT: meter provider shut down.")
    except Exception:  # noqa: BLE001
        logger.debug("OpenLIT shutdown: meter flush no-op.")


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------


def run_pipeline(
    file_path: Path,
    *,
    target_flag: str | None = None,
    no_target_flag: bool = False,
    enable_openlit: bool = False,
    categoricals_flag: str | None = None,
    no_reclassify_flag: bool = False,
    subtypes_flag: dict[str, str] | None = None,
) -> str:
    """
    Build the GroupChat and start the EDA pipeline.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the input data file.
        Must exist and be a regular file.
    target_flag : str | None
        Explicit target column name (from --target CLI arg).
    no_target_flag : bool
        If True, skip target detection (unsupervised mode).
    enable_openlit : bool
        If True, initialise OpenLIT observability before the pipeline runs.
    categoricals_flag : str | None
        Comma-separated column names to reclassify as categorical (from --categoricals).
    no_reclassify_flag : bool
        If True, skip encoded-categorical detection entirely.
    subtypes_flag : dict[str, str] | None
        Mapping of column names to "ordinal" or "nominal".  When provided
        (e.g. from Streamlit UI), overrides subtypes returned by
        ``_resolve_reclassification()``.

    Returns
    -------
    str
        The session ID for this pipeline run.  Callers (e.g. Streamlit)
        can use this to locate output artifacts via
        ``config.get_outputs_dir(session_id)``.

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

    # --- OpenLIT observability (must run BEFORE any AG2 / OpenAI call) ---
    if enable_openlit:
        # Bridge CLI flag to config so downstream code (e.g.
        # _run_comprehensive_eval) sees the authoritative value.
        import config as _cfg
        _cfg.OPENLIT_ENABLE = True
        _init_openlit()

    # --- Pre-pipeline: load data + detect target ---
    from tools.data_loader import _get_loader

    df = _get_loader(str(resolved)).load(str(resolved))
    df = df.drop_duplicates().reset_index(drop=True)

    target_info = _resolve_target(
        df, target_flag=target_flag, no_target_flag=no_target_flag,
    )
    logger.info(
        "Target resolution: column=%s, type=%s, method=%s",
        target_info.column, target_info.problem_type, target_info.detection_method,
    )

    # --- Pre-pipeline: detect encoded categoricals ---
    reclassified_cols, reclassified_subtypes = _resolve_reclassification(
        df,
        target_column=target_info.column,
        categoricals_flag=categoricals_flag,
        no_reclassify_flag=no_reclassify_flag,
    )

    # Override subtypes with Streamlit-confirmed ordinal/nominal choices
    if subtypes_flag and reclassified_cols:
        reclassified_subtypes = {
            col: subtypes_flag.get(col, reclassified_subtypes.get(col, "nominal"))
            for col in reclassified_cols
        }

    logger.info("Reclassified as categorical: %s", reclassified_cols or "(none)")

    # Initialize artifact store session (disk-backed, UUID-scoped)
    from tools._pipeline_state import init_session, clear_session, save_state
    session_id = init_session()
    ensure_output_dirs(session_id)
    logger.info("Pipeline session: %s", session_id)

    # Store target_info in artifact store for downstream tools
    save_state("target_info", target_info.model_dump_json())

    # Store reclassified categoricals for infer_dtypes() consumption
    if reclassified_cols:
        save_state("reclassified_categoricals", json.dumps(reclassified_cols))
    if reclassified_subtypes:
        save_state("reclassified_subtypes", json.dumps(reclassified_subtypes))

    # Lazy import to avoid heavy AG2 imports on --help / parse-only usage
    from orchestrator import build_group_chat

    _groupchat, manager, user_proxy, _agents, _executors, agents_list = build_group_chat()

    # Build initial message with target context
    target_ctx = ""
    if target_info.column:
        target_ctx = (
            f"\nTarget variable: '{target_info.column}' "
            f"(problem type: {target_info.problem_type})"
        )
        if target_info.problem_type == "classification":
            target_ctx += f", {target_info.n_classes} classes"
    else:
        target_ctx = "\nNo target variable identified (unsupervised analysis)."

    if target_info.has_datetime_index:
        target_ctx += "\nDatetime column detected — possible time series data."

    initial_message = (
        f"Please run the full EDA pipeline on the following data file:\n"
        f"{resolved}"
        f"{target_ctx}"
    )

    logger.info("Starting EDA pipeline for: %s", resolved)

    try:
        user_proxy.initiate_chat(manager, message=initial_message)
    finally:
        clear_session()
        if enable_openlit:
            _shutdown_openlit()

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

    # Include hallucination evaluator cost (tracked outside AG2 agents).
    from tools.findings_tools import _eval_cost_info  # noqa: E402

    cost_text = _format_cost_summary(
        agents_list, usage_dict,
        eval_cost=_eval_cost_info if _eval_cost_info else None,
    )
    cost_path = get_outputs_dir(session_id) / "cost_summary.txt"
    cost_path.write_text(cost_text, encoding="utf-8")
    logger.info("Cost summary written to %s", cost_path)

    logger.info("EDA pipeline completed.")
    return session_id


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
