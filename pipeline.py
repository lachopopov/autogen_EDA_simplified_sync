"""
pipeline.py — EDA pipeline logic, version constants, and protective controls.

This module owns `run_pipeline` and all private helpers.  `main.py` is kept
as a thin CLI entry point that re-exports the public symbols.

Architecture reference: implementation_refactoring.md
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from core import cache, concurrency, metrics
from eda_state import TargetInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version constants — baked in at import time, included in cache keys.
# ---------------------------------------------------------------------------

PIPELINE_VERSION: str = "1.0.0"

_ROOT = Path(__file__).resolve().parent


def _compute_prompt_version() -> str:
    """Return SHA-256 of the curated source files that contain prompts/logic.

    Changing any agent prompt, EDA tool, or the orchestrator will produce a
    different PROMPT_VERSION, which invalidates all existing cache entries.
    """
    files = sorted(_ROOT.glob("agents/*.py"))
    files += [
        _ROOT / "tools" / "findings_tools.py",
        _ROOT / "orchestrator.py",
    ]
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


PROMPT_VERSION: str = _compute_prompt_version()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_main_mod():
    """Return the *main* module at call time to support test monkeypatching.

    Tests patch ``main.get_outputs_dir``, ``main.ensure_run_dirs``,
    ``main.OPENLIT_ENDPOINT``, ``main._init_openlit``, etc.  All calls that
    exercise those patch points must go through this helper so the patches
    are visible at runtime.
    """
    return importlib.import_module("main")


def _format_cost_summary(
    agents_list: list,
    usage_dict: dict,
    eval_cost: dict | None = None,
) -> str:
    # Intentionally identical to original implementation in main.py
    lines: list[str] = []
    lines.append("EDA Pipeline — Cost & Timing Summary")
    lines.append("=" * 40)
    lines.append(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

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

    if eval_cost and eval_cost.get("cost", 0) > 0:
        lines.append(col_fmt.format(
            "HallucinationEval",
            eval_cost.get("model", "unknown"),
            eval_cost["cost"],
            eval_cost.get("prompt_tokens", 0),
            eval_cost.get("completion_tokens", 0),
        ))

    lines.append("")

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


def _format_timings(timings_path: Path) -> str:
    """Read timings.jsonl for the session and return a formatted timing block.

    Produces two sections:
      * Phase Timings  — pipeline-level phases (initiate_chat, cost_summary, …)
      * Agent Breakdown — per-agent spans written by the router via
        ``metrics.record_span("agent.<AgentName>", duration_ms)``

    Returns an empty string if the file does not exist or has no records.
    """
    if not timings_path.exists():
        return ""

    records: list[dict] = []
    with timings_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return ""

    pipeline_recs = [r for r in records if not r.get("phase", "").startswith("agent.")]
    agent_recs = [r for r in records if r.get("phase", "").startswith("agent.")]

    def _fmt_ms(ms: float) -> str:
        return f"{ms:>12,.1f} ms"

    lines: list[str] = []

    # ── Pipeline phases ──────────────────────────────────────────────
    lines.append("Phase Timings")
    lines.append("-" * 40)
    total_ms = 0.0
    for rec in pipeline_recs:
        phase = rec.get("phase", "unknown")
        duration_ms = rec.get("duration_ms", 0.0)
        total_ms += duration_ms
        lines.append(f"  {phase:<32}  {_fmt_ms(duration_ms)}")
    lines.append("  " + "─" * 50)
    lines.append(f"  {'total':<32}  {_fmt_ms(total_ms)}  ({total_ms / 1000:.1f} s)")
    lines.append("")

    # ── Agent breakdown ──────────────────────────────────────────────
    if agent_recs:
        lines.append("Agent Breakdown  (within initiate_chat)")
        lines.append("-" * 40)
        for rec in agent_recs:
            label = rec.get("phase", "unknown").removeprefix("agent.")
            duration_ms = rec.get("duration_ms", 0.0)
            lines.append(f"  {label:<32}  {_fmt_ms(duration_ms)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _confirm_target_interactive(candidate: TargetInfo, df) -> TargetInfo:
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


def _build_target_info(df, column: str, *, has_datetime: bool = False) -> TargetInfo:
    from tools.data_loader import _classify_target
    info = _classify_target(df, column)
    info.detection_method = "user_specified"
    info.has_datetime_index = has_datetime
    return info


def _resolve_target(df, *, target_flag: str | None, no_target_flag: bool) -> TargetInfo:
    from tools.data_loader import _has_datetime_column, detect_target

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


def _confirm_reclassify_interactive(suspects: list) -> list[str]:
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

    from tools.data_loader import detect_encoded_categoricals

    suspects = detect_encoded_categoricals(df, target_column=target_column)
    if not suspects:
        logger.info("No encoded-categorical suspects detected")
        return [], {}

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


def _init_openlit() -> None:
    try:
        import openlit  # noqa: F401
    except ImportError:
        logger.warning(
            "OpenLIT requested (--openlit) but 'openlit' package is not installed. "
            "Install with: pip install openlit"
        )
        return

    kwargs: dict = {}
    # Read through the main module so test patches on main.OPENLIT_ENDPOINT take effect.
    endpoint = _get_main_mod().OPENLIT_ENDPOINT

    if endpoint:
        kwargs["otlp_endpoint"] = endpoint
        logger.info("OpenLIT: sending traces to %s", endpoint)
    else:
        logger.info("OpenLIT: tracing to console (no OPENLIT_ENDPOINT set)")

    kwargs["disabled_instrumentors"] = ["agno"]

    pricing_path = Path(__file__).resolve().parent / "openlit_pricing.json"
    if pricing_path.exists():
        kwargs["pricing_json"] = str(pricing_path)
        logger.info("OpenLIT: using custom pricing from %s", pricing_path)

    import openlit as _ol
    _ol.init(**kwargs)
    logger.info("OpenLIT initialised — auto-tracking LLM calls, tokens, costs.")


def _shutdown_openlit() -> None:
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


def run_pipeline(
    file_path: Path,
    *,
    target_flag: str | None = None,
    no_target_flag: bool = False,
    enable_openlit: bool = False,
    categoricals_flag: str | None = None,
    no_reclassify_flag: bool = False,
) -> str:
    """Run the full EDA pipeline and return the session_id (or cache key on hit)."""
    # Resolve the main module early so all test patches on main.* are honoured.
    _main = _get_main_mod()

    resolved = file_path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    if enable_openlit:
        import config as _cfg
        _cfg.OPENLIT_ENABLE = True
        _main._init_openlit()  # through main so test patches on main._init_openlit work

    # ------------------------------------------------------------------
    # Pre-pipeline — spans here are intentional no-ops
    #
    # file_load / target_resolve / encoded_categorical_resolve must run
    # BEFORE init_session() because their outputs (file hash, target
    # column) determine the cache key.  Moving them inside the session
    # would break cache-first logic.  These phases are also cheap
    # (<100 ms) so the timing loss is not significant.
    #
    # Per-agent timing is captured inside initiate_chat by the router
    # in orchestrator.py via metrics.record_span() — see core/metrics.py
    # module docstring for the full design rationale.
    # ------------------------------------------------------------------
    from tools.data_loader import _get_loader

    with metrics.span("file_load"):
        df = _get_loader(str(resolved)).load(str(resolved))
        df = df.drop_duplicates().reset_index(drop=True)

    with metrics.span("target_resolve"):
        target_info = _resolve_target(
            df, target_flag=target_flag, no_target_flag=no_target_flag,
        )
    logger.info(
        "Target resolution: column=%s, type=%s, method=%s",
        target_info.column, target_info.problem_type, target_info.detection_method,
    )

    with metrics.span("encoded_categorical_resolve"):
        reclassified_cols, reclassified_subtypes = _resolve_reclassification(
            df,
            target_column=target_info.column,
            categoricals_flag=categoricals_flag,
            no_reclassify_flag=no_reclassify_flag,
        )
    logger.info("Reclassified as categorical: %s", reclassified_cols or "(none)")

    # ------------------------------------------------------------------
    # Cache key — always computed, dormant when EDA_MODE != 'final'
    # ------------------------------------------------------------------
    canonical_params: dict = {
        "target_flag": target_flag,
        "no_target_flag": no_target_flag,
        "categoricals_flag": categoricals_flag,
        "no_reclassify_flag": no_reclassify_flag,
        # enable_openlit intentionally excluded from cache key
    }
    key = cache.compute_key(
        resolved,
        canonical_params,
        prompt_version=PROMPT_VERSION,
        pipeline_version=PIPELINE_VERSION,
    )

    # Cache fast-path (outside the concurrency guard — hits are free)
    if cache.is_enabled():
        hit = cache.lookup(key)
        if hit is not None:
            with metrics.span("cache_hit", extra={"key": key[:16]}):
                pass
            logger.info("Cache hit: returning cached run key=%s...", key[:8])
            return key

    # ------------------------------------------------------------------
    # Heavy work — serialised by the concurrency guard
    # ------------------------------------------------------------------
    from tools._pipeline_state import clear_session, init_session, save_state

    with concurrency.pipeline_guard():
        session_id = init_session()
        try:
            _main.ensure_output_dirs(session_id)
            logger.info("Pipeline session: %s", session_id)

            save_state("target_info", target_info.model_dump_json())
            if reclassified_cols:
                save_state("reclassified_categoricals", json.dumps(reclassified_cols))
            if reclassified_subtypes:
                save_state("reclassified_subtypes", json.dumps(reclassified_subtypes))

            from orchestrator import build_group_chat

            _groupchat, manager, user_proxy, _agents, _executors, agents_list = build_group_chat()

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

            with metrics.span("initiate_chat"):
                user_proxy.initiate_chat(manager, message=initial_message)

            # Cost tracking
            from autogen import gather_usage_summary

            from tools.findings_tools import _eval_cost_info

            with metrics.span("cost_summary"):
                usage_dict = gather_usage_summary(agents_list)
                logger.info(
                    "Cost tracking: gathered usage summary from %d agents",
                    len(agents_list),
                )
                cost_text = _format_cost_summary(
                    agents_list, usage_dict,
                    eval_cost=_eval_cost_info if _eval_cost_info else None,
                )
                timings_path = _main.get_outputs_dir(session_id) / "timings.jsonl"
                timing_text = _format_timings(timings_path)
                full_summary = cost_text + timing_text
                cost_path = _main.get_outputs_dir(session_id) / "cost_summary.txt"
                cost_path.write_text(full_summary, encoding="utf-8")
                logger.info("Cost summary written to %s", cost_path)
                print(full_summary)

            if cache.is_enabled():
                cache.store(key, run_dir=_main.get_outputs_dir(session_id))

        finally:
            clear_session()
            if enable_openlit:
                _main._shutdown_openlit()  # through main so test patches apply

    logger.info("EDA pipeline completed.")
    return session_id
