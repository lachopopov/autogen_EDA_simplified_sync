"""
tools/findings_tools.py — Assemble structured EDA findings narrative.

Architecture Reference: architecture.md § 4.6, § 12.1

Public AG2-facing functions:
  - prepare_interpretation_context() -> str   (Metadata-First fact sheet)
  - save_interpretations(json)       -> str   (store LLM commentary)
  - assemble_findings(...)           -> str   (merge facts + commentary)

Design:
  - Zero AG2 imports. Zero agent references. Pure Python.
  - Accepts JSON strings (the AG2 tool contract).
  - Returns a Findings Pydantic model serialized as JSON.
  - Metadata-First Hybrid approach:
      * Tools provide deterministic output (fact sheets with 100% plot data)
      * LLM reasons deeply about them (statistical, DS/ML, business)
      * Tools validate and merge commentary into report sections
  - Iteration-aware logic:
      * REVISION_NEEDED + iteration < 2 → address flags in narrative
      * APPROVED or iteration >= 2       → finalize, mark [UNRESOLVED]
  - Future extension point: VisionCapability verification layer
    (documented, not implemented — metadata coverage is 100%)

AG2 Version: 0.10.3
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

import numpy as np
import pandas as pd

from eda_state import CriticReport, EDAResults, Findings, Interpretations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_overview_section(eda: EDAResults) -> dict[str, Any]:
    """Build the overview section from EDA describe stats."""
    describe = eda.describe
    num_columns = len(describe)
    # Extract row count from first column's 'count' if available
    row_count = 0
    if describe:
        first_col_stats = next(iter(describe.values()), {})
        row_count = int(first_col_stats.get("count", 0))
    return {
        "title": "Dataset Overview",
        "content": (
            f"The dataset contains {row_count} rows and {num_columns} columns. "
            f"Descriptive statistics were computed for all columns."
        ),
    }


def _build_missing_section(eda: EDAResults) -> dict[str, Any]:
    """Build the missing values section from EDA missing info."""
    missing = eda.missing
    total_pct = missing.total_pct
    per_column = missing.per_column
    cols_with_missing = {col: pct for col, pct in per_column.items() if pct > 0}

    if not cols_with_missing:
        content = "No missing values detected in any column."
    else:
        col_details = ", ".join(
            f"{col} ({pct:.1f}%)" for col, pct in sorted(
                cols_with_missing.items(), key=lambda x: x[1], reverse=True
            )
        )
        content = (
            f"Dataset-level missingness: {total_pct:.1f}%. "
            f"Columns with missing values: {col_details}."
        )
    return {"title": "Missing Values", "content": content}


def _build_correlation_section(eda: EDAResults) -> dict[str, Any]:
    """Build the correlation summary section."""
    corr = eda.correlation
    if not corr:
        return {"title": "Correlation Analysis", "content": "No numerical columns for correlation analysis."}

    # Find strongest off-diagonal correlation
    max_corr = 0.0
    pair = ("", "")
    cols = list(corr.keys())
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            val = abs(corr.get(col_a, {}).get(col_b, 0) or 0)
            if val > max_corr:
                max_corr = val
                pair = (col_a, col_b)

    if max_corr > 0:
        content = (
            f"Pearson correlation computed for {len(cols)} numerical columns. "
            f"Strongest correlation: {pair[0]} vs {pair[1]} (|r|={max_corr:.2f})."
        )
    else:
        content = f"Pearson correlation computed for {len(cols)} numerical columns. No notable correlations found."

    return {"title": "Correlation Analysis", "content": content}


def _build_statistical_analysis_section(eda: EDAResults) -> dict[str, Any]:
    """Build an interpretive statistical analysis section.

    Analyses distribution shape, spread, and central-tendency insights
    derived deterministically from ``describe_stats`` output.
    """
    describe = eda.describe
    if not describe:
        return {
            "title": "Statistical Analysis",
            "content": "No descriptive statistics available for analysis.",
        }

    paragraphs: list[str] = []

    # --- Identify numerical vs categorical columns ---
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col, stats in describe.items():
        if stats.get("mean") is not None:
            numeric_cols.append(col)
        elif stats.get("top") is not None:
            categorical_cols.append(col)

    # --- Distribution analysis for numeric columns ---
    high_cv_cols: list[str] = []
    narrow_iqr_cols: list[str] = []
    potential_outlier_cols: list[str] = []

    for col in numeric_cols:
        stats = describe[col]
        mean = stats.get("mean")
        std = stats.get("std")
        q25 = stats.get("25%")
        q75 = stats.get("75%")
        col_min = stats.get("min")
        col_max = stats.get("max")

        if mean is not None and std is not None and mean != 0:
            cv = abs(std / mean)
            if cv > 1.0:
                high_cv_cols.append(col)

        if q25 is not None and q75 is not None:
            iqr = q75 - q25
            if iqr == 0:
                narrow_iqr_cols.append(col)
            elif col_min is not None and col_max is not None:
                lower_fence = q25 - 1.5 * iqr
                upper_fence = q75 + 1.5 * iqr
                if col_min < lower_fence or col_max > upper_fence:
                    potential_outlier_cols.append(col)

    if numeric_cols:
        paragraphs.append(
            f"Distribution analysis was performed on {len(numeric_cols)} "
            f"numerical feature(s)."
        )

    if high_cv_cols:
        paragraphs.append(
            f"High variability detected (coefficient of variation > 1.0) in: "
            f"{', '.join(high_cv_cols)}. These features show substantial spread "
            f"relative to their mean, which may indicate heterogeneous sub-populations "
            f"or the need for normalization before modeling."
        )

    if narrow_iqr_cols:
        paragraphs.append(
            f"Near-zero interquartile range detected in: "
            f"{', '.join(narrow_iqr_cols)}. These features carry minimal information "
            f"and are candidates for removal (zero or near-zero variance)."
        )

    if potential_outlier_cols:
        paragraphs.append(
            f"Potential outliers detected (values beyond 1.5×IQR fences) in: "
            f"{', '.join(potential_outlier_cols)}. Outlier treatment "
            f"(winsorization, capping, or removal) should be considered depending "
            f"on the downstream modeling objective."
        )

    # --- Categorical analysis ---
    if categorical_cols:
        low_cardinality: list[str] = []
        for col in categorical_cols:
            stats = describe[col]
            unique = stats.get("unique")
            if unique is not None and unique <= 2:
                low_cardinality.append(col)
        if low_cardinality:
            paragraphs.append(
                f"Low-cardinality categorical feature(s): "
                f"{', '.join(low_cardinality)} (≤2 unique values). "
                f"Consider binary encoding for modeling use."
            )

    if not paragraphs:
        paragraphs.append(
            "All numerical features show standard distributions with no "
            "extreme variability, zero-variance, or outlier concerns."
        )

    return {"title": "Statistical Analysis", "content": " ".join(paragraphs)}


def _build_conclusions_section(
    eda: EDAResults,
    critic: CriticReport,
) -> dict[str, Any]:
    """Build a conclusions section synthesizing findings across all analyses.

    Deterministic: every sentence is derived from actual computed values.
    """
    conclusions: list[str] = []
    missing = eda.missing
    corr = eda.correlation
    describe = eda.describe

    # Count columns
    n_cols = len(describe) if describe else 0

    # --- Data completeness conclusion ---
    if missing.total_pct == 0:
        conclusions.append(
            "The dataset is fully complete with no missing values, "
            "requiring no imputation before analysis."
        )
    elif missing.total_pct < 5:
        conclusions.append(
            f"Overall data completeness is high ({100 - missing.total_pct:.1f}%). "
            f"Minor missingness ({missing.total_pct:.1f}%) can likely be handled "
            f"with simple imputation (mean/median for numerical, mode for categorical)."
        )
    else:
        # Find worst column
        worst_col = max(missing.per_column.items(), key=lambda x: x[1], default=("", 0))
        conclusions.append(
            f"Significant data quality concern: {missing.total_pct:.1f}% overall "
            f"missingness detected. The most affected column is '{worst_col[0]}' "
            f"at {worst_col[1]:.1f}% missing. This level of missingness may introduce "
            f"bias if not properly addressed. Multiple imputation or domain-informed "
            f"strategies are recommended over simple deletion."
        )

    # --- Multicollinearity conclusion ---
    if corr:
        high_corr_pairs: list[tuple[str, str, float]] = []
        cols = list(corr.keys())
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                val = abs(corr.get(col_a, {}).get(col_b, 0) or 0)
                if val > 0.85:
                    high_corr_pairs.append((col_a, col_b, val))
        if high_corr_pairs:
            pair_strs = [f"{a}–{b} (|r|={v:.2f})" for a, b, v in high_corr_pairs]
            conclusions.append(
                f"Multicollinearity concern: {len(high_corr_pairs)} feature pair(s) "
                f"exhibit strong linear dependence: {', '.join(pair_strs)}. "
                f"If used in regression models, this may inflate coefficient variance "
                f"and reduce interpretability. Consider PCA, VIF-based selection, "
                f"or dropping one feature from each highly correlated pair."
            )
        else:
            conclusions.append(
                "No concerning multicollinearity was detected among numerical features, "
                "suggesting independent predictive signal from each variable."
            )

    # --- Quality flags conclusion ---
    if not critic.flags:
        conclusions.append(
            "All automated quality checks passed. The dataset appears suitable "
            "for modeling with standard preprocessing."
        )
    else:
        high_flags = [f for f in critic.flags if f.severity in ("BLOCKER", "HIGH")]
        med_flags = [f for f in critic.flags if f.severity == "MEDIUM"]
        if high_flags:
            conclusions.append(
                f"{len(high_flags)} high-severity data quality issue(s) were "
                f"identified that should be addressed before production modeling. "
                f"See the Data Quality Assessment section for specifics."
            )
        if med_flags:
            conclusions.append(
                f"{len(med_flags)} medium-severity issue(s) were noted. "
                f"While not blocking, addressing these may improve model performance."
            )

    if not conclusions:
        conclusions.append(
            f"Initial EDA on {n_cols} features completed successfully. "
            f"No critical issues detected."
        )

    return {"title": "Conclusions", "content": " ".join(conclusions)}


def _build_recommendations_section(
    eda: EDAResults,
    critic: CriticReport,
) -> dict[str, Any]:
    """Build actionable recommendations and business implications.

    Each recommendation is grounded in a specific finding from the data.
    """
    recommendations: list[str] = []
    missing = eda.missing
    corr = eda.correlation

    # --- Missing data recommendations ---
    cols_with_high_missing = {
        col: pct for col, pct in missing.per_column.items() if pct > 30
    }
    cols_with_moderate_missing = {
        col: pct for col, pct in missing.per_column.items() if 5 < pct <= 30
    }

    if cols_with_high_missing:
        col_list = ", ".join(
            f"'{c}' ({p:.0f}%)" for c, p in sorted(
                cols_with_high_missing.items(), key=lambda x: x[1], reverse=True
            )
        )
        recommendations.append(
            f"HIGH PRIORITY — Investigate data collection for columns with >30% "
            f"missing: {col_list}. Business impact: models trained on heavily "
            f"imputed data may produce unreliable predictions. Consider whether "
            f"missingness is random (safe to impute) or systematic (may indicate "
            f"a data pipeline issue that needs operational remediation)."
        )

    if cols_with_moderate_missing:
        col_list = ", ".join(
            f"'{c}' ({p:.0f}%)" for c, p in sorted(
                cols_with_moderate_missing.items(), key=lambda x: x[1], reverse=True
            )
        )
        recommendations.append(
            f"MEDIUM PRIORITY — Apply appropriate imputation for moderately "
            f"missing columns: {col_list}. Validate imputation quality using "
            f"held-out data comparison."
        )

    # --- Correlation-based recommendations ---
    if corr:
        redundant_pairs: list[tuple[str, str, float]] = []
        cols = list(corr.keys())
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                val = abs(corr.get(col_a, {}).get(col_b, 0) or 0)
                if val > 0.90:
                    redundant_pairs.append((col_a, col_b, val))
        if redundant_pairs:
            recommendations.append(
                f"FEATURE ENGINEERING — {len(redundant_pairs)} near-redundant "
                f"feature pair(s) detected (|r|>0.90). For linear models, retain "
                f"only one feature per pair to avoid multicollinearity. For "
                f"tree-based models, redundancy is less critical but increases "
                f"training time without adding predictive power."
            )

    # --- Critic-driven recommendations ---
    for flag in critic.flags:
        if flag.suggestion:
            recommendations.append(
                f"DATA QUALITY — {flag.column or 'Dataset'}: {flag.suggestion}"
            )

    # --- General next-steps ---
    if not recommendations:
        recommendations.append(
            "The dataset shows good overall quality. Recommended next steps: "
            "(1) Feature engineering and selection, "
            "(2) Train/test split with stratification if applicable, "
            "(3) Baseline model training and evaluation."
        )
    else:
        recommendations.append(
            "NEXT STEPS — After addressing the above items: "
            "(1) Re-run EDA to validate improvements, "
            "(2) Proceed with feature engineering, "
            "(3) Establish baseline model performance."
        )

    # --- Business implications ---
    business_items: list[str] = []
    if missing.total_pct > 10:
        business_items.append(
            "High missingness may indicate upstream data collection issues "
            "that warrant process review with data engineering teams."
        )
    if any(f.severity in ("BLOCKER", "HIGH") for f in critic.flags):
        business_items.append(
            "High-severity quality flags suggest the data may not yet be "
            "suitable for production decision-making without remediation."
        )
    if not critic.flags and missing.total_pct < 5:
        business_items.append(
            "The dataset appears ready for predictive modeling with standard "
            "preprocessing. Time-to-value for model deployment is minimal."
        )

    if business_items:
        recommendations.append(
            "BUSINESS IMPLICATIONS — " + " ".join(business_items)
        )

    return {
        "title": "Recommendations & Business Implications",
        "content": "\n\n".join(recommendations),
    }


def _build_visualizations_section(plot_paths: list[str]) -> dict[str, Any]:
    """Build the visualizations section listing generated plots."""
    if not plot_paths:
        return {"title": "Visualizations", "content": "No visualizations were generated."}
    return {
        "title": "Visualizations",
        "content": f"{len(plot_paths)} plot(s) generated.",
        "plot_paths": plot_paths,
    }


def _build_quality_section(
    critic: CriticReport,
    is_final: bool,
) -> dict[str, Any]:
    """Build the data quality section from critic flags.

    Args:
        critic: The CriticReport with flags and status.
        is_final: True when this is the final iteration (APPROVED or forced).
    """
    flags = critic.flags
    if not flags:
        return {
            "title": "Data Quality Assessment",
            "content": "All quality checks passed. No issues detected.",
        }

    flag_lines: list[str] = []
    for flag in flags:
        col_label = flag.column if flag.column else "dataset-level"
        line = f"[{flag.severity}] {col_label}: {flag.message} (rule: {flag.rule})"
        if flag.suggestion:
            line += f" → {flag.suggestion}"
        flag_lines.append(line)

    content = f"{len(flags)} quality flag(s) raised:\n" + "\n".join(flag_lines)
    return {"title": "Data Quality Assessment", "content": content}


def _collect_unresolved(critic: CriticReport) -> list[str]:
    """Collect flag descriptions that remain unresolved after max iterations.

    Called only when iteration >= 2 and flags still exist — these are
    intrinsic data quality issues, not report issues.
    """
    unresolved: list[str] = []
    for flag in critic.flags:
        col_label = flag.column if flag.column else "dataset-level"
        line = f"[UNRESOLVED] [{flag.severity}] {col_label}: {flag.message} (rule: {flag.rule})"
        if flag.suggestion:
            line += f" → {flag.suggestion}"
        unresolved.append(line)
    return unresolved


# ---------------------------------------------------------------------------
# Metadata-First Hybrid: Interpretation context + save
# ---------------------------------------------------------------------------


def _build_histogram_metadata(df: pd.DataFrame, num_cols: list[str]) -> str:
    """Compute histogram bin data for every numeric column (30 bins).

    Returns a text block with bin edges + counts — the exact data
    matplotlib uses to render each ``hist_<column>.png`` plot.
    """
    lines: list[str] = []
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            lines.append(f"\n{col} — HISTOGRAM DATA: no non-null values")
            continue
        counts, edges = np.histogram(series.values, bins=30)
        lines.append(f"\n{col} — HISTOGRAM DATA (30 bins):")
        for i, count in enumerate(counts):
            lines.append(
                f"  [{edges[i]:.4f}, {edges[i + 1]:.4f}): {int(count)}"
            )
        # Peak / modality detection
        peaks: list[dict[str, Any]] = []
        for i in range(1, len(counts) - 1):
            if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                peaks.append({
                    "bin": f"[{edges[i]:.4f}, {edges[i + 1]:.4f})",
                    "count": int(counts[i]),
                })
        # Edge peaks (first / last bin)
        if len(counts) > 1:
            if counts[0] > counts[1]:
                peaks.insert(0, {
                    "bin": f"[{edges[0]:.4f}, {edges[1]:.4f})",
                    "count": int(counts[0]),
                })
            if counts[-1] > counts[-2]:
                peaks.append({
                    "bin": f"[{edges[-2]:.4f}, {edges[-1]:.4f})",
                    "count": int(counts[-1]),
                })
        modality = "unimodal" if len(peaks) <= 1 else f"{len(peaks)}-modal"
        lines.append(f"  Modality: {modality} ({len(peaks)} peak(s))")
        for p in peaks:
            lines.append(f"  Peak: {p['bin']} at {p['count']} obs")
        # Empty bins (gaps)
        empty = [
            f"[{edges[i]:.4f}, {edges[i + 1]:.4f})"
            for i in range(len(counts))
            if counts[i] == 0
        ]
        if empty:
            lines.append(f"  Empty bins (gaps): {', '.join(empty)}")
    return "\n".join(lines)


def _build_column_stats_block(describe: dict[str, Any]) -> str:
    """Build per-column statistics text from describe output."""
    lines: list[str] = []
    for col, stats in describe.items():
        mean = stats.get("mean")
        if mean is not None:
            # Numeric column
            median = stats.get("50%", 0)
            std = stats.get("std", 0)
            col_min = stats.get("min", 0)
            q25 = stats.get("25%", 0)
            q75 = stats.get("75%", 0)
            col_max = stats.get("max", 0)
            count = stats.get("count", 0)
            iqr = (q75 or 0) - (q25 or 0)
            cv = abs(std / mean) if mean and mean != 0 else 0
            lower_fence = (q25 or 0) - 1.5 * iqr
            upper_fence = (q75 or 0) + 1.5 * iqr
            # Skew direction from mean vs median
            if median and median != 0:
                ratio = mean / median
                if ratio > 1.05:
                    skew_dir = "RIGHT-SKEWED"
                elif ratio < 0.95:
                    skew_dir = "LEFT-SKEWED or BIMODAL"
                else:
                    skew_dir = "approximately symmetric"
            else:
                skew_dir = "undetermined"
            lines.append(
                f"  {col}: count={count}, mean={mean:.4f}, median={median:.4f}, "
                f"std={std:.4f}, min={col_min}, Q25={q25}, Q75={q75}, "
                f"max={col_max}, IQR={iqr:.4f}, CV={cv:.4f}, "
                f"lower_fence={lower_fence:.4f}, upper_fence={upper_fence:.4f}, "
                f"skew_direction={skew_dir}"
            )
        else:
            # Categorical column
            top = stats.get("top", "N/A")
            freq = stats.get("freq", 0)
            unique = stats.get("unique", 0)
            count = stats.get("count", 0)
            lines.append(
                f"  {col}: count={count}, unique={unique}, top='{top}', freq={freq}"
            )
    return "\n".join(lines)


def _build_correlation_block(corr: dict[str, Any]) -> str:
    """Build full correlation matrix text + ranked pairs."""
    if not corr:
        return "  No numerical columns for correlation."
    cols = list(corr.keys())
    lines: list[str] = [f"  Matrix ({len(cols)}×{len(cols)}):"]
    # Full matrix
    for col_a in cols:
        row_vals = []
        for col_b in cols:
            val = corr.get(col_a, {}).get(col_b, 0) or 0
            row_vals.append(f"{val:.4f}")
        lines.append(f"    {col_a}: {', '.join(row_vals)}")
    # Ranked pairs
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            val = corr.get(col_a, {}).get(col_b, 0) or 0
            pairs.append((col_a, col_b, val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    lines.append("  Ranked pairs (by |r|):")
    for a, b, v in pairs:
        strength = (
            "VERY STRONG" if abs(v) > 0.9
            else "STRONG" if abs(v) > 0.7
            else "MODERATE" if abs(v) > 0.5
            else "WEAK" if abs(v) > 0.3
            else "NEGLIGIBLE"
        )
        lines.append(f"    {a} <-> {b}: r={v:+.4f} ({strength})")
    return "\n".join(lines)


def _build_missing_block(missing_per_col: dict[str, float], total_pct: float) -> str:
    """Build missing values text — every bar height in the missing heatmap."""
    lines: list[str] = [f"  Overall completeness: {100 - total_pct:.1f}% ({total_pct:.1f}% missing)"]
    for col, pct in sorted(missing_per_col.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {col}: {pct:.2f}% missing")
    return "\n".join(lines)


def _build_critic_block(critic: CriticReport) -> str:
    """Build critic flags text."""
    if not critic.flags:
        return "  No quality flags raised. All checks passed."
    lines: list[str] = [f"  {len(critic.flags)} flag(s):"]
    for f in critic.flags:
        col_label = f.column if f.column else "dataset-level"
        line = f"  [{f.severity}] {col_label}: {f.message} (rule: {f.rule})"
        if f.suggestion:
            line += f" -> {f.suggestion}"
        lines.append(line)
    return "\n".join(lines)


def _build_plot_inventory(plot_paths: list[str]) -> str:
    """Map each plot file to its type and column."""
    from pathlib import Path

    lines: list[str] = []
    for pp in plot_paths:
        name = Path(pp).stem
        if name.startswith("hist_"):
            col = name[5:]
            lines.append(f"  {Path(pp).name} -> histogram, column={col}")
        elif name == "correlation_heatmap":
            lines.append(f"  {Path(pp).name} -> correlation heatmap (full matrix)")
        elif name == "missing_heatmap":
            lines.append(f"  {Path(pp).name} -> missing values bar chart (per-column %)")
        else:
            lines.append(f"  {Path(pp).name} -> unknown plot type")
    return "\n".join(lines)


def prepare_interpretation_context() -> str:
    """
    AG2 tool entry point.  Metadata-First Hybrid: extract ALL data points
    behind every visualization + all statistical results.

    Provides the LLM with the EXACT data used to render each plot,
    plus all computed statistics, so it can reason as a statistician
    who is reading a complete data table — not a chart summary.

    Data provided per plot type (100% pixel-coverage):
      - Histograms:         30 bin edges + 30 bin counts per column
      - Correlation heatmap: full N×N matrix (every r-value)
      - Missing heatmap:     per-column % dict (every bar height)

    Returns:
        A structured text fact sheet for LLM interpretation.
    """
    from tools._pipeline_state import is_active, load_state, \
        PipelineStateError

    if not is_active():
        raise RuntimeError(
            "prepare_interpretation_context() requires an active pipeline session."
        )

    # --- Load all artifacts ---
    desc_raw = load_state("describe_stats")
    miss_raw = load_state("missing_analysis")
    corr_raw = load_state("correlation_matrix")
    critic_raw = load_state("critic_report")
    data_raw = load_state("data_json")

    if desc_raw is None:
        raise PipelineStateError(
            "Cannot prepare context: 'describe_stats' artifact missing."
        )

    describe = json.loads(desc_raw)
    missing_info = json.loads(miss_raw) if miss_raw else {"per_column": {}, "total_pct": 0}
    corr = json.loads(corr_raw) if corr_raw else {}
    critic = CriticReport.model_validate_json(critic_raw) if critic_raw else CriticReport()

    # Load plot paths (compose from individual artifacts)
    plot_paths: list[str] = []
    for key in ("plot_histograms", "plot_correlation_heatmap", "plot_missing_heatmap"):
        raw = load_state(key)
        if raw:
            plot_paths.extend(json.loads(raw))

    # --- Build fact sheet ---
    sections: list[str] = ["=== INTERPRETATION CONTEXT (FACT SHEET) ==="]

    # Dataset overview
    num_cols = len(describe)
    first_stats = next(iter(describe.values()), {})
    row_count = int(first_stats.get("count", 0))
    sections.append(f"\nDATASET: {row_count} rows x {num_cols} columns")

    # Per-column statistics
    sections.append("\nPER-COLUMN STATISTICS:")
    sections.append(_build_column_stats_block(describe))

    # Missing values (every bar height in missing_heatmap.png)
    sections.append("\nMISSING VALUES (100% of missing_heatmap.png data):")
    sections.append(_build_missing_block(
        missing_info.get("per_column", {}),
        missing_info.get("total_pct", 0),
    ))

    # Correlation matrix (every cell in correlation_heatmap.png)
    sections.append("\nCORRELATION MATRIX (100% of correlation_heatmap.png data):")
    sections.append(_build_correlation_block(corr))

    # Histogram bin data (every bar in hist_<col>.png)
    if data_raw:
        df = pd.DataFrame(json.loads(data_raw))
        num_col_names = df.select_dtypes(include="number").columns.tolist()
        if num_col_names:
            sections.append(
                "\nHISTOGRAM BIN DATA (100% of hist_*.png data, 30 bins each):"
            )
            sections.append(_build_histogram_metadata(df, num_col_names))
    else:
        sections.append(
            "\nHISTOGRAM BIN DATA: raw data not available in artifact store."
        )

    # Critic flags
    sections.append("\nQUALITY FLAGS:")
    sections.append(_build_critic_block(critic))

    # Plot inventory
    sections.append("\nPLOT INVENTORY:")
    if plot_paths:
        sections.append(_build_plot_inventory(plot_paths))
    else:
        sections.append("  No plots generated.")

    fact_sheet = "\n".join(sections)

    logger.info(
        "Interpretation context prepared: %d chars, %d plots",
        len(fact_sheet),
        len(plot_paths),
    )

    return (
        f"{fact_sheet}\n\n"
        f"--- END OF FACT SHEET ---\n"
        f"Use this data to generate expert commentary via save_interpretations()."
    )


def save_interpretations(
    interpretations_json: Annotated[
        str,
        "JSON string matching the Interpretations schema. "
        "Keys: overview, missing_values, correlation, statistical_analysis, "
        "quality_assessment (each with 'statistical', 'ds_ml', 'business' sub-keys), "
        "plot_commentaries (list of {plot_file, statistical, ds_ml, business}), "
        "conclusions (string), recommendations_and_business_implications (string).",
    ],
) -> str:
    """
    AG2 tool entry point.  Validates and stores LLM-generated expert
    commentary in the artifact store.

    The Interpretations schema is validated via Pydantic before storage.
    If validation fails, the error is returned and assemble_findings()
    will use deterministic fallback text (safety net).

    Returns:
        Confirmation message with artifact reference.
    """
    from tools._pipeline_state import is_active, save_state, STATE_REF_PREFIX

    if not is_active():
        raise RuntimeError(
            "save_interpretations() requires an active pipeline session."
        )

    # Validate against Pydantic schema
    interp = Interpretations.model_validate_json(interpretations_json)
    validated_json = interp.model_dump_json()

    save_state("interpretations", validated_json)

    n_plots = len(interp.plot_commentaries)
    n_sections = sum(
        1 for field in (
            interp.overview,
            interp.missing_values,
            interp.correlation,
            interp.statistical_analysis,
            interp.quality_assessment,
        )
        if field is not None
    )

    logger.info(
        "Interpretations saved: %d section commentaries, %d plot commentaries",
        n_sections,
        n_plots,
    )

    return (
        f"Interpretations saved: {n_sections} section commentaries, "
        f"{n_plots} plot commentaries. "
        f"Reference: {STATE_REF_PREFIX}interpretations"
    )


# ---------------------------------------------------------------------------
# AG2-facing public function (flat callable, no OOP visible to AG2)
# ---------------------------------------------------------------------------

def assemble_findings(
    eda_results_json: Annotated[str, "JSON string of EDAResults from EDA tools"],
    critic_report_json: Annotated[str, "JSON string of CriticReport from run_critic_rules()"],
    plot_paths_json: Annotated[str, "JSON list of plot file paths from visualization tools"],
) -> str:
    """
    AG2 tool entry point. Assembles structured EDA findings narrative
    from analysis results, critic flags, and visualization paths.

    Iteration logic (architecture.md § 4.6, § 8):
      - If REVISION_NEEDED and iteration < 2: include flags in narrative
        so the next critic cycle can verify they were addressed.
      - If APPROVED or iteration >= 2: finalize findings. Any remaining
        HIGH/BLOCKER flags are marked [UNRESOLVED] — these are intrinsic
        data quality issues, not report issues.

    Returns:
        JSON string of a Findings model (sections, unresolved_flags).
    """
    # Artifact store: resolve inputs with schema validation + composition fallback
    from tools._pipeline_state import is_active, resolve, load_state, save_state, \
        STATE_REF_PREFIX, PipelineStateError

    if is_active():
        # --- Resolve eda_results_json (may require composition) ---
        try:
            eda_results_json = resolve(eda_results_json, "eda_results")
            eda = EDAResults.model_validate_json(eda_results_json)
        except (PipelineStateError, Exception):
            # Compose from individual artifacts
            desc_raw = load_state("describe_stats")
            miss_raw = load_state("missing_analysis")
            corr_raw = load_state("correlation_matrix")
            if desc_raw is None:
                raise PipelineStateError(
                    "Cannot compose EDAResults: 'describe_stats' artifact missing. "
                    "EDAAnalysisAgent may not have executed describe_stats()."
                )
            if miss_raw is None:
                raise PipelineStateError(
                    "Cannot compose EDAResults: 'missing_analysis' artifact missing. "
                    "EDAAnalysisAgent may not have executed missing_analysis()."
                )
            if corr_raw is None:
                raise PipelineStateError(
                    "Cannot compose EDAResults: 'correlation_matrix' artifact missing. "
                    "EDAAnalysisAgent may not have executed correlation_matrix()."
                )
            composed = {
                "describe": json.loads(desc_raw),
                "missing": json.loads(miss_raw),
                "correlation": json.loads(corr_raw),
            }
            eda_results_json = json.dumps(composed)
            eda = EDAResults.model_validate_json(eda_results_json)
            logger.info("EDAResults composed from individual artifacts")

        # --- Resolve critic_report_json ---
        try:
            critic_report_json = resolve(critic_report_json, "critic_report")
            critic = CriticReport.model_validate_json(critic_report_json)
        except (PipelineStateError, Exception):
            fallback = load_state("critic_report")
            if fallback is None:
                raise PipelineStateError(
                    "Cannot resolve 'critic_report' artifact. "
                    "CriticAgent may not have executed run_critic_rules()."
                )
            critic_report_json = fallback
            critic = CriticReport.model_validate_json(critic_report_json)
            logger.info("CriticReport resolved via fallback")

        # --- Resolve plot_paths_json (may require composition) ---
        try:
            plot_paths_json = resolve(plot_paths_json, "plot_paths")
            plot_paths = json.loads(plot_paths_json)
            if not isinstance(plot_paths, list):
                raise ValueError("plot_paths is not a list")
        except (PipelineStateError, Exception):
            # Compose from individual visualization artifacts
            hist = load_state("plot_histograms")
            corr_hm = load_state("plot_correlation_heatmap")
            miss_hm = load_state("plot_missing_heatmap")
            merged: list[str] = []
            if hist:
                merged.extend(json.loads(hist))
            if corr_hm:
                merged.extend(json.loads(corr_hm))
            if miss_hm:
                merged.extend(json.loads(miss_hm))
            plot_paths = merged
            plot_paths_json = json.dumps(plot_paths)
            logger.info("plot_paths composed from %d visualization artifacts", len(plot_paths))
    else:
        eda = EDAResults.model_validate_json(eda_results_json)
        critic = CriticReport.model_validate_json(critic_report_json)
        plot_paths = json.loads(plot_paths_json)

    # Determine if this is the final iteration
    is_final = critic.status == "APPROVED" or critic.iteration >= 2

    # --- Load LLM interpretations (if available) ---
    interp: Interpretations | None = None
    if is_active():
        interp_raw = load_state("interpretations")
        if interp_raw:
            try:
                interp = Interpretations.model_validate_json(interp_raw)
                logger.info("Loaded LLM interpretations for enrichment")
            except Exception:
                logger.warning("Invalid interpretations in store — using fallback")
                interp = None

    # --- Pair plots with sections ---
    from pathlib import Path

    hist_paths: list[str] = []
    corr_heatmap_paths: list[str] = []
    missing_heatmap_paths: list[str] = []
    for pp in plot_paths:
        stem = Path(pp).stem
        if stem.startswith("hist_"):
            hist_paths.append(pp)
        elif stem == "correlation_heatmap":
            corr_heatmap_paths.append(pp)
        elif stem == "missing_heatmap":
            missing_heatmap_paths.append(pp)

    # --- Helper: enrich a section with LLM commentary ---
    def _enrich(
        section: dict[str, Any],
        interp_key: str,
        paired_plots: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add LLM expert commentary and paired plots to a section."""
        if interp and (commentary := getattr(interp, interp_key, None)):
            if isinstance(commentary, dict):
                lines: list[str] = []
                if commentary.get("statistical"):
                    lines.append(
                        f"Statistical Perspective: {commentary['statistical']}"
                    )
                if commentary.get("ds_ml"):
                    lines.append(
                        f"Data Science & ML Perspective: {commentary['ds_ml']}"
                    )
                if commentary.get("business"):
                    lines.append(
                        f"Business Perspective: {commentary['business']}"
                    )
                if lines:
                    section["expert_commentary"] = "\n\n".join(lines)
            elif isinstance(commentary, str) and commentary:
                section["expert_commentary"] = commentary
        if paired_plots:
            section["plot_paths"] = paired_plots
            # Attach per-plot commentaries from interpretations
            if interp and interp.plot_commentaries:
                plot_comms = []
                for pp in paired_plots:
                    fname = Path(pp).name
                    for pc in interp.plot_commentaries:
                        if pc.plot_file == fname:
                            plot_comms.append(pc.model_dump())
                            break
                if plot_comms:
                    section["plot_commentaries"] = plot_comms
        return section

    # Build sections in report order (Option A: plots inline in parent sections)
    overview = _enrich(_build_overview_section(eda), "overview")
    missing = _enrich(
        _build_missing_section(eda), "missing_values",
        paired_plots=missing_heatmap_paths or None,
    )
    correlation = _enrich(
        _build_correlation_section(eda), "correlation",
        paired_plots=corr_heatmap_paths or None,
    )
    statistical = _enrich(
        _build_statistical_analysis_section(eda), "statistical_analysis",
        paired_plots=hist_paths or None,
    )
    quality = _enrich(
        _build_quality_section(critic, is_final), "quality_assessment",
    )

    # Conclusions & Recommendations: LLM replaces deterministic when available
    conclusions = _build_conclusions_section(eda, critic)
    if interp and interp.conclusions:
        conclusions["content"] = interp.conclusions
    recommendations = _build_recommendations_section(eda, critic)
    if interp and interp.recommendations_and_business_implications:
        recommendations["content"] = interp.recommendations_and_business_implications

    sections: list[dict[str, Any]] = [
        overview,
        missing,
        correlation,
        statistical,
        quality,
        conclusions,
        recommendations,
    ]

    # Collect unresolved flags only on final iteration with remaining issues
    unresolved: list[str] = []
    if is_final and critic.flags:
        # Only flags with severity above MEDIUM are truly "unresolved"
        high_severity_flags = [
            f for f in critic.flags if f.severity in ("BLOCKER", "HIGH")
        ]
        if high_severity_flags:
            # Create a temporary CriticReport with only high-severity flags
            high_critic = CriticReport(
                flags=high_severity_flags,
                iteration=critic.iteration,
                status=critic.status,
            )
            unresolved = _collect_unresolved(high_critic)

    findings = Findings(sections=sections, unresolved_flags=unresolved)

    logger.info(
        "Findings assembled: %d sections, %d unresolved flags, final=%s",
        len(sections),
        len(unresolved),
        is_final,
    )
    result = findings.model_dump_json()

    if is_active():
        save_state("findings", result)
        final_label = "final" if is_final else "interim"
        return (
            f"Findings assembled ({final_label}): {len(sections)} sections, "
            f"{len(unresolved)} unresolved flag(s). "
            f"Reference: {STATE_REF_PREFIX}findings"
        )
    return result
