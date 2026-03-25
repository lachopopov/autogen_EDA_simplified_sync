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
import math
from typing import Annotated, Any

import numpy as np
import pandas as pd

from eda_state import (
    CategoricalAnalysis,
    CriticReport,
    DataProfile,
    EDAResults,
    FeatureAssociations,
    Findings,
    Interpretations,
)

logger = logging.getLogger(__name__)

# Token usage captured from the comprehensive evaluation LLM call.
# Populated by _run_comprehensive_eval(); read by main._format_cost_summary().
# Survives clear_session() because it lives at module level, not in the
# artifact store.
_eval_cost_info: dict[str, Any] = {}

# Peak detection constants used in _build_histogram_metadata().
# Named constants make the algorithm self-documenting and unit-testable.
#
#   PEAK_SIGMA_THRESHOLD : A local-maximum must exceed its taller neighbour by
#       this many Poisson standard deviations to count as a genuine peak.
#       Poisson 1σ for a bin of expected size avg_bin = n_total/n_bins is
#       sqrt(avg_bin); 3.0 σ rejects ≈99.7% of random noise regardless of N.
#       Validated on iris.csv (N=150, threshold≈6.7) and adult.csv (N=32537).
#
#   PEAK_MASS_FRACTION   : A peak bin must also hold at least this fraction of
#       total observations.  Guards high-N datasets where the 3σ floor is met
#       by bins that still represent only a negligible fraction of the data.
#       At 1% the floor is 325 obs for adult age — larger than typical
#       decade-mark micro-oscillations (≈50–200 obs prominence).
PEAK_SIGMA_THRESHOLD: float = 3.0
PEAK_MASS_FRACTION: float = 0.01


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_overview_section(
    eda: EDAResults,
    shape: tuple[int, int] | None = None,
    duplicate_count: int = 0,
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
    encoded_categorical_cols: list[str] | None = None,
    encoded_categorical_subtypes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the overview section from EDA describe stats.

    Args:
        eda: Populated EDAResults (used as fallback for row/column count).
        shape: Authoritative (rows, cols) from DataProfile.  When supplied,
            this is used directly instead of inferring from describe[col][count]
            which can be 0 when describe stats are sparsely populated (W1 fix).
        duplicate_count: Number of duplicate rows removed during loading (W8).
        categorical_cols: Names of categorical columns from DataProfile.  When
            supplied, the header states the column-type breakdown (W4 fix).
        numerical_cols: Names of numerical columns from DataProfile.  When
            supplied alongside categorical_cols, both type lists are named
            explicitly so no column type is anonymous in the overview.
        encoded_categorical_cols: Columns reclassified from numerical to categorical
            (subset of categorical_cols).  When present, a transparency note is added.
        encoded_categorical_subtypes: Mapping of encoded-categorical column names
            to their measurement subtype ("nominal" or "ordinal").  When present,
            the transparency note distinguishes nominal from ordinal columns.
    """
    if shape is not None:
        row_count, num_columns = shape
    else:
        describe = eda.describe
        num_columns = len(describe)
        row_count = 0
        if describe:
            first_col_stats = next(iter(describe.values()), {})
            row_count = int(first_col_stats.get("count", 0) or 0)
    parts = [
        f"The dataset contains {row_count} rows and {num_columns} columns. "
        f"Descriptive statistics were computed for all columns."
    ]
    if categorical_cols or numerical_cols:
        n_cat = len(categorical_cols) if categorical_cols else (num_columns - len(numerical_cols or []))
        n_num = len(numerical_cols) if numerical_cols else (num_columns - n_cat)
        num_names = ", ".join(numerical_cols) if numerical_cols else ""
        cat_names = ", ".join(categorical_cols) if categorical_cols else ""
        # Add encoded-categorical annotation to the composition line
        enc_note = ""
        if encoded_categorical_cols:
            enc_names = ", ".join(encoded_categorical_cols)
            enc_note = (
                f" ({len(encoded_categorical_cols)} encoded as integers: {enc_names})"
            )
        if num_names and cat_names:
            parts.append(
                f"Column composition: {n_num} numerical ({num_names}); "
                f"{n_cat} categorical{enc_note} ({cat_names})."
            )
        elif cat_names:
            parts.append(
                f"Column composition: {n_num} numerical, {n_cat} categorical{enc_note} "
                f"({cat_names})."
            )
        else:
            parts.append(
                f"Column composition: {n_num} numerical ({num_names}), "
                f"{n_cat} categorical{enc_note}."
            )
    if encoded_categorical_cols:
        # Build subtype-aware transparency note (Stevens' measurement theory:
        # nominal vs ordinal determines valid operations on the data).
        if encoded_categorical_subtypes:
            nominal = [c for c in encoded_categorical_cols
                       if encoded_categorical_subtypes.get(c, "nominal") == "nominal"]
            ordinal = [c for c in encoded_categorical_cols
                       if encoded_categorical_subtypes.get(c) == "ordinal"]
            subtype_parts: list[str] = []
            if nominal:
                subtype_parts.append(
                    f"{len(nominal)} nominal: {', '.join(nominal)}"
                )
            if ordinal:
                subtype_parts.append(
                    f"{len(ordinal)} ordinal: {', '.join(ordinal)}"
                )
            if subtype_parts:
                subtype_detail = " (" + "; ".join(subtype_parts) + ")"
            else:
                subtype_detail = ""
            parts.append(
                f"Note: {len(encoded_categorical_cols)} column(s) are numerically "
                f"encoded but reclassified as categorical for analysis{subtype_detail}."
            )
        else:
            enc_names = ", ".join(encoded_categorical_cols)
            parts.append(
                f"Note: {enc_names} "
                f"{'are' if len(encoded_categorical_cols) > 1 else 'is'} "
                f"numerically encoded but reclassified as categorical for analysis."
            )
    if duplicate_count > 0:
        dup_pct = duplicate_count / max(row_count + duplicate_count, 1) * 100
        parts.append(
            f"{duplicate_count} duplicate row(s) ({dup_pct:.1f}% of original rows) "
            f"were detected and removed automatically before analysis."
        )
    return {
        "title": "Dataset Overview",
        "content": " ".join(parts),
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


def _build_correlation_section(
    eda: EDAResults,
    encoded_categorical_cols: list[str] | None = None,
    ordinal_spearman: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build the correlation summary section.

    Args:
        eda: EDAResults with Pearson correlation matrix.
        encoded_categorical_cols: Columns reclassified as categorical
            (excluded from Pearson matrix).  Used for a transparency note.
        ordinal_spearman: Pre-computed Spearman ρ matrix among ordinal
            encoded-categorical columns (≥3 unique values).  When provided,
            an "Ordinal Inter-Correlation" subsection is appended showing
            high inter-correlations relevant for feature selection.
    """
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

    # Transparency note: reclassified encoded-categorical columns are excluded
    # from the Pearson matrix (they operate on string-cast values post-reclassification).
    if encoded_categorical_cols:
        content += (
            f" Note: {len(encoded_categorical_cols)} column(s) reclassified as "
            f"categorical ({', '.join(encoded_categorical_cols)}) are excluded "
            f"from the Pearson matrix; see Categorical Analysis and "
            f"Feature–Target Associations for their association structure."
        )

    # Ordinal inter-correlation subsection (Spearman ρ — rank correlation,
    # makes no equal-interval assumption; appropriate for ordinal data).
    if ordinal_spearman:
        ord_cols = list(ordinal_spearman.keys())
        # Collect notable off-diagonal pairs (|ρ| ≥ 0.5)
        notable_pairs: list[tuple[str, str, float]] = []
        for i, col_a in enumerate(ord_cols):
            for col_b in ord_cols[i + 1:]:
                rho = ordinal_spearman.get(col_a, {}).get(col_b, 0.0)
                if abs(rho) >= 0.5:
                    notable_pairs.append((col_a, col_b, rho))
        notable_pairs.sort(key=lambda t: abs(t[2]), reverse=True)

        content += (
            f"\n\nOrdinal Inter-Correlation (Spearman ρ): "
            f"Computed for {len(ord_cols)} ordinal feature(s) "
            f"({', '.join(ord_cols)})."
        )
        if notable_pairs:
            pair_strs = [f"{a} ↔ {b} (ρ={v:.3f})" for a, b, v in notable_pairs]
            content += (
                f" Notable pairs (|ρ| ≥ 0.5): {'; '.join(pair_strs)}. "
                f"High inter-correlations among ordinal features suggest "
                f"multicollinearity — consider dimensionality reduction "
                f"or sequential feature selection."
            )
        else:
            content += " No pairs exceed |ρ| ≥ 0.5."

    return {"title": "Correlation Analysis", "content": content}


def _compute_target_analysis_fallback(ti: dict, data_raw: str | None) -> dict:
    """Build a full target analysis dict from target_info + raw data JSON.

    Produces the same schema as eda_tools.target_analysis(), used as a
    fallback when the LLM agent skips calling that tool.  Computes
    per_class_feature_stats on the fly from data_json so the report always
    includes per-class statistics regardless of LLM non-determinism.
    """
    col = ti.get("column", "")
    ptype = ti.get("problem_type", "unsupervised")
    total = sum(ti.get("class_counts", {}).values()) or 1

    result: dict = {
        "column": col,
        "problem_type": ptype,
        "n_classes": ti.get("n_classes", 0),
        "imbalance_ratio": ti.get("imbalance_ratio", 1.0),
        "class_distribution": {
            k: {"count": v, "pct": round(v / total * 100, 2)}
            for k, v in ti.get("class_counts", {}).items()
        },
    }

    if data_raw and ptype == "classification" and col:
        try:
            df = pd.DataFrame(json.loads(data_raw))
            if col in df.columns:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                feature_num = [c for c in num_cols if c != col]
                if feature_num:
                    per_class_stats: dict = {}
                    for cls_val, group_df in df.groupby(col):
                        cls_stats: dict = {}
                        for feat in feature_num:
                            series = group_df[feat].dropna()
                            cls_stats[feat] = {
                                "mean": round(float(series.mean()), 4),
                                "std": round(float(series.std()), 4),
                            }
                        per_class_stats[str(cls_val)] = cls_stats
                    result["per_class_feature_stats"] = per_class_stats
        except Exception:
            pass  # silently skip — per-class stats are enrichment, not critical

    return result


def _build_target_section(target_analysis_data: dict) -> dict[str, Any]:
    """Build the Target Variable Analysis report section."""
    ptype = target_analysis_data.get("problem_type", "unsupervised")
    col = target_analysis_data.get("column", "")

    if ptype == "unsupervised" or not col:
        return {
            "title": "Target Variable Analysis",
            "content": "No target variable identified — unsupervised analysis.",
        }

    paragraphs: list[str] = []

    if ptype == "classification":
        n_cls = target_analysis_data.get("n_classes", 0)
        ratio = target_analysis_data.get("imbalance_ratio", 1.0)
        dist = target_analysis_data.get("class_distribution", {})

        dist_parts = [f"{k}: {v['count']} ({v['pct']:.1f}%)" for k, v in dist.items()]
        paragraphs.append(
            f"Target variable '{col}' is a classification target with "
            f"{n_cls} classes."
        )
        if dist_parts:
            paragraphs.append(f"Class distribution: {', '.join(dist_parts)}.")

        if ratio <= 1.5:
            paragraphs.append(
                f"Classes are well-balanced (imbalance ratio: {ratio:.1f}:1)."
            )
        elif ratio <= 3:
            paragraphs.append(
                f"Moderate class imbalance detected (ratio: {ratio:.1f}:1). "
                f"Stratified splitting recommended."
            )
        else:
            paragraphs.append(
                f"Significant class imbalance detected (ratio: {ratio:.1f}:1). "
                f"Consider SMOTE, class weights, or undersampling."
            )

        # Per-class feature stats
        per_class = target_analysis_data.get("per_class_feature_stats", {})
        if per_class:
            paragraphs.append("Per-class feature statistics (mean ± std):")
            for cls_val, feat_stats in per_class.items():
                parts = [
                    f"{f}: {s['mean']:.2f}±{s['std']:.2f}"
                    for f, s in list(feat_stats.items())[:5]
                ]
                paragraphs.append(f"  {cls_val}: {', '.join(parts)}")

    elif ptype == "regression":
        stats = target_analysis_data.get("target_stats", {})
        paragraphs.append(
            f"Target variable '{col}' is a regression target."
        )
        if stats:
            paragraphs.append(
                f"Distribution: mean={stats.get('mean', 0):.2f}, "
                f"median={stats.get('median', 0):.2f}, "
                f"std={stats.get('std', 0):.2f}, "
                f"skewness={stats.get('skewness', 0):.2f}."
            )

        top_feats = target_analysis_data.get("top_correlated_features", [])
        if top_feats:
            parts = [f"{f['feature']} (r={f['correlation']:.3f})" for f in top_feats]
            paragraphs.append(f"Top correlated features: {', '.join(parts)}.")

    return {
        "title": "Target Variable Analysis",
        "content": " ".join(paragraphs),
    }


def _build_statistical_analysis_section(
    eda: EDAResults,
    critic: "CriticReport | None" = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Build an interpretive statistical analysis section.

    Analyses distribution shape, spread, and central-tendency insights
    derived deterministically from ``describe_stats`` output.

    Args:
        eda:    Populated EDAResults containing describe_stats output.
        critic: Optional CriticReport.  When supplied, columns flagged LOW
                by the ``outliers_iqr`` rule are separated from genuinely
                outlier-affected columns so the emitted text does not
                contradict the Data Quality Assessment section.
                Defaults to None (backward-compatible: all columns that
                breach IQR fences are reported uniformly).
        target_column: Optional target variable name.  When supplied, the
                target is excluded from ``narrow_iqr_cols`` and
                ``high_cv_cols`` diagnostics — its distributional
                properties are Bernoulli invariants, not data quality
                issues.
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

    # Exclude target column — its distributional properties (IQR=0 for
    # imbalanced binary, CV>1 for low-prevalence) are Bernoulli invariants,
    # not data quality issues.
    if target_column:
        high_cv_cols = [c for c in high_cv_cols if c != target_column]
        narrow_iqr_cols = [c for c in narrow_iqr_cols if c != target_column]

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

    # Cross-reference critic: columns the rule engine already rated LOW for
    # outliers_iqr have an unreliable IQR (distribution shape artefact), NOT
    # genuine outliers.  Separate them so the static text does not contradict
    # the Data Quality Assessment section which says "do not remove".
    iqr_unreliable: set[str] = set()
    if critic is not None:
        for flag in critic.flags:
            if (
                flag.rule == "outliers_iqr"
                and flag.severity == "LOW"
                and flag.column
            ):
                iqr_unreliable.add(flag.column)

    genuine_outlier_cols = [c for c in potential_outlier_cols if c not in iqr_unreliable]
    unreliable_iqr_cols  = [c for c in potential_outlier_cols if c in iqr_unreliable]

    if genuine_outlier_cols:
        paragraphs.append(
            f"Potential outliers detected (values beyond 1.5×IQR fences) in: "
            f"{', '.join(genuine_outlier_cols)}. Outlier treatment "
            f"(winsorization, capping, or removal) should be considered depending "
            f"on the downstream modeling objective."
        )

    if unreliable_iqr_cols:
        paragraphs.append(
            f"IQR method unreliable for: {', '.join(unreliable_iqr_cols)} — "
            f"the flagged rate exceeds 20%, which indicates a distribution-shape "
            f"artefact rather than genuine outliers. "
            f"See Data Quality Assessment for the full statistical explanation "
            f"and recommended action."
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
            worst_r = max(r for _, _, r in redundant_pairs)
            vif_worst = round(1 / (1 - min(worst_r, 0.9999) ** 2), 1)
            recommendations.append(
                f"FEATURE ENGINEERING — {len(redundant_pairs)} near-redundant feature "
                f"pair(s) detected (|r|>0.90). For linear models, each correlated pair "
                f"inflates VIF (worst pair: ~{vif_worst:.0f}\u00d7) \u2014 retain only one feature "
                f"per pair to avoid unstable coefficients. Tree-based models (XGBoost, "
                f"LightGBM, RandomForest) are unaffected by collinearity and can safely "
                f"use all features."
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


# ---------------------------------------------------------------------------
# Trustworthiness Assessment (from hallucination evaluation)
# ---------------------------------------------------------------------------

_TRUST_THRESHOLDS = [
    (0.3, "High Trustworthiness",
     "The AI-generated commentary is well-grounded in the source data. "
     "No significant bias, toxicity, or hallucination detected. "
     "Statistical observations, interpretations, and recommendations are "
     "consistent with the deterministic fact sheet produced by the pipeline."),
    (0.7, "Medium Trustworthiness",
     "Some claims in the AI-generated commentary may not be fully supported "
     "by the source data, or minor bias/toxicity concerns were detected. "
     "Readers should cross-check flagged sections against "
     "the raw statistics before relying on them for decisions."),
    (1.1, "Low Trustworthiness",
     "Significant issues detected in the AI-generated commentary "
     "(hallucination, bias, or toxicity). "
     "The generated text contains statements that contradict or go beyond the "
     "source data, or exhibits problematic bias/toxicity. "
     "Treat all AI-generated interpretations with caution and "
     "verify against the deterministic analysis sections above."),
]


# Map eval evaluation types to human-readable labels for the report.
_EVAL_TYPE_LABELS = {
    "hallucination": "hallucination",
    "bias_detection": "bias",
    "toxicity_detection": "toxicity",
}


def _build_trustworthiness_section(
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a report section from the comprehensive evaluation result.

    Parameters
    ----------
    eval_result : dict
        Keys: verdict, score, evaluation, classification, explanation
        (as returned by ``_run_comprehensive_eval``).

    Returns
    -------
    dict
        A Findings-compatible section dict with title + content.
    """
    score = float(eval_result.get("score", 0.0))
    verdict = eval_result.get("verdict", "unknown")
    evaluation = eval_result.get("evaluation", "none")
    classification = eval_result.get("classification", "none")
    explanation = eval_result.get("explanation", "")

    # Map score to trust level
    trust_label = "Low Trustworthiness"
    trust_description = _TRUST_THRESHOLDS[-1][2]
    for threshold, label, desc in _TRUST_THRESHOLDS:
        if score < threshold:
            trust_label = label
            trust_description = desc
            break

    # Determine the issue type label for display
    eval_label = _EVAL_TYPE_LABELS.get(evaluation, evaluation)
    if verdict == "yes":
        verdict_text = f"issue detected ({eval_label})"
    else:
        verdict_text = "no issues detected"

    lines: list[str] = [
        f"Assessment: {trust_label}",
        "",
        trust_description,
        "",
        "Evaluated scope: Hallucination + Bias + Toxicity "
        "(combined evaluation via openlit.evals.All)",
        "",
        f"Overall score: {score:.2f} (0.00 = fully grounded, "
        f"1.00 = highest risk)",
        f"Overall verdict: {verdict_text}",
    ]

    # Per-type breakdown: always show all three evaluation categories
    _EVAL_TYPES = ["hallucination", "bias_detection", "toxicity_detection"]
    _EVAL_DISPLAY = {
        "hallucination": "Hallucination",
        "bias_detection": "Bias",
        "toxicity_detection": "Toxicity",
    }
    lines.append("")
    lines.append("Per-type results:")
    for etype in _EVAL_TYPES:
        display_name = _EVAL_DISPLAY[etype]
        if verdict == "yes" and evaluation == etype:
            lines.append(
                f"  {display_name}: ✗ Issue detected "
                f"(score={score:.2f}, classification={classification})"
            )
        else:
            lines.append(f"  {display_name}: ✓ No issues detected")

    if evaluation and evaluation != "none":
        lines.append("")
        lines.append(f"Highest-risk type: {eval_label}")
    if classification and classification != "none":
        lines.append(f"Classification: {classification}")
    if explanation:
        lines.append(f"Judge explanation: {explanation}")

    return {
        "title": "Trustworthiness Assessment",
        "content": "\n".join(lines),
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
# Categorical analysis helpers (W4)
# ---------------------------------------------------------------------------


def _build_categorical_inventory(cat_analysis: CategoricalAnalysis) -> str:
    """Render categorical distributions for the fact sheet.

    Produces a text block with value counts, entropy, and target rates
    per column — the categorical analogue of histogram bin data.
    """
    if not cat_analysis.columns:
        return "  No categorical columns detected."
    lines: list[str] = []
    target_col = cat_analysis.target_column
    for col, stats in cat_analysis.columns.items():
        label = " (TARGET)" if col == target_col else ""
        lines.append(
            f"\n{col}{label} — cardinality={stats.cardinality}, "
            f"entropy={stats.entropy_bits:.4f} bits, "
            f"rare_values(<0.5%)={stats.rare_count}"
        )
        for v in stats.top_values:
            parts = [f"    '{v['value']}': {v['count']} ({v['pct']:.1f}%)"]
            if v.get("is_rare"):
                parts.append(" [RARE]")
            if v.get("target_rates"):
                rates = ", ".join(
                    f"{k}={r:.1f}%" for k, r in v["target_rates"].items()
                )
                parts.append(f"  target_rates: {rates}")
                # Wilson CI reliability annotation (W-D fix).
                # Maximum 95% CI half-width at p=0.5 (conservative bound):
                #   margin = 1.96 * sqrt(0.25 / n)  [in percentage points]
                # This flags rare-category rate estimates that look precise but
                # carry high uncertainty — e.g. Germany n=21: ±21pp means the
                # true rate could plausibly range over a 42pp window.
                # Scoped to n < 100 only; n >= 100 produces no annotation to
                # keep the fact sheet clean for high-frequency categories.
                n_cat = int(v.get("count", 0))
                if 0 < n_cat < 100:
                    margin = round(1.96 * np.sqrt(0.25 / n_cat) * 100, 0)
                    level = "VERY_LOW_N" if n_cat < 30 else "LOW_N"
                    caution = "rates unreliable" if n_cat < 30 else "interpret with caution"
                    parts.append(f"  [{level}: n={n_cat}, ±{margin:.0f}pp CI — {caution}]")
            lines.append("".join(parts))
        if stats.more_values > 0:
            lines.append(f"    ... and {stats.more_values} more value(s)")
    return "\n".join(lines)


def _build_categorical_section(
    cat_analysis: CategoricalAnalysis,
) -> dict[str, Any]:
    """Build the Categorical Analysis report section (deterministic).

    Produces summary sentences about cardinality, entropy, rare values,
    and — when a classification target is present — discriminative
    categories (highest variation in target rates).
    """
    if not cat_analysis.columns:
        return {
            "title": "Categorical Analysis",
            "content": "No categorical columns detected in the dataset.",
        }

    paragraphs: list[str] = []
    n_cols = len(cat_analysis.columns)
    paragraphs.append(
        f"Categorical analysis was performed on {n_cols} feature(s)."
    )

    # Cardinality summary
    high_card = [(c, s.cardinality) for c, s in cat_analysis.columns.items()
                 if s.cardinality > 20]
    low_card = [(c, s.cardinality) for c, s in cat_analysis.columns.items()
                if s.cardinality <= 2]
    if high_card:
        parts = [f"{c} ({n} values)" for c, n in
                 sorted(high_card, key=lambda x: x[1], reverse=True)]
        paragraphs.append(
            f"High-cardinality feature(s): {', '.join(parts)}. "
            f"One-hot encoding cost is significant; consider target-encoding, "
            f"frequency-encoding, or grouping rare levels."
        )
    if low_card:
        parts = [f"{c} ({n})" for c, n in low_card]
        paragraphs.append(
            f"Binary/low-cardinality feature(s): {', '.join(parts)}. "
            f"Binary encoding is sufficient."
        )

    # Rare-value summary
    cols_with_rare = [(c, s.rare_count) for c, s in cat_analysis.columns.items()
                      if s.rare_count > 0]
    if cols_with_rare:
        parts = [f"{c} ({n} rare)" for c, n in
                 sorted(cols_with_rare, key=lambda x: x[1], reverse=True)]
        paragraphs.append(
            f"Rare categories (<0.5% frequency) present in: {', '.join(parts)}. "
            f"Consider grouping into 'Other' to reduce noise and encoding cost."
        )

    # Entropy summary (information content)
    sorted_by_entropy = sorted(
        cat_analysis.columns.items(), key=lambda x: x[1].entropy_bits,
        reverse=True,
    )
    if sorted_by_entropy:
        top_ent = sorted_by_entropy[0]
        low_ent = sorted_by_entropy[-1]
        if n_cols > 1:
            paragraphs.append(
                f"Highest information content: {top_ent[0]} "
                f"(entropy={top_ent[1].entropy_bits:.2f} bits). "
                f"Lowest: {low_ent[0]} "
                f"(entropy={low_ent[1].entropy_bits:.2f} bits)."
            )

    # Target-rate variation (discriminative categories)
    if cat_analysis.target_column:
        discriminative: list[tuple[str, str, float, float]] = []
        for col, stats in cat_analysis.columns.items():
            if col == cat_analysis.target_column:
                continue
            # Find the first target class to measure rate variation
            target_rates_all: list[float] = []
            first_cls: str | None = None
            for v in stats.top_values:
                if v.get("target_rates"):
                    if first_cls is None:
                        first_cls = next(iter(v["target_rates"]))
                    rate = v["target_rates"].get(first_cls, 0.0)
                    target_rates_all.append(rate)
            if target_rates_all and first_cls:
                spread = max(target_rates_all) - min(target_rates_all)
                discriminative.append((col, first_cls, spread, max(target_rates_all)))

        if discriminative:
            discriminative.sort(key=lambda x: x[2], reverse=True)
            top_disc = discriminative[:3]
            disc_parts = [
                f"{c} ({spread:.1f}pp range on '{cls}')"
                for c, cls, spread, _ in top_disc
            ]
            paragraphs.append(
                f"Most discriminative feature(s) by target-rate variation: "
                f"{', '.join(disc_parts)}."
            )

            # Per-category target rates for the top-3 most discriminative columns.
            # Cap at 5 values per column to keep output bounded regardless of
            # cardinality; all classes are shown so no information is suppressed.
            for col, _cls, _spread, _ in top_disc:
                col_stats = cat_analysis.columns[col]
                row_parts = []
                for v in col_stats.top_values[:5]:
                    rates = v.get("target_rates")
                    if not rates:
                        continue
                    rate_str = ", ".join(
                        f"{cls}: {r:.1f}%" for cls, r in rates.items()
                    )
                    row_parts.append(f"{v['value']} ({rate_str})")
                if row_parts:
                    paragraphs.append(
                        f"Per-category target rates — {col}: "
                        f"{'; '.join(row_parts)}."
                    )

    return {"title": "Categorical Analysis", "content": " ".join(paragraphs)}


# ---------------------------------------------------------------------------
# Feature–target association helpers (W7)
# ---------------------------------------------------------------------------


def _build_feature_associations_fact_block(fa: FeatureAssociations) -> str:
    """Render the FEATURE–TARGET ASSOCIATIONS block for the fact sheet.

    Produces a compact table so the LLM has exact MI scores, effect sizes,
    and Borda ranks to cite precisely — no estimation or fabrication needed.
    """
    if not fa.rows:
        return "  No feature–target associations available (no target column or tool not run)."

    lines: list[str] = [
        f"  Target: {fa.target_col!r}  |  Task: {fa.task_type}  "
        f"|  Features analysed: {fa.total_features}"
    ]
    if fa.missingness_strategy:
        lines.append(f"  Missingness strategy: {fa.missingness_strategy}")
    if fa.mi_sample_note:
        lines.append(f"  NOTE: {fa.mi_sample_note}")

    header = (
        f"  {'#':<3}  {'Feature':<24}  {'Type':<12}  "
        f"{'MI':>8}  {'MI_rank':>7}  "
        f"{'EffectSize':>10}  {'ES_type':<10}  {'ES_label':<9}  {'ES_rank':>7}  "
        f"{'Borda':>5}  {'p_value':>9}"
    )
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for i, row in enumerate(fa.rows, start=1):
        lines.append(
            f"  {i:<3}  {row.feature:<24}  {row.feature_type:<12}  "
            f"{row.mi_score:>8.5f}  {row.mi_rank:>7}  "
            f"{row.effect_size:>10.4f}  {row.effect_size_type:<10}  "
            f"{row.effect_size_label:<9}  {row.effect_size_rank:>7}  "
            f"{row.borda_score:>5}  {row.p_value:>9.6f}"
        )
    return "\n".join(lines)


def _build_feature_associations_section(
    fa: FeatureAssociations,
) -> dict[str, Any]:
    """Build the Feature–Target Associations report section (deterministic).

    Presents the Borda-ranked top-N features with MI and effect size lenses,
    effect size labels (weak/moderate/strong), and a sampling note when MI
    was estimated on a subsample.
    """
    if not fa.rows:
        return {
            "title": "Feature–Target Associations",
            "content": (
                "No feature–target associations computed "
                "(no supervised target detected or tool not called)."
            ),
        }

    paragraphs: list[str] = []
    paragraphs.append(
        f"Univariate feature–target associations were computed for "
        f"{fa.total_features} feature(s) against target '{fa.target_col}' "
        f"({fa.task_type}). "
        f"Top {len(fa.rows)} features ranked by Borda score "
        f"(borda = MI rank + effect size rank; lower = more important)."
    )

    if fa.missingness_strategy:
        paragraphs.append(
            f"Missingness handling: associations computed on "
            f"{fa.missingness_strategy}."
        )

    if fa.mi_sample_note:
        paragraphs.append(fa.mi_sample_note)

    paragraphs.append(
        "NOTE: MI and effect-size estimates assume missingness is MCAR or MAR. "
        "Features with informative non-response (MNAR) may yield attenuated "
        "or biased association scores."
    )

    # Determine effect size label distribution
    label_counts: dict[str, list[str]] = {"strong": [], "moderate": [], "weak": []}
    for row in fa.rows:
        lbl = row.effect_size_label
        if lbl in label_counts:
            label_counts[lbl].append(row.feature)

    # Report top-3 by Borda
    top3 = fa.rows[:3]
    top3_parts = []
    for row in top3:
        top3_parts.append(
            f"{row.feature} (Borda={row.borda_score}, "
            f"MI={row.mi_score:.4f}, "
            f"{row.effect_size_type}={row.effect_size:.4f} [{row.effect_size_label}])"
        )
    paragraphs.append(f"Top features: {'; '.join(top3_parts)}.")

    # Strong effect size summary
    if label_counts["strong"]:
        paragraphs.append(
            f"Strong effect size (practical significance): "
            f"{', '.join(label_counts['strong'][:5])}."
        )
    if label_counts["moderate"]:
        paragraphs.append(
            f"Moderate effect size: {', '.join(label_counts['moderate'][:5])}."
        )
    if label_counts["weak"]:
        paragraphs.append(
            f"Weak effect size (low practical impact): "
            f"{', '.join(label_counts['weak'][:5])}."
        )

    # --- MI × Effect-Size quadrant analysis ---
    # Each feature is classified into one of four quadrants based on two
    # independent lenses: MI (kNN, captures any dependence including nonlinear)
    # and effect size (eta²/Cramér's V/|Pearson r|, measures linear/group-mean component).
    #
    #   Q1  HIGH MI + STRONG ES  → linear / simple relationship (expected; no alert)
    #   Q2  HIGH MI + WEAK   ES  → NONLINEAR SIGNAL: tree-based models from the start
    #   Q3  LOW  MI + STRONG ES  → SUSPICIOUS ASSOCIATION: outlier / leakage / small-n
    #   Q4  LOW  MI + WEAK   ES  → no relationship (no alert; already in weak ES summary)
    #
    # "High MI" = top half of features by MI rank: mi_rank ≤ ⌈N/2⌉.
    # "High/Low ES" = effect_size_label is "strong" / "weak" ("moderate" is neutral).
    mid_rank = math.ceil(fa.total_features / 2) if fa.total_features > 0 else 1
    nonlinear_features: list[str] = []   # Q2: high MI, weak effect size
    suspicious_features: list[str] = []  # Q3: low MI, strong effect size
    for row in fa.rows:
        high_mi = row.mi_rank <= mid_rank
        if high_mi and row.effect_size_label == "weak":
            nonlinear_features.append(
                f"{row.feature} (MI rank {row.mi_rank}/{fa.total_features}, "
                f"{row.effect_size_type}={row.effect_size:.4f} [weak])"
            )
        elif (not high_mi) and row.effect_size_label == "strong":
            suspicious_features.append(
                f"{row.feature} (MI rank {row.mi_rank}/{fa.total_features}, "
                f"{row.effect_size_type}={row.effect_size:.4f} [strong])"
            )
    if nonlinear_features:
        paragraphs.append(
            f"NONLINEAR SIGNAL -- high MI + weak effect size indicates a nonlinear "
            f"or complex relationship not captured by linear measures. "
            f"Use tree-based models (XGBoost, LightGBM, RandomForest) from the "
            f"start -- linear models (OLS, Logistic, ElasticNet) will underfit. "
            f"Affected: {'; '.join(nonlinear_features[:3])}."
        )
    if suspicious_features:
        paragraphs.append(
            f"SUSPICIOUS ASSOCIATION [red flag] -- low MI + strong effect size. "
            f"Possible causes: (a) outliers inflating eta2/Cohen's d, "
            f"(b) data leakage (feature partially encodes the target), "
            f"(c) small-n instability (wide effect-size confidence intervals). "
            f"Investigate before modeling: check scatter vs target, run a leakage "
            f"audit, and verify with bootstrapped effect-size CIs. "
            f"Affected: {'; '.join(suspicious_features[:3])}."
        )

    return {
        "title": "Feature–Target Associations",
        "content": " ".join(paragraphs),
    }


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
        # Peak / modality detection with dataset-agnostic prominence filter.
        # Design rationale (W-B fix):
        #   avg_bin = expected observations per bin under a uniform distribution.
        #   3σ Poisson floor = sqrt(avg_bin) * PEAK_SIGMA_THRESHOLD:
        #     a peak must rise this high above its taller neighbour to clear
        #     random Poisson noise.  Scales as √N — correct regardless of N.
        #   1% mass floor = n_total * PEAK_MASS_FRACTION:
        #     a peak bin must also represent at least 1% of all observations to
        #     qualify as a meaningful mode (not just a sampling ripple in high-N
        #     datasets where the 3σ bar alone would be numerically small).
        #   max(..., 1.0): degenerate guard so the threshold is ≥ 1 for tiny N.
        #   Prominence of a candidate peak = peak_count − max(left, right):
        #     using the taller neighbour gives a conservative measure of how
        #     much the peak stands above its immediate surroundings.
        n_total = int(counts.sum())
        avg_bin = n_total / max(len(counts), 1)
        prominence_threshold = max(
            np.sqrt(avg_bin) * PEAK_SIGMA_THRESHOLD,
            n_total * PEAK_MASS_FRACTION,
            1.0,
        )
        peaks: list[dict[str, Any]] = []
        for i in range(1, len(counts) - 1):
            if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                prominence = int(counts[i]) - max(int(counts[i - 1]), int(counts[i + 1]))
                if prominence >= prominence_threshold:
                    peaks.append({
                        "bin": f"[{edges[i]:.4f}, {edges[i + 1]:.4f})",
                        "count": int(counts[i]),
                    })
        # Edge peaks (first / last bin) — one-sided prominence
        if len(counts) > 1:
            if counts[0] > counts[1]:
                prominence = int(counts[0]) - int(counts[1])
                if prominence >= prominence_threshold:
                    peaks.insert(0, {
                        "bin": f"[{edges[0]:.4f}, {edges[1]:.4f})",
                        "count": int(counts[0]),
                    })
            if counts[-1] > counts[-2]:
                prominence = int(counts[-1]) - int(counts[-2])
                if prominence >= prominence_threshold:
                    peaks.append({
                        "bin": f"[{edges[-2]:.4f}, {edges[-1]:.4f})",
                        "count": int(counts[-1]),
                    })
        modality = "unimodal" if len(peaks) <= 1 else f"{len(peaks)}-modal"
        lines.append(
            f"  Modality: {modality} ({len(peaks)} peak(s); "
            f"prominence ≥ {prominence_threshold:.0f} obs "
            f"[3\u03c3 Poisson + 1% mass floor, n={n_total}])"
        )
        for p in peaks:
            lines.append(f"  Peak: {p['bin']} at {p['count']} obs")
        # Skewness annotation (W-C fix): provides the LLM with a direct scipy-
        # based distribution shape label inside HISTOGRAM BIN DATA, fulfilling
        # the DISTRIBUTION SHAPES rule in the system message which already
        # references 'skewness / modality annotations' in this block.
        # Thresholds mirror _build_column_stats_block() (Bulmer 1979 / Hair 2010)
        # for consistency: both fact-sheet locations report the same label family.
        # Guards:
        #   series.std() == 0 → constant column → skew is undefined (NaN)
        #   pd.isna(skew_val) → NaN from any other degenerate case
        skew_val = float(series.skew())
        if pd.isna(skew_val) or series.std() == 0:
            skew_label = "undetermined (constant or insufficient data)"
        else:
            abs_s = abs(skew_val)
            direction = "RIGHT" if skew_val > 0 else "LEFT"
            if abs_s < 0.5:
                skew_label = f"approximately symmetric (scipy skewness = {skew_val:.2f})"
            elif abs_s < 1.0:
                skew_label = f"slightly {direction}-SKEWED (scipy skewness = {skew_val:.2f})"
            else:
                skew_label = f"{direction}-SKEWED (scipy skewness = {skew_val:.2f})"
        lines.append(f"  Skewness: {skew_label}")
        # Empty bins (gaps)
        empty = [
            f"[{edges[i]:.4f}, {edges[i + 1]:.4f})"
            for i in range(len(counts))
            if counts[i] == 0
        ]
        if empty:
            lines.append(f"  Empty bins (gaps): {', '.join(empty)}")
        # Zero-inflation annotation: exact row counts for zero-inflated columns.
        # Provides grounding data so the LLM can cite precise non-zero counts
        # rather than estimating from histogram bins (which may group 0+non-zeros
        # in the first bin, leading to hallucinated figures).
        zero_count = int((series == 0).sum())
        nonzero_count = int(len(series) - zero_count)
        zero_pct = zero_count / len(series) * 100 if len(series) > 0 else 0
        if zero_pct >= 40:
            lines.append(
                f"  Zero-inflation: exact zero rows = {zero_count} ({zero_pct:.1f}%); "
                f"non-zero rows = {nonzero_count} ({100 - zero_pct:.1f}%)"
            )
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
            # Skew direction from scipy skewness (Fisher G1, bias-corrected).
            # Thresholds follow Bulmer (1979) and Hair et al. (2010):
            #   |skew| < 0.5  → approximately symmetric
            #   |skew| < 1.0  → slightly skewed (moderate)
            #   |skew| >= 1.0 → highly skewed
            # The transparency suffix "(skewness=X.XX)" is preserved so the LLM
            # fact sheet always shows a numeric value, not just a label.
            # Falls back to the mean/median heuristic only when skewness_scipy is
            # absent (e.g. manually constructed test dicts from older code paths)
            # to preserve backward compatibility without hiding the old bug.
            skewness_scipy = stats.get("skewness_scipy")
            if skewness_scipy is not None:
                abs_s = abs(skewness_scipy)
                direction = "RIGHT" if skewness_scipy > 0 else "LEFT"
                if abs_s < 0.5:
                    skew_dir = f"approximately symmetric (skewness={skewness_scipy:.2f})"
                elif abs_s < 1.0:
                    skew_dir = f"slightly {direction}-SKEWED (skewness={skewness_scipy:.2f})"
                else:
                    skew_dir = f"{direction}-SKEWED (skewness={skewness_scipy:.2f})"
            elif median and median != 0:
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
    from tools._pipeline_state import is_active, load_state, save_state, \
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
    target_info_raw = load_state("target_info")
    target_analysis_raw = load_state("target_analysis")

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
    for key in (
        "plot_histograms", "plot_correlation_heatmap",
        "plot_missing_heatmap", "plot_class_distribution",
    ):
        raw = load_state(key)
        if raw:
            plot_paths.extend(json.loads(raw))

    # --- Build fact sheet ---
    sections: list[str] = ["=== INTERPRETATION CONTEXT (FACT SHEET) ==="]

    # Dataset overview — use DataProfile for authoritative total column count
    # (len(describe) only counts *numerical* columns — describe_stats excludes
    # categoricals, so the DATASET line would misleadingly say e.g. "6 columns"
    # for a 15-column dataset, causing the LLM to omit categorical context).
    first_stats = next(iter(describe.values()), {})
    row_count = int(first_stats.get("count", 0))
    _num_numerical = len(describe)
    _num_categorical = 0
    _total_cols = _num_numerical  # best-effort fallback
    _dtypes_raw_fs = load_state("dtypes_json")
    _schema_raw_fs = load_state("schema_json")
    if _dtypes_raw_fs:
        try:
            from eda_state import DataProfile as _DataProfile
            _dp_fs = _DataProfile.model_validate_json(_dtypes_raw_fs)
            _total_cols = _dp_fs.shape[1]
            _num_categorical = len(_dp_fs.categorical_cols)
            row_count = _dp_fs.shape[0]  # always wins — authoritative raw shape (Strategy C)
        except Exception:
            pass
    elif _schema_raw_fs:
        try:
            from eda_state import DataProfile as _DataProfile
            _dp_fs = _DataProfile.model_validate_json(_schema_raw_fs)
            _total_cols = _dp_fs.shape[1]
            row_count = _dp_fs.shape[0]  # always wins — authoritative raw shape (Strategy C)
        except Exception:
            pass
    sections.append(
        f"\nDATASET: {row_count} rows x {_total_cols} columns "
        f"({_num_numerical} numerical, {_num_categorical} categorical)"
    )
    # Strategy B — explicit numerical anchor so the LLM cannot confuse per-column
    # non-null counts (e.g. 89 complete-case rows) with the full dataset size.
    sections.append(
        f"AUTHORITATIVE_ROW_COUNT = {row_count}  "
        f"\u2190 Use this number verbatim for dataset size in ALL commentary. "
        f"Do NOT subtract missing-row counts to derive a smaller N."
    )
    # Surface duplicate count in fact sheet (W8)
    dup_raw = load_state("duplicate_count")
    if dup_raw:
        try:
            dup_ct = int(dup_raw)
            if dup_ct > 0:
                dup_pct_fs = dup_ct / max(row_count + dup_ct, 1) * 100
                sections.append(
                    f"  Duplicate rows removed before analysis: "
                    f"{dup_ct} ({dup_pct_fs:.1f}% of original rows)"
                )
        except (ValueError, TypeError):
            pass

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

    # Categorical inventory (W4)
    cat_raw = load_state("categorical_analysis")
    if cat_raw:
        cat_analysis = CategoricalAnalysis.model_validate_json(cat_raw)
        sections.append(
            "\nCATEGORICAL INVENTORY (value counts, entropy, target rates):"
        )
        sections.append(_build_categorical_inventory(cat_analysis))
    else:
        sections.append(
            "\nCATEGORICAL INVENTORY: not yet available (analyze_categoricals not run)."
        )

    # Feature–target associations (W7): MI + effect size
    # Fallback: compute deterministically if the LLM skipped calling the tool
    # (mirrors the _compute_target_analysis_fallback pattern).
    fa_raw = load_state("feature_associations")
    if not fa_raw:
        _fa_data_raw = load_state("data_json")
        _fa_ti_raw = load_state("target_info")
        if _fa_data_raw and _fa_ti_raw:
            try:
                from tools.eda_tools import (  # noqa: PLC0415
                    compute_feature_target_associations as _fa_fn,
                )
                _fa_fn(_fa_data_raw, _fa_ti_raw)
                fa_raw = load_state("feature_associations")
                logger.info(
                    "feature_associations computed via fallback in context builder"
                )
            except Exception:
                logger.warning(
                    "feature_associations fallback failed", exc_info=True
                )
    if fa_raw:
        fa = FeatureAssociations.model_validate_json(fa_raw)
        sections.append(
            "\nFEATURE–TARGET ASSOCIATIONS (MI + effect size, Borda ranked):"
        )
        sections.append(_build_feature_associations_fact_block(fa))
    else:
        sections.append(
            "\nFEATURE–TARGET ASSOCIATIONS: not yet available "
            "(compute_feature_target_associations not run)."
        )

    # Target variable analysis
    if target_analysis_raw:
        target_analysis_data = json.loads(target_analysis_raw)
        target_section = _build_target_section(target_analysis_data)
        sections.append("\nTARGET VARIABLE ANALYSIS:")
        sections.append(f"  {target_section['content']}")
    elif target_info_raw:
        ti = json.loads(target_info_raw)
        if ti.get("column"):
            # Use the same fallback helper as assemble_findings so the LLM
            # receives full per-class stats in its context, not just
            # "not yet available".
            ta_data = _compute_target_analysis_fallback(ti, data_raw)
            target_section = _build_target_section(ta_data)
            sections.append("\nTARGET VARIABLE ANALYSIS:")
            sections.append(f"  {target_section['content']}")

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

    # Persist fact sheet for downstream hallucination evaluation
    if is_active():
        save_state("_interpretation_context", fact_sheet)

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


# ---------------------------------------------------------------------------
# Hallucination evaluation (OpenLIT programmatic evals)
# ---------------------------------------------------------------------------

def _run_comprehensive_eval(interpretations_json: str) -> dict[str, Any] | None:
    """Run OpenLIT comprehensive eval (bias + toxicity + hallucination).

    Uses ``openlit.evals.All`` to perform a combined evaluation of LLM-
    generated interpretations against the deterministic fact sheet.

    Non-blocking: logs warnings but never raises. Skipped when OpenLIT
    is disabled or the fact sheet is unavailable.

    Side effect: populates module-level ``_eval_cost_info`` with token counts
    and cost so that ``main._format_cost_summary()`` can include the evaluator
    in the pipeline cost report.

    Returns:
        Dict with keys {verdict, score, evaluation, classification, explanation}
        or None if the eval was skipped or failed.
    """
    from config import OPENLIT_ENABLE, OPENLIT_EVAL_MODEL
    if not OPENLIT_ENABLE:
        return None

    from tools._pipeline_state import is_active, load_state, save_state
    if not is_active():
        return None

    fact_sheet = load_state("_interpretation_context")
    if not fact_sheet:
        logger.debug("Skipping comprehensive eval: no fact sheet in artifact store")
        return None

    try:
        import openlit  # noqa: F811
        import openlit.evals.utils as _evals_utils

        # --- Capture token usage ---
        # openlit.evals.utils.llm_response_openai() returns only the content
        # string and discards response.usage.  We temporarily replace it with
        # an identical function that also records the token counts.
        _orig_llm_fn = _evals_utils.llm_response_openai
        captured_usage: dict[str, Any] = {}

        def _capturing_openai(prompt, model, base_url):
            """Drop-in for llm_response_openai that also captures usage."""
            from openai import OpenAI as _OAI

            client = _OAI(base_url=base_url)
            if model is None:
                model = "gpt-4o-mini"
            resp = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=_evals_utils.JsonOutput,
            )
            if hasattr(resp, "usage") and resp.usage:
                captured_usage["prompt_tokens"] = resp.usage.prompt_tokens
                captured_usage["completion_tokens"] = resp.usage.completion_tokens
                captured_usage["model"] = resp.model
            return resp.choices[0].message.content

        _evals_utils.llm_response_openai = _capturing_openai
        try:
            evals = openlit.evals.All(
                provider="openai",
                model=OPENLIT_EVAL_MODEL,
                collect_metrics=True,
            )
            result = evals.measure(
                prompt="Expert EDA interpretation of dataset based on fact sheet",
                contexts=[fact_sheet],
                text=interpretations_json,
            )
        finally:
            _evals_utils.llm_response_openai = _orig_llm_fn

        logger.info(
            "Comprehensive eval: verdict=%s, score=%.2f, evaluation=%s, "
            "classification=%s",
            result.verdict,
            result.score,
            result.evaluation,
            result.classification,
        )
        if result.verdict == "yes":
            logger.warning(
                "Issue detected (evaluation=%s, score=%.2f): %s",
                result.evaluation,
                result.score,
                result.explanation,
            )

        # --- Compute eval cost and store in module-level dict ---
        if captured_usage:
            pt = captured_usage.get("prompt_tokens", 0)
            ct = captured_usage.get("completion_tokens", 0)
            cost = _compute_eval_cost(OPENLIT_EVAL_MODEL, pt, ct)
            _eval_cost_info.clear()
            _eval_cost_info.update({
                "model": captured_usage.get("model", OPENLIT_EVAL_MODEL),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cost": cost,
            })
            logger.info(
                "Eval cost captured: model=%s, prompt=%d, completion=%d, cost=$%.4f",
                _eval_cost_info["model"], pt, ct, cost,
            )

        eval_dict: dict[str, Any] = {
            "verdict": result.verdict,
            "score": result.score,
            "evaluation": result.evaluation,
            "classification": result.classification,
            "explanation": result.explanation,
        }
        save_state("comprehensive_eval", json.dumps(eval_dict))
        return eval_dict
    except Exception:
        logger.warning("Comprehensive eval failed — skipping", exc_info=True)
        return None


def _compute_eval_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Look up pricing from openlit_pricing.json and compute cost.

    Falls back to 0.0 if the pricing file is missing or the model
    is not listed.
    """
    from pathlib import Path

    pricing_path = Path(__file__).resolve().parent.parent / "openlit_pricing.json"
    try:
        with open(pricing_path, encoding="utf-8") as f:
            pricing = json.load(f)
        p = pricing["chat"][model]
        return (prompt_tokens / 1000) * p["promptPrice"] + \
               (completion_tokens / 1000) * p["completionPrice"]
    except Exception:
        return 0.0


def save_interpretations(
    interpretations_json: Annotated[
        str,
        "JSON string matching the Interpretations schema. "
        "Keys: overview, missing_values, correlation, statistical_analysis, "
        "categorical_analysis, target_variable_analysis, "
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

    # Enforce that recommendations_and_business_implications is substantive.
    # An empty or trivially short string means the LLM omitted PART 1 / PART 2.
    # Return a recoverable error so the LLM retries with the field populated.
    if not interp.recommendations_and_business_implications or \
            len(interp.recommendations_and_business_implications.strip()) < 200:
        return (
            "Error: 'recommendations_and_business_implications' is missing or too short. "
            "This field MUST contain: "
            "PART 1 — a numbered prioritised action plan (ACTION, EXPECTED OUTCOME, "
            "RISK IF SKIPPED for each item, plus monitoring recommendation and "
            "next-step checklist); AND "
            "PART 2 — Business Problem Catalogue: 5-8 business problems each starting "
            "with a BUSINESS QUESTION, classified High/Med/Low with EDA justification, "
            "plus full PROBLEM / METRIC / RECOMMENDATIONS / BUSINESS IMPACT deep-dives "
            "for the TOP 3 HIGH-PROBABILITY problems. "
            "Please call save_interpretations() again with both parts fully populated."
        )

    validated_json = interp.model_dump_json()

    # --- Comprehensive evaluation (bias + toxicity + hallucination, OpenLIT, opt-in) ---
    _run_comprehensive_eval(validated_json)

    save_state("interpretations", validated_json)

    n_plots = len(interp.plot_commentaries)
    n_sections = sum(
        1 for field in (
            interp.overview,
            interp.missing_values,
            interp.correlation,
            interp.statistical_analysis,
            interp.categorical_analysis,
            interp.target_variable_analysis,
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
        # --- Compose EDAResults directly from individual artifacts (W1/W2/W3 fix) ---
        # Resolving via a combined 'eda_results' state-ref silently produced empty
        # fields: when the LLM passed STATE_REF:describe_stats, resolve() returned
        # the raw describe JSON and EDAResults.model_validate_json() succeeded with
        # all default-empty values because the top-level keys (describe/missing/
        # correlation) were absent.  Loading each artifact by its canonical key
        # directly guarantees correct field population regardless of what the LLM
        # passes as eda_results_json.
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
        eda = EDAResults.model_validate({
            "describe": json.loads(desc_raw),
            "missing": json.loads(miss_raw),
            "correlation": json.loads(corr_raw),
        })
        logger.info("EDAResults composed from individual artifacts")

        # Load DataProfile for the overview section — authoritative shape (W1),
        # duplicate_count (W8), categorical_cols (W4), and numerical_cols.
        _shape: tuple[int, int] | None = None
        _duplicate_count: int = 0
        _categorical_cols: list[str] | None = None
        _numerical_cols: list[str] | None = None
        _encoded_categorical_cols: list[str] | None = None
        schema_raw = load_state("schema_json")
        if schema_raw:
            try:
                _dp = DataProfile.model_validate_json(schema_raw)
                _shape = _dp.shape
                _duplicate_count = _dp.duplicate_count
                if _dp.categorical_cols:
                    _categorical_cols = _dp.categorical_cols
                if _dp.numerical_cols:
                    _numerical_cols = _dp.numerical_cols
                if _dp.encoded_categorical_cols:
                    _encoded_categorical_cols = _dp.encoded_categorical_cols
            except Exception:
                pass  # Non-critical — overview falls back to describe
        # Fallback: try dtypes_json if schema_json omitted column lists
        if _categorical_cols is None or _numerical_cols is None:
            _dtypes_raw_af = load_state("dtypes_json")
            if _dtypes_raw_af:
                try:
                    _dp2 = DataProfile.model_validate_json(_dtypes_raw_af)
                    if _categorical_cols is None and _dp2.categorical_cols:
                        _categorical_cols = _dp2.categorical_cols
                    if _numerical_cols is None and _dp2.numerical_cols:
                        _numerical_cols = _dp2.numerical_cols
                    if _encoded_categorical_cols is None and _dp2.encoded_categorical_cols:
                        _encoded_categorical_cols = _dp2.encoded_categorical_cols
                except Exception:
                    pass

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

        # --- Resolve plot_paths from individual visualization artifacts ---
        # Always compose from individual artifact keys when the session is active,
        # mirroring the EDAResults composition pattern above.  The LLM-supplied
        # plot_paths_json parameter is intentionally ignored here: it may be an
        # empty list "[]", a single-artifact reference (e.g. STATE_REF:plot_histograms
        # which would only return histogram paths), or garbage — all of which would
        # silently drop heatmap and class-distribution plots from the report.
        hist = load_state("plot_histograms")
        corr_hm = load_state("plot_correlation_heatmap")
        miss_hm = load_state("plot_missing_heatmap")
        cls_dist = load_state("plot_class_distribution")
        merged: list[str] = []
        if hist:
            merged.extend(json.loads(hist))
        if corr_hm:
            merged.extend(json.loads(corr_hm))
        if miss_hm:
            merged.extend(json.loads(miss_hm))
        if cls_dist:
            merged.extend(json.loads(cls_dist))
        plot_paths = merged
        logger.info("plot_paths composed from %d visualization artifacts", len(plot_paths))
    else:
        eda = EDAResults.model_validate_json(eda_results_json)
        critic = CriticReport.model_validate_json(critic_report_json)
        plot_paths = json.loads(plot_paths_json)
        _shape = None
        _duplicate_count = 0
        _categorical_cols = None
        _numerical_cols = None
        _encoded_categorical_cols = None

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
    target_plot_paths: list[str] = []
    for pp in plot_paths:
        stem = Path(pp).stem
        if stem.startswith("hist_"):
            hist_paths.append(pp)
        elif stem == "correlation_heatmap":
            corr_heatmap_paths.append(pp)
        elif stem == "missing_heatmap":
            missing_heatmap_paths.append(pp)
        elif stem in ("class_distribution", "target_distribution"):
            target_plot_paths.append(pp)

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

    # Load encoded-categorical subtypes for overview section (F3 fix)
    _encoded_categorical_subtypes: dict[str, str] | None = None
    if is_active():
        _subtypes_raw = load_state("reclassified_subtypes")
        if _subtypes_raw:
            try:
                _encoded_categorical_subtypes = json.loads(_subtypes_raw)
            except (json.JSONDecodeError, TypeError):
                pass

    # Build sections in report order (Option A: plots inline in parent sections)
    overview = _enrich(
        _build_overview_section(
            eda, shape=_shape, duplicate_count=_duplicate_count,
            categorical_cols=_categorical_cols,
            numerical_cols=_numerical_cols,
            encoded_categorical_cols=_encoded_categorical_cols,
            encoded_categorical_subtypes=_encoded_categorical_subtypes,
        ),
        "overview",
    )
    missing = _enrich(
        _build_missing_section(eda), "missing_values",
        paired_plots=missing_heatmap_paths or None,
    )
    # Resolve target column name for downstream sections (F2 safety fix)
    _target_column: str | None = None
    if is_active():
        _ti_raw = load_state("target_info")
        if _ti_raw:
            try:
                _target_column = json.loads(_ti_raw).get("column")
            except Exception:
                pass

    # Compute Spearman ρ among ordinal encoded-categorical columns (F1 fix).
    # Rank correlation makes no equal-interval assumption — appropriate for
    # ordinal data.  Only columns with subtype "ordinal" AND ≥3 unique values
    # contribute meaningful inter-correlations (nominal columns have no rank
    # ordering; binary columns yield degenerate rank ties).
    _ordinal_spearman: dict[str, dict[str, float]] | None = None
    if _encoded_categorical_cols and is_active():
        _data_raw_sp = load_state("data_json")
        if _data_raw_sp:
            try:
                _df_sp = pd.DataFrame(json.loads(_data_raw_sp))
                if _encoded_categorical_subtypes:
                    # Subtypes known: restrict to ordinal only.
                    _ord_cols = [
                        c for c in _encoded_categorical_cols
                        if c in _df_sp.columns
                        and _df_sp[c].nunique(dropna=True) >= 3
                        and _encoded_categorical_subtypes.get(c) == "ordinal"
                    ]
                else:
                    # Subtypes unavailable (--categoricals flag / no LLM):
                    # fall back to all columns with ≥3 unique values.
                    _ord_cols = [
                        c for c in _encoded_categorical_cols
                        if c in _df_sp.columns
                        and _df_sp[c].nunique(dropna=True) >= 3
                    ]
                if len(_ord_cols) >= 2:
                    # Columns are string-cast: convert back to numeric for Spearman
                    _df_ord = _df_sp[_ord_cols].apply(pd.to_numeric, errors="coerce")
                    _sp_matrix = _df_ord.corr(method="spearman")
                    _ordinal_spearman = {
                        col: {
                            c2: round(float(v), 4)
                            for c2, v in row.items()
                        }
                        for col, row in _sp_matrix.to_dict().items()
                    }
            except Exception:
                logger.warning("Spearman ordinal inter-correlation failed", exc_info=True)

    correlation = _enrich(
        _build_correlation_section(
            eda,
            encoded_categorical_cols=_encoded_categorical_cols,
            ordinal_spearman=_ordinal_spearman,
        ),
        "correlation",
        paired_plots=corr_heatmap_paths or None,
    )
    statistical = _enrich(
        _build_statistical_analysis_section(eda, critic, target_column=_target_column),
        "statistical_analysis",
        paired_plots=hist_paths or None,
    )

    # Categorical analysis section (W4) — between Statistical and Target
    categorical_sec: dict[str, Any] | None = None
    if is_active():
        cat_raw = load_state("categorical_analysis")
        if cat_raw:
            cat_analysis = CategoricalAnalysis.model_validate_json(cat_raw)
            categorical_sec = _enrich(
                _build_categorical_section(cat_analysis),
                "categorical_analysis",
            )

    # Feature–target associations section (W7) — after categorical, before target
    # Fallback: if prepare_interpretation_context() somehow ran without the
    # artifact (e.g. direct assemble call), compute it deterministically here.
    feature_assoc_sec: dict[str, Any] | None = None
    if is_active():
        fa_raw = load_state("feature_associations")
        if not fa_raw:
            _fa_d = load_state("data_json")
            _fa_t = load_state("target_info")
            if _fa_d and _fa_t:
                try:
                    from tools.eda_tools import (  # noqa: PLC0415
                        compute_feature_target_associations as _fa_fn2,
                    )
                    _fa_fn2(_fa_d, _fa_t)
                    fa_raw = load_state("feature_associations")
                    logger.info(
                        "feature_associations computed via fallback in assemble_findings"
                    )
                except Exception:
                    logger.warning(
                        "feature_associations fallback (assemble) failed", exc_info=True
                    )
        if fa_raw:
            fa = FeatureAssociations.model_validate_json(fa_raw)
            feature_assoc_sec = _enrich(
                _build_feature_associations_section(fa),
                "feature_associations",
            )

    quality = _enrich(
        _build_quality_section(critic, is_final), "quality_assessment",
    )

    # Target variable section (if target was detected)
    target_sec: dict[str, Any] | None = None
    if is_active():
        ta_raw = load_state("target_analysis")
        if ta_raw:
            ta_data = json.loads(ta_raw)
            target_sec = _build_target_section(ta_data)
        else:
            # Fallback: build full target section from target_info + data_json.
            # _compute_target_analysis_fallback computes per_class_feature_stats
            # on the fly so the report is complete even when the LLM skipped
            # calling target_analysis() (known non-determinism).
            ti_raw = load_state("target_info")
            if ti_raw:
                ti = json.loads(ti_raw)
                if ti.get("column"):
                    data_raw_fb = load_state("data_json")
                    ta_data = _compute_target_analysis_fallback(ti, data_raw_fb)
                    target_sec = _build_target_section(ta_data)
        if target_sec is not None:
            target_sec = _enrich(
                target_sec, "target_variable_analysis",
                paired_plots=target_plot_paths or None,
            )

    # Conclusions & Recommendations: LLM replaces deterministic when available
    conclusions = _build_conclusions_section(eda, critic)
    if interp and interp.conclusions:
        conclusions["content"] = interp.conclusions
    recommendations = _build_recommendations_section(eda, critic)
    if interp and interp.recommendations_and_business_implications:
        recommendations["content"] = interp.recommendations_and_business_implications

    # --- Trustworthiness Assessment (from comprehensive eval) ---
    trust_sec: dict[str, Any] | None = None
    if is_active():
        eval_raw = load_state("comprehensive_eval") or load_state("hallucination_eval")
        if eval_raw:
            try:
                trust_sec = _build_trustworthiness_section(json.loads(eval_raw))
                logger.info("Trustworthiness section added from comprehensive eval")
            except Exception:
                logger.warning(
                    "Failed to build trustworthiness section — skipping",
                    exc_info=True,
                )

    sections: list[dict[str, Any]] = [
        overview,
        missing,
        correlation,
        statistical,
    ]
    if categorical_sec is not None:
        sections.append(categorical_sec)
    if feature_assoc_sec is not None:
        sections.append(feature_assoc_sec)
    if target_sec is not None:
        sections.append(target_sec)
    sections.extend([
        quality,
        conclusions,
        recommendations,
    ])
    if trust_sec is not None:
        sections.append(trust_sec)

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
