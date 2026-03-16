"""
tools/eda_tools.py — Descriptive statistics, missingness, and correlation analysis.

Architecture Reference: architecture.md § 4.3, § 12.1

Public AG2-facing functions:
  - describe_stats(data_json: str) -> str
  - missing_analysis(data_json: str) -> str
  - correlation_matrix(data_json: str) -> str

Design:
  - Zero AG2 imports. Zero agent references. Pure Python.
  - Each function accepts/returns JSON strings (the AG2 tool contract).
  - Pydantic sub-models from eda_state.py validate outputs:
      * describe_stats  → validated through EDAResults(describe=...)
      * missing_analysis → validated through MissingInfo(...)
      * correlation_matrix → validated through EDAResults(correlation=...)

AG2 Version: 0.10.3
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

import pandas as pd

from eda_state import CategoricalAnalysis, CategoricalStats, EDAResults, MissingInfo, TargetInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AG2-facing public functions (flat callables)
# ---------------------------------------------------------------------------

def describe_stats(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Compute descriptive statistics: central tendency, spread, percentiles.

    Includes all column types (numerical and categorical).
    NaN values in the describe output are serialized as JSON null.

    Returns:
        JSON string of a dict mapping column names to stat dicts.
        Validated through EDAResults(describe=...) before return.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))

    if df.empty or df.columns.size == 0:
        return json.dumps({})

    desc_df = df.describe(include="all")

    # pandas to_json handles NaN → null correctly
    desc_dict = json.loads(desc_df.to_json())

    # Validate structure through Pydantic sub-model
    EDAResults(describe=desc_dict)

    logger.info(
        "Descriptive statistics computed for %d columns (%d stats each)",
        len(desc_dict),
        len(next(iter(desc_dict.values()), {})),
    )
    result = json.dumps(desc_dict)

    if is_active():
        save_state("describe_stats", result)
        n_cols = len(desc_dict)
        n_stats = len(next(iter(desc_dict.values()), {}))
        return (
            f"Computed descriptive statistics for {n_cols} columns "
            f"({n_stats} stats each). "
            f"Reference: {STATE_REF_PREFIX}describe_stats"
        )
    return result


def missing_analysis(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Compute per-column and dataset-level missing value percentages.

    Returns:
        JSON string of a MissingInfo model (per_column, total_pct).
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))

    per_column = (df.isnull().mean() * 100).round(2).to_dict()

    total_cells = df.shape[0] * df.shape[1]
    total_pct = round(
        df.isnull().sum().sum() / max(total_cells, 1) * 100, 2
    )

    info = MissingInfo(per_column=per_column, total_pct=total_pct)

    logger.info(
        "Missing analysis: %.2f%% total, %d columns checked",
        info.total_pct,
        len(info.per_column),
    )
    result = info.model_dump_json()

    if is_active():
        save_state("missing_analysis", result)
        return (
            f"Missing analysis complete: {info.total_pct:.2f}% total missing "
            f"across {len(info.per_column)} columns. "
            f"Reference: {STATE_REF_PREFIX}missing_analysis"
        )
    return result


def correlation_matrix(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Compute Pearson correlation matrix for numerical columns only.

    Non-numerical columns are excluded. If no numerical columns exist,
    returns an empty dict.

    Returns:
        JSON string of a nested dict {col: {col: corr_value, ...}, ...}.
        Validated through EDAResults(correlation=...) before return.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))
    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        corr_dict: dict = {}
    else:
        corr_df = num_df.corr(method="pearson").round(4)
        # pandas to_json handles NaN → null correctly
        corr_dict = json.loads(corr_df.to_json())

    # Validate structure through Pydantic sub-model
    EDAResults(correlation=corr_dict)

    logger.info(
        "Correlation matrix computed: %d × %d numerical columns",
        len(corr_dict),
        len(corr_dict),
    )
    result = json.dumps(corr_dict)

    if is_active():
        save_state("correlation_matrix", result)
        return (
            f"Correlation matrix computed for {len(corr_dict)} numerical columns. "
            f"Reference: {STATE_REF_PREFIX}correlation_matrix"
        )
    return result


# ---------------------------------------------------------------------------
# Target variable analysis
# ---------------------------------------------------------------------------

def target_analysis(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
    target_info_json: Annotated[str, "JSON string of TargetInfo from detect_target()"],
) -> str:
    """
    AG2 tool entry point.
    Analyse the target variable in the context of the full dataset.

    For classification:
      - Class value counts + percentages
      - Imbalance ratio
      - Per-class feature stats (mean, std for each numerical column, grouped by target)
    For regression:
      - Target distribution stats (mean, median, std, skewness, kurtosis)
      - Feature-target Pearson correlations
      - Top 3 most correlated features

    If target_info has no target column (unsupervised), returns an empty dict.

    Returns:
        JSON string with target analysis results.
    """
    import numpy as np

    # Artifact store: resolve inputs
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")
        target_info_json = resolve(target_info_json, "target_info")

    df = pd.DataFrame(json.loads(data_json))
    target_info = TargetInfo.model_validate_json(target_info_json)

    if target_info.column is None or target_info.column not in df.columns:
        result = json.dumps({"problem_type": "unsupervised"})
        if is_active():
            save_state("target_analysis", result)
            return (
                f"No target column — skipping target analysis. "
                f"Reference: {STATE_REF_PREFIX}target_analysis"
            )
        return result

    col = target_info.column
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude the target itself from feature columns if it's numerical
    feature_num = [c for c in num_cols if c != col]

    analysis: dict = {
        "column": col,
        "problem_type": target_info.problem_type,
    }

    if target_info.problem_type == "classification":
        # Class counts + percentages
        counts = df[col].value_counts()
        total = len(df)
        class_detail = {}
        for cls_val, cnt in counts.items():
            class_detail[str(cls_val)] = {
                "count": int(cnt),
                "pct": round(cnt / total * 100, 2),
            }
        analysis["class_distribution"] = class_detail
        analysis["n_classes"] = target_info.n_classes
        analysis["imbalance_ratio"] = target_info.imbalance_ratio

        # Per-class feature stats (group-by target)
        if feature_num:
            per_class_stats: dict = {}
            grouped = df.groupby(col)
            for cls_val, group_df in grouped:
                cls_stats: dict = {}
                for feat in feature_num:
                    series = group_df[feat].dropna()
                    cls_stats[feat] = {
                        "mean": round(float(series.mean()), 4),
                        "std": round(float(series.std()), 4),
                    }
                per_class_stats[str(cls_val)] = cls_stats
            analysis["per_class_feature_stats"] = per_class_stats

    elif target_info.problem_type == "regression":
        target_series = df[col].dropna()
        analysis["target_stats"] = {
            "mean": round(float(target_series.mean()), 4),
            "median": round(float(target_series.median()), 4),
            "std": round(float(target_series.std()), 4),
            "skewness": round(float(target_series.skew()), 4),
            "kurtosis": round(float(target_series.kurtosis()), 4),
            "min": round(float(target_series.min()), 4),
            "max": round(float(target_series.max()), 4),
        }

        # Feature-target correlations
        if feature_num:
            corrs = {}
            for feat in feature_num:
                valid = df[[feat, col]].dropna()
                if len(valid) > 1:
                    r = float(np.corrcoef(valid[feat], valid[col])[0, 1])
                    corrs[feat] = round(r, 4) if not np.isnan(r) else 0.0
            analysis["feature_target_correlations"] = corrs

            # Top 3 most correlated
            if corrs:
                sorted_corrs = sorted(
                    corrs.items(), key=lambda x: abs(x[1]), reverse=True,
                )
                analysis["top_correlated_features"] = [
                    {"feature": f, "correlation": r}
                    for f, r in sorted_corrs[:3]
                ]

    logger.info(
        "Target analysis complete: column='%s', type=%s",
        col, target_info.problem_type,
    )
    result = json.dumps(analysis)

    if is_active():
        save_state("target_analysis", result)
        return (
            f"Target analysis complete for '{col}' ({target_info.problem_type}). "
            f"Reference: {STATE_REF_PREFIX}target_analysis"
        )
    return result


# ---------------------------------------------------------------------------
# Categorical analysis (W4)
# ---------------------------------------------------------------------------

_RARE_THRESHOLD = 0.005  # < 0.5% — matches RareCategoryRule in critic_rules.py
_TOP_N = 10


def analyze_categoricals(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
    target_info_json: Annotated[str, "JSON string of TargetInfo from detect_target()"],
) -> str:
    """
    AG2 tool entry point.
    Compute categorical distributions: value counts (top-N), cardinality,
    Shannon entropy, rare-category count (<0.5%), and target rate per
    category (for classification targets).

    Returns:
        JSON string of a CategoricalAnalysis model.
    """
    import math

    from tools._pipeline_state import is_active, resolve, load_state, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")
        target_info_json = resolve(target_info_json, "target_info")

    df = pd.DataFrame(json.loads(data_json))

    # Determine target info
    target_info = TargetInfo.model_validate_json(target_info_json)
    target_col = target_info.column if target_info.column and target_info.column in df.columns else None
    is_classification = target_col is not None and target_info.problem_type == "classification"

    # Determine categorical columns — prefer DataProfile.categorical_cols.
    # NOTE: infer_dtypes() saves under "dtypes_json" (not "schema_json");
    # schema_json (validate_schema) does NOT populate categorical_cols.
    cat_cols: list[str] | None = None
    if is_active():
        dtypes_raw = load_state("dtypes_json")
        if dtypes_raw:
            from eda_state import DataProfile
            try:
                dp = DataProfile.model_validate_json(dtypes_raw)
                if dp.categorical_cols:          # only use if non-empty
                    cat_cols = dp.categorical_cols
            except Exception:
                pass
    if not cat_cols:                             # None OR empty list → fallback
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        empty = CategoricalAnalysis(target_column=target_col, top_n=_TOP_N)
        result = empty.model_dump_json()
        if is_active():
            save_state("categorical_analysis", result)
            return (
                f"No categorical columns detected. "
                f"Reference: {STATE_REF_PREFIX}categorical_analysis"
            )
        return result

    columns: dict[str, CategoricalStats] = {}
    n_rows = len(df)

    for col in cat_cols:
        series = df[col].dropna()
        if series.empty:
            columns[col] = CategoricalStats()
            continue

        vc = series.value_counts()
        cardinality = len(vc)

        # Shannon entropy (bits)
        probs = vc.values / vc.values.sum()
        entropy_bits = float(-sum(p * math.log2(p) for p in probs if p > 0))

        # Rare values (< 0.5%)
        freq_pct = vc / len(series)
        rare_mask = freq_pct < _RARE_THRESHOLD
        rare_count = int(rare_mask.sum())

        # Top-N value details
        top_values: list[dict] = []
        show_n = min(_TOP_N, cardinality)
        for val in vc.index[:show_n]:
            count = int(vc[val])
            pct = round(count / len(series) * 100, 2)
            is_rare = bool(freq_pct[val] < _RARE_THRESHOLD)
            entry: dict = {
                "value": str(val),
                "count": count,
                "pct": pct,
                "is_rare": is_rare,
            }
            if is_classification:
                target_rates: dict[str, float] = {}
                mask = df[col] == val
                grp = df.loc[mask, target_col].value_counts()
                grp_total = int(grp.sum())
                for cls_val, cls_cnt in grp.items():
                    target_rates[str(cls_val)] = round(
                        int(cls_cnt) / max(grp_total, 1) * 100, 2,
                    )
                entry["target_rates"] = target_rates
            top_values.append(entry)

        more_values = max(0, cardinality - show_n)

        columns[col] = CategoricalStats(
            cardinality=cardinality,
            entropy_bits=round(entropy_bits, 4),
            rare_count=rare_count,
            top_values=top_values,
            more_values=more_values,
        )

    analysis = CategoricalAnalysis(
        columns=columns,
        target_column=target_col,
        top_n=_TOP_N,
    )

    logger.info(
        "Categorical analysis complete: %d columns, target=%s",
        len(columns), target_col or "none",
    )
    result = analysis.model_dump_json()

    if is_active():
        save_state("categorical_analysis", result)
        return (
            f"Categorical analysis complete for {len(columns)} columns. "
            f"Reference: {STATE_REF_PREFIX}categorical_analysis"
        )
    return result
