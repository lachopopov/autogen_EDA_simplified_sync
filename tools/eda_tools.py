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

from eda_state import (
    AssociationRow,
    CategoricalAnalysis,
    CategoricalStats,
    EDAResults,
    FeatureAssociations,
    MissingInfo,
    TargetInfo,
)

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

    # Determine target info.
    # Guard: the LLM may pass '{}' (the documented fallback for missing
    # target_info) even when the artifact actually exists in state.  '{}' is
    # valid JSON so resolve() Tier-2 returns it as-is without hitting Tier-3.
    # If target_column ends up empty after normal resolve AND a session is
    # active, try loading the real artifact directly so that target rates and
    # the discriminative-category summary are properly computed (W4 fix).
    target_info = TargetInfo.model_validate_json(target_info_json)
    if is_active() and not target_info.column:
        _ti_raw = load_state("target_info")
        if _ti_raw:
            try:
                target_info = TargetInfo.model_validate_json(_ti_raw)
            except Exception:
                pass
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

    # ------------------------------------------------------------------
    # MISSINGNESS STRATEGY (mirrors compute_feature_target_associations)
    # Strategy B (retained_frac < 0.80): fill categorical NaN → "__MISSING__"
    # so that the missing level surfaces in per-category stats / target rates.
    # Both functions use the same threshold and the same column scope so the
    # Strategy A/B decision is always consistent between the two tools.
    # ------------------------------------------------------------------
    _numerical_cols_strat: list[str] = []
    if is_active():
        _dtypes_raw_strat = load_state("dtypes_json")
        if _dtypes_raw_strat:
            from eda_state import DataProfile
            try:
                _dp_strat = DataProfile.model_validate_json(_dtypes_raw_strat)
                _numerical_cols_strat = [
                    c for c in _dp_strat.numerical_cols if c in df.columns
                ]
            except Exception:
                pass
    if not _numerical_cols_strat:
        _numerical_cols_strat = df.select_dtypes(include="number").columns.tolist()
    # Deduplicate while preserving order (target_col may overlap cat_cols)
    _strat_cols = list(
        dict.fromkeys(cat_cols + _numerical_cols_strat + ([target_col] if target_col else []))
    )
    _n_complete_strat = df[_strat_cols].dropna().shape[0]
    _retained_frac_strat = _n_complete_strat / max(len(df), 1)

    work_df = df.copy()
    if _retained_frac_strat < 0.80:
        for _c in cat_cols:
            if _c in work_df.columns:
                work_df[_c] = work_df[_c].fillna("__MISSING__")
        logger.info(
            "analyze_categoricals: Strategy B fired (retained_frac=%.2f < 0.80) — "
            "categorical NaN → '__MISSING__'",
            _retained_frac_strat,
        )

    columns: dict[str, CategoricalStats] = {}

    for col in cat_cols:
        series = work_df[col].dropna()
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
                mask = work_df[col] == val
                grp = work_df.loc[mask, target_col].value_counts()
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


# ---------------------------------------------------------------------------
# Feature–target associations (W7): MI + effect size
# ---------------------------------------------------------------------------

# Maximum rows for kNN-based MI estimation.  Effect sizes are always computed
# on the full dataset (O(n) closed-form — no sampling needed).
_MAX_ROWS_FOR_MI = 50_000

# Effect size thresholds (weak / moderate / strong) per metric:
#   eta2:      Cohen (1988) — small=0.01, medium=0.06, large=0.14
#   cramer_v:  Cramer (1946) / conventional — small=0.10, moderate=0.30
#   pearson_r: Cohen (1988) — small=0.10, medium=0.30
_ES_THRESHOLDS: dict[str, tuple[float, float]] = {
    "eta2":      (0.01, 0.06),
    "cramer_v":  (0.10, 0.30),
    "pearson_r": (0.10, 0.30),
}


def _effect_size_label(es_type: str, value: float) -> str:
    """Classify an effect size value as weak / moderate / strong."""
    low, high = _ES_THRESHOLDS.get(es_type, (0.10, 0.30))
    if value < low:
        return "weak"
    if value < high:
        return "moderate"
    return "strong"


def _compute_eta_squared(series: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    """One-way ANOVA eta² for a numerical feature vs a categorical/class target.

    Returns (eta2, f_stat, p_value).
    eta² = SS_between / SS_total — bounded [0, 1], n-invariant.
    """
    import numpy as np
    from scipy import stats as sci_stats

    groups = [series[target == cls].dropna().values for cls in target.unique()]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0, 0.0, 1.0

    grand_mean = float(series.dropna().mean())
    ss_total = float(((series.dropna() - grand_mean) ** 2).sum())
    ss_between = float(
        sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    )
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    eta2 = max(0.0, min(1.0, round(eta2, 6)))

    try:
        f_stat, p_val = sci_stats.f_oneway(*groups)
        f_stat = float(f_stat) if not np.isnan(f_stat) else 0.0
        p_val = float(p_val) if not np.isnan(p_val) else 1.0
    except Exception:
        f_stat, p_val = 0.0, 1.0

    return eta2, round(f_stat, 4), round(p_val, 6)


def _compute_cramers_v(series: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    """Cramér's V for a categorical feature vs a categorical target.

    Returns (cramer_v, chi2_stat, p_value).
    V = sqrt(chi² / (n * (k−1))) where k = min(r, c) — bounded [0, 1].
    """
    import numpy as np
    from scipy import stats as sci_stats

    contingency = pd.crosstab(series, target)
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 0.0, 1.0

    try:
        chi2, p_val, _, _ = sci_stats.chi2_contingency(contingency)
        chi2 = float(chi2) if not np.isnan(chi2) else 0.0
        p_val = float(p_val) if not np.isnan(p_val) else 1.0
    except Exception:
        return 0.0, 0.0, 1.0

    n = contingency.values.sum()
    k = min(contingency.shape) - 1
    if k <= 0 or n <= 0:
        return 0.0, chi2, p_val

    v = float(np.sqrt(chi2 / (n * k)))
    v = max(0.0, min(1.0, round(v, 6)))
    return v, round(chi2, 4), round(p_val, 6)


def _compute_pearson_r(series: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    """Absolute Pearson correlation for a numerical feature vs a numerical target.

    Returns (|r|, t_stat_as_f_approx, p_value).
    """
    import numpy as np
    from scipy import stats as sci_stats

    valid = pd.concat([series, target], axis=1).dropna()
    if len(valid) < 3:
        return 0.0, 0.0, 1.0

    try:
        r, p_val = sci_stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        r = float(r) if not np.isnan(r) else 0.0
        p_val = float(p_val) if not np.isnan(p_val) else 1.0
    except Exception:
        return 0.0, 0.0, 1.0

    return round(abs(r), 6), round(r, 4), round(p_val, 6)


def compute_feature_target_associations(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
    target_info_json: Annotated[str, "JSON string of TargetInfo from detect_target()"],
    top_n: Annotated[int, "Number of top features to retain (default 10)"] = 10,
) -> str:
    """
    AG2 tool entry point.
    Compute univariate feature–target associations using two complementary lenses:

      1. Mutual Information (MI): kNN-estimated, detects any form of dependence
         (linear, non-linear, monotone, discrete).  Computed on a stratified
         sample when n > 50K rows to keep estimation fast.

      2. Effect Size (n-invariant, [0, 1], full dataset):
         - Numerical feature + classification  → eta²  (one-way ANOVA)
         - Categorical feature + classification → Cramér's V  (chi-square)
         - Numerical feature + regression       → |Pearson r|
         - Categorical feature + regression     → eta²  (reversed ANOVA)

    Ranking uses Borda count:
        borda_score = mi_rank + effect_size_rank   (lower = more important)

    F-statistic / chi² and p-value are stored as supplementary data only.
    They scale with n and must NOT drive the rank.

    Returns:
        JSON string of a FeatureAssociations model.
    """
    import numpy as np
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )

    from tools._pipeline_state import is_active, resolve, load_state, save_state, STATE_REF_PREFIX

    if is_active():
        data_json = resolve(data_json, "data_json")
        target_info_json = resolve(target_info_json, "target_info")

    df = pd.DataFrame(json.loads(data_json))
    target_info = TargetInfo.model_validate_json(target_info_json)

    # Guard: no target or unsupervised
    if (
        target_info.column is None
        or target_info.column not in df.columns
        or target_info.problem_type == "unsupervised"
    ):
        empty = FeatureAssociations(
            target_col=target_info.column or "",
            task_type="unknown",
            top_n=top_n,
        )
        result = empty.model_dump_json()
        if is_active():
            save_state("feature_associations", result)
            return (
                f"No target column — skipping feature association analysis. "
                f"Reference: {STATE_REF_PREFIX}feature_associations"
            )
        return result

    target_col = target_info.column
    task_type = target_info.problem_type  # "classification" | "regression"

    # Separate features from target
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        empty = FeatureAssociations(
            target_col=target_col,
            task_type=task_type,
            top_n=top_n,
            total_features=0,
        )
        result = empty.model_dump_json()
        if is_active():
            save_state("feature_associations", result)
            return (
                f"No feature columns found. "
                f"Reference: {STATE_REF_PREFIX}feature_associations"
            )
        return result

    # Determine column types — prefer DataProfile when available
    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    if is_active():
        dtypes_raw = load_state("dtypes_json")
        if dtypes_raw:
            from eda_state import DataProfile
            try:
                dp = DataProfile.model_validate_json(dtypes_raw)
                if dp.categorical_cols or dp.numerical_cols:
                    numerical_cols = [c for c in dp.numerical_cols if c in feature_cols]
                    categorical_cols = [c for c in dp.categorical_cols if c in feature_cols]
            except Exception:
                pass
    if not numerical_cols and not categorical_cols:
        numerical_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    all_feature_cols = numerical_cols + categorical_cols
    n_total = len(df)

    # -----------------------------------------------------------------------
    # MISSINGNESS STRATEGY — build a single consistent analysis_df so that
    # both MI and effect-size are computed on the same rows (Borda validity).
    #
    # Strategy A (retained_frac >= 0.80): complete-case — drop any row with
    #   a NaN in any of the analysis columns.
    # Strategy B (retained_frac < 0.80): impute — categorical NaN → the
    #   explicit "__MISSING__" level (preserves missingness signal in MI and
    #   Cramér's V); numeric NaN → column mean (conservative attenuation);
    #   then drop rows where the TARGET is still NaN.
    #
    # The retained_frac check is scoped to analysis columns only (not all df
    # columns) to avoid the wide-dataset curse of dimensionality.
    # -----------------------------------------------------------------------
    relevant_cols = all_feature_cols + [target_col]
    n_complete = df[relevant_cols].dropna().shape[0]
    retained_frac = n_complete / max(n_total, 1)
    missingness_strategy: str

    if retained_frac >= 0.80:
        analysis_df = df[relevant_cols].dropna().reset_index(drop=True)
        missingness_strategy = (
            f"complete-case (N={len(analysis_df):,}, "
            f"{retained_frac * 100:.1f}% retained)"
        )
        logger.info(
            "Missingness strategy: complete-case — %.1f%% rows retained (%d/%d)",
            retained_frac * 100, len(analysis_df), n_total,
        )
    else:
        analysis_df = df[relevant_cols].copy()
        for _col in categorical_cols:
            if _col in analysis_df.columns:
                analysis_df[_col] = analysis_df[_col].fillna("__MISSING__")
        for _col in numerical_cols:
            if _col in analysis_df.columns:
                _mean = analysis_df[_col].mean()
                analysis_df[_col] = analysis_df[_col].fillna(
                    _mean if not np.isnan(_mean) else 0.0
                )
        analysis_df = analysis_df.dropna(subset=[target_col]).reset_index(drop=True)
        missingness_strategy = (
            f"imputed (__MISSING__+mean, N={len(analysis_df):,}, "
            f"{len(analysis_df) / max(n_total, 1) * 100:.1f}% retained)"
        )
        logger.info(
            "Missingness strategy: imputation — retained_frac=%.2f < 0.80, "
            "categorical NaN → '__MISSING__', numeric NaN → mean",
            retained_frac,
        )

    n_analysis = len(analysis_df)

    # -----------------------------------------------------------------------
    # MUTUAL INFORMATION (kNN) — sample if analysis_df > _MAX_ROWS_FOR_MI.
    # Effect sizes use full analysis_df (O(n), no sampling needed).
    # Both MI and effect-size operate on analysis_df — same rows, Borda valid.
    # -----------------------------------------------------------------------
    mi_sample_size: int | None = None
    mi_sample_note: str = ""
    analysis_df_mi = analysis_df  # default: full analysis dataset

    if n_analysis > _MAX_ROWS_FOR_MI:
        mi_sample_size = _MAX_ROWS_FOR_MI
        if task_type == "classification":
            # Stratified sample: preserve class proportions
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=_MAX_ROWS_FOR_MI, random_state=42
            )
            try:
                idx, _ = next(sss.split(analysis_df, analysis_df[target_col]))
                analysis_df_mi = analysis_df.iloc[idx].reset_index(drop=True)
            except Exception:
                analysis_df_mi = analysis_df.sample(
                    n=_MAX_ROWS_FOR_MI, random_state=42
                ).reset_index(drop=True)
        else:
            analysis_df_mi = analysis_df.sample(
                n=_MAX_ROWS_FOR_MI, random_state=42
            ).reset_index(drop=True)

        mi_sample_note = (
            f"MI estimated on stratified sample of {_MAX_ROWS_FOR_MI:,} rows "
            f"(analysis N={n_analysis:,}); effect sizes on full analysis dataset."
        )
        logger.info(
            "MI sampling: %d → %d rows (%s)", n_analysis, _MAX_ROWS_FOR_MI, task_type
        )

    # Build MI feature matrix (label-encode categoricals for sklearn).
    # analysis_df_mi is already clean: Strategy A has no NaN; Strategy B has
    # "__MISSING__" strings which get valid Categorical codes (never -1).
    mi_rows = analysis_df_mi[all_feature_cols].copy()
    for col in categorical_cols:
        if col in mi_rows.columns:
            mi_rows[col] = pd.Categorical(mi_rows[col]).codes.astype(float)
            mi_rows[col] = mi_rows[col].replace(-1, float("nan"))

    mi_target = analysis_df_mi[target_col].copy()
    if task_type == "classification":
        mi_target_enc = pd.Categorical(mi_target).codes
    else:
        mi_target_enc = mi_target.astype(float)

    # Handle NaN: fill with column median/mode for MI computation only
    mi_rows_filled = mi_rows.copy()
    for col in mi_rows_filled.columns:
        if mi_rows_filled[col].isna().any():
            fill_val = mi_rows_filled[col].median()
            if np.isnan(fill_val):
                fill_val = 0.0
            mi_rows_filled[col] = mi_rows_filled[col].fillna(fill_val)

    valid_mi_mask = ~pd.isna(mi_target_enc)
    X_mi = mi_rows_filled[valid_mi_mask].values
    y_mi = mi_target_enc[valid_mi_mask]

    if len(X_mi) < 3 or len(all_feature_cols) == 0:
        mi_scores = np.zeros(len(all_feature_cols))
    else:
        try:
            if task_type == "classification":
                mi_scores = mutual_info_classif(
                    X_mi, y_mi, discrete_features=False, random_state=42
                )
            else:
                mi_scores = mutual_info_regression(
                    X_mi, y_mi.astype(float), discrete_features=False, random_state=42
                )
        except Exception:
            mi_scores = np.zeros(len(all_feature_cols))
            logger.warning("MI computation failed — using zero scores", exc_info=True)

    # -----------------------------------------------------------------------
    # EFFECT SIZE — full analysis_df, O(n), no sampling.
    # analysis_df is already complete (no NaN in features/target) so the
    # internal .dropna() in each helper is a harmless no-op.
    # -----------------------------------------------------------------------
    effect_results: dict[str, tuple[float, str, float, float]] = {}
    # returns (effect_size, es_type, stat_supplementary, p_value)

    for col in numerical_cols:
        feat_s = analysis_df[col]
        tgt_s = analysis_df[target_col]

        if task_type == "classification":
            es, f_stat, p_val = _compute_eta_squared(feat_s, tgt_s)
            effect_results[col] = (es, "eta2", f_stat, p_val)
        else:
            es, r_raw, p_val = _compute_pearson_r(feat_s, tgt_s)
            effect_results[col] = (es, "pearson_r", r_raw, p_val)

    for col in categorical_cols:
        feat_s = analysis_df[col]
        tgt_s = analysis_df[target_col]

        if task_type == "classification":
            es, chi2, p_val = _compute_cramers_v(feat_s, tgt_s)
            effect_results[col] = (es, "cramer_v", chi2, p_val)
        else:
            # Reversed: ANOVA of numerical target ~ categorical feature
            es, f_stat, p_val = _compute_eta_squared(tgt_s.astype(float), feat_s)
            effect_results[col] = (es, "eta2", f_stat, p_val)

    # -----------------------------------------------------------------------
    # RANK + BORDA FUSION
    # -----------------------------------------------------------------------
    # mi_rank: 1 = highest MI score (descending by mi_score)
    mi_score_dict = {
        col: float(mi_scores[i]) for i, col in enumerate(all_feature_cols)
    }
    mi_sorted = sorted(mi_score_dict.items(), key=lambda x: x[1], reverse=True)
    mi_rank_map = {col: rank + 1 for rank, (col, _) in enumerate(mi_sorted)}

    # effect_size_rank: 1 = largest effect size (descending)
    es_sorted = sorted(
        [(col, effect_results[col][0]) for col in all_feature_cols],
        key=lambda x: x[1],
        reverse=True,
    )
    es_rank_map = {col: rank + 1 for rank, (col, _) in enumerate(es_sorted)}

    # Build AssociationRow list
    rows: list[AssociationRow] = []
    for col in all_feature_cols:
        mi_s = mi_score_dict[col]
        mi_r = mi_rank_map[col]
        es, es_type, stat_supp, p_val = effect_results.get(col, (0.0, "eta2", 0.0, 1.0))
        es_r = es_rank_map[col]
        borda = mi_r + es_r

        rows.append(AssociationRow(
            feature=col,
            feature_type="numerical" if col in numerical_cols else "categorical",
            mi_score=round(mi_s, 6),
            mi_rank=mi_r,
            effect_size=es,
            effect_size_type=es_type,
            effect_size_label=_effect_size_label(es_type, es),
            effect_size_rank=es_r,
            borda_score=borda,
            f_stat_or_chi2=round(abs(stat_supp), 4),
            p_value=round(p_val, 6),
        ))

    # Sort by borda_score ascending, then effect_size descending as tiebreak
    rows.sort(key=lambda r: (r.borda_score, -r.effect_size))

    top_rows = rows[:top_n]

    fa = FeatureAssociations(
        target_col=target_col,
        task_type=task_type,
        rows=top_rows,
        top_n=top_n,
        total_features=len(all_feature_cols),
        mi_sample_size=mi_sample_size,
        mi_sample_note=mi_sample_note,
        missingness_strategy=missingness_strategy,
    )

    logger.info(
        "Feature associations computed: %d features, top_n=%d, task=%s, "
        "mi_sampled=%s",
        len(all_feature_cols), top_n, task_type,
        f"{mi_sample_size:,}" if mi_sample_size else "full",
    )
    result = fa.model_dump_json()

    if is_active():
        save_state("feature_associations", result)
        return (
            f"Feature–target associations computed: {len(all_feature_cols)} features "
            f"analysed, top {top_n} ranked by Borda score. "
            f"Reference: {STATE_REF_PREFIX}feature_associations"
        )
    return result
