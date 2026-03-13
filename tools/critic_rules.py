"""
tools/critic_rules.py — Strategy-pattern rule engine for CriticAgent.

Architecture Reference: architecture.md § 4.5, § 8, § 12.3

Public AG2-facing function:
  - run_critic_rules(data_json: str) -> str

Design:
  - Strategy pattern: CriticRule ABC → concrete rule classes
  - Each rule is independently unit-testable
  - DEFAULT_RULES list is the single registration point (Open/Closed principle)
  - run_critic_rules() is the only symbol AG2 ever calls
  - Zero AG2 imports. Pure Python.

Rules implemented (from architecture.md § 8 — V1 Critic Ruleset):
  1. MissingValueRule          — per-column missing: >50% BLOCKER, 30-50% HIGH, 5-30% MEDIUM
  2. DatasetMissingnessRule    — dataset-level: >30% total cells → HIGH
  3. DuplicateRowsRule         — >1% duplicate rows → HIGH, >0.1% → MEDIUM
  4. OutlierRule               — IQR method, >5% flagged in worst column → MEDIUM
  5. SkewnessRule              — |skew|>2 HIGH, 1<|skew|≤2 MEDIUM
  6. ZeroVarianceRule          — std==0 → HIGH
  7. NearZeroVarianceRule      — std < 0.01×|mean| → LOW
  8. NearPerfectCorrelationRule — |r|>0.95 HIGH, 0.85<|r|≤0.95 MEDIUM
  9. AllUniqueColumnRule       — nunique==n_rows → LOW (likely ID)
  10. SingleValueCategoricalRule — nunique==1 for non-numeric → HIGH
  11. HighCardinalityRule              — >50 unique values → HIGH, >20 → MEDIUM (W9)
  12. RareCategoryRule                 — any level < 0.5% frequency → LOW per column (W9)
  13. CategoricalNumericRedundancyRule — eta²>0.95 between cat+num column pair → HIGH (W5)

AG2 Version: 0.10.3
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Annotated, Optional

import numpy as np
import pandas as pd

from eda_state import CriticFlag, CriticReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy interface — every critic rule implements this
# ---------------------------------------------------------------------------

class CriticRule(ABC):
    """Abstract base for all critic rules (Strategy pattern)."""

    @abstractmethod
    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        """Return a list of CriticFlags for violations found (empty = no issues)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Concrete strategies — one class per rule category from architecture.md § 8
# ---------------------------------------------------------------------------

class MissingValueRule(CriticRule):
    """Per-column missing values: >50% BLOCKER, 30–50% HIGH, 5–30% MEDIUM.

    Reports ALL columns that cross a threshold (not only the worst).
    """

    name = "missing_values"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        if df.empty or df.columns.size == 0:
            return []
        missing_pct = df.isnull().mean()
        if missing_pct.max() == 0.0:
            return []
        flags: list[CriticFlag] = []
        for col, pct in missing_pct.items():
            pct = float(pct)
            if pct > 0.50:
                flags.append(CriticFlag(column=str(col), rule=self.name, severity="BLOCKER",
                                        message=f"{pct:.0%} missing", value=pct))
            elif pct > 0.30:
                flags.append(CriticFlag(column=str(col), rule=self.name, severity="HIGH",
                                        message=f"{pct:.0%} missing", value=pct))
            elif pct > 0.05:
                flags.append(CriticFlag(column=str(col), rule=self.name, severity="MEDIUM",
                                        message=f"{pct:.0%} missing", value=pct))
        # Sort: highest missing % first so the report reads in priority order
        flags.sort(key=lambda f: f.value, reverse=True)
        return flags


class DatasetMissingnessRule(CriticRule):
    """Dataset-level missingness: >30% total cells → HIGH."""

    name = "dataset_missingness"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return []
        pct = float(df.isnull().sum().sum() / total_cells)
        if pct > 0.30:
            return [CriticFlag(column=None, rule=self.name, severity="HIGH",
                              message=f"{pct:.0%} total cells missing", value=pct)]
        return []


class DuplicateRowsRule(CriticRule):
    """Duplicate rows: >1% → HIGH, >0.1% → MEDIUM.

    In pipeline mode the data passed to run_critic_rules() has already been
    deduplicated by load_data().  This rule therefore first tries to load the
    original duplicate count from the artifact store (saved before dedup).
    When the artifact is absent (unit-test / non-pipeline mode) it falls back
    to computing directly on the DataFrame.
    """

    name = "duplicate_rows"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        if df.empty:
            return []

        dup_pct: float | None = None
        dup_count: int = 0

        # --- Pipeline mode: load pre-dedup count from artifact store ---
        try:
            from tools._pipeline_state import is_active, load_state
            if is_active():
                raw = load_state("duplicate_count")
                if raw is not None:
                    dup_count = int(raw)
                    original_rows = df.shape[0] + dup_count
                    dup_pct = dup_count / max(original_rows, 1)
        except Exception:
            pass

        # --- Fallback: compute directly (unit-test / non-pipeline mode) ---
        if dup_pct is None:
            try:
                dup_pct = float(df.duplicated().mean())
                dup_count = int(df.duplicated().sum())
            except TypeError:
                dup_pct = float(df.astype(str).duplicated().mean())
                dup_count = int(df.astype(str).duplicated().sum())

        if dup_pct > 0.01:
            return [CriticFlag(
                column=None, rule=self.name, severity="HIGH",
                message=f"{dup_pct:.1%} duplicate rows ({dup_count} rows)",
                value=dup_pct,
                suggestion="Investigate data pipeline for unintended row duplication",
            )]
        if dup_pct > 0.001:
            return [CriticFlag(
                column=None, rule=self.name, severity="MEDIUM",
                message=f"{dup_pct:.1%} duplicate rows ({dup_count} rows)",
                value=dup_pct,
                suggestion="Verify duplicate rows are expected or investigate upstream",
            )]
        return []


class OutlierRule(CriticRule):
    """Outliers (IQR method): >5% of rows flagged in worst column → MEDIUM.

    For each numerical column, values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]
    are counted as outliers. Reports the worst column only.
    """

    name = "outliers_iqr"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        num = df.select_dtypes("number")
        if num.empty:
            return []
        worst_col: Optional[str] = None
        worst_pct = 0.0
        for col in num.columns:
            series = num[col].dropna()
            if series.size < 4:  # Need enough data points for IQR
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue  # Zero spread — handled by ZeroVarianceRule
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_pct = float(((series < lower) | (series > upper)).mean())
            if outlier_pct > worst_pct:
                worst_pct = outlier_pct
                worst_col = str(col)
        if worst_col is not None and worst_pct > 0.05:
            return [CriticFlag(column=worst_col, rule=self.name, severity="MEDIUM",
                              message=f"{worst_pct:.1%} outliers (IQR)", value=worst_pct)]
        return []


class SkewnessRule(CriticRule):
    """Comprehensive skewness analysis with context-aware reporting.

    Enhancements over basic threshold:
      - Reports up to 5 most skewed numerical columns (|skew| > 1)
      - Direction-aware: positive (right-skew) vs negative (left-skew)
      - Sample size adjustment: n < 30 → HIGH downgraded to MEDIUM
      - Zero-inflation detection: >20% zeros noted in message
      - Transformation suggestions based on skew direction and context

    Base thresholds: |skew|>2 HIGH, 1<|skew|≤2 MEDIUM
    """

    name = "skewness"
    _MAX_REPORTED = 5

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        num = df.select_dtypes("number")
        if num.empty:
            return []

        n_rows = len(df)

        # Collect all skewed columns with context
        skewed: list[tuple[str, float, float]] = []  # (col, skew, zeros_pct)
        for col in num.columns:
            series = num[col].dropna()
            if series.size < 3:  # Need at least 3 points for skewness
                continue
            skew_val = float(series.skew())
            if pd.isna(skew_val):
                continue
            if abs(skew_val) > 1:
                zeros_pct = float((series == 0).mean() * 100)
                skewed.append((str(col), skew_val, zeros_pct))

        if not skewed:
            return []

        # Sort by absolute skew descending, take top N
        skewed.sort(key=lambda x: abs(x[1]), reverse=True)
        skewed = skewed[: self._MAX_REPORTED]

        flags: list[CriticFlag] = []
        for col, skew_val, zeros_pct in skewed:
            abs_skew = abs(skew_val)

            # Base severity
            severity = "HIGH" if abs_skew > 2 else "MEDIUM"

            # Sample size adjustment: skewness unreliable for small samples
            if n_rows < 30 and severity == "HIGH":
                severity = "MEDIUM"

            # Direction label
            direction = (
                "positive (right-skew)" if skew_val > 0 else "negative (left-skew)"
            )

            # Build message with context
            parts = [f"skew={skew_val:.2f} ({direction})"]
            if zeros_pct > 20:
                parts.append(f"{zeros_pct:.0f}% zeros")
            if n_rows < 30:
                parts.append(f"n={n_rows}, interpret with caution")
            message = ", ".join(parts)

            # Transformation suggestion
            if zeros_pct > 20:
                suggestion = (
                    "High zero-inflation — consider zero-inflated model or log1p transform"
                )
            elif skew_val > 2:
                suggestion = "Strong positive skew — log or sqrt transform recommended"
            elif skew_val < -2:
                suggestion = (
                    "Strong negative skew — reflect and log transform recommended"
                )
            elif skew_val > 1:
                suggestion = "Moderate positive skew — sqrt transform may help"
            else:
                suggestion = "Moderate negative skew — reflect and sqrt transform may help"

            flags.append(
                CriticFlag(
                    column=col,
                    rule=self.name,
                    severity=severity,
                    message=message,
                    value=abs_skew,
                    suggestion=suggestion,
                )
            )

        return flags


class ZeroVarianceRule(CriticRule):
    """Zero-variance column: std==0 → HIGH.

    Reports the first zero-variance numerical column found.
    """

    name = "zero_variance"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        num = df.select_dtypes("number")
        if num.empty:
            return []
        stds = num.std()
        zero_var = stds[stds == 0]
        if zero_var.empty:
            return []
        col = str(zero_var.index[0])
        return [CriticFlag(column=col, rule=self.name, severity="HIGH",
                          message="zero variance (std=0)", value=0.0)]


class NearZeroVarianceRule(CriticRule):
    """Near-zero variance: std < 0.01 × |mean| → LOW.

    Columns with mean ≈ 0 are skipped (comparison undefined).
    Zero-variance columns are skipped (caught by ZeroVarianceRule).
    Reports the first matching numerical column.
    """

    name = "near_zero_variance"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        num = df.select_dtypes("number")
        if num.empty:
            return []
        for col in num.columns:
            series = num[col].dropna()
            if series.size == 0:
                continue
            std_val = float(series.std())
            mean_val = abs(float(series.mean()))
            if mean_val == 0:
                continue  # Can't compare to zero mean
            if std_val > 0 and std_val < 0.01 * mean_val:
                return [CriticFlag(
                    column=str(col), rule=self.name, severity="LOW",
                    message=f"std={std_val:.4f}, |mean|={mean_val:.4f}",
                    value=std_val,
                )]
        return []


def _eta_squared(series_cat: pd.Series, series_num: pd.Series) -> float:
    """Compute eta² (correlation ratio) between a categorical and numeric series.

    eta² = SS_between / SS_total, where SS_between is the variance in series_num
    explained by the categorical grouping in series_cat.

    Returns 0.0 when computation is not meaningful (too few rows, zero variance).
    """
    df_pair = pd.DataFrame({"cat": series_cat, "num": series_num}).dropna()
    if len(df_pair) < 2:
        return 0.0
    grand_mean = df_pair["num"].mean()
    ss_total = float(((df_pair["num"] - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0  # ZeroVarianceRule already handles this column
    ss_between = float(
        df_pair.groupby("cat")["num"]
        .apply(lambda g: len(g) * (g.mean() - grand_mean) ** 2)
        .sum()
    )
    return min(ss_between / ss_total, 1.0)  # clamp: floating-point can exceed 1.0


class NearPerfectCorrelationRule(CriticRule):
    """Near-perfect correlation: |r|>0.95 HIGH, 0.85<|r|≤0.95 MEDIUM.

    Examines all off-diagonal entries in the Pearson correlation matrix.
    """

    name = "near_perfect_correlation"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        num = df.select_dtypes("number")
        if num.shape[1] < 2:
            return []
        corr = num.corr().abs()
        # Mask diagonal (self-correlation = 1.0)
        mask = np.eye(corr.shape[0], dtype=bool)
        corr_masked = corr.where(~mask)
        max_val = float(corr_masked.max().max())
        if np.isnan(max_val):
            return []
        if max_val > 0.95:
            return [CriticFlag(column=None, rule=self.name, severity="HIGH",
                              message=f"|r|={max_val:.2f}", value=max_val)]
        if max_val > 0.85:
            return [CriticFlag(column=None, rule=self.name, severity="MEDIUM",
                              message=f"|r|={max_val:.2f}", value=max_val)]
        return []


class CategoricalNumericRedundancyRule(CriticRule):
    """Categorical–numeric redundancy: eta² > 0.95 → HIGH.

    Uses the correlation ratio (eta²) to detect when a categorical column and
    a numeric column are encoding the same concept (e.g., 'education' and
    'education-num' in the Adult Census dataset).

    eta² = SS_between / SS_total measures how much variance in the numeric
    column is explained by the categorical grouping. Values near 1.0 indicate
    the pair is essentially redundant — at least one should be dropped before
    modeling to avoid leakage or inflated feature importance.

    Guards:
      - Skips categoricals with nunique == n_rows (ID-like; eta² trivially ≈ 1.0)
      - Skips categoricals with nunique > n_rows // 2 (sparse groups: unreliable)
      - Skips numeric columns with zero variance (SS_total = 0)
      - Returns empty list when n_rows < 10

    Reports all qualifying pairs above threshold, sorted by eta² descending.
    """

    name = "categorical_numeric_redundancy"
    _ETA2_THRESHOLD = 0.95

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        num_cols = df.select_dtypes("number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if not num_cols or not cat_cols:
            return []
        n_rows = len(df)
        if n_rows < 10:
            return []
        flags: list[CriticFlag] = []
        for cat_col in cat_cols:
            nunique = int(df[cat_col].nunique())
            if nunique == n_rows or nunique > n_rows // 2:
                continue  # Skip ID-like or hyper-sparse categoricals
            for num_col in num_cols:
                eta2 = _eta_squared(df[cat_col], df[num_col])
                if eta2 > self._ETA2_THRESHOLD:
                    flags.append(CriticFlag(
                        column=None,
                        rule=self.name,
                        severity="HIGH",
                        message=(
                            f"'{cat_col}' ~ '{num_col}': eta²={eta2:.3f}"
                            f" — likely categorical encoding pair"
                        ),
                        value=round(eta2, 6),
                        suggestion=(
                            f"'{cat_col}' and '{num_col}' carry redundant information;"
                            f" consider dropping '{num_col}' or using only one for modeling"
                        ),
                    ))
        flags.sort(key=lambda f: f.value, reverse=True)
        return flags


class AllUniqueColumnRule(CriticRule):
    """All-unique column (likely ID): nunique == n_rows → LOW.

    Reports the first all-unique column found.
    """

    name = "all_unique_column"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        if df.empty:
            return []
        n_rows = len(df)
        for col in df.columns:
            if df[col].nunique() == n_rows:
                return [CriticFlag(
                    column=str(col), rule=self.name, severity="LOW",
                    message=f"all {n_rows} values unique (likely ID)",
                    value=float(n_rows),
                )]
        return []


class SingleValueCategoricalRule(CriticRule):
    """Single-value categorical: nunique == 1 → HIGH.

    Checks non-numeric columns only. Numerical constants are caught
    by ZeroVarianceRule. Reports the first matching column.
    """

    name = "single_value_categorical"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        cat = df.select_dtypes(exclude="number")
        if cat.empty:
            return []
        for col in cat.columns:
            if cat[col].nunique() == 1:
                val_repr = str(cat[col].dropna().iloc[0])
                return [CriticFlag(
                    column=str(col), rule=self.name, severity="HIGH",
                    message=f"only 1 unique value: {val_repr}",
                    value=1.0,
                )]
        return []


class HighCardinalityRule(CriticRule):
    """High-cardinality categorical: >50 unique values → HIGH, >20 → MEDIUM.

    Skips all-unique columns (already flagged as likely ID by AllUniqueColumnRule).
    Reports all columns above threshold, sorted by nunique descending.
    """

    name = "high_cardinality"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        cat = df.select_dtypes(exclude="number")
        if cat.empty or len(df) == 0:
            return []
        n_rows = len(df)
        flags: list[CriticFlag] = []
        for col in cat.columns:
            nunique = int(cat[col].nunique())
            if nunique == n_rows:
                continue  # AllUniqueColumnRule handles ID-like columns
            if nunique > 50:
                flags.append(CriticFlag(
                    column=str(col), rule=self.name, severity="HIGH",
                    message=f"{nunique} unique values — high encoding cost",
                    value=float(nunique),
                    suggestion=(
                        f"Consider target-encoding or embedding instead of one-hot; "
                        f"one-hot would add {nunique} sparse columns"
                    ),
                ))
            elif nunique > 20:
                flags.append(CriticFlag(
                    column=str(col), rule=self.name, severity="MEDIUM",
                    message=f"{nunique} unique values — moderate encoding cost",
                    value=float(nunique),
                    suggestion="Consider grouping low-frequency categories or ordinal encoding",
                ))
        flags.sort(key=lambda f: f.value, reverse=True)
        return flags


class RareCategoryRule(CriticRule):
    """Rare category levels: any level appearing in < 0.5% of rows → LOW per column.

    Only fires when n_rows >= 100 (below that the 0.5% threshold is meaningless —
    minimum frequency 1/99 ≈ 1.0% > 0.5%).
    NaN values are excluded from frequency calculation (handled by MissingValueRule).
    """

    name = "rare_category"
    _RARE_THRESHOLD = 0.005  # < 0.5%
    _MIN_ROWS = 100
    _EXAMPLE_LIMIT = 3

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        cat = df.select_dtypes(exclude="number")
        if cat.empty:
            return []
        if len(df) < self._MIN_ROWS:
            return []
        flags: list[CriticFlag] = []
        for col in cat.columns:
            series = cat[col].dropna()
            if series.empty:
                continue
            freq = series.value_counts(normalize=True)
            rare = freq[freq < self._RARE_THRESHOLD]
            if rare.empty:
                continue
            n_rare = len(rare)
            examples = [str(v) for v in rare.index[: self._EXAMPLE_LIMIT]]
            example_str = ", ".join(f"'{v}'" for v in examples)
            if n_rare > self._EXAMPLE_LIMIT:
                example_str += f" (+{n_rare - self._EXAMPLE_LIMIT} more)"
            flags.append(CriticFlag(
                column=str(col),
                rule=self.name,
                severity="LOW",
                message=f"{n_rare} rare level(s) (<0.5% each): {example_str}",
                value=float(n_rare),
                suggestion="Consider grouping rare categories into 'Other' to reduce noise",
            ))
        return flags


class ClassImbalanceRule(CriticRule):
    """Target variable class imbalance analysis.

    Only fires when target_info is available via the artifact store.
    Loads target_info from the pipeline state to inspect imbalance.

    Thresholds:
      - Imbalance ratio > 10:1 → HIGH
      - Imbalance ratio > 3:1  → MEDIUM
      - Position heuristic detection → LOW (confidence warning)
      - Regression target |skew| > 2 → MEDIUM
    """

    name = "class_imbalance"

    def check(self, df: pd.DataFrame, stats: dict) -> list[CriticFlag]:
        # Load target_info from artifact store (if available)
        try:
            from tools._pipeline_state import is_active, load_state
            if not is_active():
                return []
            raw = load_state("target_info")
            if raw is None:
                return []
        except Exception:
            return []

        from eda_state import TargetInfo
        target_info = TargetInfo.model_validate_json(raw)

        if target_info.column is None:
            return []

        flags: list[CriticFlag] = []

        # Confidence warning for position heuristic
        if target_info.detection_method == "position_heuristic":
            flags.append(CriticFlag(
                column=target_info.column,
                rule=self.name,
                severity="LOW",
                message=(
                    f"Target '{target_info.column}' detected by position heuristic "
                    f"— verify this is the correct target variable"
                ),
                value=0.0,
                suggestion="Confirm target variable or use --target CLI flag",
            ))

        if target_info.problem_type == "classification":
            ratio = target_info.imbalance_ratio
            if ratio > 10:
                flags.append(CriticFlag(
                    column=target_info.column,
                    rule=self.name,
                    severity="HIGH",
                    message=f"Severe class imbalance: {ratio:.1f}:1",
                    value=ratio,
                    suggestion="Consider SMOTE, class weights, or stratified sampling",
                ))
            elif ratio > 3:
                flags.append(CriticFlag(
                    column=target_info.column,
                    rule=self.name,
                    severity="MEDIUM",
                    message=f"Moderate class imbalance: {ratio:.1f}:1",
                    value=ratio,
                    suggestion="Consider stratified train/test split and class weights",
                ))

        elif target_info.problem_type == "regression":
            # Check regression target skewness
            if target_info.column in df.columns:
                series = df[target_info.column].dropna()
                if len(series) > 2:
                    skew_val = float(series.skew())
                    if abs(skew_val) > 2:
                        flags.append(CriticFlag(
                            column=target_info.column,
                            rule=self.name,
                            severity="MEDIUM",
                            message=(
                                f"Regression target is highly skewed "
                                f"(skewness={skew_val:.2f})"
                            ),
                            value=abs(skew_val),
                            suggestion="Consider log or Box-Cox transform",
                        ))

        return flags


# ---------------------------------------------------------------------------
# Rule registry — Open/Closed: add new rule = add new class + append here
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[CriticRule] = [
    MissingValueRule(),
    DatasetMissingnessRule(),
    DuplicateRowsRule(),
    OutlierRule(),
    SkewnessRule(),
    ZeroVarianceRule(),
    NearZeroVarianceRule(),
    NearPerfectCorrelationRule(),
    CategoricalNumericRedundancyRule(),  # W5
    AllUniqueColumnRule(),
    SingleValueCategoricalRule(),
    HighCardinalityRule(),               # W9
    RareCategoryRule(),                  # W9
    ClassImbalanceRule(),
]


# ---------------------------------------------------------------------------
# AG2-facing public function (flat callable, no OOP visible to AG2)
# ---------------------------------------------------------------------------

# Severity levels that trigger REVISION_NEEDED (architecture.md § 4.5):
# "If no flags above MEDIUM severity, output: APPROVED"
_REVISION_SEVERITIES = frozenset({"BLOCKER", "HIGH"})


def run_critic_rules(
    data_json: Annotated[str, "JSON string (records orientation) of the DataFrame to evaluate"],
) -> str:
    """
    AG2 tool entry point. Receives serialized DataFrame as JSON,
    runs all critic rules, returns CriticReport as JSON string.

    Status logic (architecture.md § 4.5):
      - REVISION_NEEDED if any flag has BLOCKER or HIGH severity
      - APPROVED if all flags are MEDIUM/LOW or no flags exist

    All OOP is encapsulated — AG2 sees only this flat function.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))

    # Sanitize columns with unhashable types (dicts/lists) that break
    # pandas hashing operations (duplicated, nunique, etc.).
    for col in df.columns:
        if df[col].dtype == object and len(df) > 0:
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
            if isinstance(sample, (dict, list)):
                df[col] = df[col].astype(str)

    flags = [f for rule in DEFAULT_RULES for f in rule.check(df, {})]
    status = (
        "REVISION_NEEDED"
        if any(f.severity in _REVISION_SEVERITIES for f in flags)
        else "APPROVED"
    )
    report = CriticReport(flags=flags, status=status)

    logger.info(
        "Critic rules: %d flags, status=%s (severities: %s)",
        len(flags),
        status,
        ", ".join(sorted({f.severity for f in flags})) or "none",
    )
    result = report.model_dump_json()

    if is_active():
        save_state("critic_report", result)
        severities = ", ".join(sorted({f.severity for f in flags})) or "none"
        return (
            f"Critic report: {len(flags)} flag(s), status={status} "
            f"(severities: {severities}). "
            f"Reference: {STATE_REF_PREFIX}critic_report"
        )
    return result
