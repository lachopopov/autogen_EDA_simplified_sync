"""
tests/test_critic_rules.py — Unit tests for tools/critic_rules.py

Tests all 10 critic rule classes independently + the run_critic_rules() AG2 function.
Validates outputs against Pydantic sub-models (CriticFlag, CriticReport).
No LLM calls — pure function tests.

NOTE: All CriticRule.check() methods return list[CriticFlag] (empty list = no issues).
"""

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from eda_state import CriticReport
from tools.critic_rules import (
    DEFAULT_RULES,
    AllUniqueColumnRule,
    CategoricalNumericRedundancyRule,
    ClassImbalanceRule,
    CriticRule,
    DatasetMissingnessRule,
    DuplicateRowsRule,
    HighCardinalityRule,
    MissingValueRule,
    NearPerfectCorrelationRule,
    NearZeroVarianceRule,
    OutlierRule,
    RareCategoryRule,
    SingleValueCategoricalRule,
    SkewnessRule,
    ZeroVarianceRule,
    _eta_squared,
    run_critic_rules,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_df():
    """A well-behaved DataFrame: no issues expected."""
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "cat": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
    })


@pytest.fixture()
def empty_df():
    """An empty DataFrame."""
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# MissingValueRule
# ---------------------------------------------------------------------------

class TestMissingValueRule:
    """Test MissingValueRule: per-column missing thresholds (all columns flagged)."""

    def test_blocker_severity(self):
        """Column with > 50% missing → BLOCKER."""
        df = pd.DataFrame({"x": [None, None, None, None, 1.0, None, None, None, None, None]})
        flags = MissingValueRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "BLOCKER"
        assert flag.rule == "missing_values"
        assert flag.value > 0.50

    def test_high_severity(self):
        """Column with 30–50% missing → HIGH."""
        # 4/10 = 40% missing
        df = pd.DataFrame({"x": [None, None, None, None, 1, 2, 3, 4, 5, 6]})
        flags = MissingValueRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert 0.30 < flag.value <= 0.50

    def test_medium_severity(self):
        """Column with 5–30% missing → MEDIUM."""
        # 2/10 = 20% missing
        df = pd.DataFrame({"x": [None, None, 3, 4, 5, 6, 7, 8, 9, 10]})
        flags = MissingValueRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"
        assert 0.05 < flag.value <= 0.30

    def test_no_flag_below_threshold(self):
        """Column with ≤ 5% missing → no flag."""
        # 0/10 = 0% missing
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        flags = MissingValueRule().check(df, {})
        assert flags == []

    def test_empty_dataframe(self, empty_df):
        """Empty DataFrame → no flag."""
        assert MissingValueRule().check(empty_df, {}) == []

    def test_all_columns_above_threshold_flagged(self):
        """All columns exceeding threshold are flagged, not just the worst."""
        df = pd.DataFrame({
            "low_missing": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # 10% → MEDIUM
            "high_missing": [None, None, None, None, None, None, 7, 8, 9, 10],  # 60% → BLOCKER
        })
        flags = MissingValueRule().check(df, {})
        assert len(flags) == 2
        severities = {f.column: f.severity for f in flags}
        assert severities["high_missing"] == "BLOCKER"
        assert severities["low_missing"] == "MEDIUM"

    def test_flags_sorted_by_missing_pct_descending(self):
        """Flags are returned highest missing % first."""
        df = pd.DataFrame({
            "a": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # 10%
            "b": [None, None, None, 4, 5, 6, 7, 8, 9, 10],  # 30% → just at boundary, no flag (≤0.30)
            "c": [None, None, None, None, 5, 6, 7, 8, 9, 10],  # 40% → HIGH
        })
        flags = MissingValueRule().check(df, {})
        assert flags[0].value >= flags[-1].value  # descending order

    def test_only_one_column_above_threshold(self):
        """Only the column crossing the threshold appears in flags."""
        df = pd.DataFrame({
            "ok": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # 5% — no flag
            "bad": [None, None, None, None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # 20% → MEDIUM
        })
        flags = MissingValueRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].column == "bad"


# ---------------------------------------------------------------------------
# DatasetMissingnessRule
# ---------------------------------------------------------------------------

class TestDatasetMissingnessRule:
    """Test DatasetMissingnessRule: >30% total cells → HIGH."""

    def test_high_severity(self):
        """More than 30% of all cells missing → HIGH."""
        # 8/20 = 40% missing
        df = pd.DataFrame({
            "a": [None, None, None, None, 5, 6, 7, 8, 9, 10],
            "b": [None, None, None, None, 5, 6, 7, 8, 9, 10],
        })
        flags = DatasetMissingnessRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert flag.column is None  # dataset-level
        assert flag.value > 0.30

    def test_no_flag_below_threshold(self, clean_df):
        """Less than 30% total missing → no flag."""
        assert DatasetMissingnessRule().check(clean_df, {}) == []

    def test_empty_dataframe(self, empty_df):
        """Empty DataFrame → no flag."""
        assert DatasetMissingnessRule().check(empty_df, {}) == []


# ---------------------------------------------------------------------------
# DuplicateRowsRule
# ---------------------------------------------------------------------------

class TestDuplicateRowsRule:
    """Test DuplicateRowsRule: >1% → HIGH, >0.1% → MEDIUM (W8 thresholds)."""

    def test_high_severity_over_1pct(self):
        """More than 1% duplicate rows → HIGH."""
        # 5 duplicates in 10 rows = 50%
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]})
        flags = DuplicateRowsRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert flag.column is None  # dataset-level
        assert "rows" in flag.message

    def test_medium_severity_between_0_1_and_1_pct(self):
        """Between 0.1% and 1% duplicate rows → MEDIUM."""
        # Build df with ~0.5% duplicate rate: 1 duplicate in 200 rows
        base = list(range(199))  # 199 unique values
        df = pd.DataFrame({"x": base + [base[0]]})  # add 1 dup → 1/200 = 0.5%
        flags = DuplicateRowsRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"
        assert flag.column is None

    def test_no_flag_no_duplicates(self, clean_df):
        """No duplicate rows → no flag."""
        assert DuplicateRowsRule().check(clean_df, {}) == []

    def test_no_flag_below_0_1_pct(self):
        """Below 0.1% duplicate rate → no flag."""
        # 1 duplicate in 2000 rows = 0.05%
        base = list(range(1999))
        df = pd.DataFrame({"x": base + [base[0]]})
        flags = DuplicateRowsRule().check(df, {})
        assert flags == []

    def test_empty_dataframe(self, empty_df):
        """Empty DataFrame → no flag."""
        assert DuplicateRowsRule().check(empty_df, {}) == []

    def test_flag_includes_row_count(self):
        """Flag message includes the raw duplicate count."""
        df = pd.DataFrame({"x": [1, 1, 1, 2, 2]})
        flags = DuplicateRowsRule().check(df, {})
        assert len(flags) == 1
        # [1,1,1,2,2]: 3 out of 5 rows are duplicates (indices 1, 2, 4)
        assert "3" in flags[0].message

    def test_suggestion_present(self):
        """Flag carries a non-empty suggestion."""
        df = pd.DataFrame({"x": [1, 1, 1, 2, 2]})
        flags = DuplicateRowsRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].suggestion

    def test_pipeline_artifact_used_over_df(self):
        """In pipeline mode the artifact-based count is used, not df.duplicated()."""
        from tools._pipeline_state import init_session, clear_session, save_state
        try:
            init_session()
            # Store 500 duplicate count in artifact — high enough to trigger HIGH
            save_state("duplicate_count", "500")
            # df has zero duplicates (already deduped by load_data)
            df = pd.DataFrame({"x": list(range(1000))})
            flags = DuplicateRowsRule().check(df, {})
            assert len(flags) == 1
            # 500 dupes / (1000 + 500) original rows ≈ 33% → HIGH
            assert flags[0].severity == "HIGH"
            assert "500" in flags[0].message
        finally:
            clear_session()

    def test_pipeline_artifact_zero_no_flag(self):
        """Artifact count of 0 → no flag even in pipeline mode."""
        from tools._pipeline_state import init_session, clear_session, save_state
        try:
            init_session()
            save_state("duplicate_count", "0")
            df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
            flags = DuplicateRowsRule().check(df, {})
            assert flags == []
        finally:
            clear_session()


# ---------------------------------------------------------------------------
# OutlierRule
# ---------------------------------------------------------------------------

class TestOutlierRule:
    """Test OutlierRule: IQR method, >5% outliers in worst column → MEDIUM."""

    def test_medium_severity(self):
        """Column with >5% outliers → MEDIUM."""
        # Normal-ish values + extreme outlier: 100 is far outside IQR
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        flags = OutlierRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"
        assert flag.rule == "outliers_iqr"

    def test_no_flag_no_outliers(self, clean_df):
        """Evenly spaced values → no outliers → no flag."""
        assert OutlierRule().check(clean_df, {}) == []

    def test_no_numerical_columns(self):
        """Only categorical columns → no flag."""
        df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"]})
        assert OutlierRule().check(df, {}) == []

    def test_few_data_points(self):
        """Fewer than 4 data points → skipped (not meaningful)."""
        df = pd.DataFrame({"x": [1, 2, 100]})
        assert OutlierRule().check(df, {}) == []


# ---------------------------------------------------------------------------
# OutlierRule — modality-aware severity (W6)
# ---------------------------------------------------------------------------

class TestOutlierRuleModality:
    """W6: multi-modal IQR outlier caveat.

    Severity: >20% outlier rate -> LOW ("IQR fences unreliable").
    5-20% -> MEDIUM (genuine anomaly signal).
    _iqr_unreliability_hint() adds a kurtosis-based message explaining WHY:
      - kurt < 0  -> platykurtic: equal-weight clusters / flat spread
      - kurt > 0, |skew| > 1 -> heavy tail / skewed (leptokurtic)
      - kurt > 0, |skew| <= 1 -> dominant spike with satellite sub-populations
    """

    def test_multimodal_downgraded_to_low(self):
        """3-cluster distribution (dominant spike): 28% outlier rate -> LOW + spike hint."""
        np.random.seed(42)
        # Dominant cluster at 40 (narrow std=0.5) drives a tight IQR (~1).
        # Clusters at 20 and 55 are far outside the fences, producing ~28%
        # IQR-outliers.  Kurt > 0, |skew| ~0.56 -> fires "dominant spike" branch.
        x = np.concatenate([
            np.random.normal(20, 0.5, 100),   # left cluster -- below fence
            np.random.normal(40, 0.5, 500),   # dominant mode -- sets narrow IQR
            np.random.normal(55, 0.5, 100),   # right cluster -- above fence
        ])
        df = pd.DataFrame({"x": x})
        flags = OutlierRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "LOW"
        assert flag.rule == "outliers_iqr"
        assert flag.value > 0.20
        assert "spike" in flag.message

    def test_hint_branches_covered_separately(self):
        """Platykurtic equal clusters expand IQR to enclose all data -> 0% outliers,
        rule never fires.  Hint branches are tested directly in TestIqrUnreliabilityHint."""
        # Three equal-weight discrete clusters at 10 / 50 / 90 ->
        # Q1=10, Q3=90, IQR=80, fences=[-110, 210] -> all values inside -> 0 flags.
        x = np.array([10.0] * 250 + [50.0] * 250 + [90.0] * 250)
        df = pd.DataFrame({"x": x})
        flags = OutlierRule().check(df, {})
        assert len(flags) == 0  # IQR wide enough to encompass all clusters

    def test_unimodal_outliers_remain_medium(self):
        """Unimodal + sparse extreme outliers (10% rate, <=20%) -> MEDIUM (unchanged)."""
        np.random.seed(42)
        # 90 obs in main cluster, 10 obs as extreme outliers.
        # Outlier rate ~10% (well below the 20% threshold) -> MEDIUM.
        x = np.concatenate([
            np.random.normal(50, 3, 90),
            np.random.normal(-90, 1, 5),    # extreme low
            np.random.normal(190, 1, 5),    # extreme high
        ])
        df = pd.DataFrame({"x": x})
        flags = OutlierRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"
        assert flag.rule == "outliers_iqr"
        assert flag.value <= 0.20

    def test_boundary_exactly_at_threshold_is_not_low(self):
        """Outlier rate equal to the 20% threshold does NOT downgrade to LOW."""
        np.random.seed(0)
        # Q1=1, Q3=3, IQR=2, fences=[-2, 6]; 20 obs at 10 > 6 => exactly 20/100 = 20.0%
        core = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3] * 7 + [1, 1, 1]  # 80 obs
        outliers = [10] * 20  # 20 obs at 10 > 6
        df = pd.DataFrame({"x": core + outliers})
        flags = OutlierRule().check(df, {})
        assert len(flags) == 1
        # 20/100 = 20.0% which is NOT > 20%, so MEDIUM
        assert flags[0].severity == "MEDIUM"


# ---------------------------------------------------------------------------
# _iqr_unreliability_hint — direct branch coverage
# ---------------------------------------------------------------------------

class TestIqrUnreliabilityHint:
    """Direct tests for each kurtosis branch in _iqr_unreliability_hint.

    Because platykurtic equal-cluster data has a wide IQR (0% outlier rate),
    the platykurtic branch is unreachable through OutlierRule in practice.
    Testing the helper directly gives precise branch coverage.
    """

    def test_platykurtic_branch(self):
        """kurt < 0 -> 'platykurtic' hint."""
        from tools.critic_rules import _iqr_unreliability_hint
        # Three equal-weight clusters -> flat spread -> kurt < 0
        x = pd.Series([10.0] * 250 + [50.0] * 250 + [90.0] * 250)
        assert x.kurt() < 0
        hint = _iqr_unreliability_hint(x)
        assert "platykurtic" in hint or "clusters" in hint

    def test_dominant_spike_branch(self):
        """kurt > 0, |skew| <= 1 -> 'spike' hint."""
        from tools.critic_rules import _iqr_unreliability_hint
        np.random.seed(42)
        # Dominant spike at 40 with small satellites
        x = pd.Series(np.concatenate([
            np.random.normal(20, 0.5, 100),
            np.random.normal(40, 0.5, 500),
            np.random.normal(55, 0.5, 100),
        ]))
        assert x.kurt() > 0 and abs(x.skew()) <= 1
        hint = _iqr_unreliability_hint(x)
        assert "spike" in hint

    def test_heavy_tail_branch(self):
        """kurt > 0, |skew| > 1 -> 'heavy tail' hint (leptokurtic)."""
        from tools.critic_rules import _iqr_unreliability_hint
        np.random.seed(42)
        x = pd.Series(np.random.pareto(1.5, 1000) * 10 + 1)
        assert x.kurt() > 0 and abs(x.skew()) > 1
        hint = _iqr_unreliability_hint(x)
        assert "heavy tail" in hint or "leptokurtic" in hint

    def test_too_few_observations(self):
        """< 8 obs -> fallback 'shape unclear' message."""
        from tools.critic_rules import _iqr_unreliability_hint
        hint = _iqr_unreliability_hint(pd.Series([1.0, 2.0, 3.0]))
        assert "unclear" in hint


# ---------------------------------------------------------------------------
# SkewnessRule — comprehensive multi-column analysis
# ---------------------------------------------------------------------------

class TestSkewnessRule:
    """Test SkewnessRule: comprehensive context-aware skewness analysis.

    Enhancements tested:
      - Multi-column reporting (up to 5)
      - Direction awareness (positive / negative)
      - Sample size adjustment (n < 30 → HIGH downgraded to MEDIUM)
      - Zero-inflation detection (>20% zeros noted)
      - Transformation suggestions populated
      - Sorted by absolute skew descending
    """

    def test_high_severity_large_sample(self):
        """|skew| > 2 with n ≥ 30 → HIGH severity retained."""
        # 30 rows: 29 × 1 then 1 × 1000 → extreme positive skew
        data = [1] * 29 + [1000]
        df = pd.DataFrame({"x": data})
        assert len(df) >= 30
        assert df["x"].skew() > 2
        flags = SkewnessRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert flag.value > 2.0
        assert flag.column == "x"
        assert flag.suggestion != ""

    def test_high_downgraded_small_sample(self):
        """|skew| > 2 with n < 30 → HIGH downgraded to MEDIUM."""
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]})
        assert len(df) < 30
        assert df["x"].skew() > 2
        flags = SkewnessRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"  # downgraded
        assert "interpret with caution" in flag.message

    def test_medium_severity(self):
        """1 < |skew| ≤ 2 → MEDIUM."""
        df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 4, 4, 5, 6, 11]})
        skew_val = df["x"].skew()
        assert 1 < abs(skew_val) <= 2, f"Test data skew={skew_val:.2f}"
        flags = SkewnessRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"

    def test_positive_direction_in_message(self):
        """Positive skew → message says 'positive (right-skew)'."""
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]})
        assert df["x"].skew() > 0
        flags = SkewnessRule().check(df, {})
        assert len(flags) == 1
        assert "positive (right-skew)" in flags[0].message

    def test_negative_direction_in_message(self):
        """Negative skew → message says 'negative (left-skew)'."""
        # Mirror: large values dominate, one small outlier
        df = pd.DataFrame({"x": [-100, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        assert df["x"].skew() < -1
        flags = SkewnessRule().check(df, {})
        assert len(flags) >= 1
        assert "negative (left-skew)" in flags[0].message

    def test_zero_inflation_noted(self):
        """>20% zeros → message includes zero-inflation context."""
        # 50% zeros, rest are high values → skewed with zero inflation
        data = [0] * 25 + [1000, 2000, 3000, 4000, 5000]
        df = pd.DataFrame({"x": data})
        pct_zero = (df["x"] == 0).mean() * 100
        assert pct_zero > 20
        assert abs(df["x"].skew()) > 1
        flags = SkewnessRule().check(df, {})
        assert len(flags) >= 1
        flag = flags[0]
        assert "zeros" in flag.message.lower()
        assert "zero-inflat" in flag.suggestion.lower()

    def test_suggestion_populated(self):
        """Every skewness flag has a non-empty suggestion."""
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]})
        flags = SkewnessRule().check(df, {})
        assert len(flags) >= 1
        for flag in flags:
            assert flag.suggestion != ""
            assert len(flag.suggestion) > 10  # not trivial

    def test_multi_column_reporting(self):
        """Multiple skewed columns → one flag per column."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "skewed_a": np.concatenate([rng.exponential(1, 27), [500, 600, 700]]),
            "skewed_b": np.concatenate([rng.exponential(2, 27), [800, 900, 1000]]),
            "normal": rng.normal(50, 10, 30),
        })
        # Verify at least one column is skewed > 1
        assert abs(df["skewed_a"].skew()) > 1 or abs(df["skewed_b"].skew()) > 1
        flags = SkewnessRule().check(df, {})
        # Should get at least 1 flag, at most 2 (normal shouldn't be flagged)
        assert len(flags) >= 1
        flagged_cols = {f.column for f in flags}
        # Normal column should not appear
        assert "normal" not in flagged_cols or abs(df["normal"].skew()) > 1

    def test_max_five_columns(self):
        """At most 5 columns reported even if more are skewed."""
        # Create 8 heavily skewed columns
        data = {f"col_{i}": [1] * 9 + [100 * (i + 1)] for i in range(8)}
        df = pd.DataFrame(data)
        flags = SkewnessRule().check(df, {})
        assert len(flags) <= 5

    def test_sorted_by_severity(self):
        """Flags sorted by absolute skew descending (worst first)."""
        df = pd.DataFrame({
            "mild": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
            "extreme": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
        })
        # Only keep columns actually flagged
        flags = SkewnessRule().check(df, {})
        if len(flags) >= 2:
            assert flags[0].value >= flags[1].value

    def test_no_flag_low_skew(self, clean_df):
        """Symmetric data → |skew| ≤ 1 → empty list."""
        flags = SkewnessRule().check(clean_df, {})
        assert flags == []

    def test_no_numerical_columns(self):
        """Only categorical columns → empty list."""
        df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"]})
        assert SkewnessRule().check(df, {}) == []

    def test_constant_column(self):
        """Constant numerical column → skew is NaN → empty list."""
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        assert SkewnessRule().check(df, {}) == []

    def test_fewer_than_3_points_skipped(self):
        """Columns with < 3 data points are skipped."""
        df = pd.DataFrame({"x": [1, 100]})
        assert SkewnessRule().check(df, {}) == []

    def test_positive_skew_suggestion_strong(self):
        """Strong positive skew → log/sqrt transform suggested."""
        data = [1] * 29 + [10000]
        df = pd.DataFrame({"x": data})
        assert df["x"].skew() > 2
        flags = SkewnessRule().check(df, {})
        assert len(flags) == 1
        assert "log" in flags[0].suggestion.lower() or "sqrt" in flags[0].suggestion.lower()

    def test_negative_skew_suggestion_strong(self):
        """Strong negative skew → reflect and log transform suggested."""
        data = [-10000] + [1] * 29
        df = pd.DataFrame({"x": data})
        assert df["x"].skew() < -2
        flags = SkewnessRule().check(df, {})
        assert len(flags) == 1
        assert "reflect" in flags[0].suggestion.lower()


# ---------------------------------------------------------------------------
# ZeroVarianceRule
# ---------------------------------------------------------------------------

class TestZeroVarianceRule:
    """Test ZeroVarianceRule: std==0 → HIGH."""

    def test_high_severity(self):
        """Column with all same values → std=0 → HIGH."""
        df = pd.DataFrame({"const": [42, 42, 42, 42, 42], "vary": [1, 2, 3, 4, 5]})
        flags = ZeroVarianceRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert flag.column == "const"

    def test_no_flag(self, clean_df):
        """All columns have variance → no flag."""
        assert ZeroVarianceRule().check(clean_df, {}) == []

    def test_no_numerical_columns(self):
        """Only categorical → no flag."""
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        assert ZeroVarianceRule().check(df, {}) == []


# ---------------------------------------------------------------------------
# NearZeroVarianceRule
# ---------------------------------------------------------------------------

class TestNearZeroVarianceRule:
    """Test NearZeroVarianceRule: std < 0.01 × |mean| → LOW."""

    def test_low_severity(self):
        """Column with tiny variance relative to mean → LOW."""
        # mean ≈ 100, std ≈ 0.045 — well below 0.01 × 100 = 1.0
        df = pd.DataFrame({"x": [100, 100, 100, 100, 100.1]})
        flags = NearZeroVarianceRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "LOW"

    def test_no_flag_normal_variance(self, clean_df):
        """Normal variance → no flag."""
        assert NearZeroVarianceRule().check(clean_df, {}) == []

    def test_mean_zero_skipped(self):
        """Columns with mean = 0 are skipped (comparison undefined)."""
        df = pd.DataFrame({"x": [-1, 0, 1, -1, 0, 1]})
        assert NearZeroVarianceRule().check(df, {}) == []

    def test_zero_std_skipped(self):
        """std=0 is handled by ZeroVarianceRule, not this rule."""
        df = pd.DataFrame({"x": [100, 100, 100, 100, 100]})
        # std = 0 → std > 0 check fails → no flag from this rule
        assert NearZeroVarianceRule().check(df, {}) == []


# ---------------------------------------------------------------------------
# NearPerfectCorrelationRule
# ---------------------------------------------------------------------------

class TestNearPerfectCorrelationRule:
    """Test NearPerfectCorrelationRule: |r|>0.95 HIGH, 0.85<|r|≤0.95 MEDIUM."""

    def test_high_severity(self):
        """|r| > 0.95 → HIGH (perfectly correlated columns)."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        flags = NearPerfectCorrelationRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert flag.column is None  # dataset-level

    def test_medium_severity(self):
        """0.85 < |r| ≤ 0.95 → MEDIUM."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 4, 3, 5]})
        corr_val = abs(df["x"].corr(df["y"]))
        # Sanity: verify r is in the MEDIUM range
        assert 0.85 < corr_val <= 0.95, f"Test data |r|={corr_val:.4f}"
        flags = NearPerfectCorrelationRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"

    def test_no_flag_low_correlation(self, clean_df):
        """Weakly correlated columns → no flag."""
        # clean_df has a = [1..10], b = [10..100] — perfectly correlated!
        # Use uncorrelated data instead
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 1, 3, 2, 4]})
        assert NearPerfectCorrelationRule().check(df, {}) == []

    def test_single_column(self):
        """Only one numerical column → can't compute correlation → no flag."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        assert NearPerfectCorrelationRule().check(df, {}) == []


# ---------------------------------------------------------------------------
# AllUniqueColumnRule
# ---------------------------------------------------------------------------

class TestAllUniqueColumnRule:
    """Test AllUniqueColumnRule: nunique==n_rows → LOW (likely ID)."""

    def test_low_severity(self):
        """Column with all unique values → LOW (likely ID)."""
        df = pd.DataFrame({"id": [101, 102, 103, 104, 105], "val": [1, 1, 2, 2, 3]})
        flags = AllUniqueColumnRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "LOW"
        assert flag.column == "id"

    def test_no_flag(self):
        """No column with all unique values → no flag."""
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3], "y": [1, 2, 1, 2, 1]})
        assert AllUniqueColumnRule().check(df, {}) == []

    def test_empty_dataframe(self, empty_df):
        """Empty DataFrame → no flag."""
        assert AllUniqueColumnRule().check(empty_df, {}) == []


# ---------------------------------------------------------------------------
# SingleValueCategoricalRule
# ---------------------------------------------------------------------------

class TestSingleValueCategoricalRule:
    """Test SingleValueCategoricalRule: non-numeric column with nunique==1 → HIGH."""

    def test_high_severity(self):
        """Categorical column with only one unique value → HIGH."""
        df = pd.DataFrame({"status": ["active", "active", "active"], "val": [1, 2, 3]})
        flags = SingleValueCategoricalRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"
        assert flag.column == "status"
        assert "active" in flag.message

    def test_no_flag(self):
        """Categorical columns with multiple unique values → no flag."""
        df = pd.DataFrame({"cat": ["a", "b", "c"], "val": [1, 2, 3]})
        assert SingleValueCategoricalRule().check(df, {}) == []

    def test_no_categorical_columns(self):
        """Only numerical columns → no flag."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        assert SingleValueCategoricalRule().check(df, {}) == []

    def test_with_nan_values(self):
        """Column with one unique non-NaN value + NaN → still flags."""
        df = pd.DataFrame({"status": ["active", None, "active"]})
        flags = SingleValueCategoricalRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "HIGH"


# ---------------------------------------------------------------------------
# DEFAULT_RULES registry
# ---------------------------------------------------------------------------

class TestDefaultRules:
    """Test the DEFAULT_RULES list."""

    def test_count(self):
        """All 14 rules are registered (11 V1 + 2 W9 + 1 W5)."""
        assert len(DEFAULT_RULES) == 14

    def test_all_critic_rule_instances(self):
        """Every item in DEFAULT_RULES is a CriticRule instance."""
        for rule in DEFAULT_RULES:
            assert isinstance(rule, CriticRule), f"{rule} is not a CriticRule"

    def test_unique_names(self):
        """Each rule has a unique name."""
        names = [rule.name for rule in DEFAULT_RULES]
        assert len(names) == len(set(names))

    def test_all_return_lists(self, clean_df):
        """Every rule.check() returns a list (not Optional)."""
        for rule in DEFAULT_RULES:
            result = rule.check(clean_df, {})
            assert isinstance(result, list), f"{rule.name}.check() returned {type(result)}"


# ---------------------------------------------------------------------------
# run_critic_rules() — AG2-facing function
# ---------------------------------------------------------------------------

class TestRunCriticRules:
    """Test run_critic_rules() entry point."""

    def test_returns_json_string(self, clean_df):
        """Function returns a valid JSON string."""
        data_json = clean_df.to_json(orient="records")
        result = run_critic_rules(data_json)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_returns_valid_critic_report(self, clean_df):
        """Output deserializes into a valid CriticReport."""
        data_json = clean_df.to_json(orient="records")
        result = run_critic_rules(data_json)
        report = CriticReport.model_validate_json(result)
        assert isinstance(report, CriticReport)
        assert report.status in ("APPROVED", "REVISION_NEEDED")

    def test_approved_no_flags(self):
        """Clean data with no issues → APPROVED, empty flags."""
        # Uncorrelated, non-unique, varied values
        df = pd.DataFrame({
            "a": [1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.8, 8.4, 9.0, 10.6],
            "b": [5, 3, 8, 1, 7, 2, 9, 4, 10, 6],
            "cat": ["x", "y", "z", "w", "v", "x", "y", "z", "w", "v"],
        })
        result = run_critic_rules(df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        assert report.status == "APPROVED"

    def test_approved_low_only(self):
        """Only LOW severity flags → APPROVED."""
        # Column with all unique values → LOW flag only
        # 'val' is uncorrelated with 'id' to avoid NearPerfectCorrelation
        df = pd.DataFrame({
            "id": list(range(10)),
            "val": [5, 3, 8, 1, 7, 2, 9, 4, 10, 6],
            "cat": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })
        result = run_critic_rules(df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        # Should have at least one LOW flag (all-unique columns)
        low_flags = [f for f in report.flags if f.severity == "LOW"]
        high_flags = [f for f in report.flags if f.severity in ("HIGH", "BLOCKER")]
        assert len(low_flags) > 0
        assert len(high_flags) == 0
        assert report.status == "APPROVED"

    def test_approved_medium_only(self):
        """Only MEDIUM severity flags → APPROVED."""
        # Single column: no correlation rule fires.
        # 1 duplicate in 200 rows = 0.5% → MEDIUM (>0.1% but ≤1%).
        a_vals = list(range(199))
        df = pd.DataFrame({"a": a_vals + [a_vals[0]]})
        result = run_critic_rules(df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        # Only MEDIUM or LOW flags → APPROVED
        high_flags = [f for f in report.flags if f.severity in ("HIGH", "BLOCKER")]
        assert len(high_flags) == 0
        assert report.status == "APPROVED"

    def test_revision_needed_high(self):
        """HIGH severity flag → REVISION_NEEDED."""
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})  # zero variance → HIGH
        result = run_critic_rules(df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        assert report.status == "REVISION_NEEDED"
        high_flags = [f for f in report.flags if f.severity == "HIGH"]
        assert len(high_flags) > 0

    def test_revision_needed_blocker(self):
        """BLOCKER severity flag → REVISION_NEEDED."""
        # 90% missing → BLOCKER
        df = pd.DataFrame({"x": [None, None, None, None, None, None, None, None, None, 1.0]})
        result = run_critic_rules(df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        assert report.status == "REVISION_NEEDED"
        blocker = [f for f in report.flags if f.severity == "BLOCKER"]
        assert len(blocker) > 0

    def test_empty_dataframe(self):
        """Empty DataFrame → APPROVED, no flags."""
        result = run_critic_rules("[]")
        report = CriticReport.model_validate_json(result)
        assert report.status == "APPROVED"
        assert len(report.flags) == 0

    def test_no_ag2_imports(self):
        """tools/critic_rules.py must have zero AG2 imports (Hard Boundary Rule P7)."""
        import tools.critic_rules as module
        source = inspect.getsource(module)
        assert "import autogen" not in source
        assert "from autogen" not in source

    def test_dict_columns_no_crash(self):
        """DataFrame with dict-valued columns must not crash (unhashable type guard)."""
        data = [
            {"a": 1, "nested": {"key": "val1"}},
            {"a": 2, "nested": {"key": "val2"}},
            {"a": 1, "nested": {"key": "val1"}},  # duplicate
        ]
        result = run_critic_rules(json.dumps(data))
        report = CriticReport.model_validate_json(result)
        assert report.status in ("APPROVED", "REVISION_NEEDED")
        # Should not raise TypeError: unhashable type: 'dict'


# ---------------------------------------------------------------------------
# ClassImbalanceRule
# ---------------------------------------------------------------------------


class TestClassImbalanceRule:
    """Test ClassImbalanceRule with mocked artifact store."""

    def _mock_pipeline(self, monkeypatch, target_info_json):
        """Wire up mocked pipeline state returning the given TargetInfo JSON."""
        monkeypatch.setattr(
            "tools.critic_rules.ClassImbalanceRule.check",
            ClassImbalanceRule.check,  # keep real method
        )
        # Patch inside the check() method's lazy import
        import tools._pipeline_state as ps
        monkeypatch.setattr(ps, "is_active", lambda: True)
        monkeypatch.setattr(ps, "load_state", lambda key: target_info_json if key == "target_info" else None)

    def test_high_severity_extreme_imbalance(self, monkeypatch, clean_df):
        from eda_state import TargetInfo
        ti = TargetInfo(
            column="cat",
            problem_type="classification",
            imbalance_ratio=15.0,
            detection_method="name_heuristic",
        )
        self._mock_pipeline(monkeypatch, ti.model_dump_json())
        rule = ClassImbalanceRule()
        flags = rule.check(clean_df, {})
        high = [f for f in flags if f.severity == "HIGH"]
        assert len(high) == 1
        assert "15.0:1" in high[0].message

    def test_medium_severity_moderate_imbalance(self, monkeypatch, clean_df):
        from eda_state import TargetInfo
        ti = TargetInfo(
            column="cat",
            problem_type="classification",
            imbalance_ratio=5.0,
            detection_method="name_heuristic",
        )
        self._mock_pipeline(monkeypatch, ti.model_dump_json())
        rule = ClassImbalanceRule()
        flags = rule.check(clean_df, {})
        med = [f for f in flags if f.severity == "MEDIUM"]
        assert len(med) == 1

    def test_no_flag_balanced(self, monkeypatch, clean_df):
        from eda_state import TargetInfo
        ti = TargetInfo(
            column="cat",
            problem_type="classification",
            imbalance_ratio=1.5,
            detection_method="name_heuristic",
        )
        self._mock_pipeline(monkeypatch, ti.model_dump_json())
        rule = ClassImbalanceRule()
        flags = rule.check(clean_df, {})
        class_flags = [f for f in flags if f.severity in ("HIGH", "MEDIUM")]
        assert len(class_flags) == 0

    def test_position_heuristic_low_warning(self, monkeypatch, clean_df):
        from eda_state import TargetInfo
        ti = TargetInfo(
            column="cat",
            problem_type="classification",
            imbalance_ratio=1.0,
            detection_method="position_heuristic",
        )
        self._mock_pipeline(monkeypatch, ti.model_dump_json())
        rule = ClassImbalanceRule()
        flags = rule.check(clean_df, {})
        low = [f for f in flags if f.severity == "LOW"]
        assert len(low) == 1
        assert "position heuristic" in low[0].message

    def test_regression_skew_medium(self, monkeypatch):
        from eda_state import TargetInfo
        # Create a heavily skewed numerical column
        df = pd.DataFrame({
            "target_price": [1] * 90 + [1000] * 10,
            "feat": list(range(100)),
        })
        ti = TargetInfo(
            column="target_price",
            problem_type="regression",
            detection_method="name_heuristic",
        )
        self._mock_pipeline(monkeypatch, ti.model_dump_json())
        rule = ClassImbalanceRule()
        flags = rule.check(df, {})
        skew_flags = [f for f in flags if "skewed" in f.message]
        assert len(skew_flags) == 1
        assert skew_flags[0].severity == "MEDIUM"

    def test_no_pipeline_returns_empty(self, clean_df):
        """Without active pipeline, should return empty list."""
        rule = ClassImbalanceRule()
        # No monkeypatch — pipeline is_active() returns False by default
        flags = rule.check(clean_df, {})
        assert flags == []

    def test_no_target_info_returns_empty(self, monkeypatch, clean_df):
        import tools._pipeline_state as ps
        monkeypatch.setattr(ps, "is_active", lambda: True)
        monkeypatch.setattr(ps, "load_state", lambda key: None)
        rule = ClassImbalanceRule()
        flags = rule.check(clean_df, {})
        assert flags == []

    def test_in_default_rules(self):
        """ClassImbalanceRule must be in DEFAULT_RULES."""
        class_names = [type(r).__name__ for r in DEFAULT_RULES]
        assert "ClassImbalanceRule" in class_names


# ---------------------------------------------------------------------------
# HighCardinalityRule (W9)
# ---------------------------------------------------------------------------

class TestHighCardinalityRule:
    """Test HighCardinalityRule: >50 unique → HIGH, >20 → MEDIUM."""

    def test_high_severity(self):
        """51 unique values in a non-ID column → HIGH."""
        df = pd.DataFrame({"cat": [str(i) for i in range(51)] * 2})
        flags = HighCardinalityRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].severity == "HIGH"
        assert flags[0].value == 51.0
        assert flags[0].column == "cat"

    def test_medium_severity(self):
        """25 unique values → MEDIUM."""
        df = pd.DataFrame({"cat": [str(i % 25) for i in range(100)]})
        flags = HighCardinalityRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].severity == "MEDIUM"
        assert flags[0].value == 25.0

    def test_no_flag_at_threshold(self):
        """Exactly 20 unique values → no flag (threshold is strictly >20)."""
        df = pd.DataFrame({"cat": [str(i % 20) for i in range(100)]})
        assert HighCardinalityRule().check(df, {}) == []

    def test_no_flag_below_threshold(self):
        """10 unique values → no flag."""
        df = pd.DataFrame({"cat": [str(i % 10) for i in range(100)]})
        assert HighCardinalityRule().check(df, {}) == []

    def test_id_column_skipped(self):
        """nunique == n_rows (ID column) → skipped, no flag."""
        df = pd.DataFrame({"id": [str(i) for i in range(100)]})
        assert HighCardinalityRule().check(df, {}) == []

    def test_multiple_columns_sorted_descending(self):
        """Multiple flagged columns → highest cardinality first."""
        df = pd.DataFrame({
            "col30": [str(i % 30) for i in range(300)],
            "col60": [str(i % 60) for i in range(300)],
        })
        flags = HighCardinalityRule().check(df, {})
        assert len(flags) == 2
        assert flags[0].column == "col60"
        assert flags[1].column == "col30"

    def test_numeric_columns_skipped(self):
        """Numeric columns are not checked for cardinality."""
        df = pd.DataFrame({"num": range(200)})
        assert HighCardinalityRule().check(df, {}) == []

    def test_empty_dataframe(self):
        """Empty DataFrame → no flag."""
        assert HighCardinalityRule().check(pd.DataFrame(), {}) == []

    def test_suggestion_present(self):
        """HIGH flag includes a non-empty suggestion."""
        df = pd.DataFrame({"cat": [str(i) for i in range(51)] * 2})
        flags = HighCardinalityRule().check(df, {})
        assert flags[0].suggestion != ""

    def test_in_default_rules(self):
        """HighCardinalityRule must be in DEFAULT_RULES."""
        class_names = [type(r).__name__ for r in DEFAULT_RULES]
        assert "HighCardinalityRule" in class_names


# ---------------------------------------------------------------------------
# RareCategoryRule (W9)
# ---------------------------------------------------------------------------

class TestRareCategoryRule:
    """Test RareCategoryRule: any level < 0.5% frequency → LOW per column."""

    def test_rare_level_flagged(self):
        """One rare level (0.2% in 500 rows) → LOW flag."""
        data = ["common_a"] * 250 + ["common_b"] * 249 + ["rare_x"]
        df = pd.DataFrame({"cat": data})
        flags = RareCategoryRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].severity == "LOW"
        assert flags[0].column == "cat"
        assert "rare_x" in flags[0].message

    def test_no_flag_when_all_common(self):
        """All categories >= 0.5% → no flag."""
        df = pd.DataFrame({"cat": [str(i % 10) for i in range(100)]})
        assert RareCategoryRule().check(df, {}) == []

    def test_small_df_skipped(self):
        """n_rows < 100 → no check (threshold meaningless at that scale)."""
        data = ["common"] * 97 + ["rare_1", "rare_2"]
        df = pd.DataFrame({"cat": data})
        assert RareCategoryRule().check(df, {}) == []

    def test_exactly_100_rows_checked(self):
        """n_rows == 100 → check is performed."""
        data = ["common"] * 99 + ["rare_x"]
        df = pd.DataFrame({"cat": data})
        # rare_x = 1% which is above 0.5% → no flag
        assert RareCategoryRule().check(df, {}) == []

    def test_truncated_examples_with_overflow(self):
        """More than 3 rare levels → shows 3 examples + '(+N more)'."""
        rare_vals = [f"rare_{i}" for i in range(5)]
        data = ["common"] * 500 + rare_vals
        df = pd.DataFrame({"cat": data})
        flags = RareCategoryRule().check(df, {})
        assert len(flags) == 1
        assert "+2 more" in flags[0].message

    def test_exactly_three_examples_no_overflow(self):
        """Exactly 3 rare levels → no '+N more' suffix."""
        rare_vals = [f"rare_{i}" for i in range(3)]
        data = ["common"] * 500 + rare_vals
        df = pd.DataFrame({"cat": data})
        flags = RareCategoryRule().check(df, {})
        assert len(flags) == 1
        assert "more" not in flags[0].message

    def test_numeric_columns_skipped(self):
        """Numeric columns are not checked for rare categories."""
        df = pd.DataFrame({"num": range(200)})
        assert RareCategoryRule().check(df, {}) == []

    def test_nan_not_counted_as_rare_level(self):
        """NaN is excluded from frequency calculation — not flagged as rare."""
        # "b" is 9/499 non-null = 1.8% which is > 0.5% → no flag
        data = ["a"] * 490 + ["b"] * 9 + [None]
        df = pd.DataFrame({"cat": data})
        assert RareCategoryRule().check(df, {}) == []

    def test_multiple_columns_each_get_own_flag(self):
        """Two columns with rare levels → two separate flags."""
        data = ["common"] * 498 + ["r1", "r2"]
        df = pd.DataFrame({"col1": data, "col2": data})
        flags = RareCategoryRule().check(df, {})
        assert len(flags) == 2
        columns_flagged = {f.column for f in flags}
        assert columns_flagged == {"col1", "col2"}

    def test_suggestion_present(self):
        """Flag includes a non-empty suggestion."""
        data = ["common"] * 499 + ["rare_x"]
        df = pd.DataFrame({"cat": data})
        flags = RareCategoryRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].suggestion != ""

    def test_in_default_rules(self):
        """RareCategoryRule must be in DEFAULT_RULES."""
        class_names = [type(r).__name__ for r in DEFAULT_RULES]
        assert "RareCategoryRule" in class_names


# ---------------------------------------------------------------------------
# _eta_squared helper (W5)
# ---------------------------------------------------------------------------

class TestEtaSquared:
    """Unit tests for the _eta_squared module-level helper."""

    def test_perfect_encoding(self):
        """Perfect group separation → eta²=1.0."""
        cat = pd.Series(["a"] * 50 + ["b"] * 50)
        num = pd.Series([0.0] * 50 + [1.0] * 50)
        assert _eta_squared(cat, num) == pytest.approx(1.0, abs=1e-9)

    def test_zero_eta(self):
        """All values same within every group → SS_between drives score.
        Uniform numeric → SS_total=0 → returns 0.0."""
        cat = pd.Series(["a", "b"] * 50)
        num = pd.Series([5.0] * 100)  # zero variance
        assert _eta_squared(cat, num) == 0.0

    def test_random_gives_low_eta(self):
        """Random numeric unrelated to groups → eta²≈0."""
        rng = np.random.default_rng(0)
        cat = pd.Series(["a", "b", "c"] * 100)
        num = pd.Series(rng.standard_normal(300))
        result = _eta_squared(cat, num)
        assert result < 0.05

    def test_nan_rows_excluded(self):
        """NaN in either series dropped; result still computes correctly."""
        cat = pd.Series(["a"] * 50 + ["b"] * 50 + [None])
        num = pd.Series([0.0] * 50 + [1.0] * 50 + [0.5])
        result = _eta_squared(cat, num)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_too_few_rows(self):
        """Fewer than 2 non-null pairs → 0.0."""
        assert _eta_squared(pd.Series(["a"]), pd.Series([1.0])) == 0.0

    def test_clamp_above_one(self):
        """Result is clamped to ≤ 1.0 regardless of floating-point overshoot."""
        # Force conditions that could cause FP > 1.0 by extreme separation
        cat = pd.Series(["x"] * 3 + ["y"] * 3)
        num = pd.Series([0.0, 0.0, 0.0, 1e15, 1e15, 1e15])
        result = _eta_squared(cat, num)
        assert result <= 1.0


# ---------------------------------------------------------------------------
# CategoricalNumericRedundancyRule (W5)
# ---------------------------------------------------------------------------

class TestCategoricalNumericRedundancyRule:
    """Test CategoricalNumericRedundancyRule: eta²>0.95 → HIGH."""

    def test_perfect_encoding_pair(self):
        """Categorical is exact int encoding of numeric → eta²=1.0 → HIGH."""
        df = pd.DataFrame({
            "cat": ["a"] * 50 + ["b"] * 50 + ["c"] * 50,
            "num": [1.0] * 50 + [2.0] * 50 + [3.0] * 50,
        })
        flags = CategoricalNumericRedundancyRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].severity == "HIGH"
        assert flags[0].column is None          # pair-level flag
        assert "cat" in flags[0].message
        assert "num" in flags[0].message
        assert flags[0].value > 0.95

    def test_no_redundancy(self):
        """Random numeric within groups → eta²≈0 → no flag."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "cat": ["a", "b", "c"] * 50,
            "num": rng.standard_normal(150).tolist(),
        })
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_below_threshold_no_flag(self):
        """Moderate group separation (eta²<0.95) → no flag."""
        df = pd.DataFrame({
            "cat": ["a"] * 50 + ["b"] * 50,
            # Overlapping ranges: many within-group obs far from group mean
            "num": list(range(50)) + list(range(25, 75)),
        })
        flags = CategoricalNumericRedundancyRule().check(df, {})
        assert flags == []

    def test_id_column_skipped(self):
        """nunique == n_rows → all-unique column skipped."""
        df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(100)],
            "num": list(range(100)),
        })
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_hyper_sparse_categorical_skipped(self):
        """nunique > n_rows // 2 → sparse groups, skip."""
        df = pd.DataFrame({
            "cat": [f"v_{i}" for i in range(60)] + [f"v_{i}" for i in range(40)],
            "num": [float(i % 3) for i in range(100)],
        })
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_no_categoricals(self):
        """Only numeric columns → no check, no flag."""
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_no_numerics(self):
        """Only categorical columns → no check, no flag."""
        df = pd.DataFrame({"cat": ["a", "b"] * 50})
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_small_dataframe_skipped(self):
        """n_rows < 10 → no check."""
        df = pd.DataFrame({
            "cat": ["a"] * 5 + ["b"] * 4,
            "num": [1.0] * 5 + [2.0] * 4,
        })
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_zero_variance_numeric_skipped(self):
        """Numeric column with std=0 → SS_total=0 → skipped, no flag."""
        df = pd.DataFrame({
            "cat": ["a"] * 50 + ["b"] * 50,
            "num": [5.0] * 100,
        })
        assert CategoricalNumericRedundancyRule().check(df, {}) == []

    def test_nan_rows_excluded(self):
        """NaN in either column dropped before eta² — result still correct."""
        df = pd.DataFrame({
            "cat": ["a"] * 50 + ["b"] * 50 + [None],
            "num": [1.0] * 50 + [2.0] * 50 + [1.5],
        })
        flags = CategoricalNumericRedundancyRule().check(df, {})
        assert len(flags) == 1
        assert flags[0].severity == "HIGH"

    def test_multiple_pairs_sorted_descending(self):
        """Two redundant pairs → both flagged, higher eta² first."""
        # Pair (cat1, num1): perfect encoding → eta²=1.0
        # Pair (cat2, num2): near-perfect but with slight noise → eta²<1.0
        df = pd.DataFrame({
            "cat1": ["x"] * 50 + ["y"] * 50 + ["z"] * 50,
            "num1": [10.0] * 50 + [20.0] * 50 + [30.0] * 50,
            "cat2": ["p"] * 50 + ["q"] * 50 + ["r"] * 50,
            # Small noise within groups but still high eta²
            "num2": [1.0] * 48 + [1.05, 1.05] + [2.0] * 48 + [2.05, 2.05] + [3.0] * 50,
        })
        flags = CategoricalNumericRedundancyRule().check(df, {})
        assert len(flags) >= 2
        # Sorted by eta² descending
        assert flags[0].value >= flags[1].value

    def test_suggestion_references_both_columns(self):
        """Flag suggestion names both the categorical and numeric columns."""
        df = pd.DataFrame({
            "education": ["a"] * 50 + ["b"] * 50,
            "education_num": [0.0] * 50 + [1.0] * 50,
        })
        flags = CategoricalNumericRedundancyRule().check(df, {})
        assert len(flags) == 1
        assert "education" in flags[0].suggestion
        assert "education_num" in flags[0].suggestion

    def test_rule_name(self):
        """Rule name string is 'categorical_numeric_redundancy'."""
        assert CategoricalNumericRedundancyRule().name == "categorical_numeric_redundancy"

    def test_in_default_rules(self):
        """CategoricalNumericRedundancyRule must be in DEFAULT_RULES."""
        class_names = [type(r).__name__ for r in DEFAULT_RULES]
        assert "CategoricalNumericRedundancyRule" in class_names
