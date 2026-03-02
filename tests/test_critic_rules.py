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
    CriticRule,
    DatasetMissingnessRule,
    DuplicateRowsRule,
    MissingValueRule,
    NearPerfectCorrelationRule,
    NearZeroVarianceRule,
    OutlierRule,
    SingleValueCategoricalRule,
    SkewnessRule,
    ZeroVarianceRule,
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
    """Test MissingValueRule: per-column missing thresholds."""

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

    def test_returns_worst_column(self):
        """Flag reports the column with highest missing %."""
        df = pd.DataFrame({
            "low_missing": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # 10%
            "high_missing": [None, None, None, None, None, None, 7, 8, 9, 10],  # 60%
        })
        flags = MissingValueRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.column == "high_missing"
        assert flag.severity == "BLOCKER"


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
    """Test DuplicateRowsRule: >1% duplicate rows → MEDIUM."""

    def test_medium_severity(self):
        """More than 1% duplicate rows → MEDIUM."""
        # 5 duplicates in 10 rows = 50%
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]})
        flags = DuplicateRowsRule().check(df, {})
        assert len(flags) == 1
        flag = flags[0]
        assert flag.severity == "MEDIUM"
        assert flag.column is None  # dataset-level

    def test_no_flag_no_duplicates(self, clean_df):
        """No duplicate rows → no flag."""
        assert DuplicateRowsRule().check(clean_df, {}) == []

    def test_empty_dataframe(self, empty_df):
        """Empty DataFrame → no flag."""
        assert DuplicateRowsRule().check(empty_df, {}) == []


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
        """All 10 V1 rules are registered."""
        assert len(DEFAULT_RULES) == 10

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
        # Duplicate rows → MEDIUM; columns uncorrelated to avoid HIGH correlation
        df = pd.DataFrame({
            "a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [5, 5, 8, 1, 7, 2, 9, 4, 3, 6],
        })
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
