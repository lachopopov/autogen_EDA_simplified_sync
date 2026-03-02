"""
tests/test_eda_tools.py — Unit tests for tools/eda_tools.py

Tests the three EDA tool functions: describe_stats, missing_analysis, correlation_matrix.
Validates outputs against Pydantic sub-models (EDAResults, MissingInfo).
No LLM calls — pure function tests.
"""

import json

import pandas as pd
import pytest

from eda_state import EDAResults, MissingInfo
from tools.eda_tools import correlation_matrix, describe_stats, missing_analysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df_json():
    """A small mixed-type DataFrame as JSON (records orientation)."""
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num_b": [10, 20, 30, 40, 50],
        "cat_a": ["x", "y", "z", "x", "y"],
    })
    return df.to_json(orient="records")


@pytest.fixture()
def missing_df_json():
    """A DataFrame with missing values as JSON."""
    df = pd.DataFrame({
        "a": [1.0, None, 3.0, None, 5.0],
        "b": [10, 20, None, 40, 50],
        "c": ["x", None, "z", None, None],
    })
    return df.to_json(orient="records")


@pytest.fixture()
def no_missing_df_json():
    """A DataFrame with zero missing values as JSON."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return df.to_json(orient="records")


@pytest.fixture()
def categorical_only_json():
    """A DataFrame with only categorical columns (no numerical)."""
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
    return df.to_json(orient="records")


@pytest.fixture()
def single_numerical_json():
    """A DataFrame with a single numerical column."""
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
    return df.to_json(orient="records")


@pytest.fixture()
def empty_df_json():
    """An empty DataFrame as JSON."""
    return "[]"


# ---------------------------------------------------------------------------
# describe_stats()
# ---------------------------------------------------------------------------

class TestDescribeStats:
    """Test describe_stats() function."""

    def test_returns_json_string(self, simple_df_json):
        result = describe_stats(simple_df_json)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_includes_all_columns(self, simple_df_json):
        result = json.loads(describe_stats(simple_df_json))
        assert "num_a" in result
        assert "num_b" in result
        assert "cat_a" in result

    def test_numerical_stats(self, simple_df_json):
        result = json.loads(describe_stats(simple_df_json))
        num_a = result["num_a"]
        assert "count" in num_a
        assert "mean" in num_a
        assert "std" in num_a
        assert "min" in num_a
        assert "max" in num_a

    def test_mean_value(self, simple_df_json):
        result = json.loads(describe_stats(simple_df_json))
        assert result["num_a"]["mean"] == 3.0

    def test_categorical_stats(self, simple_df_json):
        result = json.loads(describe_stats(simple_df_json))
        cat_a = result["cat_a"]
        assert "count" in cat_a
        assert "unique" in cat_a
        assert "top" in cat_a
        assert "freq" in cat_a

    def test_nan_serialized_as_null(self, simple_df_json):
        """Categorical columns have null for mean/std; numerical have null for unique/top."""
        result = json.loads(describe_stats(simple_df_json))
        # Numerical column should have null for 'unique' (not applicable)
        assert result["num_a"].get("unique") is None
        # Categorical column should have null for 'mean' (not applicable)
        assert result["cat_a"].get("mean") is None

    def test_validates_via_eda_results(self, simple_df_json):
        """Output must be valid as EDAResults.describe field."""
        result = json.loads(describe_stats(simple_df_json))
        eda = EDAResults(describe=result)
        assert len(eda.describe) == 3

    def test_empty_dataframe(self, empty_df_json):
        result = json.loads(describe_stats(empty_df_json))
        assert result == {}

    def test_single_column(self, single_numerical_json):
        result = json.loads(describe_stats(single_numerical_json))
        assert "val" in result
        assert result["val"]["count"] == 5.0

    def test_with_missing_values(self, missing_df_json):
        """describe() should reflect reduced count for columns with missing values."""
        result = json.loads(describe_stats(missing_df_json))
        # Column 'a' has 3 non-null values out of 5
        assert result["a"]["count"] == 3.0


# ---------------------------------------------------------------------------
# missing_analysis()
# ---------------------------------------------------------------------------

class TestMissingAnalysis:
    """Test missing_analysis() function."""

    def test_returns_json_string(self, missing_df_json):
        result = missing_analysis(missing_df_json)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_returns_valid_missing_info(self, missing_df_json):
        result = missing_analysis(missing_df_json)
        info = MissingInfo.model_validate_json(result)
        assert isinstance(info, MissingInfo)

    def test_per_column_keys(self, missing_df_json):
        info = MissingInfo.model_validate_json(missing_analysis(missing_df_json))
        assert "a" in info.per_column
        assert "b" in info.per_column
        assert "c" in info.per_column

    def test_per_column_values(self, missing_df_json):
        """Column 'a' has 2/5 missing = 40%, 'b' has 1/5 = 20%, 'c' has 3/5 = 60%."""
        info = MissingInfo.model_validate_json(missing_analysis(missing_df_json))
        assert info.per_column["a"] == 40.0
        assert info.per_column["b"] == 20.0
        assert info.per_column["c"] == 60.0

    def test_total_pct(self, missing_df_json):
        """Total: 6 missing out of 15 cells = 40%."""
        info = MissingInfo.model_validate_json(missing_analysis(missing_df_json))
        assert info.total_pct == 40.0

    def test_no_missing(self, no_missing_df_json):
        info = MissingInfo.model_validate_json(missing_analysis(no_missing_df_json))
        assert info.total_pct == 0.0
        assert all(v == 0.0 for v in info.per_column.values())

    def test_empty_dataframe(self, empty_df_json):
        info = MissingInfo.model_validate_json(missing_analysis(empty_df_json))
        assert info.total_pct == 0.0
        assert info.per_column == {}

    def test_all_missing(self):
        """DataFrame where every cell is NaN."""
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        data_json = df.to_json(orient="records")
        info = MissingInfo.model_validate_json(missing_analysis(data_json))
        assert info.total_pct == 100.0
        assert info.per_column["a"] == 100.0
        assert info.per_column["b"] == 100.0

    def test_single_column_partial_missing(self):
        df = pd.DataFrame({"x": [1, None, 3, None]})
        data_json = df.to_json(orient="records")
        info = MissingInfo.model_validate_json(missing_analysis(data_json))
        assert info.per_column["x"] == 50.0
        assert info.total_pct == 50.0


# ---------------------------------------------------------------------------
# correlation_matrix()
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    """Test correlation_matrix() function."""

    def test_returns_json_string(self, simple_df_json):
        result = correlation_matrix(simple_df_json)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_numerical_columns_only(self, simple_df_json):
        """Only num_a and num_b should appear (cat_a excluded)."""
        result = json.loads(correlation_matrix(simple_df_json))
        assert "num_a" in result
        assert "num_b" in result
        assert "cat_a" not in result

    def test_perfect_correlation(self, simple_df_json):
        """num_a and num_b are perfectly linearly correlated (both ascending)."""
        result = json.loads(correlation_matrix(simple_df_json))
        assert result["num_a"]["num_b"] == 1.0
        assert result["num_b"]["num_a"] == 1.0

    def test_diagonal_is_one(self, simple_df_json):
        result = json.loads(correlation_matrix(simple_df_json))
        assert result["num_a"]["num_a"] == 1.0
        assert result["num_b"]["num_b"] == 1.0

    def test_symmetry(self, simple_df_json):
        result = json.loads(correlation_matrix(simple_df_json))
        assert result["num_a"]["num_b"] == result["num_b"]["num_a"]

    def test_validates_via_eda_results(self, simple_df_json):
        """Output must be valid as EDAResults.correlation field."""
        result = json.loads(correlation_matrix(simple_df_json))
        eda = EDAResults(correlation=result)
        assert len(eda.correlation) == 2

    def test_categorical_only_returns_empty(self, categorical_only_json):
        """No numerical columns → empty correlation dict."""
        result = json.loads(correlation_matrix(categorical_only_json))
        assert result == {}

    def test_single_numerical_column(self, single_numerical_json):
        """Single numerical column → 1×1 correlation matrix."""
        result = json.loads(correlation_matrix(single_numerical_json))
        assert "val" in result
        assert result["val"]["val"] == 1.0

    def test_empty_dataframe(self, empty_df_json):
        result = json.loads(correlation_matrix(empty_df_json))
        assert result == {}

    def test_negative_correlation(self):
        """Perfectly negatively correlated columns."""
        df = pd.DataFrame({"up": [1, 2, 3, 4, 5], "down": [5, 4, 3, 2, 1]})
        data_json = df.to_json(orient="records")
        result = json.loads(correlation_matrix(data_json))
        assert result["up"]["down"] == -1.0

    def test_constant_column_nan_handling(self):
        """Constant column has zero variance → NaN correlation → serialized as null."""
        df = pd.DataFrame({"const": [5, 5, 5, 5], "vary": [1, 2, 3, 4]})
        data_json = df.to_json(orient="records")
        result = json.loads(correlation_matrix(data_json))
        # Correlation between constant and varying is NaN → null
        assert result["const"]["vary"] is None
        assert result["vary"]["const"] is None
        # Constant with itself is also NaN → null
        assert result["const"]["const"] is None


# ---------------------------------------------------------------------------
# Hard Boundary Rule: zero AG2 imports (architecture.md § 12.1)
# ---------------------------------------------------------------------------

class TestHardBoundaryRule:
    """Verify tools/eda_tools.py has zero AG2 imports."""

    def test_no_autogen_import(self):
        import importlib
        import inspect

        mod = importlib.import_module("tools.eda_tools")
        source = inspect.getsource(mod)
        assert "import autogen" not in source
        assert "from autogen" not in source


# ---------------------------------------------------------------------------
# End-to-end: load_data output feeds into all 3 EDA tools
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Test that load_data() output can be consumed by all 3 EDA tools."""

    @pytest.fixture()
    def csv_path(self, tmp_path):
        df = pd.DataFrame({
            "age": [25, 30, None, 40, 35],
            "salary": [50000, 60000, 70000, None, 55000],
            "dept": ["eng", "sales", "eng", "hr", None],
        })
        p = tmp_path / "employees.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_load_then_describe(self, csv_path):
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = json.loads(describe_stats(data_json))
        assert "age" in result
        assert "salary" in result
        assert "dept" in result

    def test_load_then_missing(self, csv_path):
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        info = MissingInfo.model_validate_json(missing_analysis(data_json))
        assert info.per_column["age"] == 20.0
        assert info.per_column["salary"] == 20.0
        assert info.per_column["dept"] == 20.0
        assert info.total_pct == 20.0

    def test_load_then_correlation(self, csv_path):
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = json.loads(correlation_matrix(data_json))
        assert "age" in result
        assert "salary" in result
        assert "dept" not in result  # categorical excluded

    def test_full_eda_pipeline(self, csv_path):
        """End-to-end: load → describe + missing + correlation."""
        from tools.data_loader import load_data

        data_json = load_data(csv_path)

        # All 3 tools consume the same load_data output
        desc = json.loads(describe_stats(data_json))
        miss = MissingInfo.model_validate_json(missing_analysis(data_json))
        corr = json.loads(correlation_matrix(data_json))

        # Assemble into EDAResults
        eda = EDAResults(describe=desc, missing=miss, correlation=corr)
        assert len(eda.describe) == 3
        assert eda.missing.total_pct == 20.0
        assert "age" in eda.correlation
