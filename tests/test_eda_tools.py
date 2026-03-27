"""
tests/test_eda_tools.py — Unit tests for tools/eda_tools.py

Tests the three EDA tool functions: describe_stats, missing_analysis, correlation_matrix.
Validates outputs against Pydantic sub-models (EDAResults, MissingInfo).
No LLM calls — pure function tests.
"""

import json

import pandas as pd
import pytest

from eda_state import CategoricalAnalysis, EDAResults, MissingInfo, TargetInfo
from tools.eda_tools import (
    analyze_categoricals,
    correlation_matrix,
    describe_stats,
    missing_analysis,
    target_analysis,
)


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


# ---------------------------------------------------------------------------
# target_analysis()
# ---------------------------------------------------------------------------


class TestTargetAnalysis:
    """Test target_analysis() for classification, regression, unsupervised."""

    @pytest.fixture()
    def classification_df_json(self):
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat_b": [10, 20, 30, 40, 50, 60],
            "species": ["setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica"],
        })
        return df.to_json(orient="records")

    @pytest.fixture()
    def regression_df_json(self):
        df = pd.DataFrame({
            "size": list(range(50)),
            "rooms": [x % 5 + 1 for x in range(50)],
            "price": [x * 1000.0 for x in range(50)],
        })
        return df.to_json(orient="records")

    @pytest.fixture()
    def classification_target_info_json(self):
        ti = TargetInfo(
            column="species",
            problem_type="classification",
            n_classes=3,
            class_counts={"setosa": 2, "versicolor": 2, "virginica": 2},
            imbalance_ratio=1.0,
            detection_method="name_heuristic",
        )
        return ti.model_dump_json()

    @pytest.fixture()
    def regression_target_info_json(self):
        ti = TargetInfo(
            column="price",
            problem_type="regression",
            n_classes=0,
            detection_method="name_heuristic",
        )
        return ti.model_dump_json()

    @pytest.fixture()
    def unsupervised_target_info_json(self):
        ti = TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
        )
        return ti.model_dump_json()

    def test_classification_returns_valid_json(self, classification_df_json, classification_target_info_json):
        result = json.loads(target_analysis(classification_df_json, classification_target_info_json))
        assert result["problem_type"] == "classification"
        assert result["column"] == "species"

    def test_classification_class_distribution(self, classification_df_json, classification_target_info_json):
        result = json.loads(target_analysis(classification_df_json, classification_target_info_json))
        dist = result["class_distribution"]
        assert len(dist) == 3
        for cls_val in dist.values():
            assert "count" in cls_val
            assert "pct" in cls_val

    def test_classification_per_class_stats(self, classification_df_json, classification_target_info_json):
        result = json.loads(target_analysis(classification_df_json, classification_target_info_json))
        assert "per_class_feature_stats" in result
        per_class = result["per_class_feature_stats"]
        assert "setosa" in per_class
        assert "feat_a" in per_class["setosa"]
        assert "mean" in per_class["setosa"]["feat_a"]

    def test_regression_returns_valid_json(self, regression_df_json, regression_target_info_json):
        result = json.loads(target_analysis(regression_df_json, regression_target_info_json))
        assert result["problem_type"] == "regression"
        assert result["column"] == "price"

    def test_regression_target_stats(self, regression_df_json, regression_target_info_json):
        result = json.loads(target_analysis(regression_df_json, regression_target_info_json))
        stats = result["target_stats"]
        for key in ("mean", "median", "std", "skewness", "kurtosis", "min", "max"):
            assert key in stats

    def test_regression_correlations(self, regression_df_json, regression_target_info_json):
        result = json.loads(target_analysis(regression_df_json, regression_target_info_json))
        assert "feature_target_correlations" in result
        assert "top_correlated_features" in result
        assert len(result["top_correlated_features"]) <= 3

    def test_unsupervised_returns_empty(self, classification_df_json, unsupervised_target_info_json):
        result = json.loads(target_analysis(classification_df_json, unsupervised_target_info_json))
        assert result["problem_type"] == "unsupervised"

    def test_missing_column_treated_as_unsupervised(self, classification_df_json):
        ti = TargetInfo(column="nonexistent", problem_type="classification")
        result = json.loads(target_analysis(classification_df_json, ti.model_dump_json()))
        assert result["problem_type"] == "unsupervised"


# ---------------------------------------------------------------------------
# analyze_categoricals()
# ---------------------------------------------------------------------------

class TestAnalyzeCategoricals:
    """Test analyze_categoricals() function (W4)."""

    @pytest.fixture()
    def mixed_df_json(self):
        """DataFrame with categoricals + numerics + a classification target."""
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "green", "blue",
                       "red", "blue", "red", "green", "blue"] * 10,
            "size": ["S", "M", "L", "S", "M",
                      "S", "M", "L", "S", "M"] * 10,
            "price": [10.0, 20.0, 30.0, 40.0, 50.0,
                       10.0, 20.0, 30.0, 40.0, 50.0] * 10,
            "target": ["yes", "no", "yes", "no", "yes",
                        "no", "yes", "no", "yes", "no"] * 10,
        })
        return df.to_json(orient="records")

    @pytest.fixture()
    def classification_ti_json(self):
        return TargetInfo(
            column="target", problem_type="classification",
            n_classes=2, imbalance_ratio=1.0,
        ).model_dump_json()

    @pytest.fixture()
    def unsupervised_ti_json(self):
        return TargetInfo().model_dump_json()

    def test_returns_valid_json(self, mixed_df_json, classification_ti_json):
        result = analyze_categoricals(mixed_df_json, classification_ti_json)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_validates_via_pydantic(self, mixed_df_json, classification_ti_json):
        result = analyze_categoricals(mixed_df_json, classification_ti_json)
        ca = CategoricalAnalysis.model_validate_json(result)
        assert isinstance(ca, CategoricalAnalysis)

    def test_detects_categorical_columns(self, mixed_df_json, classification_ti_json):
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(mixed_df_json, classification_ti_json)
        )
        # color, size, target are categorical (object dtype)
        assert "color" in ca.columns
        assert "size" in ca.columns
        assert "target" in ca.columns
        # price is numeric — should NOT be in columns
        assert "price" not in ca.columns

    def test_cardinality(self, mixed_df_json, classification_ti_json):
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(mixed_df_json, classification_ti_json)
        )
        assert ca.columns["color"].cardinality == 3
        assert ca.columns["size"].cardinality == 3
        assert ca.columns["target"].cardinality == 2

    def test_entropy_positive(self, mixed_df_json, classification_ti_json):
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(mixed_df_json, classification_ti_json)
        )
        for col in ("color", "size", "target"):
            assert ca.columns[col].entropy_bits > 0

    def test_top_values_populated(self, mixed_df_json, classification_ti_json):
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(mixed_df_json, classification_ti_json)
        )
        for col in ("color", "size"):
            assert len(ca.columns[col].top_values) > 0
            for v in ca.columns[col].top_values:
                assert "value" in v
                assert "count" in v
                assert "pct" in v

    def test_target_rates_present_for_classification(self, mixed_df_json, classification_ti_json):
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(mixed_df_json, classification_ti_json)
        )
        assert ca.target_column == "target"
        # Non-target categorical should have target_rates
        for v in ca.columns["color"].top_values:
            assert "target_rates" in v
            assert "yes" in v["target_rates"] or "no" in v["target_rates"]

    def test_no_target_rates_for_unsupervised(self, mixed_df_json, unsupervised_ti_json):
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(mixed_df_json, unsupervised_ti_json)
        )
        assert ca.target_column is None
        for v in ca.columns["color"].top_values:
            assert "target_rates" not in v

    def test_top_n_cap(self, classification_ti_json):
        """High-cardinality column should be capped at top-10."""
        df = pd.DataFrame({
            "city": [f"city_{i}" for i in range(500)],
            "target": ["yes", "no"] * 250,
        })
        data_json = df.to_json(orient="records")
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(data_json, classification_ti_json)
        )
        assert len(ca.columns["city"].top_values) <= 10
        assert ca.columns["city"].more_values == 500 - 10

    def test_rare_count(self):
        """Values below 0.5% threshold should be counted as rare."""
        # 200 rows: 199 are 'A', 1 is 'B' → B is 0.5%, just at boundary
        # Use 201 rows: 200 are 'A', 1 is 'B' → B is ~0.497% < 0.5% — rare
        df = pd.DataFrame({
            "x": ["A"] * 200 + ["B"],
        })
        ti = TargetInfo().model_dump_json()
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(df.to_json(orient="records"), ti)
        )
        assert ca.columns["x"].rare_count == 1

    def test_empty_categorical(self):
        """DataFrame with only numeric columns → empty columns dict."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ti = TargetInfo().model_dump_json()
        ca = CategoricalAnalysis.model_validate_json(
            analyze_categoricals(df.to_json(orient="records"), ti)
        )
        assert len(ca.columns) == 0


# ---------------------------------------------------------------------------
# Integration tests: analyze_categoricals() with active pipeline session
# ---------------------------------------------------------------------------

class TestAnalyzeCategoricalsIntegration:
    """Verify the is_active() artifact-store path (the production path).

    Pass 4 of the root-cause analysis showed that unit tests never exercise
    the pipeline-active branch, which reads DataProfile from the store.
    These tests ensure the correct artifact key ('dtypes_json') is used and
    that an empty categorical_cols falls back to select_dtypes.
    """

    @pytest.fixture()
    def mixed_df(self):
        return pd.DataFrame({
            "color": ["red", "blue", "green"] * 34,
            "size": ["S", "M", "L"] * 34,
            "price": [10.0, 20.0, 30.0] * 34,
            "target": ["yes", "no", "yes"] * 34,
        })

    def _setup_dtypes_json(self, save_state, df):
        """Save a DataProfile with categorical_cols populated under dtypes_json."""
        from eda_state import DataProfile
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        profile = DataProfile(
            shape=(df.shape[0], df.shape[1]),
            memory_mb=1.0,
            dtypes={c: str(df[c].dtype) for c in df.columns},
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
        )
        save_state("dtypes_json", profile.model_dump_json())

    def _setup_schema_json_only(self, save_state, df):
        """Save a DataProfile WITHOUT categorical_cols under schema_json only.

        Simulates validate_schema() output — categorical_cols stays [].
        """
        from eda_state import DataProfile
        profile = DataProfile(
            shape=(df.shape[0], df.shape[1]),
            memory_mb=1.0,
            dtypes={c: str(df[c].dtype) for c in df.columns},
        )
        save_state("schema_json", profile.model_dump_json())

    def test_uses_dtypes_json_when_available(self, mixed_df):
        """Pipeline path reads categorical_cols from dtypes_json artifact."""
        from tools._pipeline_state import init_session, clear_session, save_state
        try:
            init_session()
            data_json = mixed_df.to_json(orient="records")
            save_state("data_json", data_json)
            self._setup_dtypes_json(save_state, mixed_df)
            ti = TargetInfo().model_dump_json()
            save_state("target_info", ti)

            result = analyze_categoricals("STATE_REF:data_json", "STATE_REF:target_info")
            assert "STATE_REF:categorical_analysis" in result

            from tools._pipeline_state import load_state
            ca = CategoricalAnalysis.model_validate_json(load_state("categorical_analysis"))
            assert "color" in ca.columns
            assert "size" in ca.columns
            assert "target" in ca.columns
            assert "price" not in ca.columns
        finally:
            clear_session()

    def test_schema_json_without_categorical_cols_falls_back_to_dataframe(self, mixed_df):
        """Regression: schema_json has empty categorical_cols → must fall back to select_dtypes.

        This is the exact bug from the production run: validate_schema() saves
        schema_json with categorical_cols=[] and the old code returned 0 columns.
        """
        from tools._pipeline_state import init_session, clear_session, save_state
        try:
            init_session()
            data_json = mixed_df.to_json(orient="records")
            save_state("data_json", data_json)
            # Only schema_json present (no dtypes_json) — as in validate_schema only
            self._setup_schema_json_only(save_state, mixed_df)
            ti = TargetInfo().model_dump_json()
            save_state("target_info", ti)

            _result = analyze_categoricals("STATE_REF:data_json", "STATE_REF:target_info")

            from tools._pipeline_state import load_state
            ca = CategoricalAnalysis.model_validate_json(load_state("categorical_analysis"))
            # Fallback to select_dtypes must find color, size, target
            assert len(ca.columns) > 0, (
                "Bug regression: categorical_cols must not be empty when schema_json "
                "has categorical_cols=[] — fallback to select_dtypes must fire."
            )
            assert "color" in ca.columns
        finally:
            clear_session()

    def test_empty_dtypes_json_categorical_cols_falls_back(self, mixed_df):
        """dtypes_json exists but has empty categorical_cols → fallback fires."""
        from tools._pipeline_state import init_session, clear_session, save_state
        from eda_state import DataProfile
        try:
            init_session()
            data_json = mixed_df.to_json(orient="records")
            save_state("data_json", data_json)
            # DataProfile with explicitly empty categorical_cols
            profile = DataProfile(
                shape=(mixed_df.shape[0], mixed_df.shape[1]),
                memory_mb=1.0,
                dtypes={c: str(mixed_df[c].dtype) for c in mixed_df.columns},
                categorical_cols=[],     # intentionally empty
            )
            save_state("dtypes_json", profile.model_dump_json())
            ti = TargetInfo().model_dump_json()
            save_state("target_info", ti)

            _result = analyze_categoricals("STATE_REF:data_json", "STATE_REF:target_info")

            from tools._pipeline_state import load_state
            ca = CategoricalAnalysis.model_validate_json(load_state("categorical_analysis"))
            assert "color" in ca.columns, "select_dtypes fallback must populate columns"
        finally:
            clear_session()


# ---------------------------------------------------------------------------
# compute_interaction_signals() (A4)
# ---------------------------------------------------------------------------


class TestDetectColumnFamilies:
    """Test the _detect_column_families helper."""

    def test_basic_family(self):
        from tools.eda_tools import _detect_column_families
        cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        families = _detect_column_families(cols)
        assert "PAY" in families
        assert len(families["PAY"]) == 6

    def test_no_families_below_threshold(self):
        from tools.eda_tools import _detect_column_families
        cols = ["PAY_0", "PAY_1"]
        families = _detect_column_families(cols)
        assert len(families) == 0

    def test_multiple_families(self):
        from tools.eda_tools import _detect_column_families
        cols = ["PAY_0", "PAY_1", "PAY_2", "BILL_0", "BILL_1", "BILL_2"]
        families = _detect_column_families(cols)
        assert "PAY" in families
        assert "BILL" in families

    def test_non_numeric_suffix_ignored(self):
        from tools.eda_tools import _detect_column_families
        cols = ["price", "color", "size"]
        families = _detect_column_families(cols)
        assert len(families) == 0

    def test_sorted_by_index(self):
        from tools.eda_tools import _detect_column_families
        cols = ["X_3", "X_1", "X_0", "X_2"]
        families = _detect_column_families(cols)
        assert families["X"] == ["X_0", "X_1", "X_2", "X_3"]


class TestComputeInteractionSignals:
    """Test compute_interaction_signals() without pipeline session."""

    @pytest.fixture()
    def classification_df_json(self):
        """DataFrame with column families and a binary target."""
        import numpy as np
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "PAY_0": np.random.choice([0, 1, 2, 3], n),
            "PAY_1": np.random.choice([0, 1, 2, 3], n),
            "PAY_2": np.random.choice([0, 1, 2, 3], n),
            "other_feat": np.random.randn(n),
        })
        # Create a target correlated with PAY columns
        df["default"] = ((df["PAY_0"] + df["PAY_1"] + df["PAY_2"]) > 3).astype(int)
        return df.to_json(orient="records")

    @pytest.fixture()
    def no_family_df_json(self):
        """DataFrame with no column families."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45] * 20,
            "income": [50000, 60000, 70000, 80000, 90000] * 20,
            "target": [0, 1, 0, 1, 0] * 20,
        })
        return df.to_json(orient="records")

    def test_unsupervised_returns_empty(self):
        from tools.eda_tools import compute_interaction_signals
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ti = TargetInfo(column=None, problem_type="unsupervised")
        result = json.loads(compute_interaction_signals(
            df.to_json(orient="records"), ti.model_dump_json()
        ))
        assert result["signals"] == []

    def test_no_families_returns_empty_signals(self, no_family_df_json):
        from tools.eda_tools import compute_interaction_signals
        ti = TargetInfo(column="target", problem_type="classification", n_classes=2)
        result = json.loads(compute_interaction_signals(
            no_family_df_json, ti.model_dump_json()
        ))
        assert result["n_families"] == 0
        # No column families → no persistence/trajectory signals (may have cross-feature from fallback)
        family_signals = [s for s in result["signals"] if s["type"] in ("persistence_gradient", "trajectory")]
        assert len(family_signals) == 0

    def test_classification_with_families(self, classification_df_json):
        from tools.eda_tools import compute_interaction_signals
        ti = TargetInfo(column="default", problem_type="classification", n_classes=2)
        result = json.loads(compute_interaction_signals(
            classification_df_json, ti.model_dump_json()
        ))
        assert result["n_families"] == 1
        assert "PAY" in result["families_detected"]
        assert "overall_target_rate_pct" in result

    def test_regression_target_median_split(self):
        """Regression targets are median-split into binary for segment analysis."""
        from tools.eda_tools import compute_interaction_signals
        import numpy as np
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "X_0": np.random.randn(n),
            "X_1": np.random.randn(n),
            "X_2": np.random.randn(n),
            "price": np.random.randn(n) * 100 + 500,
        })
        ti = TargetInfo(column="price", problem_type="regression")
        result = json.loads(compute_interaction_signals(
            df.to_json(orient="records"), ti.model_dump_json()
        ))
        assert "overall_target_rate_pct" in result
        # Median split means ~50% target rate
        assert 40 <= result["overall_target_rate_pct"] <= 60
