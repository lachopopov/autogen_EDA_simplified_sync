"""
tests/test_visualization_tools.py — Unit tests for tools/visualization_tools.py

Tests the three visualization tool functions: plot_histograms,
plot_correlation_heatmap, plot_missing_heatmap.
Validates that PNGs are created on disk and returned paths match.
No LLM calls — pure function tests.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.visualization_tools import (
    plot_categorical_bars,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_histograms,
    plot_missing_heatmap,
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
def correlation_json():
    """A correlation matrix JSON (2 numerical columns, perfectly correlated)."""
    return json.dumps({
        "num_a": {"num_a": 1.0, "num_b": 1.0},
        "num_b": {"num_a": 1.0, "num_b": 1.0},
    })


@pytest.fixture()
def missing_json():
    """A MissingInfo JSON with per-column missing percentages."""
    return json.dumps({
        "per_column": {"a": 20.0, "b": 0.0, "c": 60.0},
        "total_pct": 26.67,
    })


@pytest.fixture()
def plots_dir(tmp_path):
    """A temporary directory for plot output."""
    return str(tmp_path / "plots")


# ---------------------------------------------------------------------------
# plot_histograms()
# ---------------------------------------------------------------------------

class TestPlotHistograms:
    """Test plot_histograms() function."""

    def test_returns_json_list(self, simple_df_json, plots_dir):
        result = plot_histograms(simple_df_json, plots_dir)
        paths = json.loads(result)
        assert isinstance(paths, list)

    def test_creates_png_per_numerical_column(self, simple_df_json, plots_dir):
        result = plot_histograms(simple_df_json, plots_dir)
        paths = json.loads(result)
        # 2 numerical columns: num_a, num_b
        assert len(paths) == 2

    def test_files_exist_on_disk(self, simple_df_json, plots_dir):
        result = plot_histograms(simple_df_json, plots_dir)
        paths = json.loads(result)
        for p in paths:
            assert Path(p).exists(), f"File not found: {p}"

    def test_file_names(self, simple_df_json, plots_dir):
        result = plot_histograms(simple_df_json, plots_dir)
        paths = json.loads(result)
        names = {Path(p).name for p in paths}
        assert "hist_num_a.png" in names
        assert "hist_num_b.png" in names

    def test_files_are_png(self, simple_df_json, plots_dir):
        result = plot_histograms(simple_df_json, plots_dir)
        paths = json.loads(result)
        for p in paths:
            assert Path(p).suffix == ".png"

    def test_files_nonzero_size(self, simple_df_json, plots_dir):
        result = plot_histograms(simple_df_json, plots_dir)
        paths = json.loads(result)
        for p in paths:
            assert Path(p).stat().st_size > 0

    def test_categorical_only_returns_empty(self, plots_dir):
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        result = plot_histograms(df.to_json(orient="records"), plots_dir)
        paths = json.loads(result)
        assert paths == []

    def test_empty_dataframe_returns_empty(self, plots_dir):
        result = plot_histograms("[]", plots_dir)
        paths = json.loads(result)
        assert paths == []

    def test_creates_output_dir(self, simple_df_json, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        result = plot_histograms(simple_df_json, nested)
        paths = json.loads(result)
        assert len(paths) == 2
        assert Path(nested).is_dir()

    def test_with_missing_values(self, plots_dir):
        """Histograms should handle NaN values (dropna before plotting)."""
        df = pd.DataFrame({"val": [1.0, None, 3.0, None, 5.0]})
        result = plot_histograms(df.to_json(orient="records"), plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()


# ---------------------------------------------------------------------------
# plot_correlation_heatmap()
# ---------------------------------------------------------------------------

class TestPlotCorrelationHeatmap:
    """Test plot_correlation_heatmap() function."""

    def test_returns_json_list(self, correlation_json, plots_dir):
        result = plot_correlation_heatmap(correlation_json, plots_dir)
        paths = json.loads(result)
        assert isinstance(paths, list)

    def test_creates_one_png(self, correlation_json, plots_dir):
        result = plot_correlation_heatmap(correlation_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1

    def test_file_exists_on_disk(self, correlation_json, plots_dir):
        result = plot_correlation_heatmap(correlation_json, plots_dir)
        paths = json.loads(result)
        assert Path(paths[0]).exists()

    def test_file_name(self, correlation_json, plots_dir):
        result = plot_correlation_heatmap(correlation_json, plots_dir)
        paths = json.loads(result)
        assert Path(paths[0]).name == "correlation_heatmap.png"

    def test_file_nonzero_size(self, correlation_json, plots_dir):
        result = plot_correlation_heatmap(correlation_json, plots_dir)
        paths = json.loads(result)
        assert Path(paths[0]).stat().st_size > 0

    def test_empty_correlation_returns_empty(self, plots_dir):
        result = plot_correlation_heatmap(json.dumps({}), plots_dir)
        paths = json.loads(result)
        assert paths == []

    def test_creates_output_dir(self, correlation_json, tmp_path):
        nested = str(tmp_path / "deep" / "dir")
        result = plot_correlation_heatmap(correlation_json, nested)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(nested).is_dir()

    def test_single_column_correlation(self, plots_dir):
        """1×1 correlation matrix (single numerical column)."""
        corr = json.dumps({"val": {"val": 1.0}})
        result = plot_correlation_heatmap(corr, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_null_values_coerced_to_float(self, plots_dir):
        """Correlation matrix with null values (NaN from JSON round-trip) must not crash."""
        corr = json.dumps({
            "a": {"a": 1.0, "b": None},
            "b": {"a": None, "b": 1.0},
        })
        result = plot_correlation_heatmap(corr, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_string_values_coerced_to_nan(self, plots_dir):
        """Correlation values that are strings should be coerced, not crash."""
        corr = json.dumps({
            "x": {"x": "1.0", "y": "0.5"},
            "y": {"x": "0.5", "y": "1.0"},
        })
        result = plot_correlation_heatmap(corr, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()


# ---------------------------------------------------------------------------
# plot_missing_heatmap()
# ---------------------------------------------------------------------------

class TestPlotMissingHeatmap:
    """Test plot_missing_heatmap() function."""

    def test_returns_json_list(self, missing_json, plots_dir):
        result = plot_missing_heatmap(missing_json, plots_dir)
        paths = json.loads(result)
        assert isinstance(paths, list)

    def test_creates_one_png(self, missing_json, plots_dir):
        result = plot_missing_heatmap(missing_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1

    def test_file_exists_on_disk(self, missing_json, plots_dir):
        result = plot_missing_heatmap(missing_json, plots_dir)
        paths = json.loads(result)
        assert Path(paths[0]).exists()

    def test_file_name(self, missing_json, plots_dir):
        result = plot_missing_heatmap(missing_json, plots_dir)
        paths = json.loads(result)
        assert Path(paths[0]).name == "missing_heatmap.png"

    def test_file_nonzero_size(self, missing_json, plots_dir):
        result = plot_missing_heatmap(missing_json, plots_dir)
        paths = json.loads(result)
        assert Path(paths[0]).stat().st_size > 0

    def test_empty_per_column_returns_empty(self, plots_dir):
        result = plot_missing_heatmap(
            json.dumps({"per_column": {}, "total_pct": 0.0}), plots_dir
        )
        paths = json.loads(result)
        assert paths == []

    def test_all_zero_missing_still_creates_chart(self, plots_dir):
        """Even when no columns have missing data, the chart is produced."""
        result = plot_missing_heatmap(
            json.dumps({"per_column": {"a": 0.0, "b": 0.0}, "total_pct": 0.0}),
            plots_dir,
        )
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_creates_output_dir(self, missing_json, tmp_path):
        nested = str(tmp_path / "x" / "y")
        result = plot_missing_heatmap(missing_json, nested)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(nested).is_dir()


# ---------------------------------------------------------------------------
# Hard Boundary Rule: zero AG2 imports (architecture.md § 12.1)
# ---------------------------------------------------------------------------

class TestHardBoundaryRule:
    """Verify tools/visualization_tools.py has zero AG2 imports."""

    def test_no_autogen_import(self):
        import importlib
        import inspect

        mod = importlib.import_module("tools.visualization_tools")
        source = inspect.getsource(mod)
        assert "import autogen" not in source
        assert "from autogen" not in source


# ---------------------------------------------------------------------------
# Matplotlib backend
# ---------------------------------------------------------------------------

class TestMatplotlibBackend:
    """Verify non-interactive backend is set."""

    def test_agg_backend(self):
        import matplotlib
        assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# End-to-end: load_data → EDA tools → visualization tools
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Test that load_data + EDA tool outputs feed into visualization tools."""

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

    def test_load_then_histograms(self, csv_path, plots_dir):
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = plot_histograms(data_json, plots_dir)
        paths = json.loads(result)
        # 2 numerical columns: age, salary
        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()

    def test_load_then_correlation_heatmap(self, csv_path, plots_dir):
        from tools.data_loader import load_data
        from tools.eda_tools import correlation_matrix

        data_json = load_data(csv_path)
        corr_json = correlation_matrix(data_json)
        result = plot_correlation_heatmap(corr_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_load_then_missing_heatmap(self, csv_path, plots_dir):
        from tools.data_loader import load_data
        from tools.eda_tools import missing_analysis

        data_json = load_data(csv_path)
        missing_json = missing_analysis(data_json)
        result = plot_missing_heatmap(missing_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_full_visualization_pipeline(self, csv_path, plots_dir):
        """End-to-end: load → EDA → all 3 visualization tools."""
        from tools.data_loader import load_data
        from tools.eda_tools import correlation_matrix, missing_analysis

        data_json = load_data(csv_path)
        corr_json = correlation_matrix(data_json)
        miss_json = missing_analysis(data_json)

        hist_paths = json.loads(plot_histograms(data_json, plots_dir))
        corr_paths = json.loads(plot_correlation_heatmap(corr_json, plots_dir))
        miss_paths = json.loads(plot_missing_heatmap(miss_json, plots_dir))

        all_paths = hist_paths + corr_paths + miss_paths
        assert len(all_paths) == 4  # 2 histograms + 1 corr + 1 missing
        for p in all_paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_class_distribution()
# ---------------------------------------------------------------------------


class TestPlotClassDistribution:
    """Test plot_class_distribution() for classification, regression, unsupervised."""

    @pytest.fixture()
    def classification_df_json(self):
        df = pd.DataFrame({
            "feat": [1, 2, 3, 4, 5, 6],
            "species": ["a", "a", "b", "b", "c", "c"],
        })
        return df.to_json(orient="records")

    @pytest.fixture()
    def regression_df_json(self):
        df = pd.DataFrame({
            "feat": list(range(50)),
            "price": [float(x) for x in range(50)],
        })
        return df.to_json(orient="records")

    @pytest.fixture()
    def classification_ti_json(self):
        from eda_state import TargetInfo
        return TargetInfo(
            column="species",
            problem_type="classification",
            n_classes=3,
            class_counts={"a": 2, "b": 2, "c": 2},
            imbalance_ratio=1.0,
            detection_method="name_heuristic",
        ).model_dump_json()

    @pytest.fixture()
    def regression_ti_json(self):
        from eda_state import TargetInfo
        return TargetInfo(
            column="price",
            problem_type="regression",
            detection_method="name_heuristic",
        ).model_dump_json()

    @pytest.fixture()
    def unsupervised_ti_json(self):
        from eda_state import TargetInfo
        return TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
        ).model_dump_json()

    def test_classification_creates_png(self, classification_df_json, classification_ti_json, plots_dir):
        result = plot_class_distribution(classification_ti_json, plots_dir, data_json=classification_df_json)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()
        assert Path(paths[0]).name == "class_distribution.png"

    def test_classification_file_nonzero(self, classification_df_json, classification_ti_json, plots_dir):
        result = plot_class_distribution(classification_ti_json, plots_dir, data_json=classification_df_json)
        paths = json.loads(result)
        assert Path(paths[0]).stat().st_size > 0

    def test_regression_creates_png(self, regression_df_json, regression_ti_json, plots_dir):
        result = plot_class_distribution(regression_ti_json, plots_dir, data_json=regression_df_json)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()
        assert Path(paths[0]).name == "target_distribution.png"

    def test_unsupervised_returns_empty(self, classification_df_json, unsupervised_ti_json, plots_dir):
        result = plot_class_distribution(unsupervised_ti_json, plots_dir, data_json=classification_df_json)
        paths = json.loads(result)
        assert paths == []

    def test_missing_column_returns_empty(self, classification_df_json, plots_dir):
        from eda_state import TargetInfo
        ti = TargetInfo(
            column="nonexistent",
            problem_type="classification",
            detection_method="name_heuristic",
        ).model_dump_json()
        result = plot_class_distribution(ti, plots_dir, data_json=classification_df_json)
        paths = json.loads(result)
        assert paths == []

    def test_creates_output_dir(self, classification_df_json, classification_ti_json, tmp_path):
        new_dir = str(tmp_path / "new" / "plots")
        result = plot_class_distribution(classification_ti_json, new_dir, data_json=classification_df_json)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(new_dir).is_dir()


# ---------------------------------------------------------------------------
# plot_categorical_bars()
# ---------------------------------------------------------------------------


class TestPlotCategoricalBars:
    """Test plot_categorical_bars() function."""

    def _make_cat_analysis_json(self, columns: dict) -> str:
        """Build a minimal CategoricalAnalysis JSON from a columns dict."""
        from eda_state import CategoricalAnalysis, CategoricalStats
        col_stats = {}
        for col, top_vals in columns.items():
            col_stats[col] = CategoricalStats(
                cardinality=len(top_vals),
                top_values=top_vals,
                more_values=0,
            )
        return CategoricalAnalysis(columns=col_stats, top_n=10).model_dump_json()

    @pytest.fixture()
    def simple_cat_json(self):
        return self._make_cat_analysis_json({
            "dept": [
                {"value": "eng", "count": 50, "pct": 50.0, "is_rare": False},
                {"value": "hr",  "count": 30, "pct": 30.0, "is_rare": False},
                {"value": "mgmt","count": 20, "pct": 20.0, "is_rare": False},
            ],
        })

    @pytest.fixture()
    def two_col_cat_json(self):
        return self._make_cat_analysis_json({
            "status": [
                {"value": "active",   "count": 80, "pct": 80.0, "is_rare": False},
                {"value": "inactive", "count": 20, "pct": 20.0, "is_rare": False},
            ],
            "region": [
                {"value": "north", "count": 60, "pct": 60.0, "is_rare": False},
                {"value": "south", "count": 25, "pct": 25.0, "is_rare": False},
                {"value": "east",  "count": 15, "pct": 15.0, "is_rare": False},
            ],
        })

    @pytest.fixture()
    def empty_cat_json(self):
        from eda_state import CategoricalAnalysis
        return CategoricalAnalysis(columns={}, top_n=10).model_dump_json()

    def test_returns_json_list(self, simple_cat_json, plots_dir):
        result = plot_categorical_bars(simple_cat_json, plots_dir)
        paths = json.loads(result)
        assert isinstance(paths, list)

    def test_creates_one_png_per_column(self, two_col_cat_json, plots_dir):
        result = plot_categorical_bars(two_col_cat_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 2

    def test_files_exist_on_disk(self, two_col_cat_json, plots_dir):
        result = plot_categorical_bars(two_col_cat_json, plots_dir)
        paths = json.loads(result)
        for p in paths:
            assert Path(p).exists(), f"File not found: {p}"

    def test_filename_prefix_is_cat(self, simple_cat_json, plots_dir):
        result = plot_categorical_bars(simple_cat_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).name == "cat_dept.png"

    def test_filename_sanitizes_spaces(self, plots_dir):
        """Column name with spaces → underscores in filename."""
        cat_json = self._make_cat_analysis_json({
            "status code": [
                {"value": "200", "count": 90, "pct": 90.0, "is_rare": False},
                {"value": "404", "count": 10, "pct": 10.0, "is_rare": False},
            ],
        })
        result = plot_categorical_bars(cat_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).name == "cat_status_code.png"

    def test_filename_sanitizes_special_chars(self, plots_dir):
        """Column with special characters → sanitized filename."""
        cat_json = self._make_cat_analysis_json({
            "col/type:raw": [
                {"value": "a", "count": 100, "pct": 100.0, "is_rare": False},
            ],
        })
        result = plot_categorical_bars(cat_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        fname = Path(paths[0]).name
        assert "/" not in fname and ":" not in fname
        assert fname.startswith("cat_")

    def test_files_are_png(self, simple_cat_json, plots_dir):
        result = plot_categorical_bars(simple_cat_json, plots_dir)
        paths = json.loads(result)
        for p in paths:
            assert Path(p).suffix == ".png"

    def test_files_nonzero_size(self, simple_cat_json, plots_dir):
        result = plot_categorical_bars(simple_cat_json, plots_dir)
        paths = json.loads(result)
        for p in paths:
            assert Path(p).stat().st_size > 0

    def test_empty_columns_returns_empty_list(self, empty_cat_json, plots_dir):
        result = plot_categorical_bars(empty_cat_json, plots_dir)
        paths = json.loads(result)
        assert paths == []

    def test_column_with_empty_top_values_is_skipped(self, plots_dir):
        """A column with no top_values entries produces no PNG."""
        from eda_state import CategoricalAnalysis, CategoricalStats
        analysis = CategoricalAnalysis(
            columns={"empty_col": CategoricalStats(cardinality=0, top_values=[])},
            top_n=10,
        )
        result = plot_categorical_bars(analysis.model_dump_json(), plots_dir)
        paths = json.loads(result)
        assert paths == []

    def test_rare_category_flag_does_not_crash(self, plots_dir):
        """At least one is_rare=True entry renders without error."""
        cat_json = self._make_cat_analysis_json({
            "rarity_col": [
                {"value": "common",  "count": 990, "pct": 99.0, "is_rare": False},
                {"value": "unusual", "count": 3,   "pct": 0.3,  "is_rare": True},
            ],
        })
        result = plot_categorical_bars(cat_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_more_values_annotation_does_not_crash(self, plots_dir):
        """more_values > 0 path completes without error."""
        from eda_state import CategoricalAnalysis, CategoricalStats
        analysis = CategoricalAnalysis(
            columns={
                "big_col": CategoricalStats(
                    cardinality=25,
                    top_values=[
                        {"value": str(i), "count": 10, "pct": 4.0, "is_rare": False}
                        for i in range(10)
                    ],
                    more_values=15,
                )
            },
            top_n=10,
        )
        result = plot_categorical_bars(analysis.model_dump_json(), plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_creates_output_dir(self, simple_cat_json, tmp_path):
        nested = str(tmp_path / "a" / "b" / "plots")
        result = plot_categorical_bars(simple_cat_json, nested)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(nested).is_dir()

    def test_end_to_end_via_analyze_categoricals(self, tmp_path):
        """analyze_categoricals() → plot_categorical_bars() full chain."""
        import pandas as pd
        from tools.eda_tools import analyze_categoricals

        df = pd.DataFrame({
            "dept":   ["eng", "hr", "eng", "sales", "hr", "eng"],
            "region": ["north", "south", "north", "north", "east", "south"],
            "age":    [25, 30, 35, 40, 45, 50],  # numeric — not categorical
        })
        data_json = df.to_json(orient="records")
        target_json = '{"column": null, "problem_type": "unsupervised", "detection_method": "none"}'

        cat_analysis_json = analyze_categoricals(data_json, target_json)
        plots_dir = str(tmp_path / "plots")
        result = plot_categorical_bars(cat_analysis_json, plots_dir)
        paths = json.loads(result)
        # dept and region are the 2 categorical columns
        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0
