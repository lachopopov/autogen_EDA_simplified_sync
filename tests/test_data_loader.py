"""
tests/test_data_loader.py — Unit tests for tools/data_loader.py

Tests the pure tools layer: DataLoader strategies, load_data(), validate_schema(),
infer_dtypes(). Zero AG2 dependency — only pandas and Pydantic.
"""

import json

import pandas as pd
import pytest

from tools.data_loader import (
    CSVLoader,
    DataLoader,
    ExcelLoader,
    ParquetLoader,
    _LOADERS,
    _get_loader,
    _has_datetime_column,
    _classify_target,
    detect_target,
    infer_dtypes,
    load_data,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny DataFrames written to various formats
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """A small DataFrame with numerical + categorical columns."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Carol", "Dave"],
        "score": [88.5, 92.0, 75.5, 88.5],
        "grade": ["A", "A", "B", "A"],
    })


@pytest.fixture()
def csv_path(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture()
def parquet_path(tmp_path, sample_df):
    p = tmp_path / "data.parquet"
    sample_df.to_parquet(p, index=False, engine="pyarrow")
    return str(p)


@pytest.fixture()
def excel_path(tmp_path, sample_df):
    p = tmp_path / "data.xlsx"
    sample_df.to_excel(p, index=False, engine="openpyxl")
    return str(p)


@pytest.fixture()
def csv_with_duplicates(tmp_path):
    """CSV with duplicate rows — load_data should drop them."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    p = tmp_path / "dupes.csv"
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Strategy Pattern — DataLoader classes
# ---------------------------------------------------------------------------

class TestDataLoaderStrategy:
    """Test the ABC + concrete strategies."""

    def test_csv_loader_supports(self):
        assert CSVLoader().supports(".csv") is True
        assert CSVLoader().supports(".parquet") is False

    def test_parquet_loader_supports(self):
        assert ParquetLoader().supports(".parquet") is True
        assert ParquetLoader().supports(".csv") is False

    def test_excel_loader_supports(self):
        assert ExcelLoader().supports(".xlsx") is True
        assert ExcelLoader().supports(".xls") is True
        assert ExcelLoader().supports(".csv") is False

    def test_csv_loader_load(self, csv_path):
        df = CSVLoader().load(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 4)

    def test_parquet_loader_load(self, parquet_path):
        df = ParquetLoader().load(parquet_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 4)

    def test_excel_loader_load(self, excel_path):
        df = ExcelLoader().load(excel_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 4)

    def test_loader_registry_not_empty(self):
        assert len(_LOADERS) >= 3

    def test_all_loaders_are_dataloader(self):
        for loader in _LOADERS:
            assert isinstance(loader, DataLoader)


class TestGetLoader:
    """Test the strategy selector."""

    def test_csv(self):
        assert isinstance(_get_loader("file.csv"), CSVLoader)

    def test_parquet(self):
        assert isinstance(_get_loader("file.parquet"), ParquetLoader)

    def test_xlsx(self):
        assert isinstance(_get_loader("file.xlsx"), ExcelLoader)

    def test_xls(self):
        assert isinstance(_get_loader("file.xls"), ExcelLoader)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            _get_loader("file.json")

    def test_case_insensitive_extension(self):
        # Path.suffix is already lowercase for most OS, but the logic does .lower()
        assert isinstance(_get_loader("FILE.CSV"), CSVLoader)


# ---------------------------------------------------------------------------
# load_data()
# ---------------------------------------------------------------------------

class TestLoadData:
    """Test the AG2-facing load_data function."""

    def test_csv_returns_json(self, csv_path):
        result = load_data(csv_path)
        records = json.loads(result)
        assert isinstance(records, list)
        assert len(records) == 4

    def test_parquet_returns_json(self, parquet_path):
        result = load_data(parquet_path)
        records = json.loads(result)
        assert len(records) == 4

    def test_excel_returns_json(self, excel_path):
        result = load_data(excel_path)
        records = json.loads(result)
        assert len(records) == 4

    def test_drops_duplicates(self, csv_with_duplicates):
        result = load_data(csv_with_duplicates)
        records = json.loads(result)
        assert len(records) == 2  # 3 rows → 2 after dedup

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_data("/nonexistent/path/file.csv")

    def test_unsupported_format(self, tmp_path):
        bad = tmp_path / "data.json"
        bad.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data(str(bad))

    def test_records_have_correct_keys(self, csv_path):
        records = json.loads(load_data(csv_path))
        assert set(records[0].keys()) == {"id", "name", "score", "grade"}

    def test_escaped_forward_slashes_sanitised(self, csv_path):
        """LLMs sometimes JSON-escape forward slashes (\\/ → /)."""
        escaped = csv_path.replace("/", "\\/")
        result = load_data(escaped)
        records = json.loads(result)
        assert isinstance(records, list)
        assert len(records) == 4


# ---------------------------------------------------------------------------
# validate_schema()
# ---------------------------------------------------------------------------

class TestValidateSchema:
    """Test the AG2-facing validate_schema function."""

    def test_returns_valid_json(self, csv_path):
        data_json = load_data(csv_path)
        result = validate_schema(data_json)
        profile = json.loads(result)
        assert "shape" in profile
        assert "dtypes" in profile
        assert "memory_mb" in profile

    def test_shape(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(validate_schema(data_json))
        assert profile["shape"] == [4, 4]  # JSON serializes tuple as list

    def test_dtypes_populated(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(validate_schema(data_json))
        assert len(profile["dtypes"]) == 4

    def test_memory_positive(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(validate_schema(data_json))
        assert profile["memory_mb"] > 0


# ---------------------------------------------------------------------------
# infer_dtypes()
# ---------------------------------------------------------------------------

class TestInferDtypes:
    """Test the AG2-facing infer_dtypes function."""

    def test_numerical_cols(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(infer_dtypes(data_json))
        assert "id" in profile["numerical_cols"]
        assert "score" in profile["numerical_cols"]

    def test_categorical_cols(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(infer_dtypes(data_json))
        assert "name" in profile["categorical_cols"]
        assert "grade" in profile["categorical_cols"]

    def test_no_overlap(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(infer_dtypes(data_json))
        num_set = set(profile["numerical_cols"])
        cat_set = set(profile["categorical_cols"])
        assert num_set.isdisjoint(cat_set)

    def test_covers_all_columns(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(infer_dtypes(data_json))
        all_cols = set(profile["numerical_cols"]) | set(profile["categorical_cols"])
        assert all_cols == {"id", "name", "score", "grade"}

    def test_shape_preserved(self, csv_path):
        data_json = load_data(csv_path)
        profile = json.loads(infer_dtypes(data_json))
        assert profile["shape"] == [4, 4]


# ---------------------------------------------------------------------------
# _has_datetime_column()
# ---------------------------------------------------------------------------

class TestHasDatetimeColumn:
    def test_datetime_dtype_detected(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "val": [1, 2],
        })
        assert _has_datetime_column(df) is True

    def test_no_datetime(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert _has_datetime_column(df) is False

    def test_string_date_column_with_date_in_name(self):
        df = pd.DataFrame({
            "order_date": ["2021-01-01", "2021-01-02"],
            "val": [1, 2],
        })
        # Should detect via name heuristic + parseable check
        assert _has_datetime_column(df) is True


# ---------------------------------------------------------------------------
# _classify_target()
# ---------------------------------------------------------------------------

class TestClassifyTarget:
    def test_classification_low_cardinality(self):
        df = pd.DataFrame({"target": ["a", "b", "c", "a", "b"]})
        info = _classify_target(df, "target")
        assert info.problem_type == "classification"
        assert info.n_classes == 3
        assert info.column == "target"

    def test_regression_high_cardinality(self):
        df = pd.DataFrame({"price": list(range(50))})
        info = _classify_target(df, "price")
        assert info.problem_type == "regression"
        assert info.n_classes == 0

    def test_imbalance_ratio(self):
        df = pd.DataFrame({"y": ["a"] * 90 + ["b"] * 10})
        info = _classify_target(df, "y")
        assert info.problem_type == "classification"
        assert info.imbalance_ratio == 9.0


# ---------------------------------------------------------------------------
# detect_target()
# ---------------------------------------------------------------------------

class TestDetectTarget:
    """Test the 4-step heuristic target detection."""

    def test_exact_keyword_target(self):
        df = pd.DataFrame({"feature_1": [1, 2, 3], "target": [0, 1, 0]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "target"
        assert result["detection_method"] == "name_heuristic"

    def test_exact_keyword_label(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "label": ["a", "b", "a"]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "label"
        assert result["detection_method"] == "name_heuristic"

    def test_exact_keyword_y(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "y"
        assert result["detection_method"] == "name_heuristic"

    def test_contains_keyword_outcome(self):
        df = pd.DataFrame({"feat": [1, 2], "patient_outcome": ["good", "bad"]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "patient_outcome"
        assert result["detection_method"] == "name_heuristic"

    def test_prefix_is_(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "is_fraud": [0, 1, 0]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "is_fraud"
        assert result["detection_method"] == "name_heuristic"

    def test_prefix_has_(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "has_subscription": [1, 0, 1]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "has_subscription"
        assert result["detection_method"] == "name_heuristic"

    def test_fallback_last_low_cardinality(self):
        # No keyword matches, last column with nunique < 10
        df = pd.DataFrame({
            "feat1": list(range(20)),
            "feat2": list(range(20, 40)),
            "status": ["ok", "fail"] * 10,
        })
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "status"
        assert result["detection_method"] == "position_heuristic"

    def test_no_target_unsupervised(self):
        # All columns have high cardinality, no keyword matches
        df = pd.DataFrame({
            "alpha": list(range(100)),
            "beta": list(range(100, 200)),
            "gamma": list(range(200, 300)),
        })
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] is None
        assert result["problem_type"] == "unsupervised"
        assert result["detection_method"] == "none"

    def test_classification_type_detected(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "class": ["a", "b", "a"]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["problem_type"] == "classification"

    def test_regression_type_detected(self):
        # "price" is an exact keyword, and column has high cardinality
        df = pd.DataFrame({
            "feat": list(range(50)),
            "price": [x * 1.5 for x in range(50)],
        })
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["problem_type"] == "regression"

    def test_exact_keyword_priority_over_contains(self):
        # "target" (exact) and a column with "outcome" (contains)
        df = pd.DataFrame({
            "patient_outcome": ["good", "bad", "good"],
            "target": [0, 1, 0],
        })
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "target"

    def test_case_insensitive(self):
        df = pd.DataFrame({"Feature": [1, 2], "TARGET": [0, 1]})
        result = json.loads(detect_target(df.to_json(orient="records")))
        assert result["column"] == "TARGET"
