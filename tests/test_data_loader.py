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
    _MAX_PROFILED_CARDINALITY,
    _build_column_profiles,
    _get_loader,
    _has_datetime_column,
    _classify_target,
    detect_encoded_categoricals,
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


# ---------------------------------------------------------------------------
# NA sentinel handling — CSVLoader
# ---------------------------------------------------------------------------

class TestCSVLoaderNASentinels:
    """Tests that CSVLoader converts common sentinel tokens to NaN at load time."""

    def test_question_mark_becomes_nan(self, tmp_path):
        """Bare '?' must be read as NaN, not a literal string."""
        p = tmp_path / "qmark.csv"
        p.write_text("a,b\n1,2\n3,?\n")
        df = CSVLoader().load(str(p))
        assert pd.isna(df.loc[1, "b"]), "'?' should be NaN"

    def test_leading_space_question_mark_becomes_nan(self, tmp_path):
        """' ?' (UCI-style leading space) must be read as NaN via skipinitialspace."""
        p = tmp_path / "spaced_qmark.csv"
        p.write_text("a,b\n1, 2\n3, ?\n")
        df = CSVLoader().load(str(p))
        assert pd.isna(df.loc[1, "b"]), "' ?' should be NaN after skipinitialspace"

    def test_other_sentinels_become_nan(self, tmp_path):
        """Common tokens (Unknown, NULL, N/A) must also become NaN."""
        p = tmp_path / "sentinels.csv"
        p.write_text("a,b,c\nUnknown,NULL,N/A\n")
        df = CSVLoader().load(str(p))
        assert pd.isna(df.loc[0, "a"]), "'Unknown' should be NaN"
        assert pd.isna(df.loc[0, "b"]), "'NULL' should be NaN"
        assert pd.isna(df.loc[0, "c"]), "'N/A' should be NaN"

    def test_clean_values_unaffected(self, tmp_path):
        """Normal string and numeric values must not be converted to NaN."""
        p = tmp_path / "clean.csv"
        p.write_text("name,score\nAlice,42\nBob,17\n")
        df = CSVLoader().load(str(p))
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[1, "score"] == 17

    def test_missing_analysis_detects_question_mark(self, tmp_path):
        """Integration: missing_analysis() must report >0% missing for a '?'-only column."""
        from tools.eda_tools import missing_analysis
        p = tmp_path / "adult_mini.csv"
        p.write_text(
            "age,workclass,occupation\n"
            "39, State-gov, Adm-clerical\n"
            "54, ?, ?\n"
            "28, Private, Prof-specialty\n"
        )
        data_json = load_data(str(p))
        result = json.loads(missing_analysis(data_json))
        assert result["per_column"]["workclass"] > 0, "workclass should have missing%>0"
        assert result["per_column"]["occupation"] > 0, "occupation should have missing%>0"
        assert result["per_column"]["age"] == 0.0, "age has no missing values"


# ---------------------------------------------------------------------------
# W8: duplicate_count artifact
# ---------------------------------------------------------------------------

class TestDuplicateCountArtifact:
    """Test that load_data() saves and validate_schema() exposes duplicate_count."""

    def test_load_data_saves_duplicate_count_artifact(self, csv_with_duplicates):
        """load_data() stores the pre-dedup duplicate count in the artifact store."""
        from tools._pipeline_state import init_session, clear_session, load_state
        try:
            init_session()
            load_data(csv_with_duplicates)
            raw = load_state("duplicate_count")
            assert raw is not None
            assert int(raw) == 1  # csv_with_duplicates has 3 rows, 1 duplicate
        finally:
            clear_session()

    def test_load_data_saves_zero_for_clean_csv(self, csv_path):
        """load_data() stores 0 when there are no duplicates."""
        from tools._pipeline_state import init_session, clear_session, load_state
        try:
            init_session()
            load_data(csv_path)
            raw = load_state("duplicate_count")
            assert int(raw) == 0
        finally:
            clear_session()

    def test_validate_schema_includes_duplicate_count(self, csv_with_duplicates):
        """validate_schema() exposes duplicate_count in the DataProfile JSON."""
        from tools._pipeline_state import init_session, clear_session, load_state
        try:
            init_session()
            data_json = load_data(csv_with_duplicates)
            validate_schema(data_json)
            # In pipeline mode validate_schema returns a STATE_REF string;
            # the DataProfile JSON is in the schema_json artifact.
            profile_json = json.loads(load_state("schema_json"))
            assert "duplicate_count" in profile_json
            assert profile_json["duplicate_count"] == 1
        finally:
            clear_session()

    def test_validate_schema_duplicate_count_zero_clean_data(self, csv_path):
        """validate_schema() returns duplicate_count=0 for a clean dataset."""
        from tools._pipeline_state import init_session, clear_session, load_state
        try:
            init_session()
            data_json = load_data(csv_path)
            validate_schema(data_json)
            profile_json = json.loads(load_state("schema_json"))
            assert profile_json.get("duplicate_count", -1) == 0
        finally:
            clear_session()


# ---------------------------------------------------------------------------
# _build_column_profiles()
# ---------------------------------------------------------------------------

class TestBuildColumnProfiles:
    """Tests for the column profiling function used by LLM detection."""

    @pytest.fixture()
    def encoded_df(self):
        """DataFrame with encoded categoricals + continuous columns."""
        return pd.DataFrame({
            "SEX": [1, 2, 1, 2, 1, 2] * 10,
            "EDUCATION": [1, 2, 3, 4, 1, 2] * 10,
            "AGE": list(range(22, 82)),
            "SALARY": [50000 + i * 1000 for i in range(60)],
            "name": ["alice"] * 60,
        })

    def test_returns_list_of_dicts(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        assert isinstance(profiles, list)
        for p in profiles:
            assert isinstance(p, dict)

    def test_includes_low_cardinality_numeric(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        names = [p["name"] for p in profiles]
        assert "SEX" in names
        assert "EDUCATION" in names

    def test_excludes_high_cardinality_numeric(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        names = [p["name"] for p in profiles]
        assert "AGE" not in names
        assert "SALARY" not in names

    def test_excludes_non_numeric(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        names = [p["name"] for p in profiles]
        assert "name" not in names

    def test_profile_keys(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        expected_keys = {"name", "dtype", "nunique", "n_rows", "sample_values",
                         "min", "max", "is_all_integer"}
        for p in profiles:
            assert set(p.keys()) == expected_keys

    def test_is_all_integer_true_for_int_cols(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        sex_profile = next(p for p in profiles if p["name"] == "SEX")
        assert sex_profile["is_all_integer"] is True

    def test_nunique_correct(self, encoded_df):
        profiles = _build_column_profiles(encoded_df)
        sex_profile = next(p for p in profiles if p["name"] == "SEX")
        assert sex_profile["nunique"] == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        profiles = _build_column_profiles(df)
        assert profiles == []

    def test_all_high_cardinality(self):
        df = pd.DataFrame({"x": list(range(100))})
        profiles = _build_column_profiles(df)
        assert profiles == []


# ---------------------------------------------------------------------------
# detect_encoded_categoricals() — mocked LLM
# ---------------------------------------------------------------------------

class TestDetectEncodedCategoricals:
    """Tests for the LLM-based encoded categorical detection (mocked API)."""

    @pytest.fixture()
    def encoded_df(self):
        return pd.DataFrame({
            "SEX": [1, 2, 1, 2, 1, 2] * 10,
            "EDUCATION": [1, 2, 3, 4, 1, 2] * 10,
            "AGE": list(range(22, 82)),
            "target": [0, 1] * 30,
        })

    def _mock_openai_response(self, suspects_json):
        """Create a mock OpenAI response object."""
        from unittest.mock import MagicMock
        mock_message = MagicMock()
        mock_message.content = json.dumps(suspects_json)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_returns_suspects_from_llm(self, encoded_df):
        from unittest.mock import MagicMock, patch
        resp = self._mock_openai_response({
            "suspects": [
                {"column": "SEX", "reason": "Binary gender code", "subtype": "nominal"},
                {"column": "EDUCATION", "reason": "Education level codes", "subtype": "ordinal"},
            ]
        })
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.return_value = resp
        with patch("openai.OpenAI", mock_client):
            suspects = detect_encoded_categoricals(encoded_df, target_column="target")
        assert len(suspects) == 2
        names = [s.column for s in suspects]
        assert "SEX" in names
        assert "EDUCATION" in names

    def test_excludes_target_column(self, encoded_df):
        from unittest.mock import MagicMock, patch
        resp = self._mock_openai_response({
            "suspects": [
                {"column": "target", "reason": "should be filtered", "subtype": "nominal"},
                {"column": "SEX", "reason": "gender code", "subtype": "nominal"},
            ]
        })
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.return_value = resp
        with patch("openai.OpenAI", mock_client):
            suspects = detect_encoded_categoricals(encoded_df, target_column="target")
        # "target" should be excluded from profiles, so LLM's suggestion is filtered
        names = [s.column for s in suspects]
        assert "target" not in names

    def test_ignores_hallucinated_columns(self, encoded_df):
        from unittest.mock import MagicMock, patch
        resp = self._mock_openai_response({
            "suspects": [
                {"column": "NONEXISTENT", "reason": "hallucinated", "subtype": "nominal"},
                {"column": "SEX", "reason": "gender code", "subtype": "nominal"},
            ]
        })
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.return_value = resp
        with patch("openai.OpenAI", mock_client):
            suspects = detect_encoded_categoricals(encoded_df, target_column="target")
        assert len(suspects) == 1
        assert suspects[0].column == "SEX"

    def test_empty_suspects(self, encoded_df):
        from unittest.mock import MagicMock, patch
        resp = self._mock_openai_response({"suspects": []})
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.return_value = resp
        with patch("openai.OpenAI", mock_client):
            suspects = detect_encoded_categoricals(encoded_df, target_column="target")
        assert suspects == []

    def test_llm_failure_returns_empty(self, encoded_df):
        from unittest.mock import MagicMock, patch
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.side_effect = RuntimeError("API error")
        with patch("openai.OpenAI", mock_client):
            suspects = detect_encoded_categoricals(encoded_df, target_column="target")
        assert suspects == []

    def test_no_low_cardinality_returns_empty(self):
        df = pd.DataFrame({"x": list(range(100)), "y": list(range(100, 200))})
        # No columns with nunique ≤ 30, so no LLM call needed
        suspects = detect_encoded_categoricals(df)
        assert suspects == []

    def test_suspect_has_correct_fields(self, encoded_df):
        from unittest.mock import MagicMock, patch
        resp = self._mock_openai_response({
            "suspects": [
                {"column": "SEX", "reason": "Binary gender code", "subtype": "nominal"},
            ]
        })
        mock_client = MagicMock()
        mock_client.return_value.chat.completions.create.return_value = resp
        with patch("openai.OpenAI", mock_client):
            suspects = detect_encoded_categoricals(encoded_df, target_column="target")
        s = suspects[0]
        assert s.column == "SEX"
        assert s.nunique == 2
        assert s.is_all_integer is True
        assert s.reason == "Binary gender code"
        assert s.subtype == "nominal"
        assert 1 in s.sample_values
        assert 2 in s.sample_values


# ---------------------------------------------------------------------------
# infer_dtypes() — reclassification via artifact store
# ---------------------------------------------------------------------------

class TestInferDtypesReclassification:
    """Test that infer_dtypes() applies reclassification from the artifact store."""

    def test_reclassified_columns_move_to_categorical(self):
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({
            "SEX": [1, 2, 1, 2],
            "AGE": [25, 30, 35, 40],
            "name": ["a", "b", "c", "d"],
        })
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["SEX"]))
            infer_dtypes(data_json)
            result = json.loads(load_state("dtypes_json"))
            # SEX should be in categorical, not numerical
            assert "SEX" in result["categorical_cols"]
            assert "SEX" not in result["numerical_cols"]
            # AGE remains numerical
            assert "AGE" in result["numerical_cols"]
        finally:
            clear_session()

    def test_encoded_categorical_cols_populated(self):
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({
            "SEX": [1, 2, 1, 2],
            "EDUCATION": [1, 2, 3, 4],
            "AGE": [25, 30, 35, 40],
        })
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["SEX", "EDUCATION"]))
            infer_dtypes(data_json)
            result = json.loads(load_state("dtypes_json"))
            assert set(result["encoded_categorical_cols"]) == {"SEX", "EDUCATION"}
        finally:
            clear_session()

    def test_no_reclassification_without_artifact(self):
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({"SEX": [1, 2], "name": ["a", "b"]})
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            # No reclassified_categoricals artifact saved
            infer_dtypes(data_json)
            result = json.loads(load_state("dtypes_json"))
            assert "SEX" in result["numerical_cols"]
            assert result["encoded_categorical_cols"] == []
        finally:
            clear_session()

    def test_invalid_column_in_artifact_ignored(self):
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({"SEX": [1, 2], "name": ["a", "b"]})
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["NONEXISTENT"]))
            infer_dtypes(data_json)
            result = json.loads(load_state("dtypes_json"))
            # NONEXISTENT should be silently ignored
            assert result["encoded_categorical_cols"] == []
            assert "SEX" in result["numerical_cols"]
        finally:
            clear_session()


class TestInferDtypesStringCasting:
    """Verify that reclassified columns are physically cast to str in the artifact store."""

    def test_artifact_data_json_has_string_values(self):
        """After casting, the data_json artifact should contain string values."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({
            "SEX": [1, 2, 1, 2],
            "AGE": [25, 30, 35, 40],
            "name": ["a", "b", "c", "d"],
        })
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["SEX"]))
            infer_dtypes(data_json)
            # Re-read data_json from artifact store and verify dtype
            stored = load_state("data_json")
            df_after = pd.DataFrame(json.loads(stored))
            assert df_after["SEX"].dtype == object
            assert df_after["SEX"].tolist() == ["1", "2", "1", "2"]
            # AGE remains numeric
            assert pd.api.types.is_numeric_dtype(df_after["AGE"])
        finally:
            clear_session()

    def test_select_dtypes_excludes_cast_columns(self):
        """Downstream select_dtypes('number') must not include cast columns."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({
            "SEX": [1, 2, 1, 2],
            "EDUCATION": [1, 2, 3, 4],
            "AGE": [25, 30, 35, 40],
        })
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["SEX", "EDUCATION"]))
            infer_dtypes(data_json)
            stored = load_state("data_json")
            df_after = pd.DataFrame(json.loads(stored))
            num_cols = df_after.select_dtypes(include="number").columns.tolist()
            obj_cols = df_after.select_dtypes(include="object").columns.tolist()
            assert "SEX" not in num_cols
            assert "EDUCATION" not in num_cols
            assert "AGE" in num_cols
            assert "SEX" in obj_cols
            assert "EDUCATION" in obj_cols
        finally:
            clear_session()

    def test_nan_preserved_through_cast(self):
        """NaN values must survive the int→str cast (not become the string 'nan')."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({
            "SEX": [1, 2, None, 1],
            "AGE": [25, 30, 35, 40],
        })
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["SEX"]))
            infer_dtypes(data_json)
            stored = load_state("data_json")
            df_after = pd.DataFrame(json.loads(stored))
            assert df_after["SEX"].dtype == object
            # Non-null values are clean strings (no ".0" suffix)
            assert df_after.loc[0, "SEX"] == "1"
            assert df_after.loc[1, "SEX"] == "2"
            # NaN preserved as actual NaN (not the string "nan")
            assert pd.isna(df_after.loc[2, "SEX"])
            assert df_after.loc[3, "SEX"] == "1"
        finally:
            clear_session()

    def test_dataprofile_dtypes_reflect_object(self):
        """DataProfile.dtypes dict should show 'object' for cast columns."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({
            "SEX": [1, 2, 1, 2],
            "AGE": [25, 30, 35, 40],
        })
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            save_state("reclassified_categoricals", json.dumps(["SEX"]))
            infer_dtypes(data_json)
            profile = json.loads(load_state("dtypes_json"))
            assert profile["dtypes"]["SEX"] == "object"
            assert profile["dtypes"]["AGE"] == "int64"
        finally:
            clear_session()

    def test_no_cast_without_reclassification(self):
        """When no columns are reclassified, data_json remains unmodified."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        df = pd.DataFrame({"SEX": [1, 2], "AGE": [25, 30]})
        data_json = df.to_json(orient="records")
        try:
            init_session()
            save_state("data_json", data_json)
            # No reclassified_categoricals artifact
            infer_dtypes(data_json)
            stored = load_state("data_json")
            # data_json unchanged — still the original
            assert stored == data_json
        finally:
            clear_session()
