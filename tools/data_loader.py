"""
tools/data_loader.py — Load, validate, and classify input data files.

Architecture Reference: architecture.md § 12.4 (Strategy Pattern — DataLoader)

Public AG2-facing functions:
  - load_data(file_path: str) -> str
  - validate_schema(data_json: str) -> str
  - infer_dtypes(data_json: str) -> str

Standalone (pre-pipeline) function:
  - detect_target(data_json: str) -> str   (heuristic target detection)

OOP layer (invisible to AG2):
  - DataLoader (ABC) → CSVLoader, ParquetLoader, ExcelLoader

Design:
  - Zero AG2 imports. Zero agent references. Pure Python.
  - Each function accepts/returns JSON strings (the AG2 tool contract).
  - Pydantic sub-models from eda_state.py validate outputs.

AG2 Version: 0.10.3
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated

import pandas as pd

from eda_state import DataProfile, TargetInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy Pattern — file format loaders
# ---------------------------------------------------------------------------

class DataLoader(ABC):
    """Strategy interface for file format loaders."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame: ...

    @abstractmethod
    def supports(self, ext: str) -> bool: ...


class CSVLoader(DataLoader):
    def load(self, path: str) -> pd.DataFrame:
        from config import NA_TOKENS
        return pd.read_csv(
            path,
            skipinitialspace=True,
            na_values=NA_TOKENS,
            keep_default_na=True,
        )

    def supports(self, ext: str) -> bool:
        return ext == ".csv"


class ParquetLoader(DataLoader):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    def supports(self, ext: str) -> bool:
        return ext == ".parquet"


class ExcelLoader(DataLoader):
    def load(self, path: str) -> pd.DataFrame:
        from config import NA_TOKENS
        return pd.read_excel(
            path,
            engine="openpyxl",
            na_values=NA_TOKENS,
            keep_default_na=True,
        )

    def supports(self, ext: str) -> bool:
        return ext in (".xlsx", ".xls")


# Registry — add new formats by appending a new strategy here.
_LOADERS: list[DataLoader] = [CSVLoader(), ParquetLoader(), ExcelLoader()]


def _get_loader(path: str) -> DataLoader:
    """Select the correct loader strategy based on file extension."""
    ext = Path(path).suffix.lower()
    for loader in _LOADERS:
        if loader.supports(ext):
            return loader
    raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# AG2-facing public functions (flat callables)
# ---------------------------------------------------------------------------

def load_data(
    file_path: Annotated[str, "Absolute or relative path to a CSV, Parquet, or Excel file"],
) -> str:
    """
    AG2 tool entry point.
    Loads CSV/Parquet/XLSX, drops duplicates, returns DataFrame as JSON string.

    Returns:
        JSON string (records orientation) of the loaded DataFrame.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    # Sanitise: LLMs sometimes JSON-escape forward slashes (e.g. \/home → /home)
    file_path = file_path.replace("\\/", "/")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Loading data from %s", file_path)
    df = _get_loader(file_path).load(file_path)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Loaded %d rows × %d columns (%.2f MB)",
                df.shape[0], df.shape[1], df.memory_usage(deep=True).sum() / 1e6)
    result = df.to_json(orient="records")

    # Artifact store: persist for downstream tools
    from tools._pipeline_state import is_active, save_state, STATE_REF_PREFIX
    if is_active():
        save_state("data_json", result)
        return (
            f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {path.name}. "
            f"Reference: {STATE_REF_PREFIX}data_json"
        )
    return result


def validate_schema(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Validates shape, dtypes, and memory footprint from loaded data.

    Returns:
        JSON string of a DataProfile (shape, dtypes, memory_mb).
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))
    profile = DataProfile(
        shape=(df.shape[0], df.shape[1]),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
    )
    logger.info("Schema validated: %d×%d, %.3f MB", *profile.shape, profile.memory_mb)
    result = profile.model_dump_json()

    if is_active():
        save_state("schema_json", result)
        return (
            f"Schema validated: {profile.shape[0]}×{profile.shape[1]}, "
            f"{profile.memory_mb:.3f} MB, {len(profile.dtypes)} columns. "
            f"Reference: {STATE_REF_PREFIX}schema_json"
        )
    return result


def infer_dtypes(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Classifies columns as numerical or categorical.

    Returns:
        JSON string of a DataProfile with numerical_cols and categorical_cols populated.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    profile = DataProfile(
        shape=(df.shape[0], df.shape[1]),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )
    logger.info("Inferred dtypes: %d numerical, %d categorical",
                len(numerical_cols), len(categorical_cols))
    result = profile.model_dump_json()

    if is_active():
        save_state("dtypes_json", result)
        return (
            f"Inferred dtypes: {len(numerical_cols)} numerical, "
            f"{len(categorical_cols)} categorical columns. "
            f"Reference: {STATE_REF_PREFIX}dtypes_json"
        )
    return result


# ---------------------------------------------------------------------------
# Target variable detection (pre-pipeline, called from main.py)
# ---------------------------------------------------------------------------

# General-purpose keywords — NOT dataset-specific.
# Ordered by specificity: exact matches first, then prefix/contains.
_EXACT_KEYWORDS: list[str] = [
    "target", "label", "y", "class", "price", "churn",
]

_CONTAINS_KEYWORDS: list[str] = [
    "target", "label", "class", "outcome",
    "diagnosis", "default", "churn", "response",
]

_PREFIX_KEYWORDS: list[str] = ["is_", "has_"]


def _has_datetime_column(df: pd.DataFrame) -> bool:
    """Return True if any column has datetime dtype or is parseable as dates."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
    # Heuristic: try object columns with 'date', 'time', 'timestamp' in name
    for col in df.select_dtypes(include="object").columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ("date", "time", "timestamp")):
            try:
                pd.to_datetime(df[col].head(5), infer_datetime_format=True)
                return True
            except (ValueError, TypeError):
                pass
    return False


def _classify_target(df: pd.DataFrame, col: str) -> TargetInfo:
    """Build a TargetInfo from a confirmed target column."""
    series = df[col]
    nunique = series.nunique()
    has_dt = _has_datetime_column(df)

    if pd.api.types.is_numeric_dtype(series) and nunique > 10:
        # Numerical continuous → regression
        return TargetInfo(
            column=col,
            problem_type="regression",
            n_classes=0,
            class_counts={},
            imbalance_ratio=1.0,
            has_datetime_index=has_dt,
        )

    # Categorical or low-cardinality numerical → classification
    counts = series.value_counts().to_dict()
    # Convert keys to strings for JSON serialisation
    counts = {str(k): int(v) for k, v in counts.items()}
    count_values = list(counts.values())
    ratio = max(count_values) / max(min(count_values), 1)

    return TargetInfo(
        column=col,
        problem_type="classification",
        n_classes=nunique,
        class_counts=counts,
        imbalance_ratio=round(ratio, 2),
        has_datetime_index=has_dt,
    )


def detect_target(data_json: str) -> str:
    """
    Heuristic target variable detection.  Called pre-pipeline from main.py.

    Detection strategy (in priority order):
      1. Exact column-name match against general keyword list
      2. Column name contains a general keyword
      3. Column name starts with a known prefix (is_, has_)
      4. Fallback: last column with nunique < 10

    If no candidate is found, returns TargetInfo with
    problem_type="unsupervised".

    This is NOT an AG2 tool — it runs before the pipeline starts.
    The result is confirmed interactively, then injected into the
    artifact store for downstream agents.

    Args:
        data_json: JSON string (records orientation) of the DataFrame.

    Returns:
        JSON string of a TargetInfo model.
    """
    df = pd.DataFrame(json.loads(data_json))
    columns_lower = {str(c).lower(): str(c) for c in df.columns}
    has_dt = _has_datetime_column(df)

    # --- Step 1: Exact keyword match ---
    for keyword in _EXACT_KEYWORDS:
        if keyword in columns_lower:
            col = columns_lower[keyword]
            info = _classify_target(df, col)
            info.detection_method = "name_heuristic"
            logger.info("Target detected (exact match): '%s'", col)
            return info.model_dump_json()

    # --- Step 2: Contains keyword match ---
    for keyword in _CONTAINS_KEYWORDS:
        for col_lower, col_orig in columns_lower.items():
            if keyword in col_lower and col_lower != keyword:
                info = _classify_target(df, col_orig)
                info.detection_method = "name_heuristic"
                logger.info("Target detected (contains '%s'): '%s'", keyword, col_orig)
                return info.model_dump_json()

    # --- Step 3: Prefix match (is_, has_) ---
    for prefix in _PREFIX_KEYWORDS:
        for col_lower, col_orig in columns_lower.items():
            if col_lower.startswith(prefix):
                info = _classify_target(df, col_orig)
                info.detection_method = "name_heuristic"
                logger.info("Target detected (prefix '%s'): '%s'", prefix, col_orig)
                return info.model_dump_json()

    # --- Step 4: Fallback — last column with nunique < 10 ---
    low_card = [
        str(c) for c in df.columns if df[c].nunique() < 10
    ]
    if low_card:
        col = low_card[-1]  # last low-cardinality column
        info = _classify_target(df, col)
        info.detection_method = "position_heuristic"
        logger.info("Target detected (low-cardinality fallback): '%s'", col)
        return info.model_dump_json()

    # --- No candidate found ---
    info = TargetInfo(
        column=None,
        problem_type="unsupervised",
        detection_method="none",
        has_datetime_index=has_dt,
    )
    logger.info("No target candidate detected — unsupervised")
    return info.model_dump_json()
