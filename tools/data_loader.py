"""
tools/data_loader.py — Load, validate, and classify input data files.

Architecture Reference: architecture.md § 12.4 (Strategy Pattern — DataLoader)

Public AG2-facing functions:
  - load_data(file_path: str) -> str
  - validate_schema(data_json: str) -> str
  - infer_dtypes(data_json: str) -> str

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

from eda_state import DataProfile

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
        return pd.read_csv(path)

    def supports(self, ext: str) -> bool:
        return ext == ".csv"


class ParquetLoader(DataLoader):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    def supports(self, ext: str) -> bool:
        return ext == ".parquet"


class ExcelLoader(DataLoader):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_excel(path, engine="openpyxl")

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
