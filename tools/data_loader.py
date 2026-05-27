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

from eda_state import DataProfile, EncodedCategoricalSuspect, TargetInfo

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
    # Count duplicates BEFORE dropping so the count is available downstream (W8).
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Loaded %d rows × %d columns (%.2f MB); %d duplicate(s) removed",
                df.shape[0], df.shape[1], df.memory_usage(deep=True).sum() / 1e6,
                dup_count)
    result = df.to_json(orient="records")

    # Artifact store: persist for downstream tools
    from tools._pipeline_state import STATE_REF_PREFIX, is_active, save_state
    if is_active():
        save_state("data_json", result)
        save_state("duplicate_count", str(dup_count))
        return (
            f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {path.name} "
            f"({dup_count} duplicate row(s) removed). "
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
    from tools._pipeline_state import STATE_REF_PREFIX, is_active, load_state, resolve, save_state
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))

    # Load original duplicate count from artifact (populated by load_data before dedup).
    dup_count = 0
    if is_active():
        dup_raw = load_state("duplicate_count")
        if dup_raw is not None:
            import contextlib
            with contextlib.suppress(ValueError, TypeError):
                dup_count = int(dup_raw)

    profile = DataProfile(
        shape=(df.shape[0], df.shape[1]),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        duplicate_count=dup_count,
    )
    logger.info("Schema validated: %d×%d, %.3f MB, %d duplicate(s)",
                *profile.shape, profile.memory_mb, profile.duplicate_count)
    result = profile.model_dump_json()

    if is_active():
        save_state("schema_json", result)
        return (
            f"Schema validated: {profile.shape[0]}×{profile.shape[1]}, "
            f"{profile.memory_mb:.3f} MB, {len(profile.dtypes)} columns, "
            f"{profile.duplicate_count} duplicate(s) in original file. "
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
    from tools._pipeline_state import STATE_REF_PREFIX, is_active, resolve, save_state
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Apply pre-pipeline reclassification (if user confirmed encoded categoricals).
    # Moves confirmed columns from numerical_cols → categorical_cols and records
    # them in encoded_categorical_cols for traceability.
    encoded_categorical_cols: list[str] = []
    if is_active():
        from tools._pipeline_state import load_state
        reclass_raw = load_state("reclassified_categoricals")
        if reclass_raw:
            try:
                reclass_list = json.loads(reclass_raw)
                for col in reclass_list:
                    if col in numerical_cols:
                        numerical_cols.remove(col)
                        categorical_cols.append(col)
                        encoded_categorical_cols.append(col)
                if encoded_categorical_cols:
                    logger.info(
                        "Reclassified %d encoded categorical(s): %s",
                        len(encoded_categorical_cols), encoded_categorical_cols,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    # PHYSICAL DTYPE CAST — convert reclassified columns from numeric
    # (int64/float64) to string (object) in the DataFrame itself.
    # Why: every downstream function that calls df.select_dtypes("number")
    # will now correctly *exclude* these columns; string dtype survives the
    # JSON round-trip (artifact store) because JSON strings → pandas object.
    if encoded_categorical_cols:
        for col in encoded_categorical_cols:
            s = df[col]
            non_null = s.dropna()
            # NaN-safe int → str: avoid "1.0" by casting to int first
            # when all non-null values are integer-valued.
            if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                df[col] = non_null.astype(int).astype(str).reindex(s.index)
            else:
                mask = s.isna()
                df[col] = s.astype(str)
                df.loc[mask, col] = None
        # Overwrite data_json in artifact store so every downstream tool
        # (describe_stats, correlation_matrix, critic_rules, plot_histograms,
        # _build_column_stats_block, …) receives the corrected dtypes.
        save_state("data_json", df.to_json(orient="records"))
        logger.info(
            "Cast %d encoded categorical column(s) to string dtype "
            "in artifact store: %s",
            len(encoded_categorical_cols), encoded_categorical_cols,
        )

    profile = DataProfile(
        shape=(df.shape[0], df.shape[1]),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        encoded_categorical_cols=encoded_categorical_cols,
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


# ---------------------------------------------------------------------------
# Encoded-categorical detection (pre-pipeline, called from main.py)
# ---------------------------------------------------------------------------

# Pre-filter: only columns with nunique ≤ this are sent to the LLM.
# Columns above this cardinality are almost certainly continuous numerics.
_MAX_PROFILED_CARDINALITY: int = 30


def _build_column_profiles(df: pd.DataFrame) -> list[dict]:
    """Build compact profiles for numeric columns suitable for LLM classification.

    Only columns with nunique ≤ _MAX_PROFILED_CARDINALITY are included (the
    rest are unambiguously continuous).  This keeps the LLM prompt small.

    Returns:
        List of dicts, each with keys: name, dtype, nunique, n_rows,
        sample_values, min, max, is_all_integer.
    """
    profiles: list[dict] = []
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        series = df[col].dropna()
        nunique = int(series.nunique())
        if nunique > _MAX_PROFILED_CARDINALITY:
            continue
        # Sample values: sorted unique values (up to 15 for readability)
        unique_vals = sorted(series.unique().tolist())
        sample = unique_vals[:15]
        is_all_int = bool((series == series.astype(int)).all()) if len(series) > 0 else False
        profiles.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "nunique": nunique,
            "n_rows": len(df),
            "sample_values": sample,
            "min": float(series.min()) if len(series) > 0 else 0.0,
            "max": float(series.max()) if len(series) > 0 else 0.0,
            "is_all_integer": is_all_int,
        })
    return profiles


_LLM_SYSTEM_PROMPT = """\
You are a senior data scientist. Given column profiles from a dataset, identify \
which numeric columns are actually encoded categorical variables (e.g., SEX \
encoded as 1/2, EDUCATION as 1/2/3/4, repayment status codes like -2,-1,0,1,...8).

For each suspected column, provide:
- column: the exact column name
- reason: one sentence explaining why (mention name semantics + value pattern)
- subtype: "nominal" or "ordinal"

Rules:
- Only flag columns you are confident about.
- Do NOT flag true continuous numerics (age, salary, amounts, balances, counts).
- Do NOT flag ID/index columns.
- Do NOT flag binary targets (0/1 class labels) — those are handled separately.
- When in doubt, do NOT flag the column.

Return valid JSON: {"suspects": [{"column": "...", "reason": "...", "subtype": "..."}]}
If no columns are suspected, return: {"suspects": []}"""


def detect_encoded_categoricals(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
) -> list[EncodedCategoricalSuspect]:
    """Detect numeric columns that are likely encoded categoricals via an LLM call.

    Pre-pipeline function called from main.py (same pattern as detect_target).
    Returns a list of suspects with reasoning; the caller handles user confirmation.

    Args:
        df: The loaded (deduplicated) DataFrame.
        target_column: If set, excluded from profiling (handled by target detection).

    Returns:
        List of EncodedCategoricalSuspect (may be empty).
    """
    profiles = _build_column_profiles(df)
    # Exclude target column from candidates (already classified by detect_target)
    if target_column:
        profiles = [p for p in profiles if p["name"] != target_column]
    if not profiles:
        logger.info("No low-cardinality numeric columns to evaluate for reclassification")
        return []

    # Build user prompt
    profile_text = json.dumps(profiles, indent=2)
    user_prompt = (
        f"Dataset has {len(df)} rows and {len(df.columns)} columns.\n"
        f"Column profiles for low-cardinality numeric columns:\n{profile_text}"
    )

    try:
        from openai import OpenAI

        from config import RECLASSIFY_MODEL
        client = OpenAI()
        resp = client.chat.completions.create(
            model=RECLASSIFY_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
    except Exception:
        logger.warning(
            "Encoded-categorical LLM detection failed — skipping reclassification",
            exc_info=True,
        )
        return []

    suspects: list[EncodedCategoricalSuspect] = []
    profile_map = {p["name"]: p for p in profiles}
    for s in result.get("suspects", []):
        col_name = s.get("column", "")
        if col_name not in profile_map:
            continue  # LLM hallucinated a column name — skip
        p = profile_map[col_name]
        suspects.append(EncodedCategoricalSuspect(
            column=col_name,
            nunique=p["nunique"],
            sample_values=p["sample_values"],
            min_val=p["min"],
            max_val=p["max"],
            is_all_integer=p["is_all_integer"],
            reason=s.get("reason", ""),
            subtype=s.get("subtype", "nominal"),
        ))

    logger.info(
        "Encoded-categorical detection: %d suspect(s) from %d profiled columns",
        len(suspects), len(profiles),
    )
    return suspects
