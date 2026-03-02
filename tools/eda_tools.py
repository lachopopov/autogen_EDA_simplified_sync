"""
tools/eda_tools.py — Descriptive statistics, missingness, and correlation analysis.

Architecture Reference: architecture.md § 4.3, § 12.1

Public AG2-facing functions:
  - describe_stats(data_json: str) -> str
  - missing_analysis(data_json: str) -> str
  - correlation_matrix(data_json: str) -> str

Design:
  - Zero AG2 imports. Zero agent references. Pure Python.
  - Each function accepts/returns JSON strings (the AG2 tool contract).
  - Pydantic sub-models from eda_state.py validate outputs:
      * describe_stats  → validated through EDAResults(describe=...)
      * missing_analysis → validated through MissingInfo(...)
      * correlation_matrix → validated through EDAResults(correlation=...)

AG2 Version: 0.10.3
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

import pandas as pd

from eda_state import EDAResults, MissingInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AG2-facing public functions (flat callables)
# ---------------------------------------------------------------------------

def describe_stats(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Compute descriptive statistics: central tendency, spread, percentiles.

    Includes all column types (numerical and categorical).
    NaN values in the describe output are serialized as JSON null.

    Returns:
        JSON string of a dict mapping column names to stat dicts.
        Validated through EDAResults(describe=...) before return.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))

    if df.empty or df.columns.size == 0:
        return json.dumps({})

    desc_df = df.describe(include="all")

    # pandas to_json handles NaN → null correctly
    desc_dict = json.loads(desc_df.to_json())

    # Validate structure through Pydantic sub-model
    EDAResults(describe=desc_dict)

    logger.info(
        "Descriptive statistics computed for %d columns (%d stats each)",
        len(desc_dict),
        len(next(iter(desc_dict.values()), {})),
    )
    result = json.dumps(desc_dict)

    if is_active():
        save_state("describe_stats", result)
        n_cols = len(desc_dict)
        n_stats = len(next(iter(desc_dict.values()), {}))
        return (
            f"Computed descriptive statistics for {n_cols} columns "
            f"({n_stats} stats each). "
            f"Reference: {STATE_REF_PREFIX}describe_stats"
        )
    return result


def missing_analysis(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Compute per-column and dataset-level missing value percentages.

    Returns:
        JSON string of a MissingInfo model (per_column, total_pct).
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))

    per_column = (df.isnull().mean() * 100).round(2).to_dict()

    total_cells = df.shape[0] * df.shape[1]
    total_pct = round(
        df.isnull().sum().sum() / max(total_cells, 1) * 100, 2
    )

    info = MissingInfo(per_column=per_column, total_pct=total_pct)

    logger.info(
        "Missing analysis: %.2f%% total, %d columns checked",
        info.total_pct,
        len(info.per_column),
    )
    result = info.model_dump_json()

    if is_active():
        save_state("missing_analysis", result)
        return (
            f"Missing analysis complete: {info.total_pct:.2f}% total missing "
            f"across {len(info.per_column)} columns. "
            f"Reference: {STATE_REF_PREFIX}missing_analysis"
        )
    return result


def correlation_matrix(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
) -> str:
    """
    AG2 tool entry point.
    Compute Pearson correlation matrix for numerical columns only.

    Non-numerical columns are excluded. If no numerical columns exist,
    returns an empty dict.

    Returns:
        JSON string of a nested dict {col: {col: corr_value, ...}, ...}.
        Validated through EDAResults(correlation=...) before return.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))
    num_df = df.select_dtypes(include="number")

    if num_df.empty:
        corr_dict: dict = {}
    else:
        corr_df = num_df.corr(method="pearson").round(4)
        # pandas to_json handles NaN → null correctly
        corr_dict = json.loads(corr_df.to_json())

    # Validate structure through Pydantic sub-model
    EDAResults(correlation=corr_dict)

    logger.info(
        "Correlation matrix computed: %d × %d numerical columns",
        len(corr_dict),
        len(corr_dict),
    )
    result = json.dumps(corr_dict)

    if is_active():
        save_state("correlation_matrix", result)
        return (
            f"Correlation matrix computed for {len(corr_dict)} numerical columns. "
            f"Reference: {STATE_REF_PREFIX}correlation_matrix"
        )
    return result
