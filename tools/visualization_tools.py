"""
tools/visualization_tools.py — Generate and save EDA visualizations.

Architecture Reference: architecture.md § 4.4, § 12.1

Public AG2-facing functions:
  - plot_histograms(data_json: str, output_dir: str) -> str
  - plot_correlation_heatmap(corr_json: str, output_dir: str) -> str
  - plot_missing_heatmap(missing_json: str, output_dir: str) -> str

Design:
  - Zero AG2 imports. Zero agent references. Pure Python.
  - Each function accepts JSON strings (the AG2 tool contract).
  - Each function saves PNG files to the specified output directory.
  - Returns a JSON list of saved file paths.
  - Matplotlib backend forced to 'Agg' (non-interactive, headless safe).

AG2 Version: 0.10.3
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend before any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AG2-facing public functions (flat callables)
# ---------------------------------------------------------------------------

def plot_histograms(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
    output_dir: Annotated[str, "Directory path where PNG files will be saved"],
) -> str:
    """
    AG2 tool entry point.
    Plot a histogram for each numerical column and save as PNG.

    Non-numerical columns are skipped. Each histogram is saved as
    ``<output_dir>/hist_<column_name>.png``.

    Returns:
        JSON list of saved file paths (absolute strings).
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        data_json = resolve(data_json, "data_json")

    df = pd.DataFrame(json.loads(data_json))
    num_cols = df.select_dtypes(include="number").columns.tolist()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(f"Histogram — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        file_path = out / f"hist_{col}.png"
        fig.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        paths.append(str(file_path))
        logger.info("Saved histogram: %s", file_path)

    logger.info("Generated %d histogram(s) in %s", len(paths), output_dir)
    result = json.dumps(paths)

    if is_active():
        save_state("plot_histograms", result)
        return (
            f"Generated {len(paths)} histogram(s) for numerical columns. "
            f"Reference: {STATE_REF_PREFIX}plot_histograms"
        )
    return result


def plot_correlation_heatmap(
    corr_json: Annotated[str, "JSON string of correlation matrix from correlation_matrix()"],
    output_dir: Annotated[str, "Directory path where PNG file will be saved"],
) -> str:
    """
    AG2 tool entry point.
    Plot a Pearson correlation heatmap and save as PNG.

    If the correlation dict is empty (no numerical columns), returns
    an empty JSON list without creating a file.

    Returns:
        JSON list of saved file paths (0 or 1 element).
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        corr_json = resolve(corr_json, "correlation_matrix")

    corr_dict = json.loads(corr_json)

    if not corr_dict:
        logger.info("Empty correlation matrix — skipping heatmap")
        return json.dumps([])

    corr_df = pd.DataFrame(corr_dict).apply(pd.to_numeric, errors="coerce")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap")

    file_path = out / "correlation_heatmap.png"
    fig.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved correlation heatmap: %s", file_path)
    result = json.dumps([str(file_path)])

    if is_active():
        save_state("plot_correlation_heatmap", result)
        return (
            f"Saved correlation heatmap to {file_path}. "
            f"Reference: {STATE_REF_PREFIX}plot_correlation_heatmap"
        )
    return result


def plot_missing_heatmap(
    missing_json: Annotated[str, "JSON string of MissingInfo from missing_analysis()"],
    output_dir: Annotated[str, "Directory path where PNG file will be saved"],
) -> str:
    """
    AG2 tool entry point.
    Plot a bar chart of per-column missing value percentages and save as PNG.

    If all columns have 0% missing, still produces the chart (showing zeros).
    If missing_json has no columns (empty per_column dict), returns an empty list.

    Returns:
        JSON list of saved file paths (0 or 1 element).
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        missing_json = resolve(missing_json, "missing_analysis")

    info = json.loads(missing_json)
    per_column: dict[str, float] = info.get("per_column", {})

    if not per_column:
        logger.info("No columns in missing info — skipping heatmap")
        return json.dumps([])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    columns = list(per_column.keys())
    values = list(per_column.values())

    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.8), 5))
    bars = ax.bar(columns, values, color="salmon", edgecolor="black", alpha=0.8)

    # Annotate bars with percentage values
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Missing Values by Column (%)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing %")
    ax.set_ylim(0, max(max(values) * 1.15, 1))  # breathing room above bars

    if len(columns) > 6:
        plt.xticks(rotation=45, ha="right")

    file_path = out / "missing_heatmap.png"
    fig.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved missing heatmap: %s", file_path)
    result = json.dumps([str(file_path)])

    if is_active():
        save_state("plot_missing_heatmap", result)
        return (
            f"Saved missing-values heatmap to {file_path}. "
            f"Reference: {STATE_REF_PREFIX}plot_missing_heatmap"
        )
    return result
