"""
tools/visualization_tools.py — Generate and save EDA visualizations.

Architecture Reference: architecture.md § 4.4, § 12.1

Public AG2-facing functions:

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
from config import get_plots_dir
from tools import _pipeline_state
from typing import Annotated

import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend before any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AG2-facing public functions (flat callables)
# ---------------------------------------------------------------------------

def plot_histograms(
    data_json: Annotated[str, "JSON string (records orientation) from load_data()"],
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

    out = get_plots_dir(_pipeline_state.get_session_id())
    out.mkdir(parents=True, exist_ok=True)

    import re

    paths: list[str] = []
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(f"Histogram — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        safe_col = re.sub(r"[^\w\-]", "_", col)
        file_path = out / f"hist_{safe_col}.png"
        fig.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        paths.append(str(file_path))
        logger.info("Saved histogram: %s", file_path)

    logger.info("Generated %d histogram(s) in %s", len(paths), out)
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

    out = get_plots_dir(_pipeline_state.get_session_id())
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

    out = get_plots_dir(_pipeline_state.get_session_id())
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


def plot_class_distribution(
    target_info_json: Annotated[str, "JSON string of TargetInfo. Pass STATE_REF:target_info in pipeline mode."],
    data_json: Annotated[str, "JSON string (records orientation) from load_data(). Auto-resolved from artifact store in pipeline mode; only needed for direct (non-pipeline) calls."] = "STATE_REF:data_json",
) -> str:
    """
    AG2 tool entry point.
    Plot target variable distribution and save as PNG.

    For classification: horizontal bar chart of class counts + percentages.
    For regression:     histogram + KDE of target distribution.
    For unsupervised:   returns empty list (no plot generated).

    In pipeline mode (is_active() == True):
      - data_json is always loaded directly from the artifact store (LLM need not supply it).
      - target_info_json is loaded from the artifact store; if absent (unsupervised dataset),
        returns an empty-list reference without raising an error.

    Returns:
        JSON list of saved file paths (0 or 1 element).
    """
    from eda_state import TargetInfo

    from tools._pipeline_state import (
        PipelineStateError,
        STATE_REF_PREFIX,
        is_active,
        load_state,
        save_state,
    )
    if is_active():
        # data_json: always load from artifact store — LLM never needs to supply this
        _data = load_state("data_json")
        if _data is None:
            raise PipelineStateError(
                "Cannot resolve artifact 'data_json'. "
                "DataPrepAgent may not have executed load_data()."
            )
        data_json = _data
        # target_info_json: graceful skip if absent (unsupervised dataset)
        _ti = load_state("target_info")
        if _ti is None:
            result = json.dumps([])
            save_state("plot_class_distribution", result)
            return (
                f"No target info in artifact store — class distribution plot skipped. "
                f"Reference: {STATE_REF_PREFIX}plot_class_distribution"
            )
        target_info_json = _ti

    df = pd.DataFrame(json.loads(data_json))
    target_info = TargetInfo.model_validate_json(target_info_json)

    if target_info.column is None or target_info.column not in df.columns:
        logger.info("No target variable — skipping class distribution plot")
        result = json.dumps([])
        if is_active():
            save_state("plot_class_distribution", result)
            return (
                f"No target variable — class distribution plot skipped. "
                f"Reference: {STATE_REF_PREFIX}plot_class_distribution"
            )
        return result

    out = get_plots_dir(_pipeline_state.get_session_id())
    out.mkdir(parents=True, exist_ok=True)

    col = target_info.column

    if target_info.problem_type == "classification":
        counts = df[col].value_counts()
        total = len(df)

        fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.6)))
        bars = ax.barh(
            [str(c) for c in counts.index],
            counts.values,
            color=plt.cm.Set2(np.linspace(0, 1, len(counts))),
            edgecolor="black",
            alpha=0.85,
        )

        # Annotate with count + percentage
        for bar, cnt in zip(bars, counts.values):
            pct = cnt / total * 100
            ax.text(
                bar.get_width() + max(counts.values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({pct:.1f}%)",
                va="center",
                fontsize=10,
            )

        ax.set_title(f"Class Distribution — {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_xlim(0, max(counts.values) * 1.25)

        file_path = out / "class_distribution.png"
        fig.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved class distribution plot: %s", file_path)
        result = json.dumps([str(file_path)])

    elif target_info.problem_type == "regression":
        target_data = df[col].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(target_data, bins=30, edgecolor="black", alpha=0.7, density=True)

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(target_data)
            x_vals = np.linspace(target_data.min(), target_data.max(), 200)
            ax.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="KDE")
            ax.legend()
        except (ImportError, np.linalg.LinAlgError):
            pass  # scipy not available or singular data

        ax.set_title(f"Target Distribution — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        file_path = out / "target_distribution.png"
        fig.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved target distribution plot: %s", file_path)
        result = json.dumps([str(file_path)])

    else:
        result = json.dumps([])

    if is_active():
        save_state("plot_class_distribution", result)
        paths = json.loads(result)
        if paths:
            return (
                f"Saved target distribution plot to {paths[0]}. "
                f"Reference: {STATE_REF_PREFIX}plot_class_distribution"
            )
        return (
            f"No target distribution plot generated. "
            f"Reference: {STATE_REF_PREFIX}plot_class_distribution"
        )
    return result


def plot_categorical_bars(
    categorical_analysis_json: Annotated[
        str,
        "JSON string of CategoricalAnalysis from analyze_categoricals(). "
        "Pass STATE_REF:categorical_analysis when running in pipeline mode.",
    ],
) -> str:
    """
    AG2 tool entry point.
    Plot a horizontal bar chart for each categorical column (top-N categories)
    and save as PNG.

    Uses the pre-computed CategoricalAnalysis artifact so no raw DataFrame
    is needed — mirrors the architecture of the other visualization tools.

    Chart design:
      - Horizontal bars so long category labels are readable.
      - Bars fill proportionally to category percentage (top_values[].pct).
      - Rare categories (<0.5%, flagged as is_rare=True) rendered in a
        distinct colour (#d9534f) to draw the analyst's attention.
      - Annotation: "<count> (<pct>%)" to the right of each bar.
      - Footer note when more_values > 0: "… and N more not shown."

    Filename: ``cat_<safe_col>.png`` where safe_col replaces every
    non-word character with "_" — avoids the space-in-filename problem.

    Returns:
        JSON list of saved file paths (one per categorical column that has
        at least one top-value entry). Empty list if no categorical columns.
    """
    import re

    from tools._pipeline_state import is_active, resolve, save_state, STATE_REF_PREFIX
    if is_active():
        categorical_analysis_json = resolve(
            categorical_analysis_json, "categorical_analysis"
        )

    from eda_state import CategoricalAnalysis as _CatAnalysis

    cat_analysis = _CatAnalysis.model_validate_json(categorical_analysis_json)

    if not cat_analysis.columns:
        logger.info("plot_categorical_bars: no categorical columns — skipping")
        result = json.dumps([])
        if is_active():
            save_state("plot_categorical_bars", result)
            return (
                f"No categorical columns — bar charts skipped. "
                f"Reference: {STATE_REF_PREFIX}plot_categorical_bars"
            )
        return result

    out = get_plots_dir(_pipeline_state.get_session_id())
    out.mkdir(parents=True, exist_ok=True)

    _NORMAL_COLOR = "#5b9bd5"   # standard bar colour
    _RARE_COLOR   = "#d9534f"   # red for rare categories

    paths: list[str] = []

    for col, stats in cat_analysis.columns.items():
        if not stats.top_values:
            logger.info("plot_categorical_bars: '%s' has no top_values — skipped", col)
            continue

        labels  = [str(entry["value"]) for entry in stats.top_values]
        counts  = [int(entry["count"])  for entry in stats.top_values]
        pcts    = [float(entry["pct"])  for entry in stats.top_values]
        is_rare = [bool(entry.get("is_rare", False)) for entry in stats.top_values]
        colors  = [_RARE_COLOR if r else _NORMAL_COLOR for r in is_rare]

        # Reverse so the most-frequent category sits at the top
        labels, counts, pcts, is_rare, colors = (
            list(reversed(labels)),
            list(reversed(counts)),
            list(reversed(pcts)),
            list(reversed(is_rare)),
            list(reversed(colors)),
        )

        n = len(labels)
        fig, ax = plt.subplots(figsize=(10, max(3, n * 0.55)))

        bars = ax.barh(labels, pcts, color=colors, edgecolor="black", alpha=0.82)

        # Annotate each bar with "count (pct%)"
        max_pct = max(pcts) if pcts else 1.0
        for bar, cnt, pct in zip(bars, counts, pcts):
            ax.text(
                bar.get_width() + max_pct * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} ({pct:.1f}%)",
                va="center",
                fontsize=9,
            )

        ax.set_title(f"Category Distribution — {col}")
        ax.set_xlabel("Percentage (%)")
        ax.set_ylabel(col)
        ax.set_xlim(0, max_pct * 1.25)

        # Footer note for truncated columns
        if stats.more_values > 0:
            ax.text(
                0.5, -0.12,
                f"… and {stats.more_values} more not shown (top {cat_analysis.top_n} displayed)",
                ha="center", va="top",
                transform=ax.transAxes,
                fontsize=8, style="italic", color="gray",
            )

        # Legend patch for rare-category colour (only if any rare bars present)
        if any(is_rare):
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor=_NORMAL_COLOR, edgecolor="black", label="Normal"),
                    Patch(facecolor=_RARE_COLOR,   edgecolor="black", label="Rare (<0.5%)"),
                ],
                loc="lower right",
                fontsize=8,
            )

        plt.tight_layout()

        safe_col = re.sub(r"[^\w\-]", "_", col)
        file_path = out / f"cat_{safe_col}.png"
        fig.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        paths.append(str(file_path))
        logger.info("Saved categorical bar chart: %s", file_path)

    logger.info(
        "plot_categorical_bars: generated %d chart(s) in %s", len(paths), out
    )
    result = json.dumps(paths)

    if is_active():
        save_state("plot_categorical_bars", result)
        return (
            f"Generated {len(paths)} categorical bar chart(s). "
            f"Reference: {STATE_REF_PREFIX}plot_categorical_bars"
        )
    return result


def plot_ordinal_heatmap(
) -> str:
    """
    AG2 tool entry point.
    Plot a Spearman rank-correlation heatmap for ordinal-encoded
    categorical columns (≥3 unique values).

    Self-contained: loads data_json, dtypes_json / schema_json, and
    reclassified_subtypes from the artifact store, computes the Spearman
    matrix internally, and saves the heatmap to output_dir.

    Returns empty list when <2 eligible ordinal columns exist.

    Returns:
        JSON list of saved file paths (0 or 1 element).
    """
    from tools._pipeline_state import (
        PipelineStateError,
        STATE_REF_PREFIX,
        is_active,
        load_state,
        save_state,
    )

    if not is_active():
        raise RuntimeError(
            "plot_ordinal_heatmap() requires an active pipeline session."
        )

    data_raw = load_state("data_json")
    if data_raw is None:
        raise PipelineStateError(
            "Cannot resolve artifact 'data_json'. "
            "DataPrepAgent may not have executed load_data()."
        )

    # Identify encoded-categorical columns
    encoded_cols: list[str] = []
    subtypes: dict[str, str] = {}

    subtypes_raw = load_state("reclassified_subtypes")
    if subtypes_raw:
        try:
            subtypes = json.loads(subtypes_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    dtypes_raw = load_state("dtypes_json") or load_state("schema_json")
    if dtypes_raw:
        try:
            from eda_state import DataProfile as _DP
            dp = _DP.model_validate_json(dtypes_raw)
            encoded_cols = list(dp.encoded_categorical_cols or [])
        except Exception:
            pass

    if not encoded_cols:
        result = json.dumps([])
        save_state("plot_ordinal_heatmap", result)
        return (
            f"No encoded-categorical columns — ordinal heatmap skipped. "
            f"Reference: {STATE_REF_PREFIX}plot_ordinal_heatmap"
        )

    df = pd.DataFrame(json.loads(data_raw))

    # Filter to ordinal columns with ≥3 unique values
    if subtypes:
        ord_cols = [
            c for c in encoded_cols
            if c in df.columns
            and df[c].nunique(dropna=True) >= 3
            and subtypes.get(c) == "ordinal"
        ]
    else:
        ord_cols = [
            c for c in encoded_cols
            if c in df.columns
            and df[c].nunique(dropna=True) >= 3
        ]

    if len(ord_cols) < 2:
        result = json.dumps([])
        save_state("plot_ordinal_heatmap", result)
        return (
            f"Fewer than 2 eligible ordinal columns — heatmap skipped. "
            f"Reference: {STATE_REF_PREFIX}plot_ordinal_heatmap"
        )

    df_ord = df[ord_cols].apply(pd.to_numeric, errors="coerce")
    sp_matrix = df_ord.corr(method="spearman")

    out = get_plots_dir(_pipeline_state.get_session_id())
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(ord_cols) * 0.9),
                                    max(5, len(ord_cols) * 0.75)))
    mask = np.triu(np.ones_like(sp_matrix, dtype=bool), k=1)
    sns.heatmap(
        sp_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Ordinal Inter-Correlation (Spearman ρ)")
    plt.tight_layout()

    file_path = out / "ordinal_spearman_heatmap.png"
    fig.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved ordinal Spearman heatmap: %s", file_path)
    result = json.dumps([str(file_path)])

    save_state("plot_ordinal_heatmap", result)
    return (
        f"Saved ordinal Spearman heatmap to {file_path}. "
        f"Reference: {STATE_REF_PREFIX}plot_ordinal_heatmap"
    )


def plot_feature_target_bars(
) -> str:
    """
    AG2 tool entry point.
    Plot a horizontal bar chart of Borda-ranked feature–target associations.

    Self-contained: loads feature_associations from the artifact store.
    Shows MI score and effect size side-by-side per feature, sorted by
    Borda rank.

    Returns empty list when no feature_associations artifact exists.

    Returns:
        JSON list of saved file paths (0 or 1 element).
    """
    from tools._pipeline_state import (
        STATE_REF_PREFIX,
        is_active,
        load_state,
        save_state,
    )

    if not is_active():
        raise RuntimeError(
            "plot_feature_target_bars() requires an active pipeline session."
        )

    fa_raw = load_state("feature_associations")
    if not fa_raw:
        result = json.dumps([])
        save_state("plot_feature_target_bars", result)
        return (
            f"No feature_associations artifact — bar chart skipped. "
            f"Reference: {STATE_REF_PREFIX}plot_feature_target_bars"
        )

    from eda_state import FeatureAssociations as _FA
    fa = _FA.model_validate_json(fa_raw)

    if not fa.rows:
        result = json.dumps([])
        save_state("plot_feature_target_bars", result)
        return (
            f"No feature–target rows — bar chart skipped. "
            f"Reference: {STATE_REF_PREFIX}plot_feature_target_bars"
        )

    out = get_plots_dir(_pipeline_state.get_session_id())
    out.mkdir(parents=True, exist_ok=True)

    # Sort by Borda score ascending (lower = more important, top at top)
    rows_sorted = sorted(fa.rows, key=lambda r: r.borda_score)

    features = [r.feature for r in rows_sorted]
    mi_scores = [r.mi_score for r in rows_sorted]
    es_values = [r.effect_size for r in rows_sorted]
    borda_scores = [r.borda_score for r in rows_sorted]

    n = len(features)
    y_pos = np.arange(n)
    bar_height = 0.35

    fig, ax1 = plt.subplots(figsize=(10, max(4, n * 0.55)))

    bars_mi = ax1.barh(
        y_pos - bar_height / 2, mi_scores, bar_height,
        label="MI score", color="#5b9bd5", edgecolor="black", alpha=0.82,
    )
    ax1.set_xlabel("MI Score")
    ax1.set_ylabel("Feature")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)

    ax2 = ax1.twiny()
    bars_es = ax2.barh(
        y_pos + bar_height / 2, es_values, bar_height,
        label=f"Effect Size ({fa.rows[0].effect_size_type})",
        color="#ff7f0e", edgecolor="black", alpha=0.82,
    )
    ax2.set_xlabel(f"Effect Size ({rows_sorted[0].effect_size_type})")

    # Annotate Borda scores
    for i, borda in enumerate(borda_scores):
        ax1.text(
            max(mi_scores) * 1.02 if mi_scores else 0.01,
            y_pos[i],
            f"Borda={borda}",
            va="center",
            fontsize=8,
            color="gray",
        )

    ax1.set_title(
        f"Feature–Target Associations (vs '{fa.target_col}', {fa.task_type})"
    )

    # Combined legend
    ax1.legend(
        handles=[bars_mi[0], bars_es[0]],
        loc="lower right",
        fontsize=8,
    )

    plt.tight_layout()

    file_path = out / "feature_target_associations.png"
    fig.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved feature–target associations chart: %s", file_path)
    result = json.dumps([str(file_path)])

    save_state("plot_feature_target_bars", result)
    return (
        f"Saved feature–target association chart to {file_path}. "
        f"Reference: {STATE_REF_PREFIX}plot_feature_target_bars"
    )
