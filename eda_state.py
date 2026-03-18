"""
eda_state.py — Pydantic sub-models for tool I/O validation + router helpers.

Architecture Reference: architecture.md § 7 (schema), § 6 (router), § 12.2 (patterns)

Design rationale (AG2-native approach):
  - Each tool validates its OWN inputs/outputs using these sub-models.
  - Tools return JSON strings that land in the GroupChat message history.
  - Agents read prior messages naturally via LLM context — no monolithic
    state object is serialized/deserialized between agents.
  - The state_flow_transition router uses a targeted helper
    (get_critic_status) to inspect only the signals it needs, rather
    than parsing a full state blob from message text.

AG2 Version: 0.10.3
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-models — used by tools (tools layer) for I/O validation
# ---------------------------------------------------------------------------

class TargetInfo(BaseModel):
    """Target variable metadata — detected heuristically, confirmed by user.

    Used by target_analysis(), ClassImbalanceRule, and the report generator
    to produce problem-type-aware EDA output.

    detection_method values:
      - "name_heuristic":     matched a general keyword in column names
      - "position_heuristic": last column + low cardinality fallback
      - "user_specified":     user provided via --target or interactive override
      - "none":               no target detected / user declined
    """

    column: str | None = None            # None = unsupervised
    problem_type: str = "unsupervised"   # "classification" | "regression" | "unsupervised"
    n_classes: int = 0                   # >0 for classification only
    class_counts: dict[str, int] = Field(default_factory=dict)
    imbalance_ratio: float = 1.0         # majority_count / minority_count
    detection_method: str = ""           # see docstring
    has_datetime_index: bool = False     # True if a datetime column detected


class DataProfile(BaseModel):
    """Schema and shape metadata produced by DataPrepAgent's tools."""

    shape: tuple[int, int] = (0, 0)
    memory_mb: float = 0.0
    dtypes: dict[str, str] = Field(default_factory=dict)
    numerical_cols: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)
    duplicate_count: int = 0  # Rows removed by load_data() dedup (W8)


class CategoricalStats(BaseModel):
    """Per-column categorical distribution summary (W4)."""

    cardinality: int = 0
    entropy_bits: float = 0.0        # Shannon entropy in bits
    rare_count: int = 0              # Values with freq < 0.5%
    top_values: list[dict[str, Any]] = Field(default_factory=list)
    more_values: int = 0             # Values beyond top-N not shown


class CategoricalAnalysis(BaseModel):
    """Full categorical inventory produced by analyze_categoricals() (W4)."""

    columns: dict[str, CategoricalStats] = Field(default_factory=dict)
    target_column: str | None = None
    top_n: int = 10


class AssociationRow(BaseModel):
    """Per-feature result row from compute_feature_target_associations() (W7).

    Two complementary lenses:
      - MI (mutual information): detects any form of dependence (kNN-estimated).
      - Effect size: measures practical magnitude in an n-invariant [0, 1] scale.

    Effect size metric selection by feature/task combination:
      - numerical feature + classification  → eta²  (SS_between/SS_total, one-way ANOVA)
      - categorical feature + classification → Cramér's V  (sqrt(chi²/(n*(k-1))))
      - numerical feature + regression       → |Pearson r|  (absolute correlation)
      - categorical feature + regression     → eta²  (reversed ANOVA: target ~ cat_feature)

    Ranking:
      borda_score = mi_rank + effect_size_rank  (lower = more important)
      F-statistic / chi² and p-value are stored as supplementary data only —
      they scale with n and must NOT drive the rank.
    """

    feature: str = ""
    feature_type: str = ""           # "numerical" | "categorical"

    # --- MI lens (kNN, possibly on stratified sample) ---
    mi_score: float = 0.0            # nats; higher = more dependence
    mi_rank: int = 0                 # 1 = highest MI

    # --- Effect size lens (full dataset, O(n), n-invariant) ---
    effect_size: float = 0.0         # [0, 1]; higher = larger practical effect
    effect_size_type: str = ""       # "eta2" | "cramer_v" | "pearson_r"
    effect_size_label: str = ""      # "weak" | "moderate" | "strong"
    effect_size_rank: int = 0        # 1 = largest effect size

    # --- Borda fusion (the ranking signal) ---
    borda_score: int = 0             # mi_rank + effect_size_rank (lower = more important)

    # --- Supplementary (informational only — NOT used in ranking) ---
    f_stat_or_chi2: float = 0.0      # raw test statistic (n-dependent)
    p_value: float = 1.0             # significance witness


class FeatureAssociations(BaseModel):
    """Top-N feature–target association results produced by
    compute_feature_target_associations() (W7).

    MI is computed on a stratified sample when n > MAX_ROWS_FOR_MI (50K)
    to keep kNN estimation fast.  Effect sizes always use the full dataset.
    """

    target_col: str = ""
    task_type: str = "unknown"       # "classification" | "regression" | "unknown"
    rows: list[AssociationRow] = Field(default_factory=list)
    top_n: int = 10
    total_features: int = 0
    mi_sample_size: int | None = None   # None = full dataset used for MI
    mi_sample_note: str = ""            # set when sampling occurred
    missingness_strategy: str = ""      # "complete-case (...)" or "imputed (...)"


class MissingInfo(BaseModel):
    """Missing-value statistics produced by EDAAnalysisAgent's tools."""

    per_column: dict[str, float] = Field(default_factory=dict)
    total_pct: float = 0.0


class EDAResults(BaseModel):
    """Statistical analysis results produced by EDAAnalysisAgent's tools."""

    describe: dict[str, Any] = Field(default_factory=dict)
    missing: MissingInfo = Field(default_factory=MissingInfo)
    correlation: dict[str, Any] = Field(default_factory=dict)


class CriticFlag(BaseModel):
    """A single quality flag raised by the Critic's rule engine."""

    column: Optional[str] = None  # None = dataset-level flag
    rule: str = ""
    severity: str = ""  # BLOCKER | HIGH | MEDIUM | LOW
    message: str = ""
    value: float = 0.0
    suggestion: str = ""  # Recommended action (e.g., "log transform recommended")


class CriticReport(BaseModel):
    """Rule-based quality assessment returned by run_critic_rules()."""

    flags: list[CriticFlag] = Field(default_factory=list)
    iteration: int = 0
    status: str = "PENDING"  # PENDING | APPROVED | REVISION_NEEDED


class Findings(BaseModel):
    """Structured narrative returned by assemble_findings()."""

    sections: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_flags: list[str] = Field(default_factory=list)


class PlotCommentary(BaseModel):
    """Three-lens expert commentary for a single plot."""

    plot_file: str = ""             # filename, e.g. "hist_sepal_length.png"
    statistical: str = ""           # statistical perspective
    ds_ml: str = ""                 # data science / ML perspective
    business: str = ""              # business perspective


class Interpretations(BaseModel):
    """LLM-generated expert commentary for all report sections.

    Validated before storage in the artifact store.
    Each key mirrors a report section.  If a key is missing or empty,
    the deterministic fallback text is used instead (safety net).

    Future extension point: a VisionCapability verification layer
    could cross-reference these text interpretations against the
    actual PNG images.  Not implemented — metadata coverage is 100%.
    """

    overview: Optional[dict[str, str]] = None
    missing_values: Optional[dict[str, str]] = None
    correlation: Optional[dict[str, str]] = None
    statistical_analysis: Optional[dict[str, str]] = None
    categorical_analysis: Optional[dict[str, str]] = None
    feature_associations: Optional[dict[str, str]] = None   # W7
    target_variable_analysis: Optional[dict[str, str]] = None
    quality_assessment: Optional[dict[str, str]] = None
    plot_commentaries: list[PlotCommentary] = Field(default_factory=list)
    conclusions: str = ""
    recommendations_and_business_implications: str = ""


# ---------------------------------------------------------------------------
# Router helper — minimal extraction for state_flow_transition
# ---------------------------------------------------------------------------
#
# AG2-native approach: the router inspects agent names and keyword
# markers in message content — exactly like the reference example
# (ag2_groupchat_state_flow_transition.py) checks for "Growth"/"Value".
#
# CriticAgent's system_message instructs it to include APPROVED or
# REVISION_NEEDED as a keyword. The iteration count is simply how
# many times CriticAgent has spoken in the chat history.
# ---------------------------------------------------------------------------

def get_critic_status(messages: list[dict[str, Any]]) -> tuple[str, int]:
    """
    Determine the CriticAgent's latest status and iteration count
    from the GroupChat message history.

    Uses the AG2-native pattern:
      - Filter messages by agent name (standard AG2 message field)
      - Check for status keywords in the last CriticAgent message
      - Count CriticAgent turns for iteration

    Used exclusively by state_flow_transition() to decide:
      - REVISION_NEEDED + iteration < 2 → loop back to CriticAgent
      - APPROVED or iteration >= 2       → proceed to ReportExporterAgent

    Returns:
        (status, iteration) — defaults to ("PENDING", 0) if CriticAgent
        has not spoken yet.
    """
    critic_messages = [m for m in messages if m.get("name") == "CriticAgent"]
    iteration = len(critic_messages)

    if not critic_messages:
        return "PENDING", 0

    last_content = critic_messages[-1].get("content") or ""

    if "APPROVED" in last_content:
        return "APPROVED", iteration
    if "REVISION_NEEDED" in last_content:
        return "REVISION_NEEDED", iteration

    return "PENDING", iteration
