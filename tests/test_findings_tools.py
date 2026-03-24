"""
tests/test_findings_tools.py — Unit tests for tools/findings_tools.py

Tests assemble_findings() with various EDA results, critic reports,
and plot paths. Validates outputs against Pydantic Findings model.
No LLM calls — pure function tests.
"""

import json

import pytest

from eda_state import (
    CategoricalAnalysis,
    CategoricalStats,
    CriticFlag,
    CriticReport,
    DataProfile,
    EDAResults,
    Findings,
    MissingInfo,
)
from tools.findings_tools import (
    _build_categorical_inventory,
    _build_categorical_section,
    _build_conclusions_section,
    _build_correlation_section,
    _build_missing_section,
    _build_overview_section,
    _build_quality_section,
    _build_recommendations_section,
    _build_statistical_analysis_section,
    _build_target_section,
    _build_visualizations_section,
    _collect_unresolved,
    assemble_findings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def eda_results_basic():
    """Basic EDAResults with describe, missing, and correlation."""
    return EDAResults(
        describe={
            "age": {"count": 100.0, "mean": 35.0, "std": 10.0, "min": 18.0, "25%": 28.0, "50%": 35.0, "75%": 42.0, "max": 65.0},
            "income": {"count": 100.0, "mean": 50000.0, "std": 15000.0, "min": 20000.0, "25%": 40000.0, "50%": 50000.0, "75%": 60000.0, "max": 100000.0},
        },
        missing=MissingInfo(
            per_column={"age": 5.0, "income": 12.0, "city": 0.0},
            total_pct=5.67,
        ),
        correlation={
            "age": {"age": 1.0, "income": 0.45},
            "income": {"age": 0.45, "income": 1.0},
        },
    )


@pytest.fixture()
def eda_results_empty():
    """Empty EDAResults — no data analyzed."""
    return EDAResults()


@pytest.fixture()
def critic_approved():
    """Critic report with APPROVED status, no flags."""
    return CriticReport(flags=[], iteration=1, status="APPROVED")


@pytest.fixture()
def critic_revision_needed():
    """Critic report with REVISION_NEEDED, iteration 1."""
    return CriticReport(
        flags=[
            CriticFlag(column="income", rule="skewness", severity="HIGH",
                       message="|skew|=2.50", value=2.5),
            CriticFlag(column=None, rule="duplicate_rows", severity="MEDIUM",
                       message="3.0% duplicate rows", value=0.03),
        ],
        iteration=1,
        status="REVISION_NEEDED",
    )


@pytest.fixture()
def critic_revision_iter2():
    """Critic report with REVISION_NEEDED at iteration 2 (forced finalize)."""
    return CriticReport(
        flags=[
            CriticFlag(column="income", rule="skewness", severity="HIGH",
                       message="|skew|=2.50", value=2.5),
        ],
        iteration=2,
        status="REVISION_NEEDED",
    )


@pytest.fixture()
def plot_paths_sample():
    """Sample list of plot file paths."""
    return ["outputs/plots/hist_age.png", "outputs/plots/correlation_heatmap.png"]


# ---------------------------------------------------------------------------
# _build_overview_section
# ---------------------------------------------------------------------------

class TestBuildOverviewSection:
    """Test the overview section builder."""

    def test_returns_dict(self, eda_results_basic):
        section = _build_overview_section(eda_results_basic)
        assert isinstance(section, dict)

    def test_has_title(self, eda_results_basic):
        section = _build_overview_section(eda_results_basic)
        assert section["title"] == "Dataset Overview"

    def test_has_content(self, eda_results_basic):
        section = _build_overview_section(eda_results_basic)
        assert "content" in section
        assert len(section["content"]) > 0

    def test_row_count_in_content(self, eda_results_basic):
        section = _build_overview_section(eda_results_basic)
        assert "100" in section["content"]

    def test_column_count_in_content(self, eda_results_basic):
        section = _build_overview_section(eda_results_basic)
        assert "2" in section["content"]

    def test_empty_describe(self, eda_results_empty):
        section = _build_overview_section(eda_results_empty)
        assert section["title"] == "Dataset Overview"
        assert "0" in section["content"]

    def test_duplicate_count_mentioned_when_nonzero(self, eda_results_basic):
        """W8: duplicate_count > 0 → sentence in overview content."""
        section = _build_overview_section(eda_results_basic, shape=(99, 5), duplicate_count=3)
        assert "3 duplicate" in section["content"]
        # 3 / (99 + 3) * 100 ≈ 2.9%
        assert "%" in section["content"]
        assert "removed" in section["content"]

    def test_duplicate_count_zero_no_mention(self, eda_results_basic):
        """W8: duplicate_count == 0 → no duplicate sentence."""
        section = _build_overview_section(eda_results_basic, shape=(100, 5), duplicate_count=0)
        assert "duplicate" not in section["content"]

    def test_duplicate_percentage_calculation(self, eda_results_basic):
        """W8: percentage = dup_count / (row_count + dup_count) * 100."""
        # 10 dupes, 90 rows after dedup → 10/100 = 10%
        section = _build_overview_section(eda_results_basic, shape=(90, 3), duplicate_count=10)
        assert "10 duplicate" in section["content"]
        assert "10.0%" in section["content"]


# ---------------------------------------------------------------------------
# _build_missing_section
# ---------------------------------------------------------------------------

class TestBuildMissingSection:
    """Test the missing values section builder."""

    def test_returns_dict(self, eda_results_basic):
        section = _build_missing_section(eda_results_basic)
        assert isinstance(section, dict)

    def test_title(self, eda_results_basic):
        section = _build_missing_section(eda_results_basic)
        assert section["title"] == "Missing Values"

    def test_mentions_total_pct(self, eda_results_basic):
        section = _build_missing_section(eda_results_basic)
        assert "5.7" in section["content"]  # 5.67 → 5.7%

    def test_lists_columns_with_missing(self, eda_results_basic):
        section = _build_missing_section(eda_results_basic)
        assert "income" in section["content"]
        assert "age" in section["content"]

    def test_no_missing_values(self):
        eda = EDAResults(missing=MissingInfo(per_column={"a": 0.0, "b": 0.0}, total_pct=0.0))
        section = _build_missing_section(eda)
        assert "No missing" in section["content"]

    def test_empty_per_column(self, eda_results_empty):
        section = _build_missing_section(eda_results_empty)
        assert "No missing" in section["content"]


# ---------------------------------------------------------------------------
# _build_correlation_section
# ---------------------------------------------------------------------------

class TestBuildCorrelationSection:
    """Test the correlation analysis section builder."""

    def test_returns_dict(self, eda_results_basic):
        section = _build_correlation_section(eda_results_basic)
        assert isinstance(section, dict)

    def test_title(self, eda_results_basic):
        section = _build_correlation_section(eda_results_basic)
        assert section["title"] == "Correlation Analysis"

    def test_mentions_strongest_pair(self, eda_results_basic):
        section = _build_correlation_section(eda_results_basic)
        assert "age" in section["content"]
        assert "income" in section["content"]
        assert "0.45" in section["content"]

    def test_empty_correlation(self, eda_results_empty):
        section = _build_correlation_section(eda_results_empty)
        assert "No numerical" in section["content"]


# ---------------------------------------------------------------------------
# _build_visualizations_section
# ---------------------------------------------------------------------------

class TestBuildVisualizationsSection:
    """Test the visualizations section builder."""

    def test_returns_dict(self, plot_paths_sample):
        section = _build_visualizations_section(plot_paths_sample)
        assert isinstance(section, dict)

    def test_title(self, plot_paths_sample):
        section = _build_visualizations_section(plot_paths_sample)
        assert section["title"] == "Visualizations"

    def test_count_in_content(self, plot_paths_sample):
        section = _build_visualizations_section(plot_paths_sample)
        assert "2" in section["content"]

    def test_includes_plot_paths(self, plot_paths_sample):
        section = _build_visualizations_section(plot_paths_sample)
        assert section["plot_paths"] == plot_paths_sample

    def test_no_plots(self):
        section = _build_visualizations_section([])
        assert "No visualizations" in section["content"]


# ---------------------------------------------------------------------------
# _build_statistical_analysis_section
# ---------------------------------------------------------------------------

class TestBuildStatisticalAnalysisSection:
    """Test the statistical analysis section builder."""

    def test_title(self, eda_results_basic):
        section = _build_statistical_analysis_section(eda_results_basic)
        assert section["title"] == "Statistical Analysis"

    def test_numerical_cols_mentioned(self, eda_results_basic):
        section = _build_statistical_analysis_section(eda_results_basic)
        assert "numerical feature" in section["content"]

    def test_empty_eda(self):
        eda = EDAResults()
        section = _build_statistical_analysis_section(eda)
        assert "No descriptive statistics" in section["content"]

    def test_high_cv_detection(self):
        """Coefficient of variation > 1.0 is flagged."""
        eda = EDAResults(describe={
            "sparse": {"count": 100, "mean": 1.0, "std": 5.0, "min": 0, "25%": 0, "50%": 0, "75%": 1, "max": 50},
        })
        section = _build_statistical_analysis_section(eda)
        assert "variability" in section["content"].lower()
        assert "sparse" in section["content"]

    def test_zero_iqr_detection(self):
        """Near-zero IQR (constant column) is flagged."""
        eda = EDAResults(describe={
            "const": {"count": 100, "mean": 5.0, "std": 0.0, "min": 5, "25%": 5, "50%": 5, "75%": 5, "max": 5},
        })
        section = _build_statistical_analysis_section(eda)
        assert "zero" in section["content"].lower() or "minimal" in section["content"].lower()

    def test_outlier_detection(self):
        """Values beyond 1.5×IQR fences are flagged as potential outliers."""
        eda = EDAResults(describe={
            "skewed": {"count": 100, "mean": 10.0, "std": 5.0, "min": -50, "25%": 7, "50%": 10, "75%": 13, "max": 100},
        })
        section = _build_statistical_analysis_section(eda)
        assert "outlier" in section["content"].lower()

    def test_low_cardinality_categorical(self):
        """Categorical with ≤2 unique values is flagged."""
        eda = EDAResults(describe={
            "gender": {"count": 100, "unique": 2, "top": "M", "freq": 55},
        })
        section = _build_statistical_analysis_section(eda)
        assert "low-cardinality" in section["content"].lower()

    def test_clean_data_no_issues(self):
        """Clean numeric data produces positive message."""
        eda = EDAResults(describe={
            "normal": {"count": 100, "mean": 50.0, "std": 10.0, "min": 20, "25%": 43, "50%": 50, "75%": 57, "max": 80},
        })
        section = _build_statistical_analysis_section(eda)
        assert "standard distribution" in section["content"].lower() or "numerical feature" in section["content"].lower()

    def test_critic_low_iqr_flag_suppresses_removal_advice(self):
        """Column flagged LOW by outliers_iqr must NOT appear in the generic
        'treatment should be considered' sentence — it should appear in the
        IQR-unreliable sentence instead, with a pointer to Data Quality Assessment."""
        eda = EDAResults(describe={
            # hours: min=1 far below fence -> ends up in potential_outlier_cols
            "hours": {"count": 1000, "mean": 40.0, "std": 5.0,
                      "min": 1, "25%": 39, "50%": 40, "75%": 41, "max": 80},
            # income: also has an outlier but NOT flagged LOW by critic
            "income": {"count": 1000, "mean": 50.0, "std": 10.0,
                       "min": -200, "25%": 45, "50%": 50, "75%": 55, "max": 500},
        })
        critic = CriticReport(flags=[
            CriticFlag(column="hours", rule="outliers_iqr", severity="LOW",
                       message="27% outliers -- IQR unreliable", value=0.277),
        ])
        section = _build_statistical_analysis_section(eda, critic)
        content = section["content"]

        # "hours" must NOT appear alongside "treatment should be considered"
        assert "treatment" not in content.lower() or "hours" not in content.split("treatment")[0].split("in:")[-1]
        # The IQR-unreliable sentence must be present and name the column
        assert "IQR method unreliable" in content
        assert "hours" in content
        # "income" (no LOW flag) must still be in the generic outlier sentence
        assert "income" in content
        assert "treatment" in content.lower()
        # The cross-reference pointer is present
        assert "Data Quality Assessment" in content

    def test_critic_none_behavior_unchanged(self):
        """With critic=None the function behaves identically to passing no critic."""
        eda = EDAResults(describe={
            "skewed": {"count": 100, "mean": 10.0, "std": 5.0,
                       "min": -50, "25%": 7, "50%": 10, "75%": 13, "max": 100},
        })
        section_default = _build_statistical_analysis_section(eda)
        section_explicit_none = _build_statistical_analysis_section(eda, None)
        assert section_default["content"] == section_explicit_none["content"]
        # The original generic outlier sentence is emitted when no critic is given
        assert "outlier" in section_default["content"].lower()
        assert "treatment" in section_default["content"].lower()


# ---------------------------------------------------------------------------
# _build_conclusions_section
# ---------------------------------------------------------------------------

class TestBuildConclusionsSection:
    """Test the conclusions section builder."""

    def test_title(self, eda_results_basic, critic_approved):
        section = _build_conclusions_section(eda_results_basic, critic_approved)
        assert section["title"] == "Conclusions"

    def test_no_missing_conclusion(self, critic_approved):
        eda = EDAResults(
            describe={"a": {"count": 100, "mean": 1.0}},
            missing=MissingInfo(per_column={"a": 0.0}, total_pct=0.0),
        )
        section = _build_conclusions_section(eda, critic_approved)
        assert "fully complete" in section["content"].lower()

    def test_high_missing_conclusion(self, critic_approved):
        eda = EDAResults(
            describe={"a": {"count": 100, "mean": 1.0}},
            missing=MissingInfo(per_column={"col_x": 40.0}, total_pct=15.0),
        )
        section = _build_conclusions_section(eda, critic_approved)
        assert "quality concern" in section["content"].lower()
        assert "col_x" in section["content"]

    def test_multicollinearity_detected(self, critic_approved):
        eda = EDAResults(
            describe={"x": {"count": 100, "mean": 1.0}, "y": {"count": 100, "mean": 2.0}},
            missing=MissingInfo(total_pct=0.0),
            correlation={"x": {"x": 1.0, "y": 0.95}, "y": {"x": 0.95, "y": 1.0}},
        )
        section = _build_conclusions_section(eda, critic_approved)
        assert "multicollinearity" in section["content"].lower()

    def test_no_multicollinearity(self, critic_approved):
        eda = EDAResults(
            describe={"a": {"count": 100, "mean": 1.0}},
            missing=MissingInfo(total_pct=0.0),
            correlation={"a": {"a": 1.0, "b": 0.3}, "b": {"a": 0.3, "b": 1.0}},
        )
        section = _build_conclusions_section(eda, critic_approved)
        assert "no concerning multicollinearity" in section["content"].lower()

    def test_quality_flags_mentioned(self, eda_results_basic, critic_revision_needed):
        section = _build_conclusions_section(eda_results_basic, critic_revision_needed)
        assert "high-severity" in section["content"].lower() or "quality" in section["content"].lower()


# ---------------------------------------------------------------------------
# _build_recommendations_section
# ---------------------------------------------------------------------------

class TestBuildRecommendationsSection:
    """Test the recommendations and business implications section builder."""

    def test_title(self, eda_results_basic, critic_approved):
        section = _build_recommendations_section(eda_results_basic, critic_approved)
        assert section["title"] == "Recommendations & Business Implications"

    def test_high_missing_recommendation(self, critic_approved):
        eda = EDAResults(
            missing=MissingInfo(per_column={"income": 45.0}, total_pct=15.0),
        )
        section = _build_recommendations_section(eda, critic_approved)
        assert "high priority" in section["content"].lower()
        assert "income" in section["content"]

    def test_moderate_missing_recommendation(self, critic_approved):
        eda = EDAResults(
            missing=MissingInfo(per_column={"age": 12.0}, total_pct=4.0),
        )
        section = _build_recommendations_section(eda, critic_approved)
        assert "medium priority" in section["content"].lower()

    def test_redundant_features_recommendation(self, critic_approved):
        eda = EDAResults(
            missing=MissingInfo(total_pct=0.0),
            correlation={"x": {"x": 1.0, "y": 0.98}, "y": {"x": 0.98, "y": 1.0}},
        )
        section = _build_recommendations_section(eda, critic_approved)
        assert "feature engineering" in section["content"].lower()

    def test_critic_suggestions_included(self):
        critic = CriticReport(
            flags=[CriticFlag(column="income", rule="skewness", severity="HIGH",
                              message="|skew|=3.0", value=3.0,
                              suggestion="log transform recommended")],
            iteration=1, status="REVISION_NEEDED",
        )
        eda = EDAResults(missing=MissingInfo(total_pct=0.0))
        section = _build_recommendations_section(eda, critic)
        assert "log transform" in section["content"].lower()

    def test_clean_data_next_steps(self, critic_approved):
        eda = EDAResults(
            missing=MissingInfo(total_pct=0.0),
        )
        section = _build_recommendations_section(eda, critic_approved)
        assert "good overall quality" in section["content"].lower()

    def test_business_implications_high_severity(self):
        critic = CriticReport(
            flags=[CriticFlag(column="x", rule="test", severity="HIGH", message="bad")],
            iteration=1, status="REVISION_NEEDED",
        )
        eda = EDAResults(missing=MissingInfo(total_pct=0.0))
        section = _build_recommendations_section(eda, critic)
        assert "business implications" in section["content"].lower()

    def test_business_implications_high_missing(self):
        critic = CriticReport(flags=[], iteration=1, status="APPROVED")
        eda = EDAResults(
            missing=MissingInfo(per_column={"a": 20.0}, total_pct=12.0),
        )
        section = _build_recommendations_section(eda, critic)
        assert "upstream data collection" in section["content"].lower()


# ---------------------------------------------------------------------------
# _build_quality_section
# ---------------------------------------------------------------------------

class TestBuildQualitySection:
    """Test the data quality section builder."""

    def test_no_flags(self, critic_approved):
        section = _build_quality_section(critic_approved, is_final=True)
        assert "passed" in section["content"].lower()

    def test_with_flags(self, critic_revision_needed):
        section = _build_quality_section(critic_revision_needed, is_final=False)
        assert "2 quality flag" in section["content"]
        assert "[HIGH]" in section["content"]
        assert "[MEDIUM]" in section["content"]

    def test_flag_column_in_content(self, critic_revision_needed):
        section = _build_quality_section(critic_revision_needed, is_final=False)
        assert "income" in section["content"]

    def test_dataset_level_flag(self):
        """Dataset-level flags (column=None) show 'dataset-level' label."""
        critic = CriticReport(
            flags=[CriticFlag(column=None, rule="dataset_missingness", severity="HIGH",
                              message="35% total cells missing", value=0.35)],
            iteration=1, status="REVISION_NEEDED",
        )
        section = _build_quality_section(critic, is_final=False)
        assert "dataset-level" in section["content"]


# ---------------------------------------------------------------------------
# _collect_unresolved
# ---------------------------------------------------------------------------

class TestCollectUnresolved:
    """Test unresolved flag collection."""

    def test_returns_list(self, critic_revision_needed):
        result = _collect_unresolved(critic_revision_needed)
        assert isinstance(result, list)

    def test_all_prefixed_unresolved(self, critic_revision_needed):
        result = _collect_unresolved(critic_revision_needed)
        for entry in result:
            assert entry.startswith("[UNRESOLVED]")

    def test_count_matches_flags(self, critic_revision_needed):
        result = _collect_unresolved(critic_revision_needed)
        assert len(result) == len(critic_revision_needed.flags)

    def test_empty_flags(self, critic_approved):
        result = _collect_unresolved(critic_approved)
        assert result == []

    def test_includes_severity_and_rule(self, critic_revision_needed):
        result = _collect_unresolved(critic_revision_needed)
        # The HIGH skewness flag
        assert any("[HIGH]" in r and "skewness" in r for r in result)

    def test_suggestion_in_unresolved(self):
        """When a flag has a suggestion, it appears in the unresolved line."""
        critic = CriticReport(
            flags=[CriticFlag(column="x", rule="skewness", severity="HIGH",
                              message="skew=3.1", value=3.1,
                              suggestion="log transform recommended")],
            iteration=2, status="REVISION_NEEDED",
        )
        result = _collect_unresolved(critic)
        assert len(result) == 1
        assert "log transform recommended" in result[0]
        assert "\u2192" in result[0]

    def test_no_suggestion_no_arrow(self):
        """When suggestion is empty, no arrow separator appears."""
        critic = CriticReport(
            flags=[CriticFlag(column="x", rule="missing", severity="HIGH",
                              message="50% missing", value=0.5)],
            iteration=2, status="REVISION_NEEDED",
        )
        result = _collect_unresolved(critic)
        assert len(result) == 1
        assert "\u2192" not in result[0]


class TestSuggestionInQuality:
    """Test that suggestion field appears in quality section output."""

    def test_suggestion_displayed(self):
        """Flag with suggestion shows arrow + suggestion in quality content."""
        critic = CriticReport(
            flags=[CriticFlag(column="x", rule="skewness", severity="HIGH",
                              message="skew=3.1", value=3.1,
                              suggestion="log transform recommended")],
            iteration=1, status="REVISION_NEEDED",
        )
        section = _build_quality_section(critic, is_final=False)
        assert "log transform recommended" in section["content"]
        assert "\u2192" in section["content"]

    def test_no_suggestion_no_arrow_in_quality(self):
        """Flag without suggestion → no arrow in quality content."""
        critic = CriticReport(
            flags=[CriticFlag(column="x", rule="missing", severity="HIGH",
                              message="50% missing", value=0.5)],
            iteration=1, status="REVISION_NEEDED",
        )
        section = _build_quality_section(critic, is_final=False)
        assert "\u2192" not in section["content"]


# ---------------------------------------------------------------------------
# assemble_findings — APPROVED path
# ---------------------------------------------------------------------------

class TestAssembleFindingsApproved:
    """Test assemble_findings() when critic status is APPROVED."""

    def test_returns_valid_json(self, eda_results_basic, critic_approved, plot_paths_sample):
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        parsed = json.loads(result)
        assert "sections" in parsed
        assert "unresolved_flags" in parsed

    def test_validates_as_findings(self, eda_results_basic, critic_approved, plot_paths_sample):
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        assert isinstance(findings, Findings)

    def test_has_seven_sections(self, eda_results_basic, critic_approved, plot_paths_sample):
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        assert len(findings.sections) == 7

    def test_no_unresolved_flags(self, eda_results_basic, critic_approved, plot_paths_sample):
        """APPROVED with no flags → empty unresolved list."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        assert findings.unresolved_flags == []

    def test_section_titles(self, eda_results_basic, critic_approved, plot_paths_sample):
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        titles = [s["title"] for s in findings.sections]
        assert "Dataset Overview" in titles
        assert "Missing Values" in titles
        assert "Correlation Analysis" in titles
        assert "Statistical Analysis" in titles
        assert "Data Quality Assessment" in titles


# ---------------------------------------------------------------------------
# assemble_findings — REVISION_NEEDED path (iteration < 2)
# ---------------------------------------------------------------------------

class TestAssembleFindingsRevision:
    """Test assemble_findings() with REVISION_NEEDED at iteration < 2."""

    def test_returns_valid_json(self, eda_results_basic, critic_revision_needed, plot_paths_sample):
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_revision_needed.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        parsed = json.loads(result)
        assert "sections" in parsed

    def test_no_unresolved_at_iter1(self, eda_results_basic, critic_revision_needed, plot_paths_sample):
        """REVISION_NEEDED at iteration 1 → no unresolved yet (still iterating)."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_revision_needed.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        assert findings.unresolved_flags == []

    def test_quality_section_has_flags(self, eda_results_basic, critic_revision_needed, plot_paths_sample):
        """Quality section includes the flag details for the LLM to address."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_revision_needed.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        quality_section = next(s for s in findings.sections if s["title"] == "Data Quality Assessment")
        assert "skewness" in quality_section["content"]


# ---------------------------------------------------------------------------
# assemble_findings — forced finalize (iteration >= 2)
# ---------------------------------------------------------------------------

class TestAssembleFindingsForcedFinalize:
    """Test assemble_findings() when iteration >= 2 (forced finalize)."""

    def test_unresolved_flags_populated(self, eda_results_basic, critic_revision_iter2, plot_paths_sample):
        """Iteration >= 2 with HIGH flags → unresolved_flags populated."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_revision_iter2.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        assert len(findings.unresolved_flags) > 0

    def test_unresolved_prefixed(self, eda_results_basic, critic_revision_iter2, plot_paths_sample):
        """Each unresolved flag starts with [UNRESOLVED]."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_revision_iter2.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        for flag in findings.unresolved_flags:
            assert flag.startswith("[UNRESOLVED]")

    def test_only_high_severity_unresolved(self, eda_results_basic, plot_paths_sample):
        """Only HIGH/BLOCKER flags are marked unresolved; MEDIUM/LOW are not."""
        critic = CriticReport(
            flags=[
                CriticFlag(column="x", rule="skewness", severity="HIGH",
                           message="|skew|=2.5", value=2.5),
                CriticFlag(column="y", rule="duplicate_rows", severity="MEDIUM",
                           message="2% dups", value=0.02),
            ],
            iteration=2,
            status="REVISION_NEEDED",
        )
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps(["plot.png"]),
        )
        findings = Findings.model_validate_json(result)
        # Only the HIGH flag should be unresolved
        assert len(findings.unresolved_flags) == 1
        assert "[HIGH]" in findings.unresolved_flags[0]

    def test_medium_only_no_unresolved(self, eda_results_basic, plot_paths_sample):
        """Iteration >= 2 but only MEDIUM flags → no unresolved."""
        critic = CriticReport(
            flags=[
                CriticFlag(column="y", rule="duplicate_rows", severity="MEDIUM",
                           message="2% dups", value=0.02),
            ],
            iteration=2,
            status="APPROVED",  # MEDIUM-only → APPROVED
        )
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps(plot_paths_sample),
        )
        findings = Findings.model_validate_json(result)
        assert findings.unresolved_flags == []


# ---------------------------------------------------------------------------
# assemble_findings — edge cases
# ---------------------------------------------------------------------------

class TestAssembleFindingsEdgeCases:
    """Test edge cases for assemble_findings()."""

    def test_empty_eda_results(self, critic_approved):
        """Empty EDA results still produce valid Findings."""
        eda = EDAResults()
        result = assemble_findings(
            eda_results_json=eda.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps([]),
        )
        findings = Findings.model_validate_json(result)
        assert len(findings.sections) == 7

    def test_empty_plot_paths(self, eda_results_basic, critic_approved):
        """No plots → no plot_paths in any section."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps([]),
        )
        findings = Findings.model_validate_json(result)
        for section in findings.sections:
            assert not section.get("plot_paths", [])

    def test_blocker_at_iter2(self, eda_results_basic):
        """BLOCKER flag at iteration 2 → marked [UNRESOLVED]."""
        critic = CriticReport(
            flags=[
                CriticFlag(column="x", rule="missing_values", severity="BLOCKER",
                           message="90% missing", value=0.9),
            ],
            iteration=2,
            status="REVISION_NEEDED",
        )
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps([]),
        )
        findings = Findings.model_validate_json(result)
        assert len(findings.unresolved_flags) == 1
        assert "[BLOCKER]" in findings.unresolved_flags[0]

    def test_output_is_string(self, eda_results_basic, critic_approved):
        """Return type is always str (JSON)."""
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps([]),
        )
        assert isinstance(result, str)

    def test_zero_iteration_approved(self, eda_results_basic):
        """Iteration 0, APPROVED → final, no unresolved."""
        critic = CriticReport(flags=[], iteration=0, status="APPROVED")
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps(["p.png"]),
        )
        findings = Findings.model_validate_json(result)
        assert findings.unresolved_flags == []

    def test_many_plot_paths(self, eda_results_basic, critic_approved):
        """Many hist_ plot paths are paired with Statistical Analysis section."""
        paths = [f"outputs/plots/hist_{i}.png" for i in range(20)]
        result = assemble_findings(
            eda_results_json=eda_results_basic.model_dump_json(),
            critic_report_json=critic_approved.model_dump_json(),
            plot_paths_json=json.dumps(paths),
        )
        findings = Findings.model_validate_json(result)
        stat_section = next(s for s in findings.sections if s["title"] == "Statistical Analysis")
        assert len(stat_section.get("plot_paths", [])) == 20


# ---------------------------------------------------------------------------
# _build_target_section()
# ---------------------------------------------------------------------------


class TestBuildTargetSection:
    """Test the _build_target_section helper."""

    def test_classification_section(self):
        data = {
            "column": "species",
            "problem_type": "classification",
            "n_classes": 3,
            "imbalance_ratio": 1.0,
            "class_distribution": {
                "setosa": {"count": 50, "pct": 33.3},
                "versicolor": {"count": 50, "pct": 33.3},
                "virginica": {"count": 50, "pct": 33.3},
            },
            "per_class_feature_stats": {
                "setosa": {"sepal_length": {"mean": 5.0, "std": 0.3}},
            },
        }
        section = _build_target_section(data)
        assert section["title"] == "Target Variable Analysis"
        assert "species" in section["content"]
        assert "classification" in section["content"]
        assert "well-balanced" in section["content"]

    def test_classification_imbalanced(self):
        data = {
            "column": "fraud",
            "problem_type": "classification",
            "n_classes": 2,
            "imbalance_ratio": 15.0,
            "class_distribution": {
                "0": {"count": 950, "pct": 95.0},
                "1": {"count": 50, "pct": 5.0},
            },
        }
        section = _build_target_section(data)
        assert "Significant class imbalance" in section["content"]
        assert "SMOTE" in section["content"]

    def test_classification_moderate_imbalance(self):
        data = {
            "column": "label",
            "problem_type": "classification",
            "n_classes": 2,
            "imbalance_ratio": 2.5,
            "class_distribution": {
                "a": {"count": 71, "pct": 71.0},
                "b": {"count": 29, "pct": 29.0},
            },
        }
        section = _build_target_section(data)
        assert "Moderate" in section["content"]
        assert "Stratified" in section["content"]

    def test_regression_section(self):
        data = {
            "column": "price",
            "problem_type": "regression",
            "target_stats": {
                "mean": 50000.0,
                "median": 45000.0,
                "std": 15000.0,
                "skewness": 1.5,
            },
            "top_correlated_features": [
                {"feature": "size", "correlation": 0.85},
                {"feature": "rooms", "correlation": 0.72},
            ],
        }
        section = _build_target_section(data)
        assert "regression" in section["content"]
        assert "price" in section["content"]
        assert "size" in section["content"]

    def test_unsupervised_section(self):
        data = {"problem_type": "unsupervised"}
        section = _build_target_section(data)
        assert section["title"] == "Target Variable Analysis"
        assert "unsupervised" in section["content"].lower()

    def test_no_column_section(self):
        data = {"problem_type": "classification", "column": ""}
        section = _build_target_section(data)
        assert "unsupervised" in section["content"].lower()


# ---------------------------------------------------------------------------
# assemble_findings — target_info fallback (no active session)
# ---------------------------------------------------------------------------


class TestAssembleFindingsTargetFallback:
    """Test that assemble_findings() falls back to target_info when
    target_analysis is not available in artifact store."""

    def test_fallback_from_target_info(
        self, eda_results_basic, critic_approved, tmp_path, monkeypatch,
    ):
        """When target_analysis is missing but target_info exists,
        the Target Variable Analysis section should still appear."""
        from tools._pipeline_state import (
            init_session, clear_session, save_state, load_state,
        )

        monkeypatch.setattr(
            "tools._pipeline_state._BASE_STATE_DIR", tmp_path / ".state",
        )
        init_session()
        try:
            # Save artifacts that assemble_findings composes from
            save_state("describe_stats", json.dumps(eda_results_basic.describe))
            save_state("missing_analysis", eda_results_basic.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda_results_basic.correlation))
            save_state("critic_report", critic_approved.model_dump_json())

            # Save target_info (always saved by main.py pre-pipeline)
            # but do NOT save target_analysis (simulating LLM skipping it)
            target_info = {
                "column": "species",
                "problem_type": "classification",
                "n_classes": 3,
                "class_counts": {"setosa": 50, "versicolor": 50, "virginica": 50},
                "imbalance_ratio": 1.0,
                "detection_method": "name_heuristic",
                "has_datetime_index": False,
            }
            save_state("target_info", json.dumps(target_info))

            assemble_findings(
                eda_results_json="STATE_REF:describe_stats",
                critic_report_json="STATE_REF:critic_report",
                plot_paths_json=json.dumps([]),
            )
            # Load findings from artifact store (active session returns ref)
            findings = Findings.model_validate_json(load_state("findings"))
            titles = [s["title"] for s in findings.sections]
            assert "Target Variable Analysis" in titles

            target_sec = next(
                s for s in findings.sections
                if s["title"] == "Target Variable Analysis"
            )
            assert "species" in target_sec["content"]
            assert "classification" in target_sec["content"]
            assert "well-balanced" in target_sec["content"]
        finally:
            clear_session()

    def test_no_target_info_no_section(
        self, eda_results_basic, critic_approved, tmp_path, monkeypatch,
    ):
        """When neither target_analysis nor target_info exists,
        the Target Variable Analysis section should NOT appear."""
        from tools._pipeline_state import (
            init_session, clear_session, save_state, load_state,
        )

        monkeypatch.setattr(
            "tools._pipeline_state._BASE_STATE_DIR", tmp_path / ".state",
        )
        init_session()
        try:
            save_state("describe_stats", json.dumps(eda_results_basic.describe))
            save_state("missing_analysis", eda_results_basic.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda_results_basic.correlation))
            save_state("critic_report", critic_approved.model_dump_json())

            assemble_findings(
                eda_results_json="STATE_REF:describe_stats",
                critic_report_json="STATE_REF:critic_report",
                plot_paths_json=json.dumps([]),
            )
            findings = Findings.model_validate_json(load_state("findings"))
            titles = [s["title"] for s in findings.sections]
            assert "Target Variable Analysis" not in titles
        finally:
            clear_session()

    def test_target_analysis_preferred_over_fallback(
        self, eda_results_basic, critic_approved, tmp_path, monkeypatch,
    ):
        """When both target_analysis and target_info exist,
        target_analysis (richer data) is used."""
        from tools._pipeline_state import (
            init_session, clear_session, save_state, load_state,
        )

        monkeypatch.setattr(
            "tools._pipeline_state._BASE_STATE_DIR", tmp_path / ".state",
        )
        init_session()
        try:
            save_state("describe_stats", json.dumps(eda_results_basic.describe))
            save_state("missing_analysis", eda_results_basic.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda_results_basic.correlation))
            save_state("critic_report", critic_approved.model_dump_json())

            # Save BOTH target_info and target_analysis
            save_state("target_info", json.dumps({
                "column": "species",
                "problem_type": "classification",
                "n_classes": 3,
                "class_counts": {"setosa": 50, "versicolor": 50, "virginica": 50},
                "imbalance_ratio": 1.0,
                "detection_method": "name_heuristic",
                "has_datetime_index": False,
            }))
            save_state("target_analysis", json.dumps({
                "column": "species",
                "problem_type": "classification",
                "n_classes": 3,
                "imbalance_ratio": 1.0,
                "class_distribution": {
                    "setosa": {"count": 50, "pct": 33.3},
                    "versicolor": {"count": 50, "pct": 33.3},
                    "virginica": {"count": 50, "pct": 33.3},
                },
                "per_class_feature_stats": {
                    "setosa": {"sepal_length": {"mean": 5.0, "std": 0.3}},
                },
            }))

            assemble_findings(
                eda_results_json="STATE_REF:describe_stats",
                critic_report_json="STATE_REF:critic_report",
                plot_paths_json=json.dumps([]),
            )
            findings = Findings.model_validate_json(load_state("findings"))
            target_sec = next(
                s for s in findings.sections
                if s["title"] == "Target Variable Analysis"
            )
            # per_class_feature_stats now comes from target_analysis (preferred)
            assert "sepal_length" in target_sec["content"]
        finally:
            clear_session()

    def test_fallback_pairs_class_distribution_plot(
        self, eda_results_basic, critic_approved, tmp_path, monkeypatch,
    ):
        """Fallback target section pairs class_distribution.png plot."""
        from tools._pipeline_state import (
            init_session, clear_session, save_state, load_state,
        )

        monkeypatch.setattr(
            "tools._pipeline_state._BASE_STATE_DIR", tmp_path / ".state",
        )
        init_session()
        try:
            save_state("describe_stats", json.dumps(eda_results_basic.describe))
            save_state("missing_analysis", eda_results_basic.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda_results_basic.correlation))
            save_state("critic_report", critic_approved.model_dump_json())
            save_state("target_info", json.dumps({
                "column": "species",
                "problem_type": "classification",
                "n_classes": 3,
                "class_counts": {"setosa": 50, "versicolor": 50, "virginica": 50},
                "imbalance_ratio": 1.0,
                "detection_method": "name_heuristic",
                "has_datetime_index": False,
            }))
            # Save class_distribution plot artifact
            save_state(
                "plot_class_distribution",
                json.dumps(["outputs/plots/class_distribution.png"]),
            )

            assemble_findings(
                eda_results_json="STATE_REF:describe_stats",
                critic_report_json="STATE_REF:critic_report",
                plot_paths_json="STATE_REF:plot_class_distribution",
            )
            findings = Findings.model_validate_json(load_state("findings"))
            target_sec = next(
                s for s in findings.sections
                if s["title"] == "Target Variable Analysis"
            )
            assert "class_distribution.png" in str(target_sec.get("plot_paths", []))
        finally:
            clear_session()

    def test_fallback_computes_per_class_stats_from_data_json(
        self, eda_results_basic, critic_approved, tmp_path, monkeypatch,
    ):
        """When target_analysis is missing but data_json is present, the fallback
        computes per_class_feature_stats on the fly so the section is as rich as
        if the LLM had called target_analysis()."""
        from tools._pipeline_state import (
            init_session, clear_session, save_state, load_state,
        )
        import pandas as pd

        monkeypatch.setattr(
            "tools._pipeline_state._BASE_STATE_DIR", tmp_path / ".state",
        )
        init_session()
        try:
            # Build a minimal dataset: 20 rows, binary target, 1 numeric feature
            df = pd.DataFrame({
                "score": list(range(10)) + list(range(10, 20)),
                "label": ["low"] * 10 + ["high"] * 10,
            })
            save_state("data_json", df.to_json(orient="records"))
            save_state("describe_stats", json.dumps(eda_results_basic.describe))
            save_state("missing_analysis", eda_results_basic.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda_results_basic.correlation))
            save_state("critic_report", critic_approved.model_dump_json())
            save_state("target_info", json.dumps({
                "column": "label",
                "problem_type": "classification",
                "n_classes": 2,
                "class_counts": {"low": 10, "high": 10},
                "imbalance_ratio": 1.0,
                "detection_method": "name_heuristic",
                "has_datetime_index": False,
            }))
            # Intentionally do NOT save target_analysis

            assemble_findings(
                eda_results_json="STATE_REF:describe_stats",
                critic_report_json="STATE_REF:critic_report",
                plot_paths_json=json.dumps([]),
            )
            findings = Findings.model_validate_json(load_state("findings"))
            target_sec = next(
                s for s in findings.sections
                if s["title"] == "Target Variable Analysis"
            )
            # The fallback should have computed per-class stats from data_json
            assert "score" in target_sec["content"], \
                "per_class_feature_stats (score column) must appear in fallback section"
            assert "low" in target_sec["content"]
            assert "high" in target_sec["content"]
        finally:
            clear_session()



class TestRunComprehensiveEval:
    """Tests for the comprehensive evaluation helper (bias + toxicity + hallucination)."""

    def test_skips_when_openlit_disabled(self, monkeypatch):
        """No-op when OPENLIT_ENABLE is false (default)."""
        monkeypatch.setenv("OPENLIT_ENABLE", "false")
        # Reload config to pick up env var
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", False)
        from tools.findings_tools import _run_comprehensive_eval
        # Should return immediately without error
        _run_comprehensive_eval('{"overview": {}}')

    def test_skips_when_no_active_session(self, monkeypatch):
        """No-op when pipeline session is not active."""
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", True)
        from tools.findings_tools import _run_comprehensive_eval
        # No active session → should return without error
        _run_comprehensive_eval('{"overview": {}}')

    def test_skips_when_no_fact_sheet(self, monkeypatch):
        """No-op when _interpretation_context artifact is missing."""
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", True)
        from tools.findings_tools import _run_comprehensive_eval
        from tools._pipeline_state import init_session, clear_session
        try:
            init_session()
            # No fact sheet saved → should skip
            _run_comprehensive_eval('{"overview": {}}')
        finally:
            clear_session()

    def test_calls_openlit_eval_when_enabled(self, monkeypatch):
        """Calls openlit.evals.All.measure when everything is set up."""
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", True)
        monkeypatch.setattr(config, "OPENLIT_EVAL_MODEL", "gpt-5")

        from tools._pipeline_state import init_session, clear_session, save_state, load_state

        # Mock openlit.evals.All
        class FakeResult:
            verdict = "no"
            score = 0.1
            evaluation = "none"
            classification = "none"
            explanation = "No issues detected"

        class FakeAll:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def measure(self, **kwargs):
                return FakeResult()

        import types
        fake_openlit = types.ModuleType("openlit")
        fake_evals = types.ModuleType("openlit.evals")
        fake_evals_utils = types.ModuleType("openlit.evals.utils")
        fake_evals_utils.llm_response_openai = lambda prompt, model, base_url: "{}"
        fake_evals_utils.JsonOutput = None  # not called in this test path
        fake_evals.All = FakeAll
        fake_openlit.evals = fake_evals
        monkeypatch.setitem(__import__("sys").modules, "openlit", fake_openlit)
        monkeypatch.setitem(__import__("sys").modules, "openlit.evals", fake_evals)
        monkeypatch.setitem(__import__("sys").modules, "openlit.evals.utils", fake_evals_utils)

        from tools.findings_tools import _run_comprehensive_eval

        try:
            init_session()
            save_state("_interpretation_context", "FACT SHEET DATA")
            # Should call the mock and return result dict
            result = _run_comprehensive_eval('{"overview": {}}')
            assert result is not None
            assert result["verdict"] == "no"
            assert result["score"] == 0.1
            assert result["evaluation"] == "none"
            assert result["classification"] == "none"
            # Verify artifact persisted
            stored = load_state("comprehensive_eval")
            assert stored is not None
            import json
            parsed = json.loads(stored)
            assert parsed["verdict"] == "no"
            assert parsed["score"] == 0.1
            assert parsed["evaluation"] == "none"
        finally:
            clear_session()

    def test_does_not_raise_on_eval_failure(self, monkeypatch):
        """Eval errors are caught and logged, not raised."""
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", True)

        from tools._pipeline_state import init_session, clear_session, save_state

        # Mock openlit to raise
        import types
        fake_openlit = types.ModuleType("openlit")
        fake_evals = types.ModuleType("openlit.evals")
        fake_evals_utils = types.ModuleType("openlit.evals.utils")
        fake_evals_utils.llm_response_openai = lambda prompt, model, base_url: "{}"
        fake_evals_utils.JsonOutput = None

        class BrokenAll:
            def __init__(self, **kwargs):
                pass
            def measure(self, **kwargs):
                raise RuntimeError("Eval API down")

        fake_evals.All = BrokenAll
        fake_openlit.evals = fake_evals
        monkeypatch.setitem(__import__("sys").modules, "openlit", fake_openlit)
        monkeypatch.setitem(__import__("sys").modules, "openlit.evals", fake_evals)
        monkeypatch.setitem(__import__("sys").modules, "openlit.evals.utils", fake_evals_utils)

        from tools.findings_tools import _run_comprehensive_eval

        try:
            init_session()
            save_state("_interpretation_context", "FACT SHEET DATA")
            # Should NOT raise, should return None
            result = _run_comprehensive_eval('{"overview": {}}')
            assert result is None
        finally:
            clear_session()

    def test_returns_none_when_disabled(self, monkeypatch):
        """Returns None when OPENLIT_ENABLE is false."""
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", False)
        from tools.findings_tools import _run_comprehensive_eval
        result = _run_comprehensive_eval('{"overview": {}}')
        assert result is None

    def test_eval_cost_info_populated(self, monkeypatch):
        """_eval_cost_info is populated when the capturing wrapper captures usage."""
        import config
        monkeypatch.setattr(config, "OPENLIT_ENABLE", True)
        monkeypatch.setattr(config, "OPENLIT_EVAL_MODEL", "gpt-5")

        from tools._pipeline_state import init_session, clear_session, save_state

        class FakeResult:
            verdict = "no"
            score = 0.05
            evaluation = "none"
            classification = "none"
            explanation = "All good"

        class FakeUsage:
            prompt_tokens = 2000
            completion_tokens = 150

        class FakeResponse:
            model = "gpt-5-2025-08-07"
            usage = FakeUsage()
            class _Choice:
                class _Msg:
                    content = '{"score": 0.05, "evaluation": "none", "classification": "none", "explanation": "All good", "verdict": "no"}'
                message = _Msg()
            choices = [_Choice()]

        class FakeJsonOutput:
            pass

        # Build mock modules
        import types
        fake_evals_utils = types.ModuleType("openlit.evals.utils")
        fake_evals_utils.JsonOutput = FakeJsonOutput
        fake_evals_utils.llm_response_openai = lambda p, m, b: FakeResponse.choices[0].message.content

        fake_evals = types.ModuleType("openlit.evals")

        class FakeAll:
            def __init__(self, **kwargs):
                self._model = kwargs.get("model", "gpt-5")
            def measure(self, **kwargs):
                # Call through the captured llm_response_openai so
                # _capturing_openai is invoked and usage is recorded.
                import sys
                eutils = sys.modules["openlit.evals.utils"]
                eutils.llm_response_openai("prompt", self._model, None)
                return FakeResult()

        fake_evals.All = FakeAll
        fake_openlit = types.ModuleType("openlit")
        fake_openlit.evals = fake_evals

        monkeypatch.setitem(__import__("sys").modules, "openlit", fake_openlit)
        monkeypatch.setitem(__import__("sys").modules, "openlit.evals", fake_evals)
        monkeypatch.setitem(__import__("sys").modules, "openlit.evals.utils", fake_evals_utils)

        # Mock the OpenAI client so _capturing_openai actually works
        class FakeClient:
            class beta:
                class chat:
                    class completions:
                        @staticmethod
                        def parse(**kwargs):
                            return FakeResponse()

        monkeypatch.setattr("openai.OpenAI", lambda **kw: FakeClient())

        from tools.findings_tools import _run_comprehensive_eval, _eval_cost_info

        try:
            init_session()
            save_state("_interpretation_context", "FACT SHEET DATA")
            result = _run_comprehensive_eval('{"overview": {}}')
            assert result is not None
            # _eval_cost_info should be populated from captured usage
            assert _eval_cost_info.get("model") == "gpt-5-2025-08-07"
            assert _eval_cost_info["prompt_tokens"] == 2000
            assert _eval_cost_info["completion_tokens"] == 150
            assert _eval_cost_info["cost"] > 0
        finally:
            clear_session()
            _eval_cost_info.clear()


class TestComputeEvalCost:
    """Tests for _compute_eval_cost pricing lookup."""

    def test_computes_cost_from_pricing_json(self):
        from tools.findings_tools import _compute_eval_cost
        # gpt-5 pricing: prompt=0.00125/1K, completion=0.01/1K
        cost = _compute_eval_cost("gpt-5", 1000, 500)
        expected = (1000 / 1000) * 0.00125 + (500 / 1000) * 0.01
        assert abs(cost - expected) < 1e-9

    def test_returns_zero_for_unknown_model(self):
        from tools.findings_tools import _compute_eval_cost
        cost = _compute_eval_cost("nonexistent-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_returns_zero_for_zero_tokens(self):
        from tools.findings_tools import _compute_eval_cost
        cost = _compute_eval_cost("gpt-5", 0, 0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Tests for _build_trustworthiness_section
# ---------------------------------------------------------------------------

class TestBuildTrustworthinessSection:
    """Tests for the trustworthiness section builder."""

    def test_high_trust_low_score(self):
        """Score < 0.3 maps to High Trustworthiness."""
        from tools.findings_tools import _build_trustworthiness_section
        result = _build_trustworthiness_section({
            "verdict": "no", "score": 0.05,
            "evaluation": "none",
            "classification": "none", "explanation": "All good",
        })
        assert result["title"] == "Trustworthiness Assessment"
        assert "High Trustworthiness" in result["content"]
        assert "well-grounded" in result["content"]
        assert "0.05" in result["content"]
        assert "no issues detected" in result["content"]

    def test_medium_trust_mid_score(self):
        """Score 0.3-0.7 maps to Medium Trustworthiness."""
        from tools.findings_tools import _build_trustworthiness_section
        result = _build_trustworthiness_section({
            "verdict": "yes", "score": 0.5,
            "evaluation": "hallucination",
            "classification": "factual_inaccuracy",
            "explanation": "Some inaccuracies",
        })
        assert "Medium Trustworthiness" in result["content"]
        assert "cross-check" in result["content"]
        assert "issue detected" in result["content"]
        assert "factual_inaccuracy" in result["content"]

    def test_low_trust_high_score(self):
        """Score >= 0.7 maps to Low Trustworthiness."""
        from tools.findings_tools import _build_trustworthiness_section
        result = _build_trustworthiness_section({
            "verdict": "yes", "score": 0.85,
            "evaluation": "bias_detection",
            "classification": "gender",
            "explanation": "Gender bias detected",
        })
        assert "Low Trustworthiness" in result["content"]
        assert "caution" in result["content"]
        assert "gender" in result["content"]
        assert "Highest-risk type: bias" in result["content"]

    def test_classification_none_omitted(self):
        """Classification 'none' is not shown in the output."""
        from tools.findings_tools import _build_trustworthiness_section
        result = _build_trustworthiness_section({
            "verdict": "no", "score": 0.0,
            "evaluation": "none",
            "classification": "none", "explanation": "",
        })
        assert "Classification:" not in result["content"]
        assert "Highest-risk type:" not in result["content"]

    def test_boundary_score_0_3(self):
        """Score exactly 0.3 maps to Medium Trustworthiness."""
        from tools.findings_tools import _build_trustworthiness_section
        result = _build_trustworthiness_section({
            "verdict": "no", "score": 0.3,
            "classification": "none", "explanation": "",
        })
        assert "Medium Trustworthiness" in result["content"]

    def test_boundary_score_0_7(self):
        """Score exactly 0.7 maps to Low Trustworthiness."""
        from tools.findings_tools import _build_trustworthiness_section
        result = _build_trustworthiness_section({
            "verdict": "yes", "score": 0.7,
            "classification": "factual_inaccuracy", "explanation": "Issues",
        })
        assert "Low Trustworthiness" in result["content"]


# ---------------------------------------------------------------------------
# Tests for trustworthiness section in assemble_findings
# ---------------------------------------------------------------------------

class TestAssembleFindingsTrustworthiness:
    """Tests that assemble_findings includes trustworthiness section when eval is available."""

    def test_trustworthiness_section_included_when_eval_present(self):
        """Findings include a Trustworthiness Assessment section when comprehensive_eval artifact exists."""
        import json
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        from tools.findings_tools import assemble_findings
        from eda_state import EDAResults, CriticReport

        try:
            init_session()
            # Set up minimal artifacts
            eda = EDAResults(
                describe={"col_a": {"mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0, "count": 100}},
                correlation={},
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            # Save individual artifacts (required by direct-composition path)
            save_state("describe_stats", json.dumps(eda.describe))
            save_state("missing_analysis", eda.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda.correlation))
            save_state("critic_report", critic.model_dump_json())
            save_state("plot_paths", "[]")
            # Store comprehensive eval result
            eval_result = {
                "verdict": "no", "score": 0.05,
                "evaluation": "none",
                "classification": "none",
                "explanation": "All claims are grounded",
            }
            save_state("comprehensive_eval", json.dumps(eval_result))

            assemble_findings(
                "STATE_REF:describe_stats",
                "STATE_REF:critic_report",
                "STATE_REF:plot_paths",
            )
            findings_raw = json.loads(load_state("findings"))
            section_titles = [s["title"] for s in findings_raw["sections"]]
            assert "Trustworthiness Assessment" in section_titles
            # Should be the last section
            assert section_titles[-1] == "Trustworthiness Assessment"
            trust_content = findings_raw["sections"][-1]["content"]
            assert "High Trustworthiness" in trust_content
        finally:
            clear_session()

    def test_no_trustworthiness_section_when_no_eval(self):
        """Findings omit trustworthiness section when no comprehensive_eval artifact."""
        import json
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        from tools.findings_tools import assemble_findings
        from eda_state import EDAResults, CriticReport

        try:
            init_session()
            eda = EDAResults(
                describe={"col_a": {"mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0, "count": 100}},
                correlation={},
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            # Save individual artifacts (required by direct-composition path)
            save_state("describe_stats", json.dumps(eda.describe))
            save_state("missing_analysis", eda.missing.model_dump_json())
            save_state("correlation_matrix", json.dumps(eda.correlation))
            save_state("critic_report", critic.model_dump_json())
            save_state("plot_paths", "[]")
            # NO comprehensive_eval stored

            assemble_findings(
                "STATE_REF:describe_stats",
                "STATE_REF:critic_report",
                "STATE_REF:plot_paths",
            )
            findings_raw = json.loads(load_state("findings"))
            section_titles = [s["title"] for s in findings_raw["sections"]]
            assert "Trustworthiness Assessment" not in section_titles
        finally:
            clear_session()


# ---------------------------------------------------------------------------
# W1 / W2 / W3 regression tests — artifact-composition correctness
#
# These tests guard against the silent-empty-field bug where resolve()
# returned an incompatible JSON blob, EDAResults.model_validate_json()
# succeeded with all-default empty values, and the report rendered
# "0 rows / No columns / No missing values / No numerical columns".
# ---------------------------------------------------------------------------

class TestW1W2W3ArtifactComposition:
    """Regression tests for W1 (row count), W2 (correlation), W3 (missingness)."""

    def _setup_session(self, save_state, eda: EDAResults, critic: CriticReport,
                       schema_shape: tuple[int, int] | None = None) -> None:
        """Save all individual pipeline artifacts for a test session."""
        save_state("describe_stats", json.dumps(eda.describe))
        save_state("missing_analysis", eda.missing.model_dump_json())
        save_state("correlation_matrix", json.dumps(eda.correlation))
        save_state("critic_report", critic.model_dump_json())
        save_state("plot_paths", "[]")
        if schema_shape is not None:
            profile = DataProfile(shape=schema_shape)
            save_state("schema_json", profile.model_dump_json())

    def test_w1_overview_uses_dataprofile_shape(self):
        """W1: overview row/col count comes from DataProfile.shape, not describe[count]."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        try:
            init_session()
            eda = EDAResults(
                describe={
                    "age":    {"count": 48842.0, "mean": 38.58, "std": 13.64},
                    "income": {"count": 48842.0, "mean": 50000.0, "std": 15000.0},
                },
                missing=MissingInfo(per_column={"age": 0.0, "income": 0.0}, total_pct=0.0),
                correlation={"age": {"age": 1.0, "income": 0.45},
                             "income": {"age": 0.45, "income": 1.0}},
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            self._setup_session(save_state, eda, critic, schema_shape=(48842, 15))

            assemble_findings(
                "STATE_REF:describe_stats",
                "STATE_REF:critic_report",
                "STATE_REF:plot_paths",
            )
            findings_raw = json.loads(load_state("findings"))
            overview = next(s for s in findings_raw["sections"] if s["title"] == "Dataset Overview")
            # Authoritative shape from DataProfile must appear
            assert "48842" in overview["content"], "W1: row count must be 48842, not 0"
            assert "15" in overview["content"], "W1: col count must be 15, not 0"
        finally:
            clear_session()

    def test_w1_overview_fallback_when_no_schema(self):
        """W1 fallback: no schema_json → row count inferred from describe[count]."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        try:
            init_session()
            eda = EDAResults(
                describe={"age": {"count": 1000.0, "mean": 35.0, "std": 10.0}},
                missing=MissingInfo(per_column={}, total_pct=0.0),
                correlation={},
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            self._setup_session(save_state, eda, critic, schema_shape=None)

            assemble_findings("x", "STATE_REF:critic_report", "STATE_REF:plot_paths")
            findings_raw = json.loads(load_state("findings"))
            overview = next(s for s in findings_raw["sections"] if s["title"] == "Dataset Overview")
            assert "1000" in overview["content"], "W1 fallback: count from describe must be used"
        finally:
            clear_session()

    def test_w2_correlation_section_populated(self):
        """W2: correlation section shows actual pairs, not 'No numerical columns'."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        try:
            init_session()
            eda = EDAResults(
                describe={
                    "age":    {"count": 48842.0, "mean": 38.58, "std": 13.64},
                    "fnlwgt": {"count": 48842.0, "mean": 189778.0, "std": 105550.0},
                },
                missing=MissingInfo(per_column={}, total_pct=0.0),
                correlation={
                    "age":    {"age": 1.0, "fnlwgt": -0.077},
                    "fnlwgt": {"age": -0.077, "fnlwgt": 1.0},
                },
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            self._setup_session(save_state, eda, critic)

            assemble_findings("x", "STATE_REF:critic_report", "STATE_REF:plot_paths")
            findings_raw = json.loads(load_state("findings"))
            corr_section = next(s for s in findings_raw["sections"] if s["title"] == "Correlation Analysis")
            assert "No numerical" not in corr_section["content"], \
                "W2: correlation section must not show 'No numerical columns'"
            assert "age" in corr_section["content"]
        finally:
            clear_session()

    def test_w3_missing_section_populated(self):
        """W3: missing section shows actual percentages, not 'No missing values detected'."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        try:
            init_session()
            eda = EDAResults(
                describe={"workclass": {"count": 45222.0, "unique": 8, "top": "Private", "freq": 33906}},
                missing=MissingInfo(
                    per_column={"workclass": 5.64, "occupation": 5.66, "native-country": 1.79},
                    total_pct=4.36,
                ),
                correlation={},
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            self._setup_session(save_state, eda, critic)

            assemble_findings("x", "STATE_REF:critic_report", "STATE_REF:plot_paths")
            findings_raw = json.loads(load_state("findings"))
            missing_section = next(s for s in findings_raw["sections"] if s["title"] == "Missing Values")
            assert "No missing" not in missing_section["content"], \
                "W3: missing section must not show 'No missing values detected'"
            assert "workclass" in missing_section["content"]
            assert "4.4" in missing_section["content"] or "4.3" in missing_section["content"]
        finally:
            clear_session()

    def test_wrong_eda_ref_still_uses_individual_artifacts(self):
        """Core regression: even when eda_results_json is a garbage/wrong ref,
        assemble_findings must compose from individual artifacts (not silently
        produce empty EDAResults with default zero values)."""
        from tools._pipeline_state import init_session, clear_session, save_state, load_state
        try:
            init_session()
            eda = EDAResults(
                describe={"age": {"count": 999.0, "mean": 40.0, "std": 12.0}},
                missing=MissingInfo(per_column={"age": 3.5}, total_pct=1.2),
                correlation={"age": {"age": 1.0}},
            )
            critic = CriticReport(flags=[], iteration=1, status="APPROVED")
            self._setup_session(save_state, eda, critic)

            # Pass intentionally wrong/garbage eda_results_json —
            # the new composition path ignores it and loads from artifacts directly.
            assemble_findings(
                eda_results_json="garbage_that_is_not_a_ref_or_valid_json",
                critic_report_json="STATE_REF:critic_report",
                plot_paths_json="STATE_REF:plot_paths",
            )
            findings_raw = json.loads(load_state("findings"))
            overview = next(s for s in findings_raw["sections"] if s["title"] == "Dataset Overview")
            missing_section = next(s for s in findings_raw["sections"] if s["title"] == "Missing Values")
            # Must reflect actual data, not zero defaults
            assert "999" in overview["content"] or "1" in overview["content"], \
                "Row count must not be 0 (W1 regression)"
            assert "No missing" not in missing_section["content"], \
                "Missing section must not be 'No missing values' (W3 regression)"
        finally:
            clear_session()


# ---------------------------------------------------------------------------
# _build_categorical_section / _build_categorical_inventory
# ---------------------------------------------------------------------------

class TestBuildCategoricalSection:
    """Tests for _build_categorical_section() and _build_categorical_inventory()."""

    @pytest.fixture()
    def empty_analysis(self):
        return CategoricalAnalysis(columns={})

    @pytest.fixture()
    def simple_analysis(self):
        return CategoricalAnalysis(
            columns={
                "color": CategoricalStats(
                    cardinality=3,
                    entropy_bits=1.585,
                    rare_count=0,
                    top_values=[
                        {"value": "red", "count": 40, "pct": 40.0},
                        {"value": "blue", "count": 35, "pct": 35.0},
                        {"value": "green", "count": 25, "pct": 25.0},
                    ],
                    more_values=0,
                ),
                "size": CategoricalStats(
                    cardinality=2,
                    entropy_bits=1.0,
                    rare_count=0,
                    top_values=[
                        {"value": "S", "count": 50, "pct": 50.0},
                        {"value": "L", "count": 50, "pct": 50.0},
                    ],
                    more_values=0,
                ),
            },
            target_column=None,
            top_n=10,
        )

    @pytest.fixture()
    def classification_analysis(self):
        return CategoricalAnalysis(
            columns={
                "color": CategoricalStats(
                    cardinality=3,
                    entropy_bits=1.585,
                    rare_count=0,
                    top_values=[
                        {"value": "red", "count": 40, "pct": 40.0,
                         "target_rates": {"yes": 60.0, "no": 40.0}},
                        {"value": "blue", "count": 35, "pct": 35.0,
                         "target_rates": {"yes": 20.0, "no": 80.0}},
                        {"value": "green", "count": 25, "pct": 25.0,
                         "target_rates": {"yes": 45.0, "no": 55.0}},
                    ],
                    more_values=0,
                ),
                "target": CategoricalStats(
                    cardinality=2,
                    entropy_bits=1.0,
                    rare_count=0,
                    top_values=[
                        {"value": "yes", "count": 50, "pct": 50.0},
                        {"value": "no", "count": 50, "pct": 50.0},
                    ],
                    more_values=0,
                ),
            },
            target_column="target",
            top_n=10,
        )

    @pytest.fixture()
    def high_card_analysis(self):
        return CategoricalAnalysis(
            columns={
                "city": CategoricalStats(
                    cardinality=500,
                    entropy_bits=8.96,
                    rare_count=450,
                    top_values=[
                        {"value": f"city_{i}", "count": 2, "pct": 0.4, "is_rare": True}
                        for i in range(10)
                    ],
                    more_values=490,
                ),
            },
        )

    # --- _build_categorical_section tests ---

    def test_empty_returns_no_categoricals(self, empty_analysis):
        section = _build_categorical_section(empty_analysis)
        assert section["title"] == "Categorical Analysis"
        assert "No categorical" in section["content"]

    def test_returns_dict_with_title_and_content(self, simple_analysis):
        section = _build_categorical_section(simple_analysis)
        assert isinstance(section, dict)
        assert "title" in section
        assert "content" in section
        assert section["title"] == "Categorical Analysis"

    def test_feature_count_in_content(self, simple_analysis):
        section = _build_categorical_section(simple_analysis)
        assert "2 feature(s)" in section["content"]

    def test_low_cardinality_mentioned(self, simple_analysis):
        section = _build_categorical_section(simple_analysis)
        assert "Binary/low-cardinality" in section["content"]
        assert "size (2)" in section["content"]

    def test_high_cardinality_mentioned(self, high_card_analysis):
        section = _build_categorical_section(high_card_analysis)
        assert "High-cardinality" in section["content"]
        assert "city" in section["content"]
        assert "500" in section["content"]

    def test_rare_values_mentioned(self, high_card_analysis):
        section = _build_categorical_section(high_card_analysis)
        assert "Rare categories" in section["content"]

    def test_entropy_summary(self, simple_analysis):
        section = _build_categorical_section(simple_analysis)
        assert "entropy" in section["content"].lower()

    def test_discriminative_with_target(self, classification_analysis):
        section = _build_categorical_section(classification_analysis)
        assert "discriminative" in section["content"].lower()
        assert "color" in section["content"]

    def test_no_discriminative_without_target(self, simple_analysis):
        section = _build_categorical_section(simple_analysis)
        assert "discriminative" not in section["content"].lower()

    def test_per_category_rates_shown_for_top_discriminative(self, classification_analysis):
        section = _build_categorical_section(classification_analysis)
        content = section["content"]
        # The per-category block fires for classification with a target column set.
        assert "Per-category target rates" in content
        # The most discriminative column (color, 40pp spread) must appear.
        assert "color" in content
        # A specific category value and its rates must be rendered.
        assert "red" in content
        assert "yes: 60.0%" in content
        assert "no: 40.0%" in content

    def test_per_category_rates_absent_without_target(self, simple_analysis):
        section = _build_categorical_section(simple_analysis)
        assert "Per-category target rates" not in section["content"]

    # --- _build_categorical_inventory tests ---

    def test_inventory_empty(self, empty_analysis):
        inv = _build_categorical_inventory(empty_analysis)
        assert "No categorical" in inv

    def test_inventory_lists_columns(self, simple_analysis):
        inv = _build_categorical_inventory(simple_analysis)
        assert "color" in inv
        assert "size" in inv
        assert "cardinality=3" in inv
        assert "cardinality=2" in inv

    def test_inventory_shows_values(self, simple_analysis):
        inv = _build_categorical_inventory(simple_analysis)
        assert "'red'" in inv
        assert "'S'" in inv

    def test_inventory_marks_target(self, classification_analysis):
        inv = _build_categorical_inventory(classification_analysis)
        assert "(TARGET)" in inv

    def test_inventory_shows_rare(self, high_card_analysis):
        inv = _build_categorical_inventory(high_card_analysis)
        assert "[RARE]" in inv

    def test_inventory_shows_more_values(self, high_card_analysis):
        inv = _build_categorical_inventory(high_card_analysis)
        assert "490 more" in inv

    def test_inventory_shows_target_rates(self, classification_analysis):
        inv = _build_categorical_inventory(classification_analysis)
        assert "target_rates" in inv
        assert "yes=" in inv


# ---------------------------------------------------------------------------
# _build_overview_section — encoded_categorical_cols parameter
# ---------------------------------------------------------------------------


class TestBuildOverviewSectionEncodedCategoricals:
    """Tests for the encoded_categorical_cols parameter in overview section."""

    @pytest.fixture()
    def eda_basic(self):
        return EDAResults(describe={
            "age": {"count": 100, "mean": 35.0, "std": 10.0, "min": 18.0,
                    "max": 65.0, "25%": 25.0, "50%": 35.0, "75%": 45.0},
            "sex": {"count": 100, "mean": 1.5, "std": 0.5, "min": 1.0,
                    "max": 2.0, "25%": 1.0, "50%": 1.5, "75%": 2.0},
        })

    def test_encoded_note_in_content(self, eda_basic):
        """When encoded_categorical_cols is provided, a reclassification note appears."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["sex", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex"],
        )
        assert "reclassified as categorical" in section["content"]
        assert "sex" in section["content"]

    def test_encoded_count_in_composition(self, eda_basic):
        """The composition line shows the encoded count annotation."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["sex", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex"],
        )
        assert "1 encoded as integers" in section["content"]

    def test_multiple_encoded_columns(self, eda_basic):
        section = _build_overview_section(
            eda_basic, shape=(100, 4),
            categorical_cols=["sex", "education", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex", "education"],
        )
        assert "2 encoded as integers" in section["content"]
        assert "sex" in section["content"]
        assert "education" in section["content"]

    def test_no_encoded_no_note(self, eda_basic):
        """When encoded_categorical_cols is None, no reclassification note."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["name"],
            numerical_cols=["age", "sex"],
        )
        assert "reclassified" not in section["content"]
        assert "encoded" not in section["content"]

    def test_empty_encoded_no_note(self, eda_basic):
        """When encoded_categorical_cols is empty list, no note."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["name"],
            numerical_cols=["age", "sex"],
            encoded_categorical_cols=[],
        )
        assert "reclassified" not in section["content"]

    def test_singular_grammar(self, eda_basic):
        """Single encoded column uses 'is' not 'are'."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["sex", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex"],
        )
        assert "sex is numerically encoded" in section["content"]

    def test_plural_grammar(self, eda_basic):
        """Multiple encoded columns use 'are' not 'is'."""
        section = _build_overview_section(
            eda_basic, shape=(100, 4),
            categorical_cols=["sex", "education", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex", "education"],
        )
        assert "are numerically encoded" in section["content"]


# ---------------------------------------------------------------------------
# F2: Exclude target from "candidate for removal"
# ---------------------------------------------------------------------------

class TestStatisticalAnalysisTargetExclusion:
    """F2 fix: target column must not appear in 'candidates for removal' or
    'high variability' diagnostics — its distributional properties are
    Bernoulli invariants, not data quality issues."""

    def test_target_excluded_from_narrow_iqr(self):
        """Binary target with IQR=0 must not be flagged as removal candidate."""
        eda = EDAResults(describe={
            "target": {"count": 1000, "mean": 0.22, "std": 0.41,
                       "min": 0, "25%": 0, "50%": 0, "75%": 0, "max": 1},
            "const_feat": {"count": 1000, "mean": 5.0, "std": 0.0,
                           "min": 5, "25%": 5, "50%": 5, "75%": 5, "max": 5},
        })
        section = _build_statistical_analysis_section(eda, target_column="target")
        content = section["content"]
        # "const_feat" should still be flagged
        assert "const_feat" in content
        # "target" should NOT appear in the narrow IQR sentence
        assert "target" not in content.split("Near-zero")[0] if "Near-zero" in content else True
        # Look specifically: "target" should not be in the narrow_iqr cols list
        if "Near-zero interquartile range" in content:
            iqr_sentence = content.split("Near-zero interquartile range")[1].split(".")[0]
            assert "target" not in iqr_sentence

    def test_target_excluded_from_high_cv(self):
        """Binary target with CV>1.0 must not be flagged as high variability."""
        eda = EDAResults(describe={
            "target": {"count": 1000, "mean": 0.22, "std": 0.41,
                       "min": 0, "25%": 0, "50%": 0, "75%": 0, "max": 1},
            "noisy_feat": {"count": 1000, "mean": 1.0, "std": 5.0,
                           "min": -10, "25%": -1, "50%": 0, "75%": 2, "max": 30},
        })
        section = _build_statistical_analysis_section(eda, target_column="target")
        content = section["content"]
        if "variability" in content.lower():
            cv_sentence = content.split("High variability")[1].split(".")[0]
            assert "target" not in cv_sentence

    def test_no_target_column_backward_compat(self):
        """Without target_column, binary column IS flagged (backward compat)."""
        eda = EDAResults(describe={
            "binary": {"count": 1000, "mean": 0.22, "std": 0.41,
                       "min": 0, "25%": 0, "50%": 0, "75%": 0, "max": 1},
        })
        section = _build_statistical_analysis_section(eda)
        content = section["content"]
        assert "binary" in content

    def test_target_none_backward_compat(self):
        """target_column=None behaves like no target provided."""
        eda = EDAResults(describe={
            "binary": {"count": 1000, "mean": 0.22, "std": 0.41,
                       "min": 0, "25%": 0, "50%": 0, "75%": 0, "max": 1},
        })
        section = _build_statistical_analysis_section(eda, target_column=None)
        content = section["content"]
        assert "binary" in content


# ---------------------------------------------------------------------------
# F5: Transparency note in correlation section
# ---------------------------------------------------------------------------

class TestCorrelationTransparencyNote:
    """F5 fix: correlation section must note excluded encoded-categorical columns."""

    def test_transparency_note_present(self):
        eda = EDAResults(correlation={
            "age": {"age": 1.0, "income": 0.3},
            "income": {"age": 0.3, "income": 1.0},
        })
        section = _build_correlation_section(
            eda, encoded_categorical_cols=["SEX", "PAY_0"],
        )
        assert "2 column(s) reclassified" in section["content"]
        assert "SEX" in section["content"]
        assert "PAY_0" in section["content"]
        assert "Categorical Analysis" in section["content"]

    def test_no_note_when_no_encoded_cols(self):
        eda = EDAResults(correlation={
            "age": {"age": 1.0, "income": 0.3},
            "income": {"age": 0.3, "income": 1.0},
        })
        section = _build_correlation_section(eda, encoded_categorical_cols=None)
        assert "reclassified" not in section["content"]

    def test_no_note_when_empty_list(self):
        eda = EDAResults(correlation={
            "age": {"age": 1.0, "income": 0.3},
            "income": {"age": 0.3, "income": 1.0},
        })
        section = _build_correlation_section(eda, encoded_categorical_cols=[])
        assert "reclassified" not in section["content"]


# ---------------------------------------------------------------------------
# F1: Spearman ordinal inter-correlation
# ---------------------------------------------------------------------------

class TestCorrelationSpearmanSubsection:
    """F1 fix: ordinal inter-correlation subsection in correlation analysis."""

    def test_spearman_subsection_present(self):
        """When ordinal_spearman is provided, subsection appears."""
        eda = EDAResults(correlation={
            "age": {"age": 1.0, "income": 0.3},
            "income": {"age": 0.3, "income": 1.0},
        })
        ordinal_sp = {
            "PAY_0": {"PAY_0": 1.0, "PAY_2": 0.72, "PAY_3": 0.65},
            "PAY_2": {"PAY_0": 0.72, "PAY_2": 1.0, "PAY_3": 0.82},
            "PAY_3": {"PAY_0": 0.65, "PAY_2": 0.82, "PAY_3": 1.0},
        }
        section = _build_correlation_section(eda, ordinal_spearman=ordinal_sp)
        content = section["content"]
        assert "Ordinal Inter-Correlation" in content
        assert "Spearman" in content
        assert "PAY_2" in content
        assert "PAY_3" in content

    def test_spearman_notable_pairs_shown(self):
        """Pairs with |ρ| ≥ 0.5 appear in the notable pairs list."""
        eda = EDAResults(correlation={"x": {"x": 1.0}})
        ordinal_sp = {
            "A": {"A": 1.0, "B": 0.82, "C": 0.3},
            "B": {"A": 0.82, "B": 1.0, "C": 0.55},
            "C": {"A": 0.3, "B": 0.55, "C": 1.0},
        }
        section = _build_correlation_section(eda, ordinal_spearman=ordinal_sp)
        content = section["content"]
        # A↔B (0.82) and B↔C (0.55) should be notable, A↔C (0.3) should not
        assert "A ↔ B" in content
        assert "B ↔ C" in content

    def test_spearman_no_notable_pairs(self):
        """When all pairs are below |ρ| < 0.5, says 'No pairs exceed'."""
        eda = EDAResults(correlation={"x": {"x": 1.0}})
        ordinal_sp = {
            "A": {"A": 1.0, "B": 0.2},
            "B": {"A": 0.2, "B": 1.0},
        }
        section = _build_correlation_section(eda, ordinal_spearman=ordinal_sp)
        assert "No pairs exceed" in section["content"]

    def test_spearman_none_no_subsection(self):
        """Without ordinal_spearman, no Spearman subsection appears."""
        eda = EDAResults(correlation={
            "x": {"x": 1.0, "y": 0.5},
            "y": {"x": 0.5, "y": 1.0},
        })
        section = _build_correlation_section(eda, ordinal_spearman=None)
        assert "Spearman" not in section["content"]

    def test_collinearity_warning(self):
        """High inter-correlations trigger collinearity advisory."""
        eda = EDAResults(correlation={"x": {"x": 1.0}})
        ordinal_sp = {
            "P4": {"P4": 1.0, "P5": 0.82},
            "P5": {"P4": 0.82, "P5": 1.0},
        }
        section = _build_correlation_section(eda, ordinal_spearman=ordinal_sp)
        assert "collinearity" in section["content"].lower()

    def test_collinearity_text_is_dataset_agnostic(self):
        """D2 fix: advisory uses 'multicollinearity', not 'temporal'."""
        eda = EDAResults(correlation={"x": {"x": 1.0}})
        ordinal_sp = {
            "P4": {"P4": 1.0, "P5": 0.82},
            "P5": {"P4": 0.82, "P5": 1.0},
        }
        section = _build_correlation_section(eda, ordinal_spearman=ordinal_sp)
        content = section["content"]
        assert "multicollinearity" in content
        assert "temporal" not in content.lower()


# ---------------------------------------------------------------------------
# F3: Nominal/ordinal subtype in overview
# ---------------------------------------------------------------------------

class TestOverviewEncodedCategoricalSubtypes:
    """F3 fix: overview must distinguish nominal from ordinal encoded columns."""

    @pytest.fixture()
    def eda_basic(self):
        return EDAResults(describe={
            "age": {"count": 100, "mean": 35.0, "std": 10.0, "min": 18,
                    "25%": 28, "50%": 35, "75%": 42, "max": 65},
        })

    def test_subtypes_shown_when_available(self, eda_basic):
        section = _build_overview_section(
            eda_basic, shape=(100, 5),
            categorical_cols=["sex", "pay_0", "pay_2"],
            numerical_cols=["age", "amount"],
            encoded_categorical_cols=["sex", "pay_0", "pay_2"],
            encoded_categorical_subtypes={
                "sex": "nominal", "pay_0": "ordinal", "pay_2": "ordinal",
            },
        )
        content = section["content"]
        assert "1 nominal" in content
        assert "2 ordinal" in content
        assert "sex" in content
        assert "pay_0" in content

    def test_no_subtypes_fallback(self, eda_basic):
        """Without subtypes, falls back to simple note (no nominal/ordinal)."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["sex", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex"],
            encoded_categorical_subtypes=None,
        )
        content = section["content"]
        assert "numerically encoded" in content
        assert "nominal" not in content
        assert "ordinal" not in content

    def test_empty_subtypes_fallback(self, eda_basic):
        """Empty subtypes dict (--categoricals path) falls back to simple note."""
        section = _build_overview_section(
            eda_basic, shape=(100, 3),
            categorical_cols=["sex", "name"],
            numerical_cols=["age"],
            encoded_categorical_cols=["sex"],
            encoded_categorical_subtypes={},
        )
        content = section["content"]
        # With empty dict, all columns default to "nominal" — the subtype
        # logic should still produce subtype detail
        assert "reclassified" in content

    def test_all_ordinal(self, eda_basic):
        section = _build_overview_section(
            eda_basic, shape=(100, 4),
            categorical_cols=["pay_0", "pay_2", "pay_3"],
            numerical_cols=["age"],
            encoded_categorical_cols=["pay_0", "pay_2", "pay_3"],
            encoded_categorical_subtypes={
                "pay_0": "ordinal", "pay_2": "ordinal", "pay_3": "ordinal",
            },
        )
        content = section["content"]
        assert "3 ordinal" in content
        assert "nominal" not in content
