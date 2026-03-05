"""
tests/test_findings_tools.py — Unit tests for tools/findings_tools.py

Tests assemble_findings() with various EDA results, critic reports,
and plot paths. Validates outputs against Pydantic Findings model.
No LLM calls — pure function tests.
"""

import json

import pytest

from eda_state import CriticFlag, CriticReport, EDAResults, Findings, MissingInfo
from tools.findings_tools import (
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
            # per_class_feature_stats only comes from target_analysis, not fallback
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
