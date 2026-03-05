"""
test_eda_state.py — Unit tests for eda_state.py

Tests Pydantic sub-models (tools-layer I/O validation) and the
AG2-native router helper (get_critic_status).
"""

from eda_state import (
    CriticFlag,
    CriticReport,
    DataProfile,
    EDAResults,
    Findings,
    Interpretations,
    MissingInfo,
    TargetInfo,
    get_critic_status,
)


# ===================================================================
# Sub-model tests — validation & serialization
# ===================================================================


class TestTargetInfo:
    """Tests for TargetInfo Pydantic model."""

    def test_defaults(self):
        ti = TargetInfo()
        assert ti.column is None
        assert ti.problem_type == "unsupervised"
        assert ti.n_classes == 0
        assert ti.class_counts == {}
        assert ti.imbalance_ratio == 1.0
        assert ti.detection_method == ""
        assert ti.has_datetime_index is False

    def test_classification(self):
        ti = TargetInfo(
            column="species",
            problem_type="classification",
            n_classes=3,
            class_counts={"setosa": 50, "versicolor": 50, "virginica": 50},
            imbalance_ratio=1.0,
            detection_method="name_heuristic",
        )
        assert ti.column == "species"
        assert ti.problem_type == "classification"
        assert ti.n_classes == 3
        assert sum(ti.class_counts.values()) == 150
        assert ti.imbalance_ratio == 1.0

    def test_regression(self):
        ti = TargetInfo(
            column="price",
            problem_type="regression",
            n_classes=0,
            detection_method="name_heuristic",
        )
        assert ti.column == "price"
        assert ti.problem_type == "regression"
        assert ti.n_classes == 0

    def test_unsupervised(self):
        ti = TargetInfo(
            column=None,
            problem_type="unsupervised",
            detection_method="none",
        )
        assert ti.column is None
        assert ti.detection_method == "none"

    def test_round_trip_json(self):
        ti = TargetInfo(
            column="label",
            problem_type="classification",
            n_classes=2,
            class_counts={"positive": 80, "negative": 20},
            imbalance_ratio=4.0,
            detection_method="user_specified",
            has_datetime_index=True,
        )
        restored = TargetInfo.model_validate_json(ti.model_dump_json())
        assert restored.column == ti.column
        assert restored.problem_type == ti.problem_type
        assert restored.n_classes == ti.n_classes
        assert restored.class_counts == ti.class_counts
        assert restored.imbalance_ratio == ti.imbalance_ratio
        assert restored.detection_method == ti.detection_method
        assert restored.has_datetime_index == ti.has_datetime_index

    def test_detection_methods(self):
        for method in ("name_heuristic", "position_heuristic", "user_specified", "none"):
            ti = TargetInfo(detection_method=method)
            assert ti.detection_method == method

    def test_has_datetime_index(self):
        ti = TargetInfo(has_datetime_index=True)
        assert ti.has_datetime_index is True


class TestDataProfile:
    def test_defaults(self):
        dp = DataProfile()
        assert dp.shape == (0, 0)
        assert dp.memory_mb == 0.0
        assert dp.dtypes == {}
        assert dp.numerical_cols == []
        assert dp.categorical_cols == []

    def test_populated(self):
        dp = DataProfile(
            shape=(100, 5),
            memory_mb=2.5,
            dtypes={"age": "int64", "name": "object"},
            numerical_cols=["age"],
            categorical_cols=["name"],
        )
        assert dp.shape == (100, 5)
        assert dp.memory_mb == 2.5
        assert len(dp.dtypes) == 2
        assert "age" in dp.numerical_cols
        assert "name" in dp.categorical_cols

    def test_round_trip_json(self):
        dp = DataProfile(shape=(50, 3), memory_mb=0.8)
        restored = DataProfile.model_validate_json(dp.model_dump_json())
        assert restored.shape == dp.shape
        assert restored.memory_mb == dp.memory_mb


class TestMissingInfo:
    def test_defaults(self):
        mi = MissingInfo()
        assert mi.per_column == {}
        assert mi.total_pct == 0.0

    def test_populated(self):
        mi = MissingInfo(per_column={"col_a": 0.15, "col_b": 0.55}, total_pct=0.12)
        assert mi.per_column["col_a"] == 0.15
        assert mi.total_pct == 0.12


class TestEDAResults:
    def test_defaults(self):
        er = EDAResults()
        assert er.describe == {}
        assert er.missing.total_pct == 0.0
        assert er.correlation == {}

    def test_nested_missing_info(self):
        er = EDAResults(
            missing=MissingInfo(per_column={"x": 0.3}, total_pct=0.3)
        )
        assert er.missing.per_column["x"] == 0.3

    def test_round_trip_json(self):
        er = EDAResults(
            describe={"col1": {"mean": 5.0}},
            missing=MissingInfo(per_column={"col1": 0.1}, total_pct=0.1),
            correlation={"col1_col2": 0.85},
        )
        restored = EDAResults.model_validate_json(er.model_dump_json())
        assert restored.describe == er.describe
        assert restored.correlation == er.correlation


class TestInterpretations:
    """Tests for Interpretations Pydantic model."""

    def test_defaults(self):
        interp = Interpretations()
        assert interp.overview is None
        assert interp.target_variable_analysis is None
        assert interp.plot_commentaries == []
        assert interp.conclusions == ""

    def test_target_variable_analysis_field(self):
        interp = Interpretations(
            target_variable_analysis={
                "statistical": "3-class classification with balanced distribution.",
                "ds_ml": "Stratified cross-validation recommended.",
                "business": "Equal representation across classes.",
            },
        )
        assert interp.target_variable_analysis is not None
        assert "3-class" in interp.target_variable_analysis["statistical"]

    def test_round_trip_with_target(self):
        interp = Interpretations(
            overview={"statistical": "s", "ds_ml": "d", "business": "b"},
            target_variable_analysis={
                "statistical": "s",
                "ds_ml": "d",
                "business": "b",
            },
        )
        restored = Interpretations.model_validate_json(interp.model_dump_json())
        assert restored.target_variable_analysis == interp.target_variable_analysis
        assert restored.overview == interp.overview


class TestCriticFlag:
    def test_defaults(self):
        cf = CriticFlag()
        assert cf.column is None
        assert cf.rule == ""
        assert cf.severity == ""
        assert cf.suggestion == ""

    def test_column_level_flag(self):
        cf = CriticFlag(
            column="income",
            rule="missing_values",
            severity="HIGH",
            message="40% missing",
            value=0.4,
        )
        assert cf.column == "income"
        assert cf.severity == "HIGH"
        assert cf.value == 0.4

    def test_dataset_level_flag(self):
        cf = CriticFlag(
            column=None,
            rule="near_perfect_correlation",
            severity="HIGH",
            message="|r|=0.98",
            value=0.98,
        )
        assert cf.column is None
        assert cf.rule == "near_perfect_correlation"

    def test_suggestion_field(self):
        """CriticFlag accepts and stores a suggestion string."""
        cf = CriticFlag(
            column="x",
            rule="skewness",
            severity="HIGH",
            message="skew=3.1",
            value=3.1,
            suggestion="log transform recommended",
        )
        assert cf.suggestion == "log transform recommended"

    def test_suggestion_default_empty(self):
        """suggestion defaults to empty string when omitted."""
        cf = CriticFlag(column="x", rule="r", severity="LOW", message="m", value=0.1)
        assert cf.suggestion == ""


class TestCriticReport:
    def test_defaults(self):
        cr = CriticReport()
        assert cr.flags == []
        assert cr.iteration == 0
        assert cr.status == "PENDING"

    def test_revision_needed(self):
        cr = CriticReport(
            flags=[
                CriticFlag(column="x", rule="skewness", severity="HIGH", message="|skew|=3.1", value=3.1)
            ],
            iteration=1,
            status="REVISION_NEEDED",
        )
        assert len(cr.flags) == 1
        assert cr.status == "REVISION_NEEDED"

    def test_approved_no_flags(self):
        cr = CriticReport(flags=[], iteration=2, status="APPROVED")
        assert cr.status == "APPROVED"
        assert cr.iteration == 2

    def test_round_trip_json(self):
        cr = CriticReport(
            flags=[CriticFlag(column="a", rule="r", severity="LOW", message="m", value=0.1)],
            iteration=1,
            status="REVISION_NEEDED",
        )
        restored = CriticReport.model_validate_json(cr.model_dump_json())
        assert restored.flags[0].column == "a"
        assert restored.status == "REVISION_NEEDED"


class TestFindings:
    def test_defaults(self):
        f = Findings()
        assert f.sections == []
        assert f.unresolved_flags == []

    def test_populated(self):
        f = Findings(
            sections=[{"title": "Summary", "content": "All good"}],
            unresolved_flags=["[UNRESOLVED] 40% missing in col_x"],
        )
        assert len(f.sections) == 1
        assert "UNRESOLVED" in f.unresolved_flags[0]


# ===================================================================
# Router helper tests — AG2-native get_critic_status
# ===================================================================


class TestGetCriticStatus:
    """Test the AG2-native router helper that inspects agent names and keywords."""

    def test_empty_messages_returns_defaults(self):
        status, iteration = get_critic_status([])
        assert status == "PENDING"
        assert iteration == 0

    def test_no_critic_messages_returns_defaults(self):
        messages = [
            {"name": "DataPrepAgent", "content": "Data loaded successfully."},
            {"name": "EDAAnalysisAgent", "content": "Stats computed."},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "PENDING"
        assert iteration == 0

    def test_first_iteration_revision_needed(self):
        messages = [
            {"name": "DataPrepAgent", "content": "Data loaded."},
            {"name": "EDAAnalysisAgent", "content": "Stats done."},
            {"name": "CriticAgent", "content": "Issues found. REVISION_NEEDED"},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "REVISION_NEEDED"
        assert iteration == 1

    def test_first_iteration_approved(self):
        messages = [
            {"name": "CriticAgent", "content": "All checks passed. APPROVED"},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "APPROVED"
        assert iteration == 1

    def test_second_iteration_approved(self):
        messages = [
            {"name": "CriticAgent", "content": "Issues found. REVISION_NEEDED"},
            {"name": "FindingsGeneratorAgent", "content": "Revised narrative."},
            {"name": "CriticAgent", "content": "All resolved. APPROVED"},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "APPROVED"
        assert iteration == 2

    def test_second_iteration_still_revision_needed(self):
        """After 2 critic turns with REVISION_NEEDED, router should force termination."""
        messages = [
            {"name": "CriticAgent", "content": "REVISION_NEEDED"},
            {"name": "FindingsGeneratorAgent", "content": "Revised."},
            {"name": "CriticAgent", "content": "Still issues. REVISION_NEEDED"},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "REVISION_NEEDED"
        assert iteration == 2  # router uses iteration >= 2 to force proceed

    def test_critic_without_keyword_returns_pending(self):
        messages = [
            {"name": "CriticAgent", "content": "Running quality checks..."},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "PENDING"
        assert iteration == 1

    def test_keyword_in_non_critic_message_ignored(self):
        """APPROVED/REVISION_NEEDED in other agents' messages must be ignored."""
        messages = [
            {"name": "FindingsGeneratorAgent", "content": "Checking if APPROVED by critic."},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "PENDING"
        assert iteration == 0

    def test_none_content_handled(self):
        messages = [
            {"name": "CriticAgent", "content": None},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "PENDING"
        assert iteration == 1

    def test_missing_content_key_handled(self):
        messages = [
            {"name": "CriticAgent"},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "PENDING"
        assert iteration == 1

    def test_keyword_embedded_in_sentence(self):
        messages = [
            {"name": "CriticAgent", "content": "After review, the result is APPROVED and ready for export."},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "APPROVED"
        assert iteration == 1

    def test_both_keywords_present_approved_wins(self):
        """If both keywords appear (unlikely but defensive), APPROVED is checked first."""
        messages = [
            {"name": "CriticAgent", "content": "Changed from REVISION_NEEDED to APPROVED"},
        ]
        status, iteration = get_critic_status(messages)
        assert status == "APPROVED"
        assert iteration == 1


# ===================================================================
# Verify removed symbols do NOT exist
# ===================================================================


class TestRemovedSymbols:
    """Ensure the non-AG2 monolith and parsers were properly removed."""

    def test_no_eda_state_class(self):
        import eda_state

        assert not hasattr(eda_state, "EDAState"), "EDAState monolith should be removed"

    def test_no_extract_eda_state(self):
        import eda_state

        assert not hasattr(eda_state, "extract_eda_state"), "extract_eda_state should be removed"

    def test_no_try_parse_state(self):
        import eda_state

        assert not hasattr(eda_state, "_try_parse_state"), "_try_parse_state should be removed"

    def test_no_try_parse_critic_report(self):
        import eda_state

        assert not hasattr(eda_state, "_try_parse_critic_report"), (
            "_try_parse_critic_report JSON parser should be removed"
        )
