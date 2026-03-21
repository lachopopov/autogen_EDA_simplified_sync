"""
tests/test_findings_generator_agent.py — Unit tests for agents/findings_generator_agent.py

Tests the agent factory, tool registration, and end-to-end wiring.
These tests use AG2 classes but do NOT make real LLM calls.
"""

import json

import pytest
from autogen import AssistantAgent, UserProxyAgent

from agents.findings_generator_agent import (
    FINDINGS_GENERATOR_SYSTEM_MESSAGE,
    create_findings_generator_agent,
    register_findings_generator_tools,
)
from eda_state import CriticReport, EDAResults, Findings, MissingInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def user_proxy():
    """A UserProxyAgent configured the same way as architecture.md § 4.1."""
    return UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
        code_execution_config=False,
    )


@pytest.fixture()
def findings_agent():
    """A FindingsGeneratorAgent created via the factory."""
    return create_findings_generator_agent()


@pytest.fixture()
def wired_pair(findings_agent, user_proxy):
    """Agent + proxy with tools registered."""
    register_findings_generator_tools(findings_agent, user_proxy)
    return findings_agent, user_proxy


# ---------------------------------------------------------------------------
# create_findings_generator_agent()
# ---------------------------------------------------------------------------

class TestCreateFindingsGeneratorAgent:
    """Test the FindingsGeneratorAgent factory."""

    def test_name(self, findings_agent):
        assert findings_agent.name == "FindingsGeneratorAgent"

    def test_system_message(self, findings_agent):
        assert "assemble_findings()" in findings_agent.system_message
        assert "prepare_interpretation_context()" in findings_agent.system_message
        assert "save_interpretations()" in findings_agent.system_message

    def test_system_message_exact(self, findings_agent):
        assert findings_agent.system_message == FINDINGS_GENERATOR_SYSTEM_MESSAGE

    def test_is_assistant_agent(self, findings_agent):
        assert isinstance(findings_agent, AssistantAgent)

    def test_no_terminate_instruction(self, findings_agent):
        """Non-terminal agents must explicitly forbid TERMINATE."""
        assert "Do NOT include the word TERMINATE" in findings_agent.system_message

    def test_max_consecutive_auto_reply(self, findings_agent):
        assert findings_agent._max_consecutive_auto_reply == 10

    def test_termination_guard(self, findings_agent):
        assert findings_agent._is_termination_msg({"content": "TERMINATE"}) is True
        assert findings_agent._is_termination_msg({"content": "keep going"}) is False


# ---------------------------------------------------------------------------
# register_findings_generator_tools()
# ---------------------------------------------------------------------------

class TestRegisterFindingsGeneratorTools:
    """Test tool registration on agent + proxy."""

    def test_function_map_has_assemble_findings(self, wired_pair):
        _, proxy = wired_pair
        assert "assemble_findings" in proxy._function_map

    def test_function_map_has_prepare_interpretation_context(self, wired_pair):
        _, proxy = wired_pair
        assert "prepare_interpretation_context" in proxy._function_map

    def test_function_map_has_save_interpretations(self, wired_pair):
        _, proxy = wired_pair
        assert "save_interpretations" in proxy._function_map

    def test_function_map_count(self, wired_pair):
        """FindingsGeneratorAgent has exactly 3 tools."""
        _, proxy = wired_pair
        assert len(proxy._function_map) == 3

    def test_llm_config_has_tool_schema(self, wired_pair):
        agent, _ = wired_pair
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 3

    def test_tool_schema_names(self, wired_pair):
        agent, _ = wired_pair
        names = {t["function"]["name"] for t in agent.llm_config["tools"]}
        assert names == {
            "prepare_interpretation_context",
            "save_interpretations",
            "assemble_findings",
        }

    def test_tool_schema_has_parameters(self, wired_pair):
        agent, _ = wired_pair
        # Find assemble_findings tool
        tool = next(
            t for t in agent.llm_config["tools"]
            if t["function"]["name"] == "assemble_findings"
        )
        params = tool["function"]["parameters"]
        assert "eda_results_json" in params.get("properties", {})
        assert "critic_report_json" in params.get("properties", {})
        assert "plot_paths_json" in params.get("properties", {})


# ---------------------------------------------------------------------------
# End-to-end: tool invocation through proxy function_map
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Test assemble_findings through the proxy function_map (no LLM)."""

    def _make_inputs(self):
        """Create standard test inputs."""
        eda = EDAResults(
            describe={"col_a": {"count": 50.0, "mean": 10.0}},
            missing=MissingInfo(per_column={"col_a": 2.0}, total_pct=2.0),
            correlation={"col_a": {"col_a": 1.0}},
        )
        critic = CriticReport(flags=[], iteration=1, status="APPROVED")
        plots = ["outputs/plots/hist_col_a.png"]
        return eda, critic, plots

    def test_assemble_findings_via_proxy(self, wired_pair):
        """Tool is callable through the proxy's function_map."""
        _, proxy = wired_pair
        fn = proxy._function_map["assemble_findings"]
        eda, critic, plots = self._make_inputs()
        result = fn(
            eda_results_json=eda.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps(plots),
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "sections" in parsed
        assert "unresolved_flags" in parsed

    def test_approved_result_via_proxy(self, wired_pair):
        """APPROVED status → no unresolved flags."""
        _, proxy = wired_pair
        fn = proxy._function_map["assemble_findings"]
        eda, critic, plots = self._make_inputs()
        result = fn(
            eda_results_json=eda.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps(plots),
        )
        findings = Findings.model_validate_json(result)
        assert findings.unresolved_flags == []

    def test_output_validates_as_findings(self, wired_pair):
        """Output always validates as a Findings Pydantic model."""
        _, proxy = wired_pair
        fn = proxy._function_map["assemble_findings"]
        eda, critic, plots = self._make_inputs()
        result = fn(
            eda_results_json=eda.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps(plots),
        )
        findings = Findings.model_validate_json(result)
        assert isinstance(findings, Findings)
        assert len(findings.sections) == 7

    def test_chained_registration_invariant(self, wired_pair):
        """Tool appears in BOTH agent LLM tools AND proxy function_map (P6)."""
        agent, proxy = wired_pair
        tool_names = {t["function"]["name"] for t in agent.llm_config.get("tools", [])}
        fn_names = set(proxy._function_map.keys())
        assert "assemble_findings" in tool_names
        assert "assemble_findings" in fn_names

    def test_empty_inputs_via_proxy(self, wired_pair):
        """Empty inputs → still produces valid Findings."""
        _, proxy = wired_pair
        fn = proxy._function_map["assemble_findings"]
        eda = EDAResults()
        critic = CriticReport(flags=[], iteration=0, status="APPROVED")
        result = fn(
            eda_results_json=eda.model_dump_json(),
            critic_report_json=critic.model_dump_json(),
            plot_paths_json=json.dumps([]),
        )
        findings = Findings.model_validate_json(result)
        assert isinstance(findings, Findings)
