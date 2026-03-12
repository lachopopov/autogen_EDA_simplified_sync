"""
tests/test_critic_agent.py — Unit tests for agents/critic_agent.py

Tests the agent factory, tool registration, and end-to-end wiring.
These tests use AG2 classes but do NOT make real LLM calls.
"""

import json

import pandas as pd
import pytest
from autogen import AssistantAgent, UserProxyAgent

from agents.critic_agent import (
    CRITIC_SYSTEM_MESSAGE,
    create_critic_agent,
    register_critic_tools,
)
from eda_state import CriticReport


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
def critic_agent():
    """A CriticAgent created via the factory."""
    return create_critic_agent()


@pytest.fixture()
def wired_pair(critic_agent, user_proxy):
    """Agent + proxy with tools registered."""
    register_critic_tools(critic_agent, user_proxy)
    return critic_agent, user_proxy


# ---------------------------------------------------------------------------
# create_critic_agent()
# ---------------------------------------------------------------------------

class TestCreateCriticAgent:
    """Test the CriticAgent factory."""

    def test_name(self, critic_agent):
        assert critic_agent.name == "CriticAgent"

    def test_system_message(self, critic_agent):
        assert "run_critic_rules()" in critic_agent.system_message
        assert "APPROVED" in critic_agent.system_message
        assert "REVISION_NEEDED" in critic_agent.system_message

    def test_system_message_exact(self, critic_agent):
        assert critic_agent.system_message == CRITIC_SYSTEM_MESSAGE

    def test_is_assistant_agent(self, critic_agent):
        assert isinstance(critic_agent, AssistantAgent)

    def test_no_terminate_instruction(self, critic_agent):
        """Non-terminal agents must explicitly forbid TERMINATE to prevent short-circuit."""
        assert "Do NOT include the word TERMINATE" in critic_agent.system_message

    def test_max_consecutive_auto_reply(self, critic_agent):
        assert critic_agent._max_consecutive_auto_reply == 10

    def test_termination_guard(self, critic_agent):
        assert critic_agent._is_termination_msg({"content": "TERMINATE"}) is True
        assert critic_agent._is_termination_msg({"content": "keep going"}) is False


# ---------------------------------------------------------------------------
# register_critic_tools() — AG2 chained-decorator registration
# ---------------------------------------------------------------------------

class TestRegisterCriticTools:
    """Test tool registration on agent + proxy."""

    def test_function_map_has_run_critic_rules(self, wired_pair):
        _, proxy = wired_pair
        assert "run_critic_rules" in proxy._function_map

    def test_function_map_count(self, wired_pair):
        """CriticAgent has exactly 1 tool."""
        _, proxy = wired_pair
        assert len(proxy._function_map) == 1

    def test_llm_config_has_tool_schema(self, wired_pair):
        agent, _ = wired_pair
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 1

    def test_tool_schema_name(self, wired_pair):
        agent, _ = wired_pair
        tool = agent.llm_config["tools"][0]
        assert tool["function"]["name"] == "run_critic_rules"

    def test_tool_schema_has_parameters(self, wired_pair):
        agent, _ = wired_pair
        tool = agent.llm_config["tools"][0]
        params = tool["function"]["parameters"]
        assert "data_json" in params.get("properties", {})


# ---------------------------------------------------------------------------
# End-to-end: tool invocation through proxy function_map
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Test run_critic_rules through the proxy function_map (no LLM)."""

    def test_run_critic_rules_via_proxy(self, wired_pair):
        """Tool is callable through the proxy's function_map."""
        _, proxy = wired_pair
        fn = proxy._function_map["run_critic_rules"]
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = fn(data_json=df.to_json(orient="records"))
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "flags" in parsed
        assert "status" in parsed

    def test_approved_result_via_proxy(self, wired_pair):
        """Clean data → APPROVED through proxy."""
        _, proxy = wired_pair
        fn = proxy._function_map["run_critic_rules"]
        df = pd.DataFrame({
            "a": [1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.8, 8.4, 9.0, 10.6],
            "b": [5, 3, 8, 1, 7, 2, 9, 4, 10, 6],
        })
        result = fn(data_json=df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        assert report.status == "APPROVED"

    def test_revision_needed_via_proxy(self, wired_pair):
        """Data with issues → REVISION_NEEDED through proxy."""
        _, proxy = wired_pair
        fn = proxy._function_map["run_critic_rules"]
        # Zero-variance column → HIGH → REVISION_NEEDED
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5], "y": [1, 2, 3, 4, 5]})
        result = fn(data_json=df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        assert report.status == "REVISION_NEEDED"

    def test_empty_df_via_proxy(self, wired_pair):
        """Empty DataFrame → APPROVED."""
        _, proxy = wired_pair
        fn = proxy._function_map["run_critic_rules"]
        result = fn(data_json="[]")
        report = CriticReport.model_validate_json(result)
        assert report.status == "APPROVED"
        assert len(report.flags) == 0

    def test_output_validates_as_critic_report(self, wired_pair):
        """Output always validates as a CriticReport Pydantic model."""
        _, proxy = wired_pair
        fn = proxy._function_map["run_critic_rules"]
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, 3]})
        result = fn(data_json=df.to_json(orient="records"))
        report = CriticReport.model_validate_json(result)
        assert isinstance(report, CriticReport)
        for flag in report.flags:
            assert flag.severity in ("BLOCKER", "HIGH", "MEDIUM", "LOW")

    def test_chained_registration_invariant(self, wired_pair):
        """Tool appears in BOTH agent LLM tools AND proxy function_map (P6)."""
        agent, proxy = wired_pair
        # Agent side: LLM knows about the tool
        tool_names = {t["function"]["name"] for t in agent.llm_config.get("tools", [])}
        # Proxy side: proxy can execute the tool
        fn_names = set(proxy._function_map.keys())
        # Both must contain run_critic_rules
        assert "run_critic_rules" in tool_names
        assert "run_critic_rules" in fn_names
