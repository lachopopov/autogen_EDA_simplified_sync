"""
tests/test_eda_analysis_agent.py — Unit tests for agents/eda_analysis_agent.py

Tests the agent factory, tool registration, and end-to-end wiring.
These tests use AG2 classes but do NOT make real LLM calls.
"""

import json

import pandas as pd
import pytest
from autogen import AssistantAgent, UserProxyAgent

from agents.eda_analysis_agent import (
    EDA_ANALYSIS_SYSTEM_MESSAGE,
    create_eda_analysis_agent,
    register_eda_analysis_tools,
)
from eda_state import EDAResults, MissingInfo


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
def eda_agent():
    """An EDAAnalysisAgent created via the factory."""
    return create_eda_analysis_agent()


@pytest.fixture()
def wired_pair(eda_agent, user_proxy):
    """Agent + proxy with tools registered."""
    register_eda_analysis_tools(eda_agent, user_proxy)
    return eda_agent, user_proxy


@pytest.fixture()
def csv_path(tmp_path):
    """A small CSV file with mixed types and some missing values."""
    df = pd.DataFrame({
        "age": [25, 30, None, 40, 35],
        "salary": [50000, 60000, 70000, None, 55000],
        "dept": ["eng", "sales", "eng", "hr", None],
    })
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# create_eda_analysis_agent()
# ---------------------------------------------------------------------------

class TestCreateEDAAnalysisAgent:
    """Test the EDAAnalysisAgent factory."""

    def test_name(self, eda_agent):
        assert eda_agent.name == "EDAAnalysisAgent"

    def test_system_message(self, eda_agent):
        assert "describe_stats()" in eda_agent.system_message
        assert "missing_analysis()" in eda_agent.system_message
        assert "correlation_matrix()" in eda_agent.system_message

    def test_system_message_exact(self, eda_agent):
        assert eda_agent.system_message == EDA_ANALYSIS_SYSTEM_MESSAGE

    def test_is_assistant_agent(self, eda_agent):
        assert isinstance(eda_agent, AssistantAgent)

    def test_no_terminate_instruction(self, eda_agent):
        """Non-terminal agents must explicitly forbid TERMINATE to prevent short-circuit."""
        assert "Do NOT include the word TERMINATE" in eda_agent.system_message

    def test_max_consecutive_auto_reply(self, eda_agent):
        assert eda_agent._max_consecutive_auto_reply == 5

    def test_termination_guard(self, eda_agent):
        assert eda_agent._is_termination_msg({"content": "TERMINATE"}) is True
        assert eda_agent._is_termination_msg({"content": "keep going"}) is False


# ---------------------------------------------------------------------------
# register_eda_analysis_tools() — AG2 chained-decorator registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Test that tools are properly wired to agent + proxy."""

    def test_four_tools_on_agent(self, wired_pair):
        agent, _ = wired_pair
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 4

    def test_tool_names(self, wired_pair):
        agent, _ = wired_pair
        tool_names = {t["function"]["name"] for t in agent.llm_config["tools"]}
        assert tool_names == {"describe_stats", "missing_analysis", "correlation_matrix", "target_analysis"}

    def test_tool_descriptions(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            desc = tool["function"]["description"]
            assert len(desc) > 10, f"Description too short: {desc}"

    def test_executor_has_all_tools(self, wired_pair):
        _, proxy = wired_pair
        assert "describe_stats" in proxy._function_map
        assert "missing_analysis" in proxy._function_map
        assert "correlation_matrix" in proxy._function_map
        assert "target_analysis" in proxy._function_map

    def test_describe_stats_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["describe_stats"]["function"]["parameters"]
        assert "data_json" in params["properties"]

    def test_missing_analysis_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["missing_analysis"]["function"]["parameters"]
        assert "data_json" in params["properties"]

    def test_correlation_matrix_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["correlation_matrix"]["function"]["parameters"]
        assert "data_json" in params["properties"]


# ---------------------------------------------------------------------------
# End-to-end tool execution (pure function calls — no LLM)
# ---------------------------------------------------------------------------

class TestToolExecution:
    """Test that registered tools can be called through the executor's function map."""

    def test_describe_stats_via_executor(self, wired_pair, csv_path):
        _, proxy = wired_pair
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = proxy._function_map["describe_stats"](data_json)
        parsed = json.loads(result)
        assert "age" in parsed
        assert "salary" in parsed

    def test_missing_analysis_via_executor(self, wired_pair, csv_path):
        _, proxy = wired_pair
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = proxy._function_map["missing_analysis"](data_json)
        info = MissingInfo.model_validate_json(result)
        assert info.per_column["age"] == 20.0

    def test_correlation_matrix_via_executor(self, wired_pair, csv_path):
        _, proxy = wired_pair
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = proxy._function_map["correlation_matrix"](data_json)
        parsed = json.loads(result)
        assert "age" in parsed
        assert "salary" in parsed
        assert "dept" not in parsed  # categorical excluded

    def test_full_pipeline(self, wired_pair, csv_path):
        """End-to-end: load → describe + missing + correlation, all via executor function map."""
        _, proxy = wired_pair
        from tools.data_loader import load_data

        # Step 1: load data
        data_json = load_data(csv_path)
        records = json.loads(data_json)
        assert len(records) == 5

        # Step 2: describe
        desc = json.loads(proxy._function_map["describe_stats"](data_json))
        assert len(desc) == 3  # 3 columns

        # Step 3: missing
        miss = MissingInfo.model_validate_json(
            proxy._function_map["missing_analysis"](data_json)
        )
        assert miss.total_pct == 20.0

        # Step 4: correlation
        corr = json.loads(proxy._function_map["correlation_matrix"](data_json))
        assert "age" in corr

        # Assemble into EDAResults (what the agent would do)
        eda = EDAResults(describe=desc, missing=miss, correlation=corr)
        assert len(eda.describe) == 3
        assert eda.missing.total_pct == 20.0
        assert "age" in eda.correlation
