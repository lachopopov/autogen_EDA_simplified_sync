"""
tests/test_data_prep_agent.py — Unit tests for agents/data_prep_agent.py

Tests the agent factory, tool registration, and end-to-end wiring.
These tests use AG2 classes but do NOT make real LLM calls.
"""

import json

import pandas as pd
import pytest
from autogen import AssistantAgent, UserProxyAgent

from agents import make_agent
from agents.data_prep_agent import (
    DATA_PREP_SYSTEM_MESSAGE,
    create_data_prep_agent,
    register_data_prep_tools,
)


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
def data_prep_agent():
    """A DataPrepAgent created via the factory."""
    return create_data_prep_agent()


@pytest.fixture()
def wired_pair(data_prep_agent, user_proxy):
    """Agent + proxy with tools registered."""
    register_data_prep_tools(data_prep_agent, user_proxy)
    return data_prep_agent, user_proxy


@pytest.fixture()
def csv_path(tmp_path):
    """A tiny CSV file for tool execution tests."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# make_agent() factory
# ---------------------------------------------------------------------------

class TestMakeAgent:
    """Test the base factory function."""

    def test_returns_assistant_agent(self):
        a = make_agent("Test", "You are a test agent.")
        assert isinstance(a, AssistantAgent)

    def test_name(self):
        a = make_agent("FooAgent", "msg")
        assert a.name == "FooAgent"

    def test_system_message(self):
        a = make_agent("FooAgent", "custom system msg")
        assert "custom system msg" in a.system_message

    def test_max_consecutive_auto_reply(self):
        a = make_agent("FooAgent", "msg")
        assert a._max_consecutive_auto_reply == 5

    def test_termination_guard(self):
        a = make_agent("FooAgent", "msg")
        assert a._is_termination_msg({"content": "TERMINATE"}) is True
        assert a._is_termination_msg({"content": "keep going"}) is False
        assert a._is_termination_msg({"content": None}) is False


# ---------------------------------------------------------------------------
# create_data_prep_agent()
# ---------------------------------------------------------------------------

class TestCreateDataPrepAgent:
    """Test the DataPrepAgent factory specifically."""

    def test_name(self, data_prep_agent):
        assert data_prep_agent.name == "DataPrepAgent"

    def test_system_message(self, data_prep_agent):
        assert "load_data()" in data_prep_agent.system_message
        assert "validate_schema()" in data_prep_agent.system_message
        assert "infer_dtypes()" in data_prep_agent.system_message

    def test_system_message_exact(self, data_prep_agent):
        assert data_prep_agent.system_message == DATA_PREP_SYSTEM_MESSAGE

    def test_is_assistant_agent(self, data_prep_agent):
        assert isinstance(data_prep_agent, AssistantAgent)

    def test_no_terminate_instruction(self, data_prep_agent):
        """Non-terminal agents must explicitly forbid TERMINATE to prevent short-circuit."""
        assert "Do NOT include the word TERMINATE" in data_prep_agent.system_message


# ---------------------------------------------------------------------------
# register_data_prep_tools() — AG2 two-step registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Test that tools are properly wired to agent + proxy."""

    def test_three_tools_on_agent(self, wired_pair):
        agent, _ = wired_pair
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 3

    def test_tool_names(self, wired_pair):
        agent, _ = wired_pair
        tool_names = {t["function"]["name"] for t in agent.llm_config["tools"]}
        assert tool_names == {"load_data", "validate_schema", "infer_dtypes"}

    def test_tool_descriptions(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            desc = tool["function"]["description"]
            assert len(desc) > 10, f"Description too short: {desc}"

    def test_executor_has_all_tools(self, wired_pair):
        _, proxy = wired_pair
        assert "load_data" in proxy._function_map
        assert "validate_schema" in proxy._function_map
        assert "infer_dtypes" in proxy._function_map

    def test_load_data_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["load_data"]["function"]["parameters"]
        assert "file_path" in params["properties"]

    def test_validate_schema_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["validate_schema"]["function"]["parameters"]
        assert "data_json" in params["properties"]

    def test_infer_dtypes_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["infer_dtypes"]["function"]["parameters"]
        assert "data_json" in params["properties"]


# ---------------------------------------------------------------------------
# End-to-end tool execution (pure function calls — no LLM)
# ---------------------------------------------------------------------------

class TestToolExecution:
    """Test that registered tools can be called through the executor's function map."""

    def test_load_data_via_executor(self, wired_pair, csv_path):
        _, proxy = wired_pair
        fn = proxy._function_map["load_data"]
        result = fn(csv_path)
        records = json.loads(result)
        assert len(records) == 3
        assert set(records[0].keys()) == {"a", "b"}

    def test_validate_schema_via_executor(self, wired_pair, csv_path):
        _, proxy = wired_pair
        data_json = proxy._function_map["load_data"](csv_path)
        result = proxy._function_map["validate_schema"](data_json)
        profile = json.loads(result)
        assert profile["shape"] == [3, 2]
        assert "a" in profile["dtypes"]

    def test_infer_dtypes_via_executor(self, wired_pair, csv_path):
        _, proxy = wired_pair
        data_json = proxy._function_map["load_data"](csv_path)
        result = proxy._function_map["infer_dtypes"](data_json)
        profile = json.loads(result)
        assert "a" in profile["numerical_cols"]
        assert "b" in profile["categorical_cols"]

    def test_full_pipeline(self, wired_pair, csv_path):
        """End-to-end: load → validate → infer, all via executor function map."""
        _, proxy = wired_pair
        fmap = proxy._function_map

        # Step 1: load
        data_json = fmap["load_data"](csv_path)
        records = json.loads(data_json)
        assert len(records) == 3

        # Step 2: validate
        schema_json = fmap["validate_schema"](data_json)
        schema = json.loads(schema_json)
        assert schema["shape"] == [3, 2]

        # Step 3: infer
        dtypes_json = fmap["infer_dtypes"](data_json)
        dtypes = json.loads(dtypes_json)
        assert len(dtypes["numerical_cols"]) + len(dtypes["categorical_cols"]) == 2
