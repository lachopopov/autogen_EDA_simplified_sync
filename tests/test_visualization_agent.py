"""
tests/test_visualization_agent.py — Unit tests for agents/visualization_agent.py

Tests the agent factory, tool registration, and end-to-end wiring.
These tests use AG2 classes but do NOT make real LLM calls.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from autogen import AssistantAgent, UserProxyAgent

from agents.visualization_agent import (
    VISUALIZATION_SYSTEM_MESSAGE,
    create_visualization_agent,
    register_visualization_tools,
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
def viz_agent():
    """A VisualizationAgent created via the factory."""
    return create_visualization_agent()


@pytest.fixture()
def wired_pair(viz_agent, user_proxy):
    """Agent + proxy with tools registered."""
    register_visualization_tools(viz_agent, user_proxy)
    return viz_agent, user_proxy


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


@pytest.fixture()
def plots_dir(tmp_path):
    """A temporary directory for plot output."""
    return str(tmp_path / "plots")


# ---------------------------------------------------------------------------
# create_visualization_agent()
# ---------------------------------------------------------------------------

class TestCreateVisualizationAgent:
    """Test the VisualizationAgent factory."""

    def test_name(self, viz_agent):
        assert viz_agent.name == "VisualizationAgent"

    def test_system_message(self, viz_agent):
        assert "plot_histograms()" in viz_agent.system_message
        assert "plot_correlation_heatmap()" in viz_agent.system_message
        assert "plot_missing_heatmap()" in viz_agent.system_message

    def test_system_message_exact(self, viz_agent):
        assert viz_agent.system_message == VISUALIZATION_SYSTEM_MESSAGE

    def test_is_assistant_agent(self, viz_agent):
        assert isinstance(viz_agent, AssistantAgent)

    def test_no_terminate_instruction(self, viz_agent):
        """Non-terminal agents must explicitly forbid TERMINATE to prevent short-circuit."""
        assert "Do NOT include the word TERMINATE" in viz_agent.system_message

    def test_max_consecutive_auto_reply(self, viz_agent):
        assert viz_agent._max_consecutive_auto_reply == 10

    def test_termination_guard(self, viz_agent):
        assert viz_agent._is_termination_msg({"content": "TERMINATE"}) is True
        assert viz_agent._is_termination_msg({"content": "keep going"}) is False


# ---------------------------------------------------------------------------
# register_visualization_tools() — AG2 chained-decorator registration
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
        assert tool_names == {
            "plot_histograms",
            "plot_correlation_heatmap",
            "plot_missing_heatmap",
            "plot_class_distribution",
        }

    def test_tool_descriptions(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            desc = tool["function"]["description"]
            assert len(desc) > 10, f"Description too short: {desc}"

    def test_executor_has_all_tools(self, wired_pair):
        _, proxy = wired_pair
        assert "plot_histograms" in proxy._function_map
        assert "plot_correlation_heatmap" in proxy._function_map
        assert "plot_missing_heatmap" in proxy._function_map
        assert "plot_class_distribution" in proxy._function_map

    def test_plot_histograms_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["plot_histograms"]["function"]["parameters"]
        assert "data_json" in params["properties"]
        assert "output_dir" in params["properties"]

    def test_plot_correlation_heatmap_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["plot_correlation_heatmap"]["function"]["parameters"]
        assert "corr_json" in params["properties"]
        assert "output_dir" in params["properties"]

    def test_plot_missing_heatmap_schema(self, wired_pair):
        agent, _ = wired_pair
        tools_by_name = {t["function"]["name"]: t for t in agent.llm_config["tools"]}
        params = tools_by_name["plot_missing_heatmap"]["function"]["parameters"]
        assert "missing_json" in params["properties"]
        assert "output_dir" in params["properties"]


# ---------------------------------------------------------------------------
# End-to-end tool execution (pure function calls — no LLM)
# ---------------------------------------------------------------------------

class TestToolExecution:
    """Test that registered tools can be called through the executor's function map."""

    def test_plot_histograms_via_executor(self, wired_pair, csv_path, plots_dir):
        _, proxy = wired_pair
        from tools.data_loader import load_data

        data_json = load_data(csv_path)
        result = proxy._function_map["plot_histograms"](data_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 2  # age, salary
        for p in paths:
            assert Path(p).exists()

    def test_plot_correlation_via_executor(self, wired_pair, csv_path, plots_dir):
        _, proxy = wired_pair
        from tools.data_loader import load_data
        from tools.eda_tools import correlation_matrix

        data_json = load_data(csv_path)
        corr_json = correlation_matrix(data_json)
        result = proxy._function_map["plot_correlation_heatmap"](corr_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_plot_missing_via_executor(self, wired_pair, csv_path, plots_dir):
        _, proxy = wired_pair
        from tools.data_loader import load_data
        from tools.eda_tools import missing_analysis

        data_json = load_data(csv_path)
        miss_json = missing_analysis(data_json)
        result = proxy._function_map["plot_missing_heatmap"](miss_json, plots_dir)
        paths = json.loads(result)
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_full_pipeline(self, wired_pair, csv_path, plots_dir):
        """End-to-end: load → EDA tools → all 3 visualization tools via executor."""
        _, proxy = wired_pair
        from tools.data_loader import load_data
        from tools.eda_tools import correlation_matrix, missing_analysis

        data_json = load_data(csv_path)
        corr_json = correlation_matrix(data_json)
        miss_json = missing_analysis(data_json)

        hist = json.loads(proxy._function_map["plot_histograms"](data_json, plots_dir))
        corr = json.loads(proxy._function_map["plot_correlation_heatmap"](corr_json, plots_dir))
        miss = json.loads(proxy._function_map["plot_missing_heatmap"](miss_json, plots_dir))

        all_paths = hist + corr + miss
        assert len(all_paths) == 4  # 2 hist + 1 corr + 1 missing
        for p in all_paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0
