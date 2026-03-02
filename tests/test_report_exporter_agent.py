"""
tests/test_report_exporter_agent.py — Unit tests for agents/report_exporter_agent.py

Tests the agent factory, tool registration, and end-to-end wiring.
These tests use AG2 classes but do NOT make real LLM calls.
"""

import json
from pathlib import Path

import pytest
from autogen import AssistantAgent, UserProxyAgent

from agents.report_exporter_agent import (
    REPORT_EXPORTER_SYSTEM_MESSAGE,
    create_report_exporter_agent,
    register_report_exporter_tools,
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
def report_agent():
    """A ReportExporterAgent created via the factory."""
    return create_report_exporter_agent()


@pytest.fixture()
def wired_pair(report_agent, user_proxy):
    """Agent + proxy with tools registered."""
    register_report_exporter_tools(report_agent, user_proxy)
    return report_agent, user_proxy


# ---------------------------------------------------------------------------
# create_report_exporter_agent()
# ---------------------------------------------------------------------------

class TestCreateReportExporterAgent:
    """Test the ReportExporterAgent factory."""

    def test_name(self, report_agent):
        assert report_agent.name == "ReportExporterAgent"

    def test_system_message(self, report_agent):
        assert "render_pdf()" in report_agent.system_message
        assert "render_ipynb()" in report_agent.system_message

    def test_system_message_exact(self, report_agent):
        assert report_agent.system_message == REPORT_EXPORTER_SYSTEM_MESSAGE

    def test_is_assistant_agent(self, report_agent):
        assert isinstance(report_agent, AssistantAgent)

    def test_terminate_instruction(self, report_agent):
        """ReportExporterAgent is the ONLY agent that says TERMINATE."""
        assert "TERMINATE" in report_agent.system_message
        # It should NOT say "Do NOT include the word TERMINATE"
        assert "Do NOT include the word TERMINATE" not in report_agent.system_message

    def test_max_consecutive_auto_reply(self, report_agent):
        assert report_agent._max_consecutive_auto_reply == 5

    def test_termination_guard(self, report_agent):
        assert report_agent._is_termination_msg({"content": "TERMINATE"}) is True
        assert report_agent._is_termination_msg({"content": "keep going"}) is False


# ---------------------------------------------------------------------------
# register_report_exporter_tools()
# ---------------------------------------------------------------------------

class TestRegisterReportExporterTools:
    """Test tool registration on agent + proxy."""

    def test_function_map_has_render_pdf(self, wired_pair):
        _, proxy = wired_pair
        assert "render_pdf" in proxy._function_map

    def test_function_map_has_render_ipynb(self, wired_pair):
        _, proxy = wired_pair
        assert "render_ipynb" in proxy._function_map

    def test_function_map_count(self, wired_pair):
        """ReportExporterAgent has exactly 2 tools."""
        _, proxy = wired_pair
        assert len(proxy._function_map) == 2

    def test_llm_config_has_tool_schemas(self, wired_pair):
        agent, _ = wired_pair
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 2

    def test_tool_schema_names(self, wired_pair):
        agent, _ = wired_pair
        names = {t["function"]["name"] for t in agent.llm_config["tools"]}
        assert names == {"render_pdf", "render_ipynb"}

    def test_render_pdf_schema_has_parameters(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            if tool["function"]["name"] == "render_pdf":
                params = tool["function"]["parameters"]
                assert "findings_json" in params.get("properties", {})
                assert "output_dir" in params.get("properties", {})
                return
        pytest.fail("render_pdf tool schema not found")

    def test_render_ipynb_schema_has_parameters(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            if tool["function"]["name"] == "render_ipynb":
                params = tool["function"]["parameters"]
                assert "findings_json" in params.get("properties", {})
                assert "output_dir" in params.get("properties", {})
                return
        pytest.fail("render_ipynb tool schema not found")


# ---------------------------------------------------------------------------
# End-to-end: tool invocation through proxy function_map
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Test render_pdf / render_ipynb through the proxy function_map (no LLM)."""

    @staticmethod
    def _make_findings_json() -> str:
        """Create a standard Findings JSON for testing."""
        return json.dumps({
            "sections": [
                {"title": "Dataset Overview", "content": "50 rows, 3 columns."},
                {"title": "Missing Values", "content": "col_a: 2.0% missing"},
                {"title": "Correlation Analysis", "content": "col_a vs col_a: r=1.00"},
                {"title": "Statistical Analysis", "content": "Distribution analysis was performed on 2 numerical feature(s)."},
                {"title": "Data Quality Assessment", "content": "All quality checks passed."},
                {"title": "Conclusions", "content": "The dataset is fully complete with no missing values."},
                {"title": "Recommendations & Business Implications", "content": "The dataset shows good overall quality."},
            ],
            "unresolved_flags": [],
        })

    def test_render_pdf_via_proxy(self, wired_pair, tmp_path):
        """render_pdf callable through the proxy's function_map."""
        _, proxy = wired_pair
        fn = proxy._function_map["render_pdf"]
        result = fn(
            findings_json=self._make_findings_json(),
            output_dir=str(tmp_path),
        )
        assert isinstance(result, str)
        assert result.endswith("report.pdf")
        assert Path(result).exists()

    def test_render_ipynb_via_proxy(self, wired_pair, tmp_path):
        """render_ipynb callable through the proxy's function_map."""
        _, proxy = wired_pair
        fn = proxy._function_map["render_ipynb"]
        result = fn(
            findings_json=self._make_findings_json(),
            output_dir=str(tmp_path),
        )
        assert isinstance(result, str)
        assert result.endswith("report.ipynb")
        assert Path(result).exists()

    def test_pdf_magic_bytes_via_proxy(self, wired_pair, tmp_path):
        """PDF produced via proxy starts with %PDF- magic bytes."""
        _, proxy = wired_pair
        fn = proxy._function_map["render_pdf"]
        result = fn(
            findings_json=self._make_findings_json(),
            output_dir=str(tmp_path),
        )
        with open(result, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_ipynb_valid_notebook_via_proxy(self, wired_pair, tmp_path):
        """IPYNB produced via proxy is a valid nbformat v4 notebook."""
        import nbformat

        _, proxy = wired_pair
        fn = proxy._function_map["render_ipynb"]
        result = fn(
            findings_json=self._make_findings_json(),
            output_dir=str(tmp_path),
        )
        nb = nbformat.read(result, as_version=4)
        assert nb.nbformat == 4

    def test_chained_registration_invariant(self, wired_pair):
        """Both tools appear in agent LLM tools AND proxy function_map (P6)."""
        agent, proxy = wired_pair
        tool_names = {t["function"]["name"] for t in agent.llm_config.get("tools", [])}
        fn_names = set(proxy._function_map.keys())
        for name in ("render_pdf", "render_ipynb"):
            assert name in tool_names, f"{name} missing from LLM tools"
            assert name in fn_names, f"{name} missing from function_map"

    def test_pdf_and_ipynb_coexist_via_proxy(self, wired_pair, tmp_path):
        """Both exports can be produced in the same directory."""
        _, proxy = wired_pair
        fj = self._make_findings_json()
        pdf = proxy._function_map["render_pdf"](findings_json=fj, output_dir=str(tmp_path))
        ipynb = proxy._function_map["render_ipynb"](findings_json=fj, output_dir=str(tmp_path))
        assert Path(pdf).exists()
        assert Path(ipynb).exists()
        assert pdf != ipynb
