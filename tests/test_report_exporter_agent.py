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
    _build_system_message,
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
    """Agent + proxy with tools registered (IPYNB_EXPORT=false, default)."""
    register_report_exporter_tools(report_agent, user_proxy)
    return report_agent, user_proxy


@pytest.fixture()
def wired_pair_ipynb(monkeypatch, user_proxy):
    """Agent + proxy with all 3 tools registered (IPYNB_EXPORT=True forced).

    Monkeypatches agents.report_exporter_agent.IPYNB_EXPORT before creating
    the agent and registering tools, so both _build_system_message() and the
    conditional registration branch in register_report_exporter_tools() see
    the patched value without needing a module reload.
    """
    import agents.report_exporter_agent as m
    monkeypatch.setattr(m, "IPYNB_EXPORT", True)
    agent = create_report_exporter_agent()
    register_report_exporter_tools(agent, user_proxy)
    return agent, user_proxy


# ---------------------------------------------------------------------------
# create_report_exporter_agent()
# ---------------------------------------------------------------------------

class TestCreateReportExporterAgent:
    """Test the ReportExporterAgent factory."""

    def test_name(self, report_agent):
        assert report_agent.name == "ReportExporterAgent"

    def test_system_message(self, report_agent):
        assert "render_pdf" in report_agent.system_message
        assert "render_markdown" in report_agent.system_message
        # IPYNB_EXPORT=false (default in tests) → render_ipynb must NOT appear;
        # its omission is the unambiguous instruction to the LLM not to call it.
        assert "render_ipynb" not in report_agent.system_message

    def test_system_message_exact(self, report_agent):
        # REPORT_EXPORTER_SYSTEM_MESSAGE is computed from IPYNB_EXPORT at module
        # load; create_report_exporter_agent() calls _build_system_message(IPYNB_EXPORT)
        # with the same value → both strings are identical.
        assert report_agent.system_message == REPORT_EXPORTER_SYSTEM_MESSAGE

    def test_system_message_ipynb_enabled(self, monkeypatch):
        """When IPYNB_EXPORT=true, system message explicitly instructs render_ipynb."""
        import agents.report_exporter_agent as m
        monkeypatch.setattr(m, "IPYNB_EXPORT", True)
        agent = create_report_exporter_agent()
        assert "render_ipynb" in agent.system_message

    def test_build_system_message_disabled_has_no_ipynb(self):
        """_build_system_message(False) must not mention render_ipynb at all."""
        msg = _build_system_message(False)
        assert "render_ipynb" not in msg

    def test_build_system_message_enabled_has_ipynb(self):
        """_build_system_message(True) must instruct the agent to call render_ipynb."""
        msg = _build_system_message(True)
        assert "render_ipynb" in msg

    def test_is_assistant_agent(self, report_agent):
        assert isinstance(report_agent, AssistantAgent)

    def test_terminate_instruction(self, report_agent):
        """ReportExporterAgent is the ONLY agent that says TERMINATE."""
        assert "TERMINATE" in report_agent.system_message
        # It should NOT say "Do NOT include the word TERMINATE"
        assert "Do NOT include the word TERMINATE" not in report_agent.system_message

    def test_max_consecutive_auto_reply(self, report_agent):
        assert report_agent._max_consecutive_auto_reply == 10

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

    def test_function_map_has_render_markdown(self, wired_pair):
        _, proxy = wired_pair
        assert "render_markdown" in proxy._function_map

    def test_function_map_count(self, wired_pair):
        """When IPYNB_EXPORT=false (default), exactly 2 tools are registered."""
        _, proxy = wired_pair
        assert len(proxy._function_map) == 2

    def test_llm_config_has_tool_schemas(self, wired_pair):
        agent, _ = wired_pair
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 2

    def test_tool_schema_names(self, wired_pair):
        agent, _ = wired_pair
        names = {t["function"]["name"] for t in agent.llm_config["tools"]}
        assert names == {"render_pdf", "render_markdown"}

    def test_render_ipynb_absent_when_disabled(self, wired_pair):
        """render_ipynb must not appear in schema or function_map when IPYNB=false."""
        agent, proxy = wired_pair
        assert "render_ipynb" not in proxy._function_map
        schema_names = {t["function"]["name"] for t in agent.llm_config.get("tools", [])}
        assert "render_ipynb" not in schema_names

    def test_render_pdf_schema_has_parameters(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            if tool["function"]["name"] == "render_pdf":
                params = tool["function"]["parameters"]
                assert "findings_json" in params.get("properties", {})
                return
        pytest.fail("render_pdf tool schema not found")

    def test_render_markdown_schema_has_parameters(self, wired_pair):
        agent, _ = wired_pair
        for tool in agent.llm_config["tools"]:
            if tool["function"]["name"] == "render_markdown":
                params = tool["function"]["parameters"]
                assert "findings_json" in params.get("properties", {})
                return
        pytest.fail("render_markdown tool schema not found")


class TestRegisterReportExporterToolsWithIPYNB:
    """Tool registration tests when IPYNB_EXPORT=true (uses wired_pair_ipynb)."""

    def test_function_map_has_render_ipynb(self, wired_pair_ipynb):
        _, proxy = wired_pair_ipynb
        assert "render_ipynb" in proxy._function_map

    def test_function_map_count_with_ipynb(self, wired_pair_ipynb):
        """When IPYNB_EXPORT=true, exactly 3 tools are registered."""
        _, proxy = wired_pair_ipynb
        assert len(proxy._function_map) == 3

    def test_llm_config_has_tool_schemas_with_ipynb(self, wired_pair_ipynb):
        agent, _ = wired_pair_ipynb
        tools = agent.llm_config.get("tools", [])
        assert len(tools) == 3

    def test_tool_schema_names_with_ipynb(self, wired_pair_ipynb):
        agent, _ = wired_pair_ipynb
        names = {t["function"]["name"] for t in agent.llm_config["tools"]}
        assert names == {"render_pdf", "render_markdown", "render_ipynb"}

    def test_render_ipynb_schema_has_parameters(self, wired_pair_ipynb):
        agent, _ = wired_pair_ipynb
        for tool in agent.llm_config["tools"]:
            if tool["function"]["name"] == "render_ipynb":
                params = tool["function"]["parameters"]
                assert "findings_json" in params.get("properties", {})
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

    def test_render_markdown_via_proxy(self, wired_pair, tmp_path):
        """render_markdown callable through the proxy's function_map."""
        _, proxy = wired_pair
        fn = proxy._function_map["render_markdown"]
        result = fn(
            findings_json=self._make_findings_json(),
            output_dir=str(tmp_path),
        )
        assert isinstance(result, str)
        assert result.endswith("report.md")
        assert Path(result).exists()

    def test_render_ipynb_via_proxy(self, wired_pair_ipynb, tmp_path):
        """render_ipynb callable through the proxy's function_map (IPYNB_EXPORT=true)."""
        _, proxy = wired_pair_ipynb
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

    def test_ipynb_valid_notebook_via_proxy(self, wired_pair_ipynb, tmp_path):
        """IPYNB produced via proxy is a valid nbformat v4 notebook."""
        import nbformat

        _, proxy = wired_pair_ipynb
        fn = proxy._function_map["render_ipynb"]
        result = fn(
            findings_json=self._make_findings_json(),
            output_dir=str(tmp_path),
        )
        nb = nbformat.read(result, as_version=4)
        assert nb.nbformat == 4

    def test_chained_registration_invariant_base(self, wired_pair):
        """IPYNB disabled: render_pdf + render_markdown in both schema and function_map (P6)."""
        agent, proxy = wired_pair
        tool_names = {t["function"]["name"] for t in agent.llm_config.get("tools", [])}
        fn_names = set(proxy._function_map.keys())
        for name in ("render_pdf", "render_markdown"):
            assert name in tool_names, f"{name} missing from LLM tools"
            assert name in fn_names, f"{name} missing from function_map"
        assert "render_ipynb" not in tool_names
        assert "render_ipynb" not in fn_names

    def test_chained_registration_invariant_ipynb(self, wired_pair_ipynb):
        """IPYNB enabled: all 3 tools in both schema and function_map (P6)."""
        agent, proxy = wired_pair_ipynb
        tool_names = {t["function"]["name"] for t in agent.llm_config.get("tools", [])}
        fn_names = set(proxy._function_map.keys())
        for name in ("render_pdf", "render_markdown", "render_ipynb"):
            assert name in tool_names, f"{name} missing from LLM tools (ipynb)"
            assert name in fn_names, f"{name} missing from function_map (ipynb)"

    def test_pdf_and_ipynb_coexist_via_proxy(self, wired_pair_ipynb, tmp_path):
        """Both exports can be produced in the same directory (IPYNB_EXPORT=true)."""
        _, proxy = wired_pair_ipynb
        fj = self._make_findings_json()
        pdf = proxy._function_map["render_pdf"](findings_json=fj, output_dir=str(tmp_path))
        ipynb = proxy._function_map["render_ipynb"](findings_json=fj, output_dir=str(tmp_path))
        assert Path(pdf).exists()
        assert Path(ipynb).exists()
        assert pdf != ipynb
