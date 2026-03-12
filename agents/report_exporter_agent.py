"""
agents/report_exporter_agent.py — ReportExporterAgent factory + tool registration.

Architecture Reference: architecture.md § 4.7, § 5.1, § 12.1, § 12.6

Role: Generate the final EDA report files (PDF + Markdown + optional IPYNB).
Tools: render_pdf(), render_markdown(), render_ipynb()
Output: outputs/report.pdf, outputs/report.md, optionally outputs/report.ipynb

This is the ONLY agent authorized to emit TERMINATE (architecture.md § 5.1).
All other agents explicitly suppress TERMINATE in their system messages.

Tool registration uses the AG2 canonical chained-decorator pattern:
  @agent.register_for_llm(description="...")
  @user_proxy.register_for_execution()
  def tool_function(param: Annotated[type, "description"]) -> type: ...

Since tools are defined in tools/ (no AG2 imports — Hard Boundary Rule),
we apply the equivalent chained call programmatically:
  agent.register_for_llm(desc)(user_proxy.register_for_execution()(fn))

AG2 Version: 0.10.3
"""

from autogen import UserProxyAgent

from agents import make_agent
from tools.report_tools import render_ipynb, render_markdown, render_pdf

# System message matches architecture.md § 4.7.
# This is the ONLY agent whose system_message includes "TERMINATE".
# § 5.1: "ReportExporterAgent MUST emit TERMINATE after export"
REPORT_EXPORTER_SYSTEM_MESSAGE = """\
Generate the final EDA report files.
Use render_pdf(findings_json=..., output_dir="outputs/") — always pass output_dir="outputs/" exactly.
Use render_markdown(findings_json=..., output_dir="outputs/") — always call this unconditionally alongside render_pdf().
Use render_ipynb() if IPYNB_EXPORT=true in environment, with output_dir="outputs/" as well.
When a tool returns a confirmation message with "Reference: STATE_REF:...", the tool has SUCCEEDED.
Do NOT re-call the same tool. Do NOT copy large JSON.
Ground your answers only on data returned by your tools. If you do not have the facts, state "No info available at this stage." Do NOT invent or fabricate any statistics, numbers, or findings.
After export, reply with exactly one word: TERMINATE"""


def create_report_exporter_agent():
    """Factory: return a configured ReportExporterAgent instance."""
    return make_agent(
        name="ReportExporterAgent",
        system_message=REPORT_EXPORTER_SYSTEM_MESSAGE,
    )


def register_report_exporter_tools(agent, user_proxy: UserProxyAgent) -> None:
    """
    Wire ReportExporterAgent's tools using the AG2 canonical chained-decorator pattern.

    Equivalent to:
        @agent.register_for_llm(description="...")
        @user_proxy.register_for_execution()
        def tool_fn(...): ...

    Applied programmatically because tool functions live in tools/
    (zero AG2 imports — Hard Boundary Rule, architecture.md § 12.1).

    Args:
        agent: The ReportExporterAgent (AssistantAgent) instance.
        user_proxy: The UserProxyAgent that executes all tools.
    """
    # --- render_pdf ---
    agent.register_for_llm(
        description=(
            "Render EDA findings as a PDF report. Accepts Findings JSON "
            "from assemble_findings() and output directory path. "
            "Returns the path to the generated PDF file."
        )
    )(user_proxy.register_for_execution()(render_pdf))

    # --- render_markdown ---
    agent.register_for_llm(
        description=(
            "Render EDA findings as a plain Markdown report (.md). Accepts "
            "Findings JSON from assemble_findings() and output directory path. "
            "Always call this unconditionally — Markdown is the LLM-readable output. "
            "Returns the path to the generated report.md file."
        )
    )(user_proxy.register_for_execution()(render_markdown))

    # --- render_ipynb ---
    agent.register_for_llm(
        description=(
            "Render EDA findings as a Jupyter notebook (.ipynb). Accepts "
            "Findings JSON from assemble_findings() and output directory path. "
            "Only call this when IPYNB_EXPORT=true. "
            "Returns the path to the generated IPYNB file."
        )
    )(user_proxy.register_for_execution()(render_ipynb))
