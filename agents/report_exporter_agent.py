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
from config import IPYNB_EXPORT
from tools.report_tools import render_ipynb, render_markdown, render_pdf


def _build_system_message(ipynb_enabled: bool) -> str:
    """Build the ReportExporterAgent system message for the given IPYNB state.

    Resolves the IPYNB gate at call time so the LLM receives a concrete,
    unambiguous imperative rather than a conditional it cannot evaluate
    (e.g. "if IPYNB_EXPORT=true in environment" — the LLM has no OS context).

    Called at module load to produce the public constant, and called again
    inside create_report_exporter_agent() so that tests which monkeypatch
    the module-level IPYNB_EXPORT attribute always get the correct message
    for the patched value without needing to reload the module.
    """
    parts = [
        "Generate the final EDA report files.",
        'Use render_pdf(findings_json=..., output_dir="outputs/") — always pass output_dir="outputs/" exactly.',
        'Use render_markdown(findings_json=..., output_dir="outputs/") — always call this unconditionally alongside render_pdf().',
    ]
    if ipynb_enabled:
        parts.append(
            'Use render_ipynb(findings_json=..., output_dir="outputs/") '
            "to also export the report as a Jupyter notebook."
        )
    parts += [
        'When a tool returns a confirmation message with "Reference: STATE_REF:...", the tool has SUCCEEDED.',
        "Do NOT re-call the same tool. Do NOT copy large JSON.",
        (
            "Ground your answers only on data returned by your tools. "
            'If you do not have the facts, state "No info available at this stage." '
            "Do NOT invent or fabricate any statistics, numbers, or findings."
        ),
        "After export, reply with exactly one word: TERMINATE",
    ]
    return "\n".join(parts)


# System message matches architecture.md § 4.7.
# This is the ONLY agent whose system_message includes "TERMINATE".
# § 5.1: "ReportExporterAgent MUST emit TERMINATE after export"
#
# Computed at module load from the live IPYNB_EXPORT value so that:
#   1. Tests importing this constant get the same value as create_report_exporter_agent().
#   2. render_ipynb is mentioned only when IPYNB_EXPORT=true, removing the
#      unresolvable "if … in environment" conditional the LLM previously received.
REPORT_EXPORTER_SYSTEM_MESSAGE: str = _build_system_message(IPYNB_EXPORT)


def create_report_exporter_agent():
    """Factory: return a configured ReportExporterAgent instance.

    Calls _build_system_message(IPYNB_EXPORT) directly (rather than using
    the module-level constant) so that tests which monkeypatch
    agents.report_exporter_agent.IPYNB_EXPORT receive the correct runtime
    message without needing to reload the module.
    """
    return make_agent(
        name="ReportExporterAgent",
        system_message=_build_system_message(IPYNB_EXPORT),
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

    # --- render_ipynb (conditional) ---
    # Only added to the LLM tool schema when IPYNB_EXPORT=true.
    # Structural defense: if the tool is not in the schema, the model
    # physically cannot call it — no system-message pressure or context-window
    # compression can produce a call to a function that doesn't exist in the
    # function definitions the model receives.  This complements the system-
    # message-level instruction produced by _build_system_message(), giving
    # dual-layer enforcement: schema forbids + message is silent (disabled),
    # schema permits + message instructs (enabled).
    if IPYNB_EXPORT:
        agent.register_for_llm(
            description=(
                "Render EDA findings as a Jupyter notebook (.ipynb). Accepts "
                "Findings JSON from assemble_findings() and output directory path. "
                "Returns the path to the generated IPYNB file."
            )
        )(user_proxy.register_for_execution()(render_ipynb))
