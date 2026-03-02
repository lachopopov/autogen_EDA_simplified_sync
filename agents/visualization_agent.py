"""
agents/visualization_agent.py — VisualizationAgent factory + tool registration.

Architecture Reference: architecture.md § 4.4, § 12.1, § 12.6

Role: Generate and save all visualizations to outputs/plots/.
Tools: plot_histograms(), plot_correlation_heatmap(), plot_missing_heatmap()
Output: plot_paths — list of saved PNG file paths

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
from tools.visualization_tools import (
    plot_correlation_heatmap,
    plot_histograms,
    plot_missing_heatmap,
)

# System message matches architecture.md § 4.4.
# Explicit "Do NOT use TERMINATE" prevents accidental pipeline short-circuit
# (only ReportExporterAgent is allowed to emit TERMINATE — architecture.md § 4.7, § 5).
VISUALIZATION_SYSTEM_MESSAGE = """\
Generate and save all visualizations to outputs/plots/.
Call all three tools in a SINGLE parallel tool_calls message:
  plot_histograms(), plot_correlation_heatmap(), plot_missing_heatmap().
When a tool returns a confirmation message with "Reference: STATE_REF:...", the tool has SUCCEEDED.
Do NOT re-call the same tool. After receiving results, emit a brief text summary and advance.
Do NOT copy large JSON.
Keep your text summary under 3 sentences. Do not offer options or next-step suggestions — the pipeline advances automatically.
Ground your answers only on data returned by your tools. If you do not have the facts, state "No info available at this stage." Do NOT invent or fabricate any statistics, numbers, or findings.
Do NOT include the word TERMINATE in your response."""


def create_visualization_agent():
    """Factory: return a configured VisualizationAgent instance."""
    return make_agent(
        name="VisualizationAgent",
        system_message=VISUALIZATION_SYSTEM_MESSAGE,
    )


def register_visualization_tools(agent, user_proxy: UserProxyAgent) -> None:
    """
    Wire VisualizationAgent's tools using the AG2 canonical chained-decorator pattern.

    Equivalent to:
        @agent.register_for_llm(description="...")
        @user_proxy.register_for_execution()
        def tool_fn(...): ...

    Applied programmatically because tool functions live in tools/
    (zero AG2 imports — Hard Boundary Rule, architecture.md § 12.1).

    Args:
        agent: The VisualizationAgent (AssistantAgent) instance.
        user_proxy: The UserProxyAgent that executes all tools.
    """
    # --- plot_histograms ---
    agent.register_for_llm(
        description="Plot histograms for all numerical columns. Saves PNGs to output_dir. Returns JSON list of file paths."
    )(user_proxy.register_for_execution()(plot_histograms))

    # --- plot_correlation_heatmap ---
    agent.register_for_llm(
        description="Plot Pearson correlation heatmap. Saves PNG to output_dir. Returns JSON list of file paths."
    )(user_proxy.register_for_execution()(plot_correlation_heatmap))

    # --- plot_missing_heatmap ---
    agent.register_for_llm(
        description="Plot missing values bar chart by column. Saves PNG to output_dir. Returns JSON list of file paths."
    )(user_proxy.register_for_execution()(plot_missing_heatmap))
