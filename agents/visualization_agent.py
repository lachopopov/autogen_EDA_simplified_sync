"""
agents/visualization_agent.py — VisualizationAgent factory + tool registration.

Architecture Reference: architecture.md § 4.4, § 12.1, § 12.6

Role: Generate and save all visualizations.
Tools: plot_histograms(), plot_correlation_heatmap(), plot_missing_heatmap(), plot_class_distribution()
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
    plot_categorical_bars,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_target_bars,
    plot_histograms,
    plot_missing_heatmap,
    plot_ordinal_heatmap,
)

# System message matches architecture.md § 4.4.
# Explicit "Do NOT use TERMINATE" prevents accidental pipeline short-circuit
# (only ReportExporterAgent is allowed to emit TERMINATE — architecture.md § 4.7, § 5).
VISUALIZATION_SYSTEM_MESSAGE = """\
Generate and save all visualizations.
Call ALL seven tools in a SINGLE parallel tool_calls message — do not omit any:
  1. plot_histograms()
  2. plot_correlation_heatmap()
  3. plot_missing_heatmap()
  4. plot_class_distribution()
  5. plot_categorical_bars()
  6. plot_ordinal_heatmap()
  7. plot_feature_target_bars()
Every tool above MUST appear in your tool_calls list.
plot_class_distribution() requires only target_info_json=STATE_REF:target_info.
Do NOT supply data_json for plot_class_distribution() — it is loaded automatically from the artifact store.
ALWAYS call plot_class_distribution(); it handles the no-target case internally and returns a reference even when no plot is generated.
plot_categorical_bars() requires categorical_analysis_json — load it from artifact store (STATE_REF:categorical_analysis).
If no categorical_analysis artifact exists, skip plot_categorical_bars().
plot_ordinal_heatmap() and plot_feature_target_bars() require no arguments — they are self-contained and load all data from the artifact store.
ALWAYS call plot_ordinal_heatmap() and plot_feature_target_bars(); they handle the no-data case internally.
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
        description="Plot histograms for all numerical columns. Saves PNGs to the internal session directory. Returns JSON list of file paths."
    )(user_proxy.register_for_execution()(plot_histograms))

    # --- plot_correlation_heatmap ---
    agent.register_for_llm(
        description="Plot Pearson correlation heatmap. Saves PNG to the internal session directory. Returns JSON list of file paths."
    )(user_proxy.register_for_execution()(plot_correlation_heatmap))

    # --- plot_missing_heatmap ---
    agent.register_for_llm(
        description="Plot missing values bar chart by column. Saves PNG to the internal session directory. Returns JSON list of file paths."
    )(user_proxy.register_for_execution()(plot_missing_heatmap))

    # --- plot_class_distribution ---
    agent.register_for_llm(
        description="Plot target variable distribution: bar chart for classification, histogram+KDE for regression. Requires target_info_json from artifact store. Saves PNG to the internal session directory."
    )(user_proxy.register_for_execution()(plot_class_distribution))

    # --- plot_categorical_bars ---
    agent.register_for_llm(
        description="Plot horizontal bar charts for each categorical column showing top-N category frequencies and percentages. Requires categorical_analysis_json from artifact store (STATE_REF:categorical_analysis). Saves one PNG per column to the internal session directory."
    )(user_proxy.register_for_execution()(plot_categorical_bars))

    # --- plot_ordinal_heatmap ---
    agent.register_for_llm(
        description="Plot Spearman rank-correlation heatmap for ordinal-encoded categorical columns. Self-contained: loads data from artifact store. Requires no arguments."
    )(user_proxy.register_for_execution()(plot_ordinal_heatmap))

    # --- plot_feature_target_bars ---
    agent.register_for_llm(
        description="Plot horizontal bar chart of Borda-ranked feature–target associations (MI score + effect size). Self-contained: loads feature_associations from artifact store. Requires no arguments."
    )(user_proxy.register_for_execution()(plot_feature_target_bars))
