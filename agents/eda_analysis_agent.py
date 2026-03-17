"""
agents/eda_analysis_agent.py — EDAAnalysisAgent factory + tool registration.

Architecture Reference: architecture.md § 4.3, § 12.1, § 12.6

Role: Perform descriptive statistical analysis on the loaded data.
Tools: describe_stats(), missing_analysis(), correlation_matrix(), target_analysis(),
       analyze_categoricals()
Output: EDAResults (stats dict, missing dict, correlation dict) + target analysis
       + categorical inventory

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
from tools.eda_tools import (
    analyze_categoricals,
    compute_feature_target_associations,
    correlation_matrix,
    describe_stats,
    missing_analysis,
    target_analysis,
)

# System message matches architecture.md § 4.3.
# Explicit "Do NOT use TERMINATE" prevents accidental pipeline short-circuit
# (only ReportExporterAgent is allowed to emit TERMINATE — architecture.md § 4.7, § 5).
EDA_ANALYSIS_SYSTEM_MESSAGE = """\
Perform descriptive statistical analysis on the loaded data.
Call all six tools in a SINGLE parallel tool_calls message:
  describe_stats(), missing_analysis(), correlation_matrix(), target_analysis(),
  analyze_categoricals(), compute_feature_target_associations().
Pass the data reference from load_data() directly to each tool. Do NOT copy large JSON.
target_analysis(), analyze_categoricals(), and compute_feature_target_associations()
also require target_info_json — load it from artifact store (STATE_REF:target_info).
If no target_info artifact exists, skip target_analysis() and
compute_feature_target_associations(), but still call analyze_categoricals()
(pass an empty TargetInfo JSON: {}).
When a tool returns a confirmation message with "Reference: STATE_REF:...", the tool has SUCCEEDED.
Do NOT re-call the same tool. After receiving results, emit a brief text summary and advance.
Keep your text summary under 3 sentences. Do not offer options or next-step suggestions — the pipeline advances automatically.
Ground your answers only on data returned by your tools. If you do not have the facts, state "No info available at this stage." Do NOT invent or fabricate any statistics, numbers, or findings.
Do NOT include the word TERMINATE in your response."""


def create_eda_analysis_agent():
    """Factory: return a configured EDAAnalysisAgent instance."""
    return make_agent(
        name="EDAAnalysisAgent",
        system_message=EDA_ANALYSIS_SYSTEM_MESSAGE,
    )


def register_eda_analysis_tools(agent, user_proxy: UserProxyAgent) -> None:
    """
    Wire EDAAnalysisAgent's tools using the AG2 canonical chained-decorator pattern.

    Equivalent to:
        @agent.register_for_llm(description="...")
        @user_proxy.register_for_execution()
        def tool_fn(...): ...

    Applied programmatically because tool functions live in tools/
    (zero AG2 imports — Hard Boundary Rule, architecture.md § 12.1).

    Args:
        agent: The EDAAnalysisAgent (AssistantAgent) instance.
        user_proxy: The UserProxyAgent that executes all tools.
    """
    # --- describe_stats ---
    agent.register_for_llm(
        description="Compute descriptive statistics: central tendency, spread, percentiles for all columns."
    )(user_proxy.register_for_execution()(describe_stats))

    # --- missing_analysis ---
    agent.register_for_llm(
        description="Compute per-column and dataset-level missing value percentages. Returns MissingInfo JSON."
    )(user_proxy.register_for_execution()(missing_analysis))

    # --- correlation_matrix ---
    agent.register_for_llm(
        description="Compute Pearson correlation matrix for numerical columns. Returns nested dict JSON."
    )(user_proxy.register_for_execution()(correlation_matrix))

    # --- target_analysis ---
    agent.register_for_llm(
        description="Analyse target variable: class distribution, imbalance ratio, per-class feature stats (classification) or target correlations (regression). Requires target_info_json from artifact store."
    )(user_proxy.register_for_execution()(target_analysis))

    # --- analyze_categoricals (W4) ---
    agent.register_for_llm(
        description=(
            "Compute categorical distributions: value counts (top-10), cardinality, "
            "Shannon entropy, rare-category count, and target rate per category "
            "(classification). Requires target_info_json from artifact store."
        )
    )(user_proxy.register_for_execution()(analyze_categoricals))

    # --- compute_feature_target_associations (W7) ---
    agent.register_for_llm(
        description=(
            "Compute univariate feature-target associations using two lenses: "
            "(1) Mutual Information (MI, kNN-based, detects any dependence); "
            "(2) Effect size (n-invariant [0,1]: eta² for numerical/classification, "
            "Cramér's V for categorical/classification, |Pearson r| for numerical/regression, "
            "eta² reversed for categorical/regression). "
            "Ranks features by Borda score = mi_rank + effect_size_rank (lower = more important). "
            "Requires data_json and target_info_json from artifact store."
        )
    )(user_proxy.register_for_execution()(compute_feature_target_associations))
