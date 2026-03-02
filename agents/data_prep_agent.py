"""
agents/data_prep_agent.py — DataPrepAgent factory + tool registration.

Architecture Reference: architecture.md § 4.2, § 12.1, § 12.6

Role: Load and validate the input data file.
Tools: load_data(), validate_schema(), infer_dtypes()
Output: DataProfile (shape, dtypes, memory_mb, column classification)

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
from tools.data_loader import infer_dtypes, load_data, validate_schema

# System message matches architecture.md § 4.2.
# Explicit "Do NOT use TERMINATE" prevents accidental pipeline short-circuit
# (only ReportExporterAgent is allowed to emit TERMINATE — architecture.md § 4.7, § 5).
DATA_PREP_SYSTEM_MESSAGE = """\
Load and validate the input file.
Use load_data() to load CSV/Parquet/XLSX.
Use validate_schema() to check column types and shape.
Use infer_dtypes() to classify columns as numerical/categorical.
When a tool returns a confirmation message with "Reference: STATE_REF:...", the tool has SUCCEEDED.
Do NOT re-call the same tool. Move on to the next tool or emit a text summary to advance.
Do NOT copy large JSON blobs. Just pass the reference.
Keep your text summary under 3 sentences. Do not offer options or next-step suggestions — the pipeline advances automatically.
Ground your answers only on data returned by your tools. If you do not have the facts, state "No info available at this stage." Do NOT invent or fabricate any statistics, numbers, or findings.
Do NOT include the word TERMINATE in your response."""


def create_data_prep_agent():
    """Factory: return a configured DataPrepAgent instance."""
    return make_agent(name="DataPrepAgent", system_message=DATA_PREP_SYSTEM_MESSAGE)


def register_data_prep_tools(agent, user_proxy: UserProxyAgent) -> None:
    """
    Wire DataPrepAgent's tools using the AG2 canonical chained-decorator pattern.

    Equivalent to:
        @agent.register_for_llm(description="...")
        @user_proxy.register_for_execution()
        def tool_fn(...): ...

    Applied programmatically because tool functions live in tools/
    (zero AG2 imports — Hard Boundary Rule, architecture.md § 12.1).

    Args:
        agent: The DataPrepAgent (AssistantAgent) instance.
        user_proxy: The UserProxyAgent that executes all tools.
    """
    # --- load_data ---
    agent.register_for_llm(
        description="Load a CSV, Parquet, or Excel file. Returns DataFrame as JSON string."
    )(user_proxy.register_for_execution()(load_data))

    # --- validate_schema ---
    agent.register_for_llm(
        description="Validate shape, dtypes, and memory footprint. Returns DataProfile JSON."
    )(user_proxy.register_for_execution()(validate_schema))

    # --- infer_dtypes ---
    agent.register_for_llm(
        description="Classify columns as numerical or categorical. Returns DataProfile JSON."
    )(user_proxy.register_for_execution()(infer_dtypes))
