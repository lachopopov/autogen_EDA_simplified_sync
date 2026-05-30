"""
agents/critic_agent.py — CriticAgent factory + tool registration.

Architecture Reference: architecture.md § 4.5, § 12.1, § 12.6

Role: Apply rule-based quality checks to EDA results.
Tools: run_critic_rules() — pure deterministic rule engine, no LLM reasoning on checks.
Output: CriticReport — flags list, iteration count, status (APPROVED | REVISION_NEEDED)

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
from tools.critic_rules import run_critic_rules

# System message matches architecture.md § 4.5.
# Explicit "Do NOT use TERMINATE" prevents accidental pipeline short-circuit
# (only ReportExporterAgent is allowed to emit TERMINATE — architecture.md § 4.7, § 5).
CRITIC_SYSTEM_MESSAGE = """\
Apply rule-based quality checks to the EDA results.
Use run_critic_rules() with the data reference from load_data().
Do NOT invent additional rules beyond what run_critic_rules() returns.
When a tool returns a confirmation message with "Reference: STATE_REF:...", the tool has SUCCEEDED.
Do NOT re-call the same tool. Move on to emit your status summary.
Summarise the quality flags returned by run_critic_rules(). Do not determine approval status — the pipeline handles routing automatically.
Do NOT copy large JSON.
Keep your text summary under 3 sentences. Do not offer options or next-step suggestions — the pipeline advances automatically.
Ground your answers only on data returned by your tools. If you do not have the facts, state "No info available at this stage." Do NOT invent or fabricate any statistics, numbers, or findings.
Do NOT include the word TERMINATE in your response."""


def create_critic_agent():
    """Factory: return a configured CriticAgent instance."""
    return make_agent(
        name="CriticAgent",
        system_message=CRITIC_SYSTEM_MESSAGE,
    )


def register_critic_tools(agent, user_proxy: UserProxyAgent) -> None:
    """
    Wire CriticAgent's tools using the AG2 canonical chained-decorator pattern.

    Equivalent to:
        @agent.register_for_llm(description="...")
        @user_proxy.register_for_execution()
        def tool_fn(...): ...

    Applied programmatically because tool functions live in tools/
    (zero AG2 imports — Hard Boundary Rule, architecture.md § 12.1).

    Args:
        agent: The CriticAgent (AssistantAgent) instance.
        user_proxy: The UserProxyAgent that executes all tools.
    """
    # --- run_critic_rules ---
    agent.register_for_llm(
        description=(
            "Run all critic rules on the DataFrame. "
            "Returns CriticReport JSON with flags and APPROVED/REVISION_NEEDED status."
        )
    )(user_proxy.register_for_execution()(run_critic_rules))
