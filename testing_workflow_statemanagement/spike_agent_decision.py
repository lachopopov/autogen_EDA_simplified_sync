"""
spike_agent_decision.py — Approach A: Agent-as-decision-point.

Minimal 2-stage pipeline: DataPrep → EDA on iris.csv.

Pattern:
  - AssistantAgent sends tool_call → route to its dedicated executor
  - Executor always returns to its AssistantAgent
  - AssistantAgent sends TEXT (no tool_call) → advance to next stage
  - The AGENT decides when it's done (by not making more tool calls)

Run:
    conda run -n ag2_env python testing_workflow_statemanagement/spike_agent_decision.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autogen import GroupChat, GroupChatManager, UserProxyAgent  # noqa: E402

from agents.data_prep_agent import create_data_prep_agent, register_data_prep_tools  # noqa: E402
from agents.eda_analysis_agent import (  # noqa: E402
    create_eda_analysis_agent,
    register_eda_analysis_tools,
)

# ---------------------------------------------------------------------------
# Data file
# ---------------------------------------------------------------------------
IRIS_CSV = PROJECT_ROOT / "test_data" / "iris.csv"
assert IRIS_CSV.exists(), f"Missing test data: {IRIS_CSV}"

# ---------------------------------------------------------------------------
# 1. Create agents
# ---------------------------------------------------------------------------

# Initializer — sends the task, never speaks again
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
    code_execution_config=False,
)

# Pipeline agents
data_prep_agent = create_data_prep_agent()
eda_analysis_agent = create_eda_analysis_agent()

# Dedicated executors (one per stage)
data_prep_executor = UserProxyAgent(
    name="DataPrepExecutor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
    code_execution_config=False,
)

eda_executor = UserProxyAgent(
    name="EDAAnalysisExecutor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
    code_execution_config=False,
)

# ---------------------------------------------------------------------------
# 2. Register tools on executors (not on user_proxy!)
# ---------------------------------------------------------------------------
register_data_prep_tools(data_prep_agent, data_prep_executor)
register_eda_analysis_tools(eda_analysis_agent, eda_executor)

print(f"DataPrepExecutor tools: {list(data_prep_executor._function_map.keys())}")
print(f"EDAAnalysisExecutor tools: {list(eda_executor._function_map.keys())}")


# ---------------------------------------------------------------------------
# 3. StateFlow router — Agent decides
# ---------------------------------------------------------------------------

def _has_tool_call(message: dict) -> bool:
    """Return True if message requests a tool / function call."""
    if message.get("tool_calls"):
        return True
    if message.get("function_call"):
        return True
    return False


def state_flow_agent_decision(last_speaker, groupchat):
    """Agent-as-decision-point: tool_call → executor; text → advance."""
    messages = groupchat.messages
    name = last_speaker.name

    # Initializer → DataPrepAgent
    if name == "user_proxy":
        return data_prep_agent

    # DataPrepAgent: tool_call → executor; text → advance to EDA
    if name == "DataPrepAgent":
        if messages and _has_tool_call(messages[-1]):
            return data_prep_executor
        return eda_analysis_agent

    # DataPrepExecutor → always back to DataPrepAgent
    if name == "DataPrepExecutor":
        return data_prep_agent

    # EDAAnalysisAgent: tool_call → executor; text → end
    if name == "EDAAnalysisAgent":
        if messages and _has_tool_call(messages[-1]):
            return eda_executor
        return None  # end of pipeline

    # EDAAnalysisExecutor → always back to EDAAnalysisAgent
    if name == "EDAAnalysisExecutor":
        return eda_analysis_agent

    return None


# ---------------------------------------------------------------------------
# 4. Build GroupChat and run
# ---------------------------------------------------------------------------

groupchat = GroupChat(
    agents=[user_proxy, data_prep_agent, data_prep_executor,
            eda_analysis_agent, eda_executor],
    messages=[],
    max_round=30,
    speaker_selection_method=state_flow_agent_decision,
)

manager = GroupChatManager(groupchat=groupchat, name="chat_manager")

print("\n" + "=" * 70)
print("APPROACH A: Agent-as-decision-point")
print("=" * 70 + "\n")

user_proxy.initiate_chat(
    manager,
    message=(
        f"Please run the full data preparation and EDA pipeline on:\n"
        f"{IRIS_CSV}"
    ),
)

print("\n" + "=" * 70)
print("APPROACH A COMPLETE")
print(f"Total messages: {len(groupchat.messages)}")
print("=" * 70)
