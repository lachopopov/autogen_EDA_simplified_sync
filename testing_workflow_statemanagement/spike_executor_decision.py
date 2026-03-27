"""
spike_executor_decision.py — Approach B: Executor-as-decision-point.

Minimal 2-stage pipeline: DataPrep → EDA on iris.csv.

Pattern (mirrors the reference notebook):
  - AssistantAgent ALWAYS routes to its executor (unconditional)
  - Executor runs the tool, then DECIDES:
    - error in result → retry (back to AssistantAgent)
    - success → back to AssistantAgent for more work
  - To advance pipeline, the AssistantAgent must send a specific
    sentinel phrase ("DATA_PREP_COMPLETE" / "EDA_COMPLETE") which
    the router detects.

Run:
    conda run -n ag2_env python testing_workflow_statemanagement/spike_executor_decision.py
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
# 1. Create agents (with modified system messages for sentinel phrases)
# ---------------------------------------------------------------------------

# Initializer — sends the task, never speaks again
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
    code_execution_config=False,
)

# Use existing agent factories (system messages already instruct tool usage)
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
# 2. Register tools on executors
# ---------------------------------------------------------------------------
register_data_prep_tools(data_prep_agent, data_prep_executor)
register_eda_analysis_tools(eda_analysis_agent, eda_executor)

print(f"DataPrepExecutor tools: {list(data_prep_executor._function_map.keys())}")
print(f"EDAAnalysisExecutor tools: {list(eda_executor._function_map.keys())}")


# ---------------------------------------------------------------------------
# 3. StateFlow router — Executor decides
# ---------------------------------------------------------------------------

def _has_tool_call(message: dict) -> bool:
    """Return True if message requests a tool / function call."""
    if message.get("tool_calls"):
        return True
    if message.get("function_call"):
        return True
    return False


def state_flow_executor_decision(last_speaker, groupchat):
    """Executor-as-decision-point: mimics the reference notebook pattern."""
    messages = groupchat.messages
    name = last_speaker.name
    last_content = messages[-1].get("content", "") if messages else ""

    # Initializer → DataPrepAgent
    if name == "user_proxy":
        return data_prep_agent

    # DataPrepAgent → ALWAYS to executor (unconditional, like coder→executor)
    if name == "DataPrepAgent":
        if messages and _has_tool_call(messages[-1]):
            # Has a tool call → executor runs it
            return data_prep_executor
        else:
            # Text message (no tool call) — agent is done with tools
            # Advance to next stage
            return eda_analysis_agent

    # DataPrepExecutor → DECIDES based on result
    if name == "DataPrepExecutor":
        if last_content and ("error" in last_content.lower()
                             or "exitcode: 1" in last_content):
            # Error → retry: send back to assistant to fix
            print("  [EXECUTOR DECISION] DataPrepExecutor: ERROR detected → retry")
            return data_prep_agent
        else:
            # Success → return to agent for next tool or summary
            print("  [EXECUTOR DECISION] DataPrepExecutor: SUCCESS → back to agent")
            return data_prep_agent

    # EDAAnalysisAgent → to executor or end
    if name == "EDAAnalysisAgent":
        if messages and _has_tool_call(messages[-1]):
            return eda_executor
        else:
            # Text message → done
            return None

    # EDAAnalysisExecutor → DECIDES based on result
    if name == "EDAAnalysisExecutor":
        if last_content and ("error" in last_content.lower()
                             or "exitcode: 1" in last_content):
            print("  [EXECUTOR DECISION] EDAExecutor: ERROR detected → retry")
            return eda_analysis_agent
        else:
            print("  [EXECUTOR DECISION] EDAExecutor: SUCCESS → back to agent")
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
    speaker_selection_method=state_flow_executor_decision,
)

manager = GroupChatManager(groupchat=groupchat, name="chat_manager")

print("\n" + "=" * 70)
print("APPROACH B: Executor-as-decision-point")
print("=" * 70 + "\n")

user_proxy.initiate_chat(
    manager,
    message=(
        f"Please run the full data preparation and EDA pipeline on:\n"
        f"{IRIS_CSV}"
    ),
)

print("\n" + "=" * 70)
print("APPROACH B COMPLETE")
print(f"Total messages: {len(groupchat.messages)}")
print("=" * 70)
