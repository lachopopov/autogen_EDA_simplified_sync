"""
orchestrator.py — GroupChat + GroupChatManager + state_flow_transition router.

Architecture Reference: architecture.md § 4.1, § 5, § 6, § 11

Wires all agents into a deterministic pipeline using the AG2 StateFlow
pattern.  Each pipeline stage is an (AssistantAgent, executor) pair:

  user_proxy (init)
    → DataPrepAgent ⇄ DataPrepExecutor → (text)
    → EDAAnalysisAgent ⇄ EDAAnalysisExecutor → (text)
    → VisualizationAgent ⇄ VisualizationExecutor → (text)
    → CriticAgent ⇄ CriticExecutor → (text)
    → FindingsGeneratorAgent ⇄ FindingsGeneratorExecutor → (text)
      ↻ CriticAgent (≤ 2 cycles)
    → ReportExporterAgent ⇄ ReportExporterExecutor → (text)
    → TERMINATE

Each AssistantAgent suggests tool calls.  Its dedicated executor
UserProxyAgent executes them (tools are in executor._function_map).
When the AssistantAgent sends text (no tool calls), the router
advances to the next pipeline stage.

Three-Layer Termination Guard (§ 5):
  Layer 1 — Keyword:   is_termination_msg checks for "TERMINATE"
  Layer 2 — GroupChat:  max_round (default 50, configurable via MAX_ROUNDS)
  Layer 3 — State:      get_critic_status() iteration ≥ 2 → force to ReportExporter

AG2 Version: 0.10.3
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

from agents.critic_agent import create_critic_agent, register_critic_tools
from agents.data_prep_agent import create_data_prep_agent, register_data_prep_tools
from agents.eda_analysis_agent import (
    create_eda_analysis_agent,
    register_eda_analysis_tools,
)
from agents.findings_generator_agent import (
    create_findings_generator_agent,
    register_findings_generator_tools,
)
from agents.report_exporter_agent import (
    create_report_exporter_agent,
    register_report_exporter_tools,
)
from agents.visualization_agent import (
    create_visualization_agent,
    register_visualization_tools,
)
from config import MAX_ROUNDS
from core import metrics
from eda_state import get_critic_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Creation
# ---------------------------------------------------------------------------


def _create_user_proxy() -> UserProxyAgent:
    """
    Create the UserProxyAgent — conversation initiator.

    Architecture § 4.1:
      - human_input_mode="NEVER": fully autonomous
      - max_consecutive_auto_reply=0: only sends the initial message
      - code_execution_config=False: no code execution
      - is_termination_msg: Layer 1 keyword check for "TERMINATE"
    """
    return UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
        code_execution_config=False,
    )


def _create_executor(name: str) -> UserProxyAgent:
    """
    Create an executor UserProxyAgent — executes tool calls for one stage.

    Each AssistantAgent in the pipeline has a dedicated executor that
    holds the tools (via _function_map) for that stage.
    The executor auto-replies with tool results and never asks for
    human input.
    """
    return UserProxyAgent(
        name=name,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
        code_execution_config=False,
    )


# ---------------------------------------------------------------------------
# StateFlow Router — architecture.md § 6
# ---------------------------------------------------------------------------


def _make_state_flow_transition(
    *,
    user_proxy: UserProxyAgent,
    data_prep_agent: AssistantAgent,
    data_prep_executor: UserProxyAgent,
    eda_analysis_agent: AssistantAgent,
    eda_executor: UserProxyAgent,
    visualization_agent: AssistantAgent,
    viz_executor: UserProxyAgent,
    critic_agent: AssistantAgent,
    critic_executor: UserProxyAgent,
    findings_generator_agent: AssistantAgent,
    findings_executor: UserProxyAgent,
    report_exporter_agent: AssistantAgent,
    report_executor: UserProxyAgent,
):
    """
    Factory: creates a state_flow_transition closure with agent references.

    Returns a callable with the signature expected by GroupChat's
    speaker_selection_method parameter:
        (last_speaker, groupchat) -> Agent | None

    AG2 StateFlow pattern (per the reference notebook):
      Each pipeline stage is an (AssistantAgent, executor) pair.
        - AssistantAgent sends tool_calls → route to its executor
        - Executor executes tool → route back to its AssistantAgent
        - AssistantAgent sends text (no tool calls) → advance pipeline

    The router is deterministic — no LLM call for speaker selection.

    Termination signals:
      - Returning None after ReportExporterAgent text → end conversation
      - get_critic_status() iteration >= 2 → force to ReportExporter
        (Layer 3 of the termination guard, § 5).
    """

    def _has_tool_call(message: dict) -> bool:
        """Return True if *message* requests a tool / function call."""
        if message.get("tool_calls"):
            return True
        return bool(message.get("function_call"))

    # Per-agent wall-clock timers.
    # Maps agent name → perf_counter() value recorded when the agent was
    # first routed to.  Cleared when the agent's span is written.
    # Closure-local — one dict per build_group_chat() call, thread-safe
    # for single-threaded AG2 groupchat execution.
    _agent_timers: dict[str, float] = {}

    def _start_agent(name: str) -> None:
        _agent_timers[name] = time.perf_counter()

    def _finish_agent(name: str) -> None:
        start = _agent_timers.pop(name, None)
        if start is not None:
            metrics.record_span(
                f"agent.{name}",
                (time.perf_counter() - start) * 1000,
            )

    def flush_agent_timers() -> None:
        """Write spans for any agents still in-flight when the groupchat ends.

        AG2 terminates the conversation (via ``is_termination_msg``) before
        calling ``state_flow_transition`` for the final ReportExporterAgent
        message.  This means ``_finish_agent("ReportExporterAgent")`` is
        never reached from the router.  Call this function immediately after
        ``user_proxy.initiate_chat()`` returns to capture any pending spans.
        """
        for agent_name in list(_agent_timers):
            _finish_agent(agent_name)

    def state_flow_transition(last_speaker, groupchat):
        """Deterministic speaker selection using (agent, executor) pairs."""
        messages = groupchat.messages
        name = last_speaker.name

        # ── Initializer: start pipeline ────────────────────────────
        if name == "user_proxy":
            _start_agent("DataPrepAgent")
            return data_prep_agent

        # ── DataPrepAgent ⇄ DataPrepExecutor ───────────────────────
        if name == "DataPrepAgent":
            if messages and _has_tool_call(messages[-1]):
                return data_prep_executor
            _finish_agent("DataPrepAgent")
            _start_agent("EDAAnalysisAgent")
            return eda_analysis_agent

        if name == "DataPrepExecutor":
            return data_prep_agent

        # ── EDAAnalysisAgent ⇄ EDAAnalysisExecutor ─────────────────
        if name == "EDAAnalysisAgent":
            if messages and _has_tool_call(messages[-1]):
                return eda_executor
            _finish_agent("EDAAnalysisAgent")
            _start_agent("VisualizationAgent")
            return visualization_agent

        if name == "EDAAnalysisExecutor":
            return eda_analysis_agent

        # ── VisualizationAgent ⇄ VisualizationExecutor ─────────────
        if name == "VisualizationAgent":
            if messages and _has_tool_call(messages[-1]):
                return viz_executor
            _finish_agent("VisualizationAgent")
            _start_agent("CriticAgent")
            return critic_agent

        if name == "VisualizationExecutor":
            return visualization_agent

        # ── CriticAgent ⇄ CriticExecutor ───────────────────────────
        if name == "CriticAgent":
            if messages and _has_tool_call(messages[-1]):
                return critic_executor
            _finish_agent("CriticAgent")
            _start_agent("FindingsGeneratorAgent")
            return findings_generator_agent

        if name == "CriticExecutor":
            return critic_agent

        # ── FindingsGeneratorAgent ⇄ FindingsGeneratorExecutor ─────
        if name == "FindingsGeneratorAgent":
            if messages and _has_tool_call(messages[-1]):
                return findings_executor
            # Decision: critic loop or advance to report
            status, iteration = get_critic_status(messages)

            if status == "REVISION_NEEDED" and iteration < 2:
                logger.info(
                    "Critic loop: REVISION_NEEDED at iteration %d → "
                    "routing back to CriticAgent",
                    iteration,
                )
                _finish_agent("FindingsGeneratorAgent")
                _start_agent("CriticAgent")
                return critic_agent  # loop back

            if status == "REVISION_NEEDED" and iteration >= 2:
                logger.warning(
                    "Critic loop force-exit: iteration %d >= 2 → "
                    "routing to ReportExporterAgent (Layer 3 guard)",
                    iteration,
                )

            _finish_agent("FindingsGeneratorAgent")
            _start_agent("ReportExporterAgent")
            return report_exporter_agent  # approved or forced termination

        if name == "FindingsGeneratorExecutor":
            return findings_generator_agent

        # ── ReportExporterAgent ⇄ ReportExporterExecutor ───────────
        if name == "ReportExporterAgent":
            if messages and _has_tool_call(messages[-1]):
                return report_executor
            _finish_agent("ReportExporterAgent")
            return None  # end of conversation

        if name == "ReportExporterExecutor":
            return report_exporter_agent

        # Safety: unknown speaker → stop
        logger.error("Unknown speaker '%s' — returning None", name)
        return None

    return state_flow_transition, flush_agent_timers


# ---------------------------------------------------------------------------
# Build GroupChat + GroupChatManager
# ---------------------------------------------------------------------------


def build_group_chat() -> tuple[
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    dict[str, AssistantAgent],
    dict[str, UserProxyAgent],
    list[AssistantAgent],
    Callable[[], None],
]:
    """
    Wire all agents, register all tools, and build the GroupChat.

    Returns:
        (groupchat, manager, user_proxy, agents_dict, executors_dict,
         agents_list, flush_agent_timers)

        agents_dict maps AssistantAgent names to instances.
        executors_dict maps executor names to their UserProxyAgent instances.
        agents_list is a list of all AssistantAgent instances (for cost tracking).
        flush_agent_timers() must be called immediately after
        ``user_proxy.initiate_chat()`` returns; it writes the wall-clock span
        for any agent (typically ReportExporterAgent) that was in-flight when
        AG2 terminated the groupchat before the router could call _finish_agent.

    Architecture References:
      - § 4.1: UserProxyAgent config
      - § 5: Three-layer termination guard
      - § 6: state_flow_transition router
      - § 11: Chained-decorator tool registration
    """
    # --- 1. Create agents ---
    user_proxy = _create_user_proxy()

    data_prep = create_data_prep_agent()
    eda_analysis = create_eda_analysis_agent()
    visualization = create_visualization_agent()
    critic = create_critic_agent()
    findings = create_findings_generator_agent()
    report_exporter = create_report_exporter_agent()

    agents_dict = {
        "DataPrepAgent": data_prep,
        "EDAAnalysisAgent": eda_analysis,
        "VisualizationAgent": visualization,
        "CriticAgent": critic,
        "FindingsGeneratorAgent": findings,
        "ReportExporterAgent": report_exporter,
    }

    # --- 2. Create executor proxies (one per AssistantAgent) ---
    data_prep_executor = _create_executor("DataPrepExecutor")
    eda_executor = _create_executor("EDAAnalysisExecutor")
    viz_executor = _create_executor("VisualizationExecutor")
    critic_executor = _create_executor("CriticExecutor")
    findings_executor = _create_executor("FindingsGeneratorExecutor")
    report_executor = _create_executor("ReportExporterExecutor")

    executors_dict = {
        "DataPrepExecutor": data_prep_executor,
        "EDAAnalysisExecutor": eda_executor,
        "VisualizationExecutor": viz_executor,
        "CriticExecutor": critic_executor,
        "FindingsGeneratorExecutor": findings_executor,
        "ReportExporterExecutor": report_executor,
    }

    # --- 3. Register tools on each executor (P6: chained-decorator) ---
    register_data_prep_tools(data_prep, data_prep_executor)
    register_eda_analysis_tools(eda_analysis, eda_executor)
    register_visualization_tools(visualization, viz_executor)
    register_critic_tools(critic, critic_executor)
    register_findings_generator_tools(findings, findings_executor)
    register_report_exporter_tools(report_exporter, report_executor)

    for exec_name, executor in executors_dict.items():
        logger.info(
            "Registered %d tools on %s: %s",
            len(executor._function_map),
            exec_name,
            list(executor._function_map.keys()),
        )

    # --- 4. Build state_flow_transition router (§ 6) ---
    router, flush_agent_timers = _make_state_flow_transition(
        user_proxy=user_proxy,
        data_prep_agent=data_prep,
        data_prep_executor=data_prep_executor,
        eda_analysis_agent=eda_analysis,
        eda_executor=eda_executor,
        visualization_agent=visualization,
        viz_executor=viz_executor,
        critic_agent=critic,
        critic_executor=critic_executor,
        findings_generator_agent=findings,
        findings_executor=findings_executor,
        report_exporter_agent=report_exporter,
        report_executor=report_executor,
    )

    # --- 5. Build GroupChat (§ 5, Layer 2: max_round) ---
    all_agents = [
        user_proxy,
        data_prep, data_prep_executor,
        eda_analysis, eda_executor,
        visualization, viz_executor,
        critic, critic_executor,
        findings, findings_executor,
        report_exporter, report_executor,
    ]

    groupchat = GroupChat(
        agents=all_agents,
        messages=[],
        max_round=MAX_ROUNDS,              # Layer 2 — absolute ceiling
        speaker_selection_method=router,    # § 6 — deterministic routing
    )

    # --- 6. Build GroupChatManager ---
    manager = GroupChatManager(
        groupchat=groupchat,
        name="chat_manager",
    )

    logger.info(
        "GroupChat built: %d agents, max_round=%d",
        len(all_agents),
        MAX_ROUNDS,
    )

    # Build agents_list for cost tracking (AssistantAgents only, not executors)
    agents_list = [
        data_prep,
        eda_analysis,
        visualization,
        critic,
        findings,
        report_exporter,
    ]

    return groupchat, manager, user_proxy, agents_dict, executors_dict, agents_list, flush_agent_timers
