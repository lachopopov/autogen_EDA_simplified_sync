"""
tests/test_orchestrator.py — Unit tests for orchestrator.py

Tests the GroupChat assembly, state_flow_transition router, tool registration
on dedicated executors, and three-layer termination guard.
NO LLM calls — uses mock messages and direct function invocation.

Architecture: each pipeline stage is (AssistantAgent, executor UserProxyAgent)
pair. Tools are registered on the executor, not the shared user_proxy.
"""

import pytest
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

from orchestrator import (
    _create_executor,
    _create_user_proxy,
    _make_state_flow_transition,
    build_group_chat,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def group_chat_components():
    """Build the full GroupChat and return all components (6-tuple)."""
    groupchat, manager, user_proxy, agents_dict, executors_dict, agents_list, flush_agent_timers = build_group_chat()
    return groupchat, manager, user_proxy, agents_dict, executors_dict, agents_list


@pytest.fixture()
def user_proxy():
    """A standalone UserProxyAgent (conversation initiator)."""
    return _create_user_proxy()


@pytest.fixture()
def mock_agents():
    """Create lightweight mock agents for routing tests."""
    names = [
        "DataPrepAgent",
        "EDAAnalysisAgent",
        "VisualizationAgent",
        "CriticAgent",
        "FindingsGeneratorAgent",
        "ReportExporterAgent",
    ]
    agents = {}
    for name in names:
        agents[name] = type("MockAgent", (), {"name": name})()
    return agents


@pytest.fixture()
def mock_executors():
    """Create lightweight mock executors for routing tests."""
    names = [
        "DataPrepExecutor",
        "EDAAnalysisExecutor",
        "VisualizationExecutor",
        "CriticExecutor",
        "FindingsGeneratorExecutor",
        "ReportExporterExecutor",
    ]
    executors = {}
    for name in names:
        executors[name] = type("MockExecutor", (), {"name": name})()
    return executors


@pytest.fixture()
def mock_user_proxy():
    """A mock user_proxy for routing tests."""
    return type("MockAgent", (), {"name": "user_proxy"})()


@pytest.fixture()
def router(mock_agents, mock_executors, mock_user_proxy):
    """A state_flow_transition function with mock agents and executors."""
    router_fn, _flush = _make_state_flow_transition(
        user_proxy=mock_user_proxy,
        data_prep_agent=mock_agents["DataPrepAgent"],
        data_prep_executor=mock_executors["DataPrepExecutor"],
        eda_analysis_agent=mock_agents["EDAAnalysisAgent"],
        eda_executor=mock_executors["EDAAnalysisExecutor"],
        visualization_agent=mock_agents["VisualizationAgent"],
        viz_executor=mock_executors["VisualizationExecutor"],
        critic_agent=mock_agents["CriticAgent"],
        critic_executor=mock_executors["CriticExecutor"],
        findings_generator_agent=mock_agents["FindingsGeneratorAgent"],
        findings_executor=mock_executors["FindingsGeneratorExecutor"],
        report_exporter_agent=mock_agents["ReportExporterAgent"],
        report_executor=mock_executors["ReportExporterExecutor"],
    )
    return router_fn


@pytest.fixture()
def mock_groupchat():
    """A GroupChat-like object with a messages list for router tests."""
    return type("MockGroupChat", (), {"messages": []})()


# ---------------------------------------------------------------------------
# _create_user_proxy()
# ---------------------------------------------------------------------------

class TestCreateUserProxy:
    """Test UserProxyAgent creation (conversation initiator)."""

    def test_name(self, user_proxy):
        assert user_proxy.name == "user_proxy"

    def test_human_input_mode(self, user_proxy):
        assert user_proxy.human_input_mode == "NEVER"

    def test_max_consecutive_auto_reply(self, user_proxy):
        """Initiator only sends one message, max_consecutive_auto_reply=0."""
        assert user_proxy._max_consecutive_auto_reply == 0

    def test_code_execution_disabled(self, user_proxy):
        assert user_proxy._code_execution_config is False

    def test_is_user_proxy_agent(self, user_proxy):
        assert isinstance(user_proxy, UserProxyAgent)

    def test_termination_msg_positive(self, user_proxy):
        assert user_proxy._is_termination_msg({"content": "Done. TERMINATE"}) is True

    def test_termination_msg_negative(self, user_proxy):
        assert user_proxy._is_termination_msg({"content": "keep going"}) is False

    def test_termination_msg_none_content(self, user_proxy):
        assert user_proxy._is_termination_msg({"content": None}) is False


# ---------------------------------------------------------------------------
# _create_executor()
# ---------------------------------------------------------------------------

class TestCreateExecutor:
    """Test executor UserProxyAgent creation (tool runner for each stage)."""

    def test_name(self):
        executor = _create_executor("TestExecutor")
        assert executor.name == "TestExecutor"

    def test_human_input_mode(self):
        executor = _create_executor("TestExecutor")
        assert executor.human_input_mode == "NEVER"

    def test_max_consecutive_auto_reply(self):
        """Executors allow up to 20 auto-replies to absorb retries and critic-loop iterations."""
        executor = _create_executor("TestExecutor")
        assert executor._max_consecutive_auto_reply == 20

    def test_code_execution_disabled(self):
        executor = _create_executor("TestExecutor")
        assert executor._code_execution_config is False

    def test_is_user_proxy_agent(self):
        executor = _create_executor("TestExecutor")
        assert isinstance(executor, UserProxyAgent)

    def test_termination_msg_check(self):
        executor = _create_executor("TestExecutor")
        assert executor._is_termination_msg({"content": "TERMINATE"}) is True
        assert executor._is_termination_msg({"content": "ok"}) is False


# ---------------------------------------------------------------------------
# state_flow_transition — Linear Pipeline Routing (text, no tool calls)
# ---------------------------------------------------------------------------

class TestStateFlowLinearRouting:
    """Test deterministic linear routing when agents emit text (no tool calls)."""

    def test_user_proxy_to_data_prep(self, router, mock_groupchat):
        speaker = type("S", (), {"name": "user_proxy"})()
        result = router(speaker, mock_groupchat)
        assert result.name == "DataPrepAgent"

    def test_data_prep_to_eda(self, router, mock_groupchat):
        speaker = type("S", (), {"name": "DataPrepAgent"})()
        result = router(speaker, mock_groupchat)
        assert result.name == "EDAAnalysisAgent"

    def test_eda_to_visualization(self, router, mock_groupchat):
        speaker = type("S", (), {"name": "EDAAnalysisAgent"})()
        result = router(speaker, mock_groupchat)
        assert result.name == "VisualizationAgent"

    def test_visualization_to_critic(self, router, mock_groupchat):
        speaker = type("S", (), {"name": "VisualizationAgent"})()
        result = router(speaker, mock_groupchat)
        assert result.name == "CriticAgent"

    def test_critic_to_findings(self, router, mock_groupchat):
        speaker = type("S", (), {"name": "CriticAgent"})()
        result = router(speaker, mock_groupchat)
        assert result.name == "FindingsGeneratorAgent"

    def test_report_exporter_returns_none(self, router, mock_groupchat):
        """ReportExporterAgent → None signals end of conversation."""
        speaker = type("S", (), {"name": "ReportExporterAgent"})()
        result = router(speaker, mock_groupchat)
        assert result is None

    def test_unknown_speaker_returns_none(self, router, mock_groupchat):
        """Unknown speaker → None (safety fallback)."""
        speaker = type("S", (), {"name": "UnknownAgent"})()
        result = router(speaker, mock_groupchat)
        assert result is None


# ---------------------------------------------------------------------------
# state_flow_transition — Executor Round-Trip Routing
# ---------------------------------------------------------------------------

class TestStateFlowExecutorRouting:
    """Test executor routing: Executor always returns to its AssistantAgent."""

    @pytest.mark.parametrize("executor_name,expected_agent", [
        ("DataPrepExecutor", "DataPrepAgent"),
        ("EDAAnalysisExecutor", "EDAAnalysisAgent"),
        ("VisualizationExecutor", "VisualizationAgent"),
        ("CriticExecutor", "CriticAgent"),
        ("FindingsGeneratorExecutor", "FindingsGeneratorAgent"),
        ("ReportExporterExecutor", "ReportExporterAgent"),
    ])
    def test_executor_returns_to_agent(self, router, mock_groupchat,
                                        executor_name, expected_agent):
        speaker = type("S", (), {"name": executor_name})()
        result = router(speaker, mock_groupchat)
        assert result.name == expected_agent


# ---------------------------------------------------------------------------
# state_flow_transition — Tool-Call Routing (agent → executor)
# ---------------------------------------------------------------------------

class TestToolCallRouting:
    """Test tool-call routing: AssistantAgent → its dedicated executor."""

    @pytest.mark.parametrize("agent_name,expected_executor", [
        ("DataPrepAgent", "DataPrepExecutor"),
        ("EDAAnalysisAgent", "EDAAnalysisExecutor"),
        ("VisualizationAgent", "VisualizationExecutor"),
        ("CriticAgent", "CriticExecutor"),
        ("FindingsGeneratorAgent", "FindingsGeneratorExecutor"),
        ("ReportExporterAgent", "ReportExporterExecutor"),
    ])
    def test_tool_calls_route_to_executor(self, router, agent_name, expected_executor):
        """When agent sends tool_calls, route to its dedicated executor."""
        speaker = type("S", (), {"name": agent_name})()
        gc = type("GC", (), {
            "messages": [
                {"name": agent_name, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "some_tool", "arguments": "{}"}}
                ]}
            ]
        })()
        result = router(speaker, gc)
        assert result.name == expected_executor

    def test_function_call_legacy_format(self, router):
        """When agent sends function_call (legacy format), route to executor."""
        speaker = type("S", (), {"name": "EDAAnalysisAgent"})()
        gc = type("GC", (), {
            "messages": [
                {"name": "EDAAnalysisAgent",
                 "function_call": {"name": "describe_stats", "arguments": "{}"}}
            ]
        })()
        result = router(speaker, gc)
        assert result.name == "EDAAnalysisExecutor"

    def test_empty_tool_calls_advances_pipeline(self, router):
        """Empty tool_calls list = text message → advance pipeline."""
        speaker = type("S", (), {"name": "DataPrepAgent"})()
        gc = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "content": "Done.", "tool_calls": []}
            ]
        })()
        result = router(speaker, gc)
        assert result.name == "EDAAnalysisAgent"

    def test_sequential_tool_calls_round_trip(self, router):
        """Agent makes tool call → executor → agent → tool call → executor."""
        # Step 1: DataPrepAgent sends tool_call → DataPrepExecutor
        speaker1 = type("S", (), {"name": "DataPrepAgent"})()
        gc1 = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "load_data", "arguments": "{}"}}
                ]}
            ]
        })()
        assert router(speaker1, gc1).name == "DataPrepExecutor"

        # Step 2: DataPrepExecutor returns → DataPrepAgent
        speaker2 = type("S", (), {"name": "DataPrepExecutor"})()
        gc2 = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "load_data", "arguments": "{}"}}
                ]},
                {"name": "DataPrepExecutor", "content": "loaded data"},
            ]
        })()
        assert router(speaker2, gc2).name == "DataPrepAgent"

        # Step 3: DataPrepAgent sends another tool_call → DataPrepExecutor
        gc3 = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "load_data", "arguments": "{}"}}
                ]},
                {"name": "DataPrepExecutor", "content": "loaded data"},
                {"name": "DataPrepAgent", "tool_calls": [
                    {"id": "call_2", "type": "function",
                     "function": {"name": "validate_schema", "arguments": "{}"}}
                ]},
            ]
        })()
        assert router(speaker1, gc3).name == "DataPrepExecutor"

    def test_no_tool_call_advances_pipeline(self, router):
        """When agent sends text (no tool calls), advance to next stage."""
        speaker = type("S", (), {"name": "DataPrepAgent"})()
        gc = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "content": "Data preparation complete."}
            ]
        })()
        result = router(speaker, gc)
        assert result.name == "EDAAnalysisAgent"


# ---------------------------------------------------------------------------
# state_flow_transition — Critic Loop Logic
# ---------------------------------------------------------------------------

class TestStateFlowCriticLoop:
    """Test the critic loop decision at FindingsGeneratorAgent."""

    def test_approved_routes_to_report(self, router):
        """APPROVED → proceed to ReportExporterAgent."""
        gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "All checks passed. APPROVED"},
            ]
        })()
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = router(speaker, gc)
        assert result.name == "ReportExporterAgent"

    def test_revision_needed_iter1_loops_back(self, router):
        """REVISION_NEEDED at iteration 1 → loop back to CriticAgent."""
        gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "Issues found. REVISION_NEEDED"},
            ]
        })()
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = router(speaker, gc)
        assert result.name == "CriticAgent"

    def test_revision_needed_iter2_forces_report(self, router):
        """REVISION_NEEDED at iteration >= 2 → force to ReportExporterAgent (Layer 3)."""
        gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "Issues found. REVISION_NEEDED"},
                {"name": "FindingsGeneratorAgent", "content": "Revised findings."},
                {"name": "CriticAgent", "content": "Still issues. REVISION_NEEDED"},
            ]
        })()
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = router(speaker, gc)
        assert result.name == "ReportExporterAgent"

    def test_pending_routes_to_report(self, router, mock_groupchat):
        """PENDING (no critic messages) → proceed to ReportExporterAgent."""
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = router(speaker, mock_groupchat)
        assert result.name == "ReportExporterAgent"

    def test_exactly_two_iterations_forces_exit(self, router):
        """Exactly iteration == 2 with REVISION_NEEDED → force exit."""
        gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
            ]
        })()
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = router(speaker, gc)
        assert result.name == "ReportExporterAgent"

    def test_approved_after_one_loop(self, router):
        """APPROVED on second iteration → proceed normally."""
        gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
                {"name": "FindingsGeneratorAgent", "content": "Revised."},
                {"name": "CriticAgent", "content": "Looks good. APPROVED"},
            ]
        })()
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = router(speaker, gc)
        assert result.name == "ReportExporterAgent"


# ---------------------------------------------------------------------------
# build_group_chat() — Assembly
# ---------------------------------------------------------------------------

class TestBuildGroupChat:
    """Test the full GroupChat assembly."""

    def test_returns_6_tuple(self, group_chat_components):
        groupchat, manager, user_proxy, agents_dict, executors_dict, agents_list = group_chat_components
        assert isinstance(groupchat, GroupChat)
        assert isinstance(manager, GroupChatManager)
        assert isinstance(user_proxy, UserProxyAgent)
        assert isinstance(agents_dict, dict)
        assert isinstance(executors_dict, dict)
        assert isinstance(agents_list, list)

    def test_agent_count(self, group_chat_components):
        """13 agents in GroupChat: 1 user_proxy + 6 AssistantAgents + 6 executors."""
        groupchat, _, _, _, _, _ = group_chat_components
        assert len(groupchat.agents) == 13

    def test_agent_names(self, group_chat_components):
        """All expected agent and executor names are present."""
        groupchat, _, _, _, _, _ = group_chat_components
        names = {a.name for a in groupchat.agents}
        expected = {
            "user_proxy",
            "DataPrepAgent", "DataPrepExecutor",
            "EDAAnalysisAgent", "EDAAnalysisExecutor",
            "VisualizationAgent", "VisualizationExecutor",
            "CriticAgent", "CriticExecutor",
            "FindingsGeneratorAgent", "FindingsGeneratorExecutor",
            "ReportExporterAgent", "ReportExporterExecutor",
        }
        assert names == expected

    def test_agents_dict_has_six_entries(self, group_chat_components):
        """agents_dict has 6 AssistantAgents (excludes user_proxy and executors)."""
        _, _, _, agents_dict, _, _ = group_chat_components
        assert len(agents_dict) == 6

    def test_executors_dict_has_six_entries(self, group_chat_components):
        """executors_dict has 6 executor UserProxyAgents."""
        _, _, _, _, executors_dict, _ = group_chat_components
        assert len(executors_dict) == 6

    def test_all_agents_are_assistant_agents(self, group_chat_components):
        """All pipeline agents are AssistantAgent instances."""
        _, _, _, agents_dict, _, _ = group_chat_components
        for name, agent in agents_dict.items():
            assert isinstance(agent, AssistantAgent), f"{name} is not AssistantAgent"

    def test_all_executors_are_user_proxy_agents(self, group_chat_components):
        """All executors are UserProxyAgent instances."""
        _, _, _, _, executors_dict, _ = group_chat_components
        for name, executor in executors_dict.items():
            assert isinstance(executor, UserProxyAgent), f"{name} is not UserProxyAgent"

    def test_max_round(self, group_chat_components):
        """GroupChat max_round matches the MAX_ROUNDS used at build time."""
        groupchat, _, _, _, _, _ = group_chat_components
        from orchestrator import MAX_ROUNDS as ORCHESTRATOR_MAX_ROUNDS
        assert groupchat.max_round == ORCHESTRATOR_MAX_ROUNDS

    def test_max_round_positive(self, group_chat_components):
        """MAX_ROUNDS is a positive integer (Layer 2 guard)."""
        groupchat, _, _, _, _, _ = group_chat_components
        assert groupchat.max_round > 0

    def test_speaker_selection_is_callable(self, group_chat_components):
        """speaker_selection_method is a callable (our router function)."""
        groupchat, _, _, _, _, _ = group_chat_components
        assert callable(groupchat.speaker_selection_method)

    def test_manager_name(self, group_chat_components):
        """GroupChatManager has the expected name."""
        _, manager, _, _, _, _ = group_chat_components
        assert manager.name == "chat_manager"


# ---------------------------------------------------------------------------
# Tool Registration — Tools wired to dedicated executors
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Verify tools are registered on their dedicated executor (not user_proxy)."""

    def test_user_proxy_has_no_tools(self, group_chat_components):
        """user_proxy (initiator) should have no tools registered."""
        _, _, user_proxy, _, _, _ = group_chat_components
        assert len(user_proxy._function_map) == 0

    def test_total_tool_count_across_executors(self, group_chat_components):
        """20 total tools distributed across 6 executors (render_ipynb excluded when IPYNB_EXPORT=false)."""
        _, _, _, _, executors_dict, _ = group_chat_components
        total = sum(len(e._function_map) for e in executors_dict.values())
        assert total == 22

    @pytest.mark.parametrize("executor_name,expected_tools", [
        ("DataPrepExecutor", {"load_data", "validate_schema", "infer_dtypes"}),
        ("EDAAnalysisExecutor", {"describe_stats", "missing_analysis", "correlation_matrix", "target_analysis", "analyze_categoricals", "compute_feature_target_associations"}),
        ("VisualizationExecutor", {"plot_histograms", "plot_correlation_heatmap", "plot_missing_heatmap", "plot_class_distribution", "plot_categorical_bars", "plot_ordinal_heatmap", "plot_feature_target_bars"}),
        ("CriticExecutor", {"run_critic_rules"}),
        ("FindingsGeneratorExecutor", {"assemble_findings", "prepare_interpretation_context", "save_interpretations"}),
        ("ReportExporterExecutor", {"render_pdf", "render_markdown"}),  # render_ipynb excluded when IPYNB_EXPORT=false
    ])
    def test_executor_has_correct_tools(self, group_chat_components,
                                         executor_name, expected_tools):
        _, _, _, _, executors_dict, _ = group_chat_components
        executor = executors_dict[executor_name]
        registered = set(executor._function_map.keys())
        assert registered == expected_tools, (
            f"{executor_name}: expected {expected_tools}, got {registered}"
        )

    def test_each_tool_is_callable(self, group_chat_components):
        """Every registered tool on every executor is a callable."""
        _, _, _, _, executors_dict, _ = group_chat_components
        for exec_name, executor in executors_dict.items():
            for tool_name, fn in executor._function_map.items():
                assert callable(fn), f"{exec_name}/{tool_name} is not callable"


# ---------------------------------------------------------------------------
# Agent LLM Config — Each agent has its own tools in llm_config
# ---------------------------------------------------------------------------

class TestAgentLLMConfig:
    """Verify each agent's llm_config has the correct tool schemas."""

    def _get_tool_names(self, agent):
        tools = agent.llm_config.get("tools", [])
        return {t["function"]["name"] for t in tools}

    def test_data_prep_has_3_tools(self, group_chat_components):
        _, _, _, agents, _, _ = group_chat_components
        names = self._get_tool_names(agents["DataPrepAgent"])
        assert names == {"load_data", "validate_schema", "infer_dtypes"}

    def test_eda_analysis_has_5_tools(self, group_chat_components):
        _, _, _, agents, _, _ = group_chat_components
        names = self._get_tool_names(agents["EDAAnalysisAgent"])
        assert names == {"describe_stats", "missing_analysis", "correlation_matrix", "target_analysis", "analyze_categoricals", "compute_feature_target_associations"}

    def test_visualization_has_7_tools(self, group_chat_components):
        _, _, _, agents, _, _ = group_chat_components
        names = self._get_tool_names(agents["VisualizationAgent"])
        assert names == {"plot_histograms", "plot_correlation_heatmap", "plot_missing_heatmap", "plot_class_distribution", "plot_categorical_bars", "plot_ordinal_heatmap", "plot_feature_target_bars"}

    def test_critic_has_1_tool(self, group_chat_components):
        _, _, _, agents, _, _ = group_chat_components
        names = self._get_tool_names(agents["CriticAgent"])
        assert names == {"run_critic_rules"}

    def test_findings_has_3_tools(self, group_chat_components):
        _, _, _, agents, _, _ = group_chat_components
        names = self._get_tool_names(agents["FindingsGeneratorAgent"])
        assert names == {"assemble_findings", "prepare_interpretation_context", "save_interpretations"}

    def test_report_has_2_tools(self, group_chat_components):
        """When IPYNB_EXPORT=false (default), ReportExporterAgent has 2 tools."""
        _, _, _, agents, _, _ = group_chat_components
        names = self._get_tool_names(agents["ReportExporterAgent"])
        assert names == {"render_pdf", "render_markdown"}


# ---------------------------------------------------------------------------
# Three-Layer Termination Guard
# ---------------------------------------------------------------------------

class TestThreeLayerTerminationGuard:
    """Verify all three termination layers are wired correctly."""

    def test_layer1_keyword_on_user_proxy(self, group_chat_components):
        """Layer 1: user_proxy checks for TERMINATE keyword."""
        _, _, user_proxy, _, _, _ = group_chat_components
        assert user_proxy._is_termination_msg({"content": "TERMINATE"}) is True
        assert user_proxy._is_termination_msg({"content": "ok"}) is False

    def test_layer1_keyword_on_executors(self, group_chat_components):
        """Layer 1: all executors also check for TERMINATE keyword."""
        _, _, _, _, executors_dict, _ = group_chat_components
        for name, executor in executors_dict.items():
            assert executor._is_termination_msg({"content": "TERMINATE"}) is True, (
                f"{name} should detect TERMINATE"
            )
            assert executor._is_termination_msg({"content": "ok"}) is False

    def test_layer2_max_round(self, group_chat_components):
        """Layer 2: GroupChat max_round is set (positive integer ceiling)."""
        groupchat, _, _, _, _, _ = group_chat_components
        assert groupchat.max_round > 0

    def test_layer3_critic_force_exit(self, group_chat_components):
        """Layer 3: router forces to ReportExporter when iteration >= 2."""
        groupchat, _, _, _, _, _ = group_chat_components
        mock_gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
            ]
        })()
        speaker = type("S", (), {"name": "FindingsGeneratorAgent"})()
        result = groupchat.speaker_selection_method(speaker, mock_gc)
        assert result.name == "ReportExporterAgent"

    def test_only_report_exporter_has_terminate(self, group_chat_components):
        """Only ReportExporterAgent's system_message contains 'TERMINATE' instruction."""
        _, _, _, agents, _, _ = group_chat_components
        for name, agent in agents.items():
            if name == "ReportExporterAgent":
                assert "TERMINATE" in agent.system_message
                assert "Do NOT include the word TERMINATE" not in agent.system_message
            else:
                assert "Do NOT include the word TERMINATE" in agent.system_message

    def test_all_assistant_agents_have_max_auto_reply_10(self, group_chat_components):
        """All AssistantAgents have max_consecutive_auto_reply=10 (3 tools + retries + final text)."""
        _, _, _, agents, _, _ = group_chat_components
        for name, agent in agents.items():
            assert agent._max_consecutive_auto_reply == 10, (
                f"{name} max_consecutive_auto_reply != 10"
            )


# ---------------------------------------------------------------------------
# Chained Registration Invariant (P6)
# ---------------------------------------------------------------------------

class TestChainedRegistrationInvariant:
    """Every tool must appear in BOTH the agent's LLM tools AND executor function_map."""

    AGENT_EXECUTOR_PAIRS = {
        "DataPrepAgent": "DataPrepExecutor",
        "EDAAnalysisAgent": "EDAAnalysisExecutor",
        "VisualizationAgent": "VisualizationExecutor",
        "CriticAgent": "CriticExecutor",
        "FindingsGeneratorAgent": "FindingsGeneratorExecutor",
        "ReportExporterAgent": "ReportExporterExecutor",
    }

    def test_all_tools_in_both_sides(self, group_chat_components):
        """Each agent's LLM tools also appear in its executor's function_map."""
        _, _, _, agents, executors, _ = group_chat_components

        for agent_name, executor_name in self.AGENT_EXECUTOR_PAIRS.items():
            agent = agents[agent_name]
            executor = executors[executor_name]
            agent_tools = {
                t["function"]["name"]
                for t in agent.llm_config.get("tools", [])
            }
            fn_names = set(executor._function_map.keys())
            missing = agent_tools - fn_names
            assert not missing, (
                f"{agent_name}: tools {missing} in LLM config but not in "
                f"{executor_name} function_map"
            )

    def test_no_extra_tools_in_executors(self, group_chat_components):
        """Executors have no extra tools beyond what their agent declares."""
        _, _, _, agents, executors, _ = group_chat_components

        for agent_name, executor_name in self.AGENT_EXECUTOR_PAIRS.items():
            agent = agents[agent_name]
            executor = executors[executor_name]
            agent_tools = {
                t["function"]["name"]
                for t in agent.llm_config.get("tools", [])
            }
            fn_names = set(executor._function_map.keys())
            extra = fn_names - agent_tools
            assert not extra, (
                f"{executor_name}: extra tools {extra} not in {agent_name} LLM config"
            )


# ---------------------------------------------------------------------------
# Router End-to-End — Full pipeline sequence
# ---------------------------------------------------------------------------

class TestRouterEndToEnd:
    """Simulate a full happy-path pipeline through the router."""

    def test_happy_path_linear_sequence(self, group_chat_components):
        """Simulate the linear pipeline (text only, no tool calls):
        user_proxy → DataPrep → EDA → Viz → Critic → Findings → Report → None.
        """
        groupchat, _, _, _, _, _ = group_chat_components
        router = groupchat.speaker_selection_method

        speakers = {name: type("S", (), {"name": name})() for name in [
            "user_proxy", "DataPrepAgent", "EDAAnalysisAgent",
            "VisualizationAgent", "CriticAgent", "FindingsGeneratorAgent",
            "ReportExporterAgent",
        ]}

        mock_gc = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "All checks passed. APPROVED"},
            ]
        })()

        expected_sequence = [
            ("user_proxy", "DataPrepAgent"),
            ("DataPrepAgent", "EDAAnalysisAgent"),
            ("EDAAnalysisAgent", "VisualizationAgent"),
            ("VisualizationAgent", "CriticAgent"),
            ("CriticAgent", "FindingsGeneratorAgent"),
            ("FindingsGeneratorAgent", "ReportExporterAgent"),
            ("ReportExporterAgent", None),
        ]

        for from_name, expected_next in expected_sequence:
            result = router(speakers[from_name], mock_gc)
            if expected_next is None:
                assert result is None, f"After {from_name}, expected None"
            else:
                assert result.name == expected_next, (
                    f"After {from_name}, expected {expected_next}, got {result.name}"
                )

    def test_tool_call_round_trip_sequence(self, group_chat_components):
        """Simulate DataPrepAgent tool_call → DataPrepExecutor → DataPrepAgent → advance."""
        groupchat, _, _, _, _, _ = group_chat_components
        router = groupchat.speaker_selection_method

        # Step 1: DataPrepAgent sends tool_call → DataPrepExecutor
        gc1 = type("GC", (), {
            "messages": [{"name": "DataPrepAgent", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "load_data", "arguments": "{}"}}
            ]}]
        })()
        speaker1 = type("S", (), {"name": "DataPrepAgent"})()
        result1 = router(speaker1, gc1)
        assert result1.name == "DataPrepExecutor"

        # Step 2: DataPrepExecutor returns → DataPrepAgent
        gc2 = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "load_data", "arguments": "{}"}}
                ]},
                {"name": "DataPrepExecutor", "content": "data loaded"},
            ]
        })()
        speaker2 = type("S", (), {"name": "DataPrepExecutor"})()
        result2 = router(speaker2, gc2)
        assert result2.name == "DataPrepAgent"

        # Step 3: DataPrepAgent sends text → advance to EDAAnalysisAgent
        gc3 = type("GC", (), {
            "messages": [
                {"name": "DataPrepAgent", "content": "Data profile ready."}
            ]
        })()
        result3 = router(speaker1, gc3)
        assert result3.name == "EDAAnalysisAgent"

    def test_one_revision_loop_sequence(self, group_chat_components):
        """Simulate a pipeline with one critic revision loop."""
        groupchat, _, _, _, _, _ = group_chat_components
        router = groupchat.speaker_selection_method

        speakers = {name: type("S", (), {"name": name})() for name in [
            "FindingsGeneratorAgent", "ReportExporterAgent",
        ]}

        # First pass: REVISION_NEEDED at iteration 1
        gc_revision = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
            ]
        })()
        result = router(speakers["FindingsGeneratorAgent"], gc_revision)
        assert result.name == "CriticAgent"  # loops back

        # Second pass: APPROVED at iteration 2
        gc_approved = type("GC", (), {
            "messages": [
                {"name": "CriticAgent", "content": "REVISION_NEEDED"},
                {"name": "CriticAgent", "content": "APPROVED"},
            ]
        })()
        result = router(speakers["FindingsGeneratorAgent"], gc_approved)
        assert result.name == "ReportExporterAgent"  # proceeds
