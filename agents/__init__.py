"""
agents — Factory functions for AG2 agent instantiation.

Architecture Reference: architecture.md § 12.6

Design:
  - make_agent() is the ONLY way to create agents in this project.
  - Never subclass AssistantAgent for computation logic.
  - All agents share: LLM_CONFIG, max_consecutive_auto_reply, termination guard.
  - Each agents/*.py file calls make_agent() with a specific system_message.

AG2 Version: 0.10.3
"""

from __future__ import annotations

from typing import Any

from autogen import AssistantAgent

from config import LLM_CONFIG

# Keyword-based termination guard — shared by all agents.
# Layer 1 of the three-layer termination system (architecture.md § 5).
def _TERMINATION_GUARD(x: dict) -> bool:
    return "TERMINATE" in (x.get("content") or "")


def make_agent(
    name: str,
    system_message: str,
    llm_config: dict[str, Any] | None = None,
) -> AssistantAgent:
    """
    Factory function — creates a consistently configured AssistantAgent.

    Enforces:
      - Shared LLM_CONFIG (model selection via EDA_MODE) unless
        *llm_config* is provided (e.g. gpt-5-mini for interpretation).
      - max_consecutive_auto_reply = 10  (allows 3 tools + retries + final text)
      - Keyword-based termination guard ("TERMINATE" in content)

    Never subclasses AssistantAgent.
    """
    return AssistantAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config if llm_config is not None else LLM_CONFIG,
        max_consecutive_auto_reply=10,
        is_termination_msg=_TERMINATION_GUARD,
    )
