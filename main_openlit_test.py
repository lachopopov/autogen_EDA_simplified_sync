import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so OPENAI_API_KEY is available
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# --- OpenLIT observability setup ---
# openlit 1.36.8 has a known SyntaxError in its agno instrumentor — disabled here.
# Also patched: __init__.py (configured_tracer) and async_agno.py (return in async gen).
import openlit

openlit.init(
    otlp_endpoint="http://127.0.0.1:4318",
    disabled_instrumentors=["agno"],
)

from autogen import AssistantAgent, LLMConfig, UserProxyAgent

# --- Base configuration shared across all models ---
# Note: temperature is NOT set here. gpt-5-nano and gpt-5-mini only support
# the default temperature (1). Setting temperature=0.0 causes a 400 error.
_BASE: dict = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "cache_seed": None,  # ephemeral cache — no stale outputs across runs
}

llm_config = LLMConfig({"model": "gpt-5-nano", **_BASE})
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

# Start the chat
user_proxy.initiate_chat(
    assistant,
    message="Tell me a joke about NVDA and TESLA stock prices.",
)