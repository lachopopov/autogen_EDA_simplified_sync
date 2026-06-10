"""
tools/_eval_telemetry.py — Telemetry-only helpers for the EDA pipeline.

NOT in PROMPT_VERSION hash. Changes here do not invalidate the cache.

Contents:
  - _eval_cost_info    — module-level cost accumulator, read by pipeline.py
  - EvalCostCapture    — context manager: monkey-patches openlit eval utils
                         to capture token usage, restores on exit
  - _compute_eval_cost — pricing JSON lookup (arithmetic only)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Token usage captured from the comprehensive evaluation LLM call.
# Populated by EvalCostCapture.finalize(); read by pipeline._format_cost_summary().
# Survives clear_session() because it lives at module level, not in the
# artifact store.
_eval_cost_info: dict[str, Any] = {}


class EvalCostCapture:
    """Context manager that monkey-patches openlit.evals.utils.llm_response_openai
    to capture token usage and cost during a comprehensive eval call.

    Usage::

        with EvalCostCapture() as capture:
            evals = openlit.evals.All(...)
            result = evals.measure(...)

        capture.finalize(eval_model)   # populates _eval_cost_info

    The monkey-patch is active only inside the ``with`` block (restored on exit).
    """

    def __init__(self) -> None:
        self._orig_fn: Any = None
        self._captured: dict[str, Any] = {}

    def _capturing_openai(self, prompt: str, model: str | None, base_url: str) -> str:
        """Drop-in for llm_response_openai that also captures usage.

        Mirrors the SDK helper to intercept ``resp.usage`` token counts
        for pipeline cost tracking.  The original SDK function returns
        only the content string and discards usage metadata.
        """
        from openai import OpenAI as _OAI

        client = _OAI(base_url=base_url)
        if model is None:
            model = "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        if hasattr(resp, "usage") and resp.usage:
            self._captured["prompt_tokens"] = resp.usage.prompt_tokens
            self._captured["completion_tokens"] = resp.usage.completion_tokens
            self._captured["model"] = resp.model
            # Capture OpenAI prompt cache savings (prompt_tokens_details may be None)
            _details = getattr(resp.usage, "prompt_tokens_details", None)
            self._captured["cached_tokens"] = (
                getattr(_details, "cached_tokens", 0) or 0
            ) if _details else 0
        return resp.choices[0].message.content

    def __enter__(self) -> EvalCostCapture:
        import openlit.evals.utils as _evals_utils

        self._orig_fn = _evals_utils.llm_response_openai
        _evals_utils.llm_response_openai = self._capturing_openai
        return self

    def __exit__(self, *args: object) -> None:
        import openlit.evals.utils as _evals_utils

        _evals_utils.llm_response_openai = self._orig_fn

    def finalize(self, eval_model: str) -> dict[str, Any]:
        """Compute cost from captured usage, populate _eval_cost_info, log.

        Returns the captured usage dict for callers that need it.
        """
        if self._captured:
            pt = self._captured.get("prompt_tokens", 0)
            ct = self._captured.get("completion_tokens", 0)
            cost = _compute_eval_cost(eval_model, pt, ct)
            _eval_cost_info.clear()
            _eval_cost_info.update({
                "model": self._captured.get("model", eval_model),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cached_tokens": self._captured.get("cached_tokens", 0),
                "cost": cost,
            })
            logger.info(
                "Eval cost captured: model=%s, prompt=%d, completion=%d, cost=$%.4f",
                _eval_cost_info["model"], pt, ct, cost,
            )
        return dict(self._captured)


def _compute_eval_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Look up pricing from openlit_pricing.json and compute cost.

    Falls back to 0.0 if the pricing file is missing or the model
    is not listed.
    """
    pricing_path = Path(__file__).resolve().parent.parent / "openlit_pricing.json"
    try:
        with open(pricing_path, encoding="utf-8") as f:
            pricing = json.load(f)
        p = pricing["chat"][model]
        return (prompt_tokens / 1000) * p["promptPrice"] + \
               (completion_tokens / 1000) * p["completionPrice"]
    except Exception:
        return 0.0
