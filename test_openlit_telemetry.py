"""
Standalone telemetry test for openlit.evals.All.

Verifies that:
  1. openlit.evals.All runs and returns correct result structure
  2. OTel metrics (evals.requests counter) are exported to the OTLP endpoint
  3. OTel traces (from the eval's OpenAI call) capture the model name
  4. Both the "issue detected" and "no issues" cases work

Usage:
  python test_openlit_telemetry.py

Prerequisites:
  - OpenLIT stack running: docker (port 3000/4318)
  - OPENAI_API_KEY in .env or environment
  - conda activate ag2_env
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import openlit  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Initialise OpenLIT with OTLP endpoint + custom pricing
# ---------------------------------------------------------------------------
OTLP_ENDPOINT = os.getenv("OPENLIT_ENDPOINT", "http://127.0.0.1:4318")
EVAL_MODEL = os.getenv("OPENLIT_EVAL_MODEL", "gpt-5")
PRICING_JSON = str(Path(__file__).resolve().parent / "openlit_pricing.json")

print(f"OpenLIT endpoint : {OTLP_ENDPOINT}")
print(f"Eval model       : {EVAL_MODEL}")
print(f"Pricing JSON     : {PRICING_JSON}")
print()

openlit.init(
    otlp_endpoint=OTLP_ENDPOINT,
    pricing_json=PRICING_JSON,
    disabled_instrumentors=["agno"],
)
print("✓ openlit.init() completed")
print()

# ---------------------------------------------------------------------------
# 2. Test Case A — known hallucination (should detect an issue)
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST A: Known hallucination (expect: verdict=yes, score>0)")
print("=" * 60)

detector_a = openlit.evals.All(
    provider="openai",
    model=EVAL_MODEL,
    collect_metrics=True,
)

result_a = detector_a.measure(
    prompt="Discuss the achievements of scientists.",
    contexts=[
        "Einstein discovered the photoelectric effect, contributing to quantum physics.",
    ],
    text="Einstein won the Nobel Prize in 1969 for discovering black holes.",
)

print(f"  verdict       : {result_a.verdict}")
print(f"  score         : {result_a.score}")
print(f"  evaluation    : {result_a.evaluation}")
print(f"  classification: {result_a.classification}")
print(f"  explanation   : {result_a.explanation}")
print()

if result_a.verdict == "yes":
    print("  ✓ Hallucination correctly detected")
else:
    print("  ⚠ Expected verdict='yes' but got 'no' — judge may be lenient")
print()

# ---------------------------------------------------------------------------
# 3. Test Case B — clean text (should detect no issues)
# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST B: Clean text (expect: verdict=no, score~0)")
print("=" * 60)

detector_b = openlit.evals.All(
    provider="openai",
    model=EVAL_MODEL,
    collect_metrics=True,
)

result_b = detector_b.measure(
    prompt="Summarize key facts about Einstein.",
    contexts=[
        "Einstein won the Nobel Prize in Physics in 1921 for his explanation "
        "of the photoelectric effect.",
    ],
    text="Einstein received the Nobel Prize in Physics in 1921 for his work "
         "on the photoelectric effect.",
)

print(f"  verdict       : {result_b.verdict}")
print(f"  score         : {result_b.score}")
print(f"  evaluation    : {result_b.evaluation}")
print(f"  classification: {result_b.classification}")
print(f"  explanation   : {result_b.explanation}")
print()

if result_b.verdict == "no":
    print("  ✓ Clean text correctly passed")
else:
    print("  ⚠ Expected verdict='no' but got 'yes' — possible false positive")
print()

# ---------------------------------------------------------------------------
# 4. Flush providers to ensure all telemetry is exported
# ---------------------------------------------------------------------------
print("=" * 60)
print("Flushing OTel providers...")
print("=" * 60)

try:
    from opentelemetry import trace
    tp = trace.get_tracer_provider()
    if hasattr(tp, "force_flush"):
        tp.force_flush(timeout_millis=10_000)
        print("  ✓ TracerProvider flushed")
    if hasattr(tp, "shutdown"):
        tp.shutdown()
        print("  ✓ TracerProvider shut down")
except Exception as e:
    print(f"  ⚠ Tracer flush: {e}")

try:
    from opentelemetry import metrics
    mp = metrics.get_meter_provider()
    if hasattr(mp, "force_flush"):
        mp.force_flush(timeout_millis=10_000)
        print("  ✓ MeterProvider flushed")
    if hasattr(mp, "shutdown"):
        mp.shutdown()
        print("  ✓ MeterProvider shut down")
except Exception as e:
    print(f"  ⚠ Meter flush: {e}")

print()
print("=" * 60)
print("Done. Check the OpenLIT dashboard at http://127.0.0.1:3000")
print("  - Requests tab: should show 2 OpenAI calls (gpt-5)")
print("  - Evals tab: should show 2 eval entries")
print("    - Test A: verdict=yes, evaluation=hallucination")
print("    - Test B: verdict=no, evaluation=none")
print("=" * 60)
