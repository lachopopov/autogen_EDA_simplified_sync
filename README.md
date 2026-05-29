> [!WARNING]
> ## ⛔ ARCHIVED — FAILED EXPERIMENT BRANCH
>
> **This branch is a preserved historical record of a failed latency-optimisation attempt.**
> It was never merged into `main` and will never be. The code here exists solely so future
> engineers can read exactly what was tried, inspect the actual implementation, and understand
> precisely why it failed — without repeating the same experiment.
>
> **Do not build on this branch. Do not rebase it. Do not open PRs from it.**
> Switch to `main` for the production codebase.

---

# `feat/findings-dag` — DAG Parallelisation Experiment Post-Mortem

**Experiment period:** May 2026
**Hypothesis:** Parallelise gpt-5 LLM calls in `FindingsGeneratorAgent` by splitting the fact
sheet into 20 sections and running them concurrently via an async DAG. Total LLM time would
collapse from T_gpt5 × sequential_sections to T_gpt5 for the slowest section only.
**Result:** 377 s on iris.csv vs 390 s baseline. ~13 s improvement. Not a meaningful gain.
**Verdict:** Archived. Wrong axis parallelised.

---

## The Governing Principle — Quality Trumps Latency

> **Report quality is the non-negotiable constraint of this project.**
> Any latency-reduction proposal that degrades the quality of the generated report is
> rejected automatically, regardless of how much time it saves.

This principle was established explicitly during the architectural debate that produced this
experiment, and it is the single most important decision rule for evaluating any future
optimisation attempt. It is stated here — at the top, before any technical detail — because
it is the reason every significant latency lever was ultimately blocked.

**What "quality" means concretely:**
- The PART1 action plan (numbered items with ACTION / EXPECTED OUTCOME / RISK IF SKIPPED) must be coherent across all dataset features simultaneously
- The PART2 business problem catalogue (10–20 problems with quantified impact) must draw on the full picture — not a truncated or section-local view
- Every number the LLM cites in its commentary must come from deterministic data the tool provided — not from inference over a partial fact sheet
- Any missing figure the LLM does not see, it will fabricate (proven empirically)

**Consequence:** If an optimisation forces the LLM to reason over less data, fewer sections,
or section-local context — it is a quality regression, not an optimisation. It is rejected.

---

## Table of Contents

- [1. Why This Was Attempted — The Latency Problem](#1-why-this-was-attempted--the-latency-problem)
- [2. What Was Tried — The DAG Design](#2-what-was-tried--the-dag-design)
- [3. Why It Failed — Four Root Causes](#3-why-it-failed--four-root-causes)
- [4. Three Alternative Proposals Also Rejected](#4-three-alternative-proposals-also-rejected)
- [5. The Constraint Table](#5-the-constraint-table)
- [6. The Architectural Principle Established](#6-the-architectural-principle-established)
- [7. What to Do Instead](#7-what-to-do-instead)

---

## 1. Why This Was Attempted — The Latency Problem

The production pipeline (`main` branch) was functionally correct at Phase 6 (1 227 tests
passing). Attention turned to a core operational scaling problem: total pipeline latency grows
with dataset width with no clean ceiling.

### Measured timings — iris.csv baseline (150 rows × 5 columns)

| Stage | Time | % of total |
|---|---|---|
| DataPrepAgent (load, validate, infer dtypes) | ~12.5 s | 3% |
| EDAAnalysisAgent (describe, missing, correlation) | ~29 s | 7% |
| VisualizationAgent (histograms, heatmaps, plots) | ~14.3 s | 4% |
| CriticAgent (rules + critique) | ~11.5 s | 3% |
| **FindingsGeneratorAgent** (`prepare_interpretation_context()` + gpt-5 call) | **~304 s** | **78%** |
| ReportExporterAgent (render PDF + IPYNB) | ~18 s | 5% |
| **Total** | **~390 s** | **100%** |

### The scaling formula

`prepare_interpretation_context()` builds a fact sheet that grows:
- **O(cols)** for histogram metadata — ~150 tokens per column
- **O(cols²)** for the full N×N correlation matrix

At 5 columns (iris): ~6.5 K tokens → ~304 s gpt-5 call.
At 50 columns: ~70 K tokens → projected ~900 s for `FindingsGeneratorAgent` alone, ~1 000 s total.

**The pipeline becomes unusable at moderate dataset widths.** This is what motivated the
search for a latency fix — subject to the quality constraint stated above.

---

## 2. What Was Tried — The DAG Design

**Branch:** `feat/findings-dag` (this branch)

### Hypothesis

Split the ~6.5 K-token fact sheet into 20 sections (one per plot / correlation cluster /
target variable section). Run each section as an independent gpt-5 call concurrently via an
async DAG with four waves:

- **Wave 1:** Independent sections (histograms, missing analysis) — fully parallel
- **Wave 2:** Sections that depend on Wave 1 outputs (distribution comparisons)
- **Wave 3:** Cross-section synthesis (correlation + target)
- **Wave 4:** Final integration — assemble PART1 action plan + PART2 business problems

**Expected wall-clock time:** T_slowest_section ≈ 30 s (one gpt-5 call).
**Expected saving:** ~304 s → ~30 s for FindingsGeneratorAgent. Total: ~390 s → ~120 s.

### Key additions in this branch

- `tools/findings_dag.py` — async DAG executor, wave scheduling, `_call_with_retry()`
- `DagSectionOutput` — Pydantic schema for per-section structured output
- `build_section_prompt()` — injects JSON schema into each section prompt
- Modified `agents/findings_generator_agent.py` — replaced single LLM call with DAG orchestration

---

## 3. Why It Failed — Four Root Causes

### Root Cause 1 — Wave 4 Bug (+120 s wasted)

The DAG's final wave (wave 4) called `_call_with_retry()` directly with a raw `wave4_context`
dict, bypassing `build_section_prompt()`. This meant no JSON schema was injected into the
prompt. gpt-5 returned free text instead of structured JSON → `ValidationError` → 4 failed
serial retry calls × ~30 s each = **+120 s added, not saved**.

The DAG's nominal latency improvement (parallel LLM calls in waves 1–3) was entirely consumed
by the Wave 4 bug overhead. Without the bug, the DAG might have measured ~257 s — but that
brings the next root causes into force.

### Root Cause 2 — `DagSectionOutput` Schema Prevents the Quality Contract

`FindingsGeneratorAgent` delivers two mandatory output sections on `main`:

- **PART1:** Numbered action plan — each item has ACTION / EXPECTED OUTCOME / RISK IF SKIPPED
- **PART2:** 10–20 business problems with quantified impact and confidence scores

This PART1/PART2 contract requires a **single LLM call that sees ALL sections simultaneously**.
The overall recommendations must be coherent across correlation analysis, distribution analysis,
target variable analysis, and critic findings.

The `DagSectionOutput` Pydantic schema forces each parallel call to produce section-local
output only. It structurally cannot produce cross-section recommendations — the schema does not
allow it, and the section-local context window does not contain sibling section data.

Fixing the schema would require giving every parallel call the entire fact sheet, which removes
the latency benefit entirely. **This is a quality regression — rejected per the governing
principle.**

### Root Cause 3 — Cross-Section Incoherence Is Structural, Not a Prompt Problem

Even if the Wave 4 bug were fixed and the schema relaxed:

A 20-call DAG where call #7 (correlation section) does not see the output of call #3
(distribution section) cannot produce coherent cross-feature recommendations. The LLM for each
section lacks context from sibling sections. This is not fixable by prompt engineering — it is
a fundamental constraint of parallelising reasoning across independent LLM calls. The calls are
stateless. There is no shared scratchpad.

The only fix is to give every call the full context — which collapses all 20 calls into 1 call
with the same token cost and the same latency as the baseline. **Any partial-context shortcut
is a quality regression — rejected per the governing principle.**

### Root Cause 4 — LLM Provider Rate Limiting Negates Parallel Wall-Clock Gain *(Unexpected Finding)*

**What was expected:** 20 parallel gpt-5 calls → wall-clock time ≈ T_slowest_call ≈ 30 s.

**What actually happens:** OpenAI's API enforces rate limits per minute (tokens-per-minute and
requests-per-minute). Firing 20 concurrent gpt-5 calls simultaneously triggers
`429 Too Many Requests` throttling. `_call_with_retry()` backs off and retries — adding
latency, not removing it. In the limit, 20 "parallel" calls that are all throttled and
serialised by the API end up taking **longer** than a single sequential call, because each
retry cycle burns backoff time on top of the original call duration.

**Why this is unexpected:** Theoretical reasoning assumes the bottleneck is compute (the GPU
doing inference). For an API-accessed LLM, the bottleneck is the API quota — a shared,
rate-limited resource across all callers. Parallelism at the client side does not create
parallelism at the provider side; it creates quota contention.

The iris DAG measured 377 s vs. 390 s baseline — a 13 s "improvement" that was noise, not
signal, and would have been negative if the Wave 4 bug had not masked the throttling overhead.

**Generalizable rule for future application building:** When an LLM is accessed via a
rate-limited API (not self-hosted), client-side LLM call parallelism does not reliably improve
wall-clock latency and may worsen it through backoff overhead. The only reliable parallel axis
is **data computation** (numpy, pandas, scikit-learn) — never LLM API calls.

---

## 4. Three Alternative Proposals Also Rejected

After archiving the DAG, three remaining latency levers were evaluated against the `main`
branch (not implemented on this branch). The quality constraint was applied to each.

### Proposal A — ProcessPoolExecutor on EDA Compute Phase

Add `ProcessPoolExecutor` inside `prepare_interpretation_context()` to run the 5 EDA
computations concurrently instead of serially.

**Estimated benefit:** EDA compute ~29 s → ~8 s for iris. Saves ~20 s from 390 s total = **5%**.

**Why not pursued now:** Compute is not the bottleneck. `FindingsGeneratorAgent` is 78% of
total time and is unchanged by ProcessPoolExecutor on the compute phase. The improvement is
marginal. Crucially, this proposal does **not** touch report quality — it is not rejected on
quality grounds, only on impact grounds. It remains viable for future implementation,
especially for the clustering branch where compute is the genuine bottleneck.

### Proposal B — Adaptive Fact Sheet Cap

Rank features by signal (mutual information × 0.5 + max absolute correlation × 0.3 +
(1 − missing_rate) × 0.2) and give top-N features full histogram metadata; remaining features
receive only a 1-line summary. Cap the correlation matrix to top-25 pairs by |r|.

**Why rejected — quality grounds:** Every number the LLM does not see, it fabricates. The
Metadata-First Hybrid architecture was specifically designed to give the LLM 100% data
coverage — full histograms, full N×N correlation, full missing percentages — because any
truncation recreates the blind-spot hallucination problem the architecture was built to solve.
Truncating columns to summary rows is a quality regression. **Rejected per the governing
principle.**

### Proposal C — Faster Model for FindingsGeneratorAgent

Switch `FindingsGeneratorAgent` from gpt-5 to gpt-5-mini. T_per_call drops from ~30 s to ~5 s.
For iris (6.5 K tokens): `FindingsGeneratorAgent` ~304 s → ~50 s. Total: ~390 s → ~136 s.

**Status — quality concern, not yet measured:** gpt-5 is hardcoded for `FindingsGeneratorAgent`
because PART1/PART2 commentary quality is the pipeline's primary differentiating output.
gpt-5-mini has been observed to produce shallower, less coherent PART2 sections. However,
this has not been measured objectively. This is the only lever not rejected outright — it
requires a PART1/PART2 quality regression test framework before a decision can be made. If
measurement shows no quality degradation, the 75% latency saving is worth taking.

---

## 5. The Constraint Table

| Lever | Latency impact | Quality impact | Status |
|---|---|---|---|
| Parallelise LLM calls (this DAG branch) | Theoretically large — negated by rate limiting | Cross-section incoherence — quality regression | **Rejected** — both quality and latency fail |
| Reduce fact sheet tokens (adaptive cap) | Proportional to reduction | Hallucination on truncated columns — quality regression | **Rejected** — quality regression |
| ProcessPoolExecutor on EDA compute | ~5% of total | None — does not touch LLM input | **Not rejected** — marginal impact, viable for clustering branch |
| Faster model for FindingsGenerator | ~75% reduction | Unknown — not yet measured objectively | **Deferred** — measure quality degradation first |
| Accept latency as-is | 0% | None | **Default** — current state of `main` |

**The honest conclusion:** The two levers with meaningful latency impact both cause quality
regressions and are blocked by the governing principle. The one lever with no quality impact
(ProcessPoolExecutor on compute) saves only 5%. **Classification pipeline latency scales
linearly with dataset width with no clean fix that preserves quality.**

For iris (5 cols): 390 s is acceptable for a development tool.
For 50-col datasets: ~1 000 s. This is a known, documented limitation — not a bug.

---

## 6. The Architectural Principle Established

> **N parallel DATA computations → 1 monolithic LLM call.**

This is the correct axis of parallelism for this pipeline architecture:

- The **tool** is the sensor — deterministic, parallelisable, testable, zero hallucination risk
- The **LLM** is the brain — sequential, coherence-requiring, cannot be split without quality loss

**Correct application:** ProcessPoolExecutor *inside* a single tool function, invisible to AG2.
The GroupChat still sees one tool call → one tool result. The parallelism is infrastructure,
not orchestration. Quality is unaffected.

**Incorrect application (this branch):** Multiple tool calls → multiple LLM agent turns →
multiple gpt-5 API calls → rate limit contention + cross-section incoherence = quality
regression + no latency gain.

---

## 7. What to Do Instead

Switch to `main`. The production pipeline is on `main`. It is functionally correct, fully
tested (1 227 tests), and produces high-quality PART1/PART2 findings.

If latency becomes unacceptable at large dataset widths, the levers to revisit (in order of
expected impact, filtered by the quality constraint):

1. **Measure faster model first** — Build a PART1/PART2 quality regression test framework.
   Measure gpt-5 vs. gpt-5-mini objectively on PART1 coherence and PART2 depth. If no
   measurable degradation, the 75% latency saving is the only clean win available.
2. **ProcessPoolExecutor on EDA compute** — Low-risk, ~5% saving for classification, genuine
   benefit for the future clustering branch. Safe to implement anytime without quality risk.
3. **Self-hosted LLM** — Removes the rate-limiting constraint (Root Cause 4). Makes
   client-side parallelism viable in principle — but Root Causes 2 and 3 (schema + incoherence)
   remain architectural blockers regardless of hosting.

**Do not reopen the DAG approach** without a solution to Root Causes 2 and 3. Rate limiting
can be mitigated. Cross-section incoherence and the schema conflict cannot be patched — they
require a fundamentally different output contract that does not violate the quality constraint.

---

*Branch created: May 2026. Archived: May 2026.*
*Test count at time of experiment: 1 227 — unchanged by this branch. No tests were added or modified here.*

---

> **Branch scope note:** The README below this line on `main` documents the production CLI
> pipeline. That content has been replaced on this branch by the post-mortem above, since this
> branch has no production use. See `main` for the full production README.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [How It Works](#how-it-works-business-problem-catalogue)
  - [For Data Analysts](#for-data-analysts)
- [Architecture Overview](#architecture-overview)
- [Output Files & Artifacts](#output-files--artifacts)
- [Deployment Recommendations](#deployment-recommendations)
- [Development & Testing](#development--testing)
- [Future Optimizations](#future-optimizations)
- [Observability (OpenLIT)](#observability-openlit)
- [Hallucination, Toxicity and Bias Evaluation](#hallucination-toxicity-and-bias-evaluation)
- [Troubleshooting](#troubleshooting)
- [For Developers & Engineers](#for-developers--engineers)
- [License & Attribution](#license--attribution)

---

## Quick Start

### What You Get

A comprehensive exploratory data analysis report that identifies and ranks **solvable business problems** in your dataset:
- **Tier 1:** Strategic overview of 5–20 potential business questions your data can answer
- **Tier 2:** Deep-dive analysis of the top 3 high-probability problems with metrics and recommendations
- **Quality assessment:** Data readiness verdict + flagged issues + remediation steps
- **Visualizations:** Histograms, correlation heatmaps, categorical distributions, feature importance rankings
- **Trustworthiness Guarantee:** When OpenLIT is enabled (`OPENLIT_ENABLE=true`), every report is automatically evaluated for hallucination using an external judge model (gpt-5). The report includes a Trustworthiness Assessment section so you know how much to rely on the AI-generated insights.

**Output formats:** Professional PDF + Markdown + interactive Jupyter notebook (optional) + individual PNG plots + cost summary

**Bottom line:** You get a data analysis report from AI agents that you can actually trust.

### Command Line (CLI)

For DS/ML experts and GenAI engineers building custom analyses:

```bash
python main.py your_dataset.csv
```

Results appear in `outputs/` (PDF, Markdown, plots, cost summary).

**Why CLI?** Integrate into scripts, automate batch processing, extend with custom tools.

> **Want a browser UI?** See the `streamlit-deploy` branch — it contains a Streamlit Cloud deployment with file upload, interactive results, and one-click PDF/notebook download.



---

## Installation

### Prerequisites

- **Python 3.10+**
- **OpenAI API key** (for LLM calls)
- **Optional**: conda or uv for environment management

### Option 1: Conda (Recommended)

```bash
conda create -n ag2_env python=3.12
conda activate ag2_env
pip install -r requirements.txt
```

### Option 2: uv (Fast Alternative)

```bash
# Create virtual environment
uv venv ag2_env
source ag2_env/bin/activate  # or: ag2_env\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Configure OpenAI API Key

Create a `.env` file in the project root:

```bash
# .env

# --- Required ---
OPENAI_API_KEY=sk-your-actual-key-here

# --- Model selection (optional) ---
# dev = gpt-5-mini (fast, ~$0.05–0.15/run)  |  final = gpt-5 (quality, ~$0.25–0.75/run)
# EDA_MODE=dev or EDA_MODE=final

# --- CSV / Excel missing-value sentinels (optional) ---
# Comma-separated list overrides the built-in default set (which includes "?").
# Set to a custom list if your data uses non-standard sentinels.
# Set to empty (CSV_NA_TOKENS=) to disable all custom sentinel conversion.
# CSV_NA_TOKENS=?,Unknown,NULL,N/A,missing

# --- Observability (optional) ---
# OPENLIT_ENABLE=true
# OPENLIT_ENDPOINT=http://127.0.0.1:4318
# OPENLIT_EVAL_MODEL=gpt-5    # judge model — must be stronger than the evaluated model

# --- Report export (optional) ---
# IPYNB_EXPORT=true           # also export Jupyter notebook alongside PDF

# --- Pipeline tuning (optional) ---
# MAX_CRITIC_ITERATIONS=2     # max critic↔revision loops before forcing report export
# MAX_ROUNDS=70               # absolute ceiling on GroupChat rounds
```



---

## Usage

### How It Works: Business Problem Catalogue

The pipeline's core output is a **Business Problem Catalogue** — a ranked list of all solvable business problems your data can answer, with deep-dive recommendations for high-probability ones.

**Two-Tier Structure:**

**Tier 1 — Strategic Overview (All Problems)**
- All potential business questions ranked by probability: HIGH / MEDIUM / LOW
- Each problem includes: business question + one-sentence EDA justification
- Example: "Churn prediction (HIGH): customer tenure shows strong negative correlation"
- **Purpose:** Gives stakeholders a complete landscape of opportunities

**Tier 2 — Tactical Deep-Dives (Top 3 High-Probability Problems)**
- Only the highest-probability problems receive full analysis:
  - **PROBLEM statement** + EDA evidence
  - **METRIC:** Key success measure (e.g., "churn rate = % customers lost per quarter")
  - **RECOMMENDATIONS:** 2–3 actions + expected impact
  - **BUSINESS IMPACT:** ROI or value driver
- **Purpose:** Action-ready guidance for immediate execution

**Why two tiers?**
- Token efficiency: Deep-dives consume 100 tokens each; brief listings — only 20
- Signal quality: HIGH-probability problems are more likely to yield ROI
- Cognitive load: Stakeholders focus on high-impact items first

**The Report Includes:**
- Full Tier 1 + Tier 2 structure in `report.pdf` and `report.md`
- Classification-specific analysis: Borda-ranked features (sorted by MI + effect size dual-lens voting), per-class statistics, feature-target interactions
- Data quality assessment: 14 automated quality rules (multicollinearity, outliers, imbalance, skewness, class imbalance)
- Visualizations: Histograms, correlation heatmaps, feature importance, missing data patterns, categorical distributions

### For Data Analysts

#### Scenario 1: Basic EDA Report

```bash
python main.py path/to/your_dataset.csv
```

> **Try it now with an included dataset:** `python main.py test_data/iris.csv`

**What happens:**
1. DataPrepAgent loads and validates your CSV
2. EDAAnalysisAgent computes statistics, missing data, correlations
3. VisualizationAgent generates 7+ plots (histograms, heatmaps, feature associations, categorical distributions)
4. CriticAgent flags data quality issues (high correlations, outliers, etc.)
5. FindingsGeneratorAgent synthesizes 3-lens insights (statistical, ML, business)
6. ReportExporterAgent renders PDF with inline plots + commentary

**Output files:**
- `outputs/report.pdf` — professional PDF report (statistics, visualizations, quality flags, findings, recommendations)
- `outputs/report.md` — Markdown version (machine-readable, for version control and downstream systems)
- `outputs/report.ipynb` — interactive Jupyter notebook (optional, if IPYNB_EXPORT=true)
- `outputs/plots/` — individual PNG files (histograms, heatmaps, feature importance, categorical distributions)
- `outputs/cost_summary.txt` — token usage and cost breakdown

#### Scenario 2: Custom Dataset Formats

The pipeline supports CSV, Parquet, and Excel files:

```bash
# Parquet file
python main.py path/to/dataset.parquet

# Excel spreadsheet (single sheet)
python main.py path/to/analysis.xlsx
```

> **Try included format examples:** `python main.py test_data/iris.parquet` or `python main.py test_data/iris.xlsx`

#### Scenario 3: Development vs. Production Models

By default, the pipeline uses `gpt-5-mini` (fast, cost-effective for testing).

In final mode, `FindingsGeneratorAgent` switches to `gpt-5` (higher quality); all other agents remain on `gpt-5-mini` to control cost (~5× cost for the findings step only):

```bash
# Development (default, gpt-5-mini)
python main.py test_data/iris.csv

# Production-ready reporting (gpt-5)
EDA_MODE=final python main.py test_data/iris.csv
```

**Cost & timing reference:**
- Dev mode: ~30–60 seconds, ~$0.05–0.15 per run
- Final mode: ~60–120 seconds, ~$0.25–0.75 per run

#### Scenario 4: Example Datasets

Sample datasets are included in `test_data/`:

- **iris.csv** — Classic Iris dataset (150 rows, 5 columns, balanced, multicollinear)
- **stress_critic.csv** — Edge case dataset for quality thresholds (tests high outlier counts, missing data)
- **adult.csv** — UCI census dataset (250 rows sampled); `workclass` and `occupation` use `" ?"` (space + question mark) as missing-value sentinels — validates automatic sentinel-to-NaN conversion
- **default_of_credit_card_clients.csv** — UCI credit card default dataset (30,000 rows, 25 columns); tests large dataset handling and encoded categorical detection
- **strategy_b_synthetic.csv** — Synthetic binary-classification dataset (120 rows, 4 columns); tests mixed categorical + numeric pipelines
- **iris.parquet** — Parquet format of the Iris dataset; validates Parquet file loading
- **iris.xlsx** — Excel format of the Iris dataset; validates Excel file loading

```bash
# Quick test
python main.py test_data/iris.csv

# Stress test the quality rules
python main.py test_data/stress_critic.csv

# Sentinel handling test (? missing values)
python main.py test_data/adult.csv
```

> **Note:** If your dataset uses `"?"` as a *legitimate* value (not a missing sentinel), override via: `CSV_NA_TOKENS=NA,NULL,none python main.py your_data.csv` or add `CSV_NA_TOKENS=NA,NULL,none` to your `.env`.

---

## Architecture Overview

### 6-Stage Pipeline (Classification Task Focused)

1. **DataPrepAgent** ← Loads, validates, type-infers data; detects encoded categoricals
2. **EDAAnalysisAgent** ← Computes 60+ statistics, correlations, missing patterns; per-class statistics for classification targets
3. **VisualizationAgent** ← Generates 7+ plots: histograms, heatmaps, categorical distributions, feature importance
4. **CriticAgent** ← Flags 14 data quality rules (multicollinearity, outliers, imbalance, rare categories, skewness, class imbalance, etc.)
5. **FindingsGeneratorAgent** ← LLM synthesizes TWO outputs:
   - **ACTION PLAN:** Numbered recommendations (HIGH/MEDIUM/LOW priority) with action, expected outcome, and risk if skipped
   - **BUSINESS PROBLEM CATALOGUE:** Tier 1 (all problems ranked by probability) + Tier 2 (top-3 problems deep-dives with metrics + ROI)
6. **ReportExporterAgent** ← Renders PDF + Markdown + optional Jupyter notebook with inline plots

**Classification-Task Specialization:**
- **Per-class target statistics:** Mean/std of each feature broken down by target class (imbalance ratio reported)
- **Borda-ranked features:** Dual-lens voting combines Mutual Information (kNN-estimated, detects any dependence) + effect size (η² numerical/classification, Cramér's V categorical), ranked by Borda score (lower = more important)
- **Feature-target associations:** Detects nonlinear signals (high MI + weak effect size → use tree-based models) and suspicious associations (low MI + strong effect size → investigate outliers/leakage)
- **Interaction detection:** Trajectories (monotone patterns across feature levels), segments (cross-feature cohorts with distinct target rates), portfolio concentration (which quantiles drive outcomes)

### Metadata-First Hybrid

**Problem:** Raw data overflows LLM context. Vision hallucinations on exact values.

**Solution:** Intermediate fact blocks delivered to FindingsGeneratorAgent:
- Histogram bin counts + edges (complete shape DNA)
- 5-number summaries per column
- Correlation matrix (all cell values)
- Missing percentages per column
- Critic flags

**LLM receives:** ~6.5K tokens of deterministic facts → synthesizes 3-lens insights without hallucination.

**Result:** 100% data coverage + expert-quality interpretation.

### Two-State Separation

| Aspect | Conversation State | Pipeline State |
|--------|------------------|-----------------|
| **Lives in** | AG2 message history (ephemeral) | Artifact store on disk (persistent) |
| **Size** | Few KB (tokens) | Up to 50MB (DataFrames, plots) |
| **LLM role** | Author/consumer (reasoning) | Neither (infrastructure) |
| **Example** | "I'll call describe_stats next" | 17KB DataFrame JSON, 898B stats dict |

**Why this matters:** Small LLMs (gpt-5-mini dev mode) cannot copy large JSON from messages into tool parameters. Instead, tools save to disk and return `STATE_REF:key` references. Downstream tools load from disk. LLM only handles 30-char references, not 15KB blobs.



---

## Output Files & Artifacts

### Generated on Every Run

#### 1. **report.pdf** (Main Output)
- **Location:** `outputs/report.pdf`
- **Size:** ~170 KB
- **Format:** Professional 7-section PDF
- **Contents:**
  - Executive overview
  - Missing data analysis
  - Correlation analysis
  - Statistical analysis
  - Data quality assessment (critic flags)
  - Conclusions (data-readiness verdict, business consequences, risks, action plan)
  - Recommendations (5 numbered items with ACTION/OUTCOME/RISK)
- **Audience:** Non-technical stakeholders, decision-makers

#### 2. **report.md** (Markdown Report)
- **Location:** `outputs/report.md`
- **Format:** Plain Markdown text
- **Contents:** Identical structure to PDF (sections + expert commentary from LLM)
- **Use:** Machine-readable for downstream systems, documentation, version control, feeding to other LLMs
- **Audience:** DS/ML experts, technical teams, automation, continuous integration

#### 3. **Visualizations** (Plots)
- **Location:** `outputs/plots/`
- **Files:**
  - `hist_<column>.png` (1 per numeric column; 30 bins)
  - `correlation_heatmap.png` (Pearson r matrix, if N>1 numeric columns)
  - `missing_heatmap.png` (per-column missing % bar chart, if any missing data)
  - `class_distribution.png` (target class counts + imbalance ratio, if classification)
  - `target_distribution.png` (target value histogram + KDE overlay, if regression)
  - `feature_target_associations.png` (Borda-ranked features with MI + effect size)
  - `cat_<column>.png` (1 per categorical column; top values + target rates)
  - `ordinal_spearman_heatmap.png` (Spearman ρ matrix for ordinal features, if present)
- **Size:** ~50–200 KB total (embedded in PDF, also standalone for presentations/dashboards)
- **Audience:** Reports, presentations, documentation, dashboard integration

#### 4. **cost_summary.txt** (Cost Report)
- **Location:** `outputs/cost_summary.txt`
- **Format:** Plain text
- **Contents:** Per-agent token usage, cost breakdown by model, and grand total (including hallucination eval if OpenLIT enabled)
- **Audience:** Finance tracking, cost analysis, billing

#### 5. **Session Artifacts** (Developer Access)
- **Location:** `outputs/.pipeline_state/<session-uuid>/`
- **Contents:**
  - `data_json.json` — DataFrame as records JSON (raw data for reproducibility)
  - `schema_json.json` — Column metadata (names, types, memory usage)
  - `describe_stats.json` — per-column statistics (count, mean, std, min, 25%/50%/75%, max; plus skewness_scipy for numeric columns)
  - `missing_analysis.json` — Null/NaN percentages per column
  - `correlation_matrix.json` — N×N Pearson correlation matrix
  - `categorical_analysis.json` — Per-column categorical distributions (cardinality, entropy, rare values, target rates)
  - `feature_associations.json` — Borda-ranked features with MI (kNN-estimated) + effect size (η²/Cramér's V)
  - `interaction_signals.json` — Multivariate patterns (persistence gradients, trajectories, cross-feature segments, portfolio concentration)
  - `critic_report.json` — Quality flags (14 data quality rules: multicollinearity, outliers, imbalance, skewness, class imbalance, etc.)
  - `interpretations.json` — LLM-generated expert commentary (statistical, DS/ML, and business perspectives)
- **Access:** Via `tools._pipeline_state.load_state(key)` in Python (see developers section for examples)
- **Audience:** Developers, automated workflows, data scientists building custom analyses

### Optional Outputs

#### **report.ipynb** (Jupyter Notebook Export)
If IPYNB export is enabled:
- **Location:** `outputs/report.ipynb`
- **Format:** Interactive Jupyter notebook
- **Contents:** Markdown cells + inline plots + findings
- **Use:** Iterative analysis, sharing with analysts

---

## Deployment Recommendations

### Local Development

**Current mode** — CLI-based, single-file analysis:

```bash
python main.py path/to/your_dataset.csv
```

✅ Simple, no infrastructure required  
✅ Fast feedback loop for prototyping  
⚠️ No concurrent requests, state file cleanup manual  

### Web UI (Streamlit)

A browser-based interface is maintained on the `streamlit-deploy` branch (`streamlit_app.py`). That branch has its own README with full setup and usage instructions — see it for Streamlit Cloud deployment, file upload workflow, and UI-specific configuration.

For building a custom web UI on top of `main`, the recommended architecture:

```
Client (Streamlit/Gradio UI)
    │
    ├─→ File upload widget
    ├─→ Model selection (dev/final)
    ├─→ Progress bar (agent activity)
    └─→ Report viewer (PDF/HTML)
         │
         ▼
    Backend (FastAPI/Flask)
         │
         ├─→ Session management (UUID-based)
         ├─→ Call main.run_pipeline()
         ├─→ Cleanup on completion
         └─→ Serve output files (PDF, plots)
```

**Key considerations:**
- Use `contextvars.ContextVar` for session isolation (multi-worker)
- Implement cleanup task (delete `.pipeline_state/<uuid>` after download)
- Add timeout guards (max 5 min per analysis)
- Rate limiting on API endpoint

### Cloud / API Deployment

**Recommended stack:**
- **Compute:** Docker container (Python 3.12 + dependencies)
- **Queue:** Celery/RQ for async jobs
- **Storage:** S3/GCS for output files + session artifacts
- **Orchestration:** Kubernetes (auto-scale on queue depth)

**Scalability roadmap:**
- Phase 1 (current): Single-process CLI
- Phase 2 (next): Async job queue + REST API
- Phase 3 (future): Distributed agent orchestration (async planning, parallel API calls)

See [Future Optimizations](#future-optimizations) for async architectural notes.

---

## Development & Testing

### Setting Up Your Dev Environment

```bash
# 1. Clone repo
git clone <repo-url>
cd autogen_simplified_EDA_tool

# 2. Create environment
conda create -n ag2_env python=3.12
conda activate ag2_env
pip install -r requirements.txt

# 3. Install dev tools (optional)
pip install ruff mypy black pre-commit

# 4. Configure pre-commit hooks (optional)
pre-commit install

# 5. Run tests
pytest tests/ -v --tb=short
```

### Key Project Files

- **[main.py](main.py)** — CLI entry point
- **[orchestrator.py](orchestrator.py)** — AG2 GroupChat assembly & routing
- **[agents/](agents/)** — Agent factories (DataPrepAgent, EDAAnalysisAgent, etc.)
- **[tools/](tools/)** — Pure-Python tool functions (pandas, matplotlib, analysis)
- **[tools/_pipeline_state.py](tools/_pipeline_state.py)** — Artifact store implementation
- **[eda_state.py](eda_state.py)** — Pydantic models (state schema)
- **[tests/](tests/)** — 1,000+ unit tests

### Code Quality

```bash
# Format code
black agents/ tools/ tests/

# Lint Python
ruff check --fix agents/ tools/ tests/

# Type checking
mypy agents/ tools/ --strict

# Run tests
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Current state:** 1,014 tests passing, ruff clean, zero linting errors.

### Adding New Tests

```python
# tests/test_my_new_feature.py
import pytest
from tools.my_tool import my_function

class TestMyFeature:
    def test_basic_case(self):
        result = my_function("input")
        assert result == "expected"

    def test_edge_case(self):
        with pytest.raises(ValueError):
            my_function(None)

# Run it
pytest tests/test_my_new_feature.py -v
```

---

## Future Optimizations

### Async Architecture Analysis

**Current:** Synchronous agent orchestration (agents run sequentially).

**Potential gains from async:**

#### 1. **Concurrent Agent Planning** (Est. Savings: 20–30%)
Currently, agents plan serially:
```
DataPrepAgent (2s) → EDAAnalysisAgent (3s) → VisualizationAgent (4s) = 9s total
```

With async, agents could plan in parallel during execution handoff:
```
DataPrepAgent (2s) + [EDAAnalysisAgent planning (0.5s) in parallel] = 2.5s
→ EDAAnalysisAgent (3s) + [VisualizationAgent planning (0.3s) in parallel] = 3.3s
= ~5.8s total (35% faster)
```

**Implementation:** Replace `initiate_chat()` with async handlers on speaker transitions.

#### 2. **Parallel LLM API Calls** (Est. Savings: 40–50%)
Currently, tools call LLM sequentially (one agent finishes, next begins).

With async `aiohttp`:
```
FindingsGeneratorAgent makes 3 LLM calls (reasoning + section synthesis)
Sequential: 50s + 40s + 35s = 125s total
Parallel: max(50s, 40s, 35s) = 50s total (60% faster)
```

**Implementation:** Wrap LLM calls in `asyncio.gather()`, use OpenAI async client.

#### 3. **Concurrent Plot Generation** (Est. Savings: 15–25%)
Currently, 6 plots are generated ~sequentially (matplotlib threads contend).

With async file I/O:
```
Plot 1 (300ms) + Plot 2 (300ms) + ... + Plot 6 (300ms) = 1800ms
With async I/O scheduling: max(300ms) + overhead = ~400ms (77% faster)
```

**Implementation:** Matplotlib in TkAgg backend (threadsafe), async PNG encoding & disk writes.

### Roadmap (No Implementation Required)

| Phase | Focus | Est. Speedup | Complexity |
|-------|-------|--------------|-----------|
| **1 (Current)** | Sync orchestration | — | Low |
| **2 (Next)** | Concurrent planning during handoff | 20–30% | Medium |
| **3 (Future)** | Parallel LLM API calls + async I/O | 40–60% | High |
| **4 (Advanced)** | Distributed agent + GPU-accelerated EDA | 2–3× | Very High |

### Implementation Notes (Not Executed)

If async is implemented:
- Replace `UserProxyAgent.initiate_chat()` with async event loop
- Use `asyncio` for LLM call batching (OpenAI async client)
- Keep tools synchronous (pandas is not async-safe)
- Use `contextvars` for session isolation per concurrent request
- Test with `pytest-asyncio` fixture model

**Do NOT implement without:** benchmarking gains on realistic data sizes (async overhead may exceed gains on small datasets like iris.csv).

---

## Observability (OpenLIT)

The pipeline supports **OpenLIT** for LLM observability — tracing every agent call, token usage, and cost in a visual dashboard.

### Prerequisites

1. **Docker** installed and running
2. **OpenLIT stack** deployed (OTEL collector + dashboard):

```bash
# Deploy OpenLIT (one-time)
docker run -d --name openlit \
  -p 3000:3000 -p 4317:4317 -p 4318:4318 \
  ghcr.io/openlit/openlit:latest
```

3. **openlit SDK** installed (already in `requirements.txt`):

```bash
pip install openlit
```

### Configuration

Add to your `.env` file:

```bash
OPENLIT_ENABLE=true
OPENLIT_ENDPOINT=http://127.0.0.1:4318
OPENLIT_EVAL_MODEL=gpt-5    # judge for hallucination eval; must be stronger than main model
```

Or use CLI flags to override:

```bash
# Enable OpenLIT for this run (overrides .env)
python main.py test_data/iris.csv --openlit

# Disable OpenLIT for this run (overrides .env)
python main.py test_data/iris.csv --no-openlit
```

### Accessing the Dashboard

Open **http://127.0.0.1:3000** in your browser.

**Default credentials:**
- Email: `user@openlit.io`
- Password: `openlituser`

The dashboard shows:
- **Request traces** — full call chain per agent
- **Token usage** — input/output tokens per LLM call
- **Cost tracking** — per-request and cumulative costs

### Hallucination Evaluation

The pipeline includes **automated hallucination detection** for FindingsGenerator output using OpenLIT's programmatic evaluations. When OpenLIT is enabled, the LLM-generated interpretations are evaluated against the deterministic fact sheet (ground truth) using a stronger judge model.

**How it works:**
1. `prepare_interpretation_context()` (called by **FindingsGeneratorExecutor**) produces a deterministic fact sheet: all statistics, histogram bin data, correlation matrix, missing percentages, critic flags
2. **FindingsGeneratorAgent** (gpt-5-mini in `dev` mode / gpt-5 in `final` mode, controlled by `EDA_MODE`) generates expert commentary grounded in the fact sheet
3. `save_interpretations()` (called by **FindingsGeneratorExecutor**, only when OpenLIT session is active) runs `openlit.evals.All` with the judge model (`OPENLIT_EVAL_MODEL`, default `gpt-5`), performing a **combined hallucination + bias + toxicity evaluation** against the fact sheet as ground truth
4. Evaluation results are persisted in the artifact store (`comprehensive_eval` key) and logged via the OTel tracer
5. `assemble_findings()` builds a **Trustworthiness Assessment** section at the end of the report based on the persisted eval score

**Trustworthiness levels** (based on comprehensive eval score — hallucination + bias + toxicity):

| Score Range | Level | Meaning |
|---|---|---|
| 0.0 – 0.3 | **High Trustworthiness** | Commentary is well-grounded in the source data |
| 0.3 – 0.7 | **Medium Trustworthiness** | Some claims may not be fully supported; cross-check recommended |
| 0.7 – 1.0 | **Low Trustworthiness** | Significant hallucination detected; treat with caution |

**Telemetry:** `_shutdown_openlit()` flushes both the `TracerProvider` and `MeterProvider` before exit, ensuring all pending spans and metrics are exported to the OTLP collector before the process terminates (default `PeriodicExportingMetricReader` interval is 60 s — longer than a typical pipeline run).

**Non-blocking:** The evaluation logs warnings but never fails the pipeline.

**Configure the judge model** via environment variable:

```bash
# Default: gpt-5 (recommended — must be stronger than the evaluated model)
OPENLIT_EVAL_MODEL=gpt-5
```

**Cost impact:** ~$0.10 per run (one additional gpt-5 call).

### Custom Pricing for New Models

OpenLIT's default pricing JSON may not include newer models like `gpt-5-mini` and `gpt-5`. The project includes `openlit_pricing.json` with correct pricing:

```json
{
  "chat": {
    "gpt-5-nano": {"promptPrice": 0.00005, "completionPrice": 0.0004},
    "gpt-5-mini": {"promptPrice": 0.00025, "completionPrice": 0.002},
    "gpt-5":      {"promptPrice": 0.00125, "completionPrice": 0.01}
  }
}
```

> Versioned model aliases (e.g. `gpt-5-mini-2025-08-07`) are also included. Edit `openlit_pricing.json` directly to add further entries — prices are per-token.

> **Agno instrumentor:** `openlit.init()` is called with `disabled_instrumentors=["agno"]` since this project does not use the Agno framework. This is a precaution — openlit 1.39.0+ also auto-skips instrumentors whose packages are not installed, so it is effectively a no-op if `agno` is absent from the environment.

---

## Hallucination, Toxicity and Bias Evaluation

### Hallucination, Bias, and Toxicity — All Evaluated (Combined)

All three are evaluated in a **single pass** via `openlit.evals.All` when `OPENLIT_ENABLE=true`. There are no separate evaluator calls — the combined evaluator returns one unified score, verdict, and a per-type breakdown (Hallucination / Bias / Toxicity) that is embedded in the report's **Trustworthiness Assessment** section.

When the judge model finds no issues, the report states: *"No significant bias, toxicity, or hallucination detected."*

See [Observability → Hallucination Evaluation](#hallucination-evaluation) for the full flow, scoring table, and configuration. The scope described there covers all three dimensions.

---

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Ensure `.env` file exists in project root with valid key
- Or set env var: `export OPENAI_API_KEY="sk-..."`

**"File not found: iris.csv"**
- Use full path or place file in project root
- Or copy to `test_data/`: `cp my_data.csv test_data/`

**"gpt-5-mini not available" / HTTP 400 error**
- Check OpenAI account has access to latest models
- Verify API key is correct: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

**Tests fail with "PipelineStateError"**
- Session cleanup may have left stale state dirs
- `rm -rf outputs/.pipeline_state/` and retry

**Missing values showing as 0% despite known missing data (`?`, `Unknown`, etc.)**
- Confirm your CSV uses one of the built-in sentinel tokens (`?`, `Unknown`, `NULL`, `N/A`, etc. — see `config.py → NA_TOKENS`)
- For custom sentinels not in the default list: `CSV_NA_TOKENS=my_token,another python main.py data.csv`
- For datasets where `"?"` is a *legitimate* category value (not missing), exclude it: `CSV_NA_TOKENS=NA,NULL,None,nan`
- `skipinitialspace=True` is applied at load time — leading spaces in fields (e.g., `" ?"` in UCI-style CSVs) are automatically stripped before sentinel matching

**PDF not generated**
- Check `outputs/plots/` directory exists and is writable
- Verify `report.pdf` isn't open in another process (Windows)

---

## For Developers & Engineers

### Understanding the Architecture

This project uses **AG2 StateFlow** — a deterministic agent orchestration pattern:

```
user_proxy (initiator)
    ├─→ DataPrepAgent ⇄ DataPrepExecutor
    ├─→ EDAAnalysisAgent ⇄ EDAAnalysisExecutor
    ├─→ VisualizationAgent ⇄ VisualizationExecutor
    ├─→ CriticAgent ⇄ CriticExecutor
    ├─→ FindingsGeneratorAgent ⇄ FindingsGeneratorExecutor
    └─→ ReportExporterAgent ⇄ ReportExporterExecutor
```

**Key principles:**
1. **Agent = Brain** — Decides which tools to call
2. **Executor = Hands** — Runs the tool, returns result
3. **Two-State Separation** — LLM message history ≠ artifact store (state stored on disk)
4. **Metadata-First Hybrid** — Deterministic fact sheets + LLM interpretation (prevents hallucination)

### Extending the System

Create a pure-Python tool in `tools/` and register it programmatically in the orchestrator. See `agents/data_prep_agent.py` for examples.

### Running Tests

```bash
pytest tests/ -v
```

**Coverage:** 1,000+ tests across 17 test files (tools, agents, orchestrator, state management, end-to-end).

### Model Selection & Tuning

```bash
# Fast (gpt-5-mini, dev mode)
python main.py data.csv

# High quality (gpt-5, final mode)  
EDA_MODE=final python main.py data.csv
```

Pricing: gpt-5-mini ($0.25/$2.00 per 1M tokens) vs. gpt-5 ($1.25/$10.00).

---

## License & Attribution

**AG2 Framework:** [Apache 2.0](https://github.com/ag2ai/ag2)  
**Project:** [GNU General Public License v3.0](LICENSE)

Copyright (C) 2026 Lachezar Popov

This project is free software: you can redistribute it and/or modify it under
the terms of the **GNU General Public License v3.0** as published by the Free
Software Foundation. Any derivative work distributed to others must also be
released under GPL-v3 — this ensures the codebase and all improvements remain
open. See [LICENSE](LICENSE) for the full terms.





---

## Contributing

Contributions welcome! See CONTRIBUTING.md (not yet included in this repository) or:

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Run full test suite
5. Commit & push
