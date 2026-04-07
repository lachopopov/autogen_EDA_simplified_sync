# EDA Multi-Agent AG2 Report Generator For Classification Tasks

**A production-ready exploratory data analysis system powered by autonomous LLM agents.**

Automated end-to-end statistical analysis, visualization, quality assessment, and report generation using AI agents coordinated through AG2 StateFlow. Engineered for small-to-medium datasets (100–100K rows) with expert-validated Metadata-First Hybrid architecture.


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

### For Business Stakeholders & Analysts

#### Option 1: Web UI (Streamlit Cloud) — Recommended for Business Users

Access the app directly from your browser (no installation needed):

1. **Upload your dataset** — CSV, Parquet, or Excel format (first row = feature names)
2. **Confirm target variable** — AI agents suggest a target; you confirm, reassign, or skip
3. **Validate encoded categoricals** — AI detects numeric columns that might be encoded categories; you confirm each and label it as nominal or ordinal
4. **Run the analysis** — AI agents generate a complete report (~2 minutes)
5. **View results in UI** — See the plots, Markdown version, and cost analysis by clicking Results section
6. **Download results** — PDF report, Markdown version, Jupyter Notebook, and cost summary
7. **Check trustworthiness** — See the hallucination score in the report (requires OpenLIT to be enabled)

**Why Streamlit Cloud?** No software installation, instant access, intuitive interface, results viewable in UI and ready to download immediately.

#### Option 2: Command Line (CLI) — For Local Development

For DS/ML experts and GenAI engineers building custom analyses:

```bash
python main.py your_dataset.csv
```

Results appear in `outputs/` (PDF, Markdown, plots, cost summary).

**Why CLI?** Integrate into scripts, automate batch processing, extend with custom tools.



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

A Streamlit implementation is included (`streamlit_app.py`). See [Quick Start → Option 1](#option-1-web-ui-streamlit-cloud--recommended-for-business-users) for usage instructions.

For reference, the underlying architecture:

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

#### Hallucination, Toxicity and Bias Evaluation

All three are evaluated in a single pass via `openlit.evals.All` when `OPENLIT_ENABLE=true`. There are no separate evaluator calls — the combined evaluator returns one unified score, verdict, and a per-type breakdown (Hallucination / Bias / Toxicity) that is embedded in the report's **Trustworthiness Assessment** section.

When the judge model finds no issues, the report states: _"No significant bias, toxicity, or hallucination detected."_

See [Observability → Hallucination Evaluation](#hallucination-evaluation) for the full flow, scoring table, and configuration. The scope described there covers all three dimensions.

---

The pipeline includes **automated hallucination detection** for FindingsGenerator output using OpenLIT's programmatic evaluations. When OpenLIT is enabled, the LLM-generated interpretations are evaluated against the deterministic fact sheet (ground truth) using a stronger judge model.

**How it works:**
1. `prepare_interpretation_context()` (called by **FindingsGeneratorExecutor**) produces a deterministic fact sheet: all statistics, histogram bin data, correlation matrix, missing percentages, critic flags
2. **FindingsGeneratorAgent** (gpt-5-mini in `dev` mode / gpt-5 in `final` mode, controlled by `EDA_MODE`) generates expert commentary grounded in the fact sheet
3. `save_interpretations()` (called by **FindingsGeneratorExecutor**, only when OpenLIT session is active) runs `openlit.evals.Hallucination` with the judge model (`OPENLIT_EVAL_MODEL`, default `gpt-5`) comparing the generated text against the fact sheet as ground truth
4. Evaluation results are persisted in the artifact store (`hallucination_eval` key) and forwarded as OTel metrics to the OpenLIT dashboard via `collect_metrics=True`
5. `assemble_findings()` builds a **Trustworthiness Assessment** section at the end of the report based on the persisted eval score

**Trustworthiness levels** (based on hallucination score):

| Score Range | Level | Meaning |
|---|---|---|
| 0.0 – 0.3 | **High Trustworthiness** | Commentary is well-grounded in the source data |
| 0.3 – 0.7 | **Medium Trustworthiness** | Some claims may not be fully supported; cross-check recommended |
| 0.7 – 1.0 | **Low Trustworthiness** | Significant hallucination detected; treat with caution |

**Telemetry:** `_shutdown_openlit()` flushes both the `TracerProvider` and `MeterProvider` before exit, ensuring the eval counter created by `collect_metrics=True` is exported to the OTLP collector (default `PeriodicExportingMetricReader` interval is 60 s — longer than a typical pipeline run).

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

### Known Issues (openlit 1.36.8)

Three bugs exist in openlit 1.36.8 that are patched locally in the conda environment:

1. **`async_agno.py` line 783** — `return await result` inside an async generator (invalid Python). Patched to `await result`.
2. **`__init__.py` tracer=None** — `config.update_config()` passes user-provided `otel_tracer` (always None) instead of the internally created `configured_tracer`. Patched to pass `configured_tracer`.
3. **`evals/utils.py` line 155** — `temperature=0.0` hardcoded in `client.beta.chat.completions.parse()`. gpt-5 family models reject `temperature=0.0` with HTTP 400. Patched by removing the `temperature` parameter.

Additionally, the **Agno instrumentor** is disabled (`disabled_instrumentors=["agno"]`) since AG2 does not use the Agno framework, and the buggy instrumentor would cause initialization failures.

> **Note:** These patches live in the installed package and will be lost on `pip install --upgrade openlit`. Re-apply them if upgrading, or check if the upstream fix has been released.

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
**Project:** MIT (your choice)





---

## Contributing

Contributions welcome! See CONTRIBUTING.md (not yet included in this repository) or:

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Run full test suite
5. Commit & push
