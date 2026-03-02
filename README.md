# AG2 Multi-Agent EDA Pipeline

**A production-ready exploratory data analysis system powered by autonomous LLM agents.**

Automated end-to-end statistical analysis, visualization, quality assessment, and report generation using AI agents coordinated through AG2 StateFlow. Engineered for small-to-medium datasets (100–100K rows) with expert-validated Metadata-First Hybrid architecture.

**Latest:** 625 unit tests passing, live smoke tests validated on iris.csv and stress_critic.csv. Conclusions and recommendations fortified with business-actionable insights.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [For Data Analysts](#for-data-analysts)
  - [For Developers & Engineers](#for-developers--engineers)
- [Architecture Overview](#architecture-overview)
- [Output Files & Artifacts](#output-files--artifacts)
- [Deployment Recommendations](#deployment-recommendations)
- [Development & Testing](#development--testing)
- [Future Optimizations](#future-optimizations)
- [License & Attribution](#license--attribution)

---

## Quick Start

### For Data Analysts

Get a full EDA report in 2 minutes:

```bash
# 1. Activate the environment
conda activate ag2_env

# 2. Run the pipeline on your data
python main.py your_dataset.csv

# 3. Open the report
open outputs/report.pdf
```

**Output**: A professional PDF report with statistics, visualizations, quality flags, conclusions, and business recommendations.

### For Developers

Set up the development environment and run tests:

```bash
# 1. Clone and navigate
cd autogen_simplified_EDA_tool

# 2. Install dependencies
conda env create -f environment.yml
conda activate ag2_env

# 3. Configure OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Run the test suite
pytest tests/ -v

# 5. Execute a smoke test
python main.py test_data/iris.csv
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **OpenAI API key** (for LLM calls)
- **Optional**: conda or uv for environment management

### Option 1: Conda (Recommended)

```bash
# Create environment from provided YAML
conda env create -f environment.yml
conda activate ag2_env
```

**Or** manually create and install:

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
OPENAI_API_KEY=sk-your-actual-key-here
```

**Security note:** Never commit `.env` to version control. Add it to `.gitignore` (already done).

Alternatively, set via environment variable:

```bash
export OPENAI_API_KEY="sk-your-key"
```

---

## Usage

### For Data Analysts

#### Scenario 1: Basic EDA Report

```bash
python main.py data/sales.csv
```

**What happens:**
1. DataPrepAgent loads and validates your CSV
2. EDAAnalysisAgent computes statistics, missing data, correlations
3. VisualizationAgent generates 6+ charts (histograms, heatmaps)
4. CriticAgent flags data quality issues (high correlations, outliers, etc.)
5. FindingsGeneratorAgent synthesizes 3-lens insights (statistical, ML, business)
6. ReportExporterAgent renders PDF with inline plots + commentary

**Output files:**
- `outputs/report.pdf` — professional 7-page report (PDF format)
- `outputs/report.html` — interactive HTML version (if IPYNB export available)
- `outputs/findings.json` — raw findings data (for programmatic access)
- `outputs/plots/` — individual PNG files for each chart

#### Scenario 2: Custom Dataset Formats

The pipeline supports CSV, Parquet, and Excel files:

```bash
# Parquet file
python main.py data/dataset.parquet

# Excel spreadsheet (single sheet)
python main.py data/analysis.xlsx
```

#### Scenario 3: Development vs. Production Models

By default, the pipeline uses `gpt-5-nano` (fast, cost-effective for testing).

For final validation, switch to `gpt-5-mini` (higher quality, ~5× cost):

```bash
# Development (default, gpt-5-nano)
python main.py test_data/iris.csv

# Production-ready reporting (gpt-5-mini)
EDA_MODE=final python main.py test_data/iris.csv
```

**Cost & timing reference:**
- Dev mode: ~30–60 seconds, ~$0.01 per run
- Final mode: ~60–120 seconds, ~$0.10 per run

#### Scenario 4: Example Datasets

Sample datasets are included in `test_data/`:

- **iris.csv** — Classic Iris dataset (150 rows, 5 columns, balanced, multicollinear)
- **stress_critic.csv** — Edge case dataset for quality thresholds (tests high outlier counts, missing data)

```bash
# Quick test
python main.py test_data/iris.csv

# Stress test the system
python main.py test_data/stress_critic.csv
```

---

### For Developers & Engineers

#### Understanding the Architecture

This project uses **AG2 StateFlow** — a deterministic agent orchestration pattern:

```
user_proxy (initiator)
    │
    ├─→ DataPrepAgent ⇄ DataPrepExecutor (load_data, validate_schema, infer_dtypes)
    │
    ├─→ EDAAnalysisAgent ⇄ EDAAnalysisExecutor (describe_stats, missing_analysis, correlation_matrix)
    │
    ├─→ VisualizationAgent ⇄ VisualizationExecutor (plot_histograms, plot_heatmaps, etc.)
    │
    ├─→ CriticAgent ⇄ CriticExecutor (run_critic_rules)
    │
    ├─→ FindingsGeneratorAgent ⇄ FindingsGeneratorExecutor (prepare_interpretation_context, save_interpretations)
    │
    └─→ ReportExporterAgent ⇄ ReportExporterExecutor (render_pdf, finalize_report)
```

**Key architectural principles:**
1. **Agent = Brain** — Decides which tools to call and when to advance
2. **Executor = Hands** — Runs the tool, returns result, zero decision-making
3. **Two-State Separation** — LLM message history ≠ artifact store (see Lessons 16–23)
4. **Metadata-First Hybrid** — Deterministic fact sheets + LLM interpretation (Lesson 26)

**Reference docs:**
- [architecture.md](architecture.md) — System design, 13 sections
- [lessons_learned.md](lessons_learned.md) — 26 engineering principles (Lessons 1–26)

#### Extending the System: Add a New Tool

**Step 1:** Create a pure-Python tool in `tools/`:

```python
# tools/custom_analysis.py
def custom_analysis(data_json: str) -> str:
    """Your analysis logic here."""
    import json
    df = pd.DataFrame(json.loads(data_json))
    result = {...}  # compute something
    return json.dumps(result)
```

**Step 2:** Register it programmatically in an agent factory (see Lesson #11):

```python
# agents/custom_agent.py
from tools.custom_analysis import custom_analysis

def register_custom_tools(agent, executor):
    agent.register_for_llm(
        description="Analyze custom aspects of the data."
    )(executor.register_for_execution()(custom_analysis))
```

**Step 3:** Add the agent to the orchestrator and route it:

```python
# orchestrator.py
custom_agent, custom_executor = make_agent(...), _create_executor(...)
register_custom_tools(custom_agent, custom_executor)

# In speaker_selection_method():
# elif last_speaker == some_agent:
#     return custom_agent if agent_needs_custom_analysis else next_agent
```

**Step 4:** Test it:

```bash
pytest tests/test_custom_agent.py -v
```

#### Accessing Intermediate Artifacts

During a pipeline run, all intermediate results are saved in an artifact store:

```
outputs/.pipeline_state/<session-uuid>/
├── data_json.json               # Loaded DataFrame as JSON
├── schema_json.json             # Column types, memory profile
├── describe_stats.json          # Descriptive statistics (13 metrics per column)
├── missing_analysis.json        # Missingness percentages
├── correlation_matrix.json      # Correlation matrix (NxN)
├── plot_histograms.json         # Plot file paths
├── plot_correlation_heatmap.json
├── plot_missing_heatmap.json
├── critic_report.json           # Quality flags + metadata
└── findings.json                # Assembled findings with LLM commentary
```

**Access these in Python:**

```python
from tools._pipeline_state import load_state
import json

session_data = load_state("describe_stats")
stats = json.loads(session_data)
print(stats["sepal_length"])
```

#### Understanding the Findings JSON

The `findings.json` artifact contains the synthesized report in structured form:

```json
{
  "overview": "string",
  "missing_data_analysis": "string",
  "correlation_analysis": "string",
  "statistical_analysis": "string",
  "data_quality_assessment": "string",
  "conclusions": "string (4-part: verdict + findings + risks + action)",
  "recommendations_and_business_implications": "string (5 numbered items with ACTION/OUTCOME/RISK)",
  "plot_commentaries": {
    "histogram_sepal_length.png": "3-lens commentary (statistical, ML, business)",
    ...
  }
}
```

**Use this for:**
- Automated report generation in other formats (Word, HTML, email)
- Feeding findings to downstream systems
- Archival and audit trails

#### Running the Test Suite

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_eda_analysis_agent.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Filter by marker (e.g., only critic tests)
pytest -m "not slow" tests/
```

**Test inventory:** 625 tests covering tools, agents, orchestrator, state management, and end-to-end integration.

#### Performance & Cost Tuning

Adjust model selection via environment:

```bash
# Fast iteration (gpt-5-nano, ~$0.01/run)
EDA_MODE=dev python main.py data.csv

# High quality (gpt-5-mini, ~$0.10/run)
EDA_MODE=final python main.py data.csv
```

**Pricing reference (per 1M tokens):**
- gpt-5-nano: $0.05 (input) / $0.40 (output)
- gpt-5-mini: $0.25 (input) / $2.00 (output)

Typical run: ~2K input tokens, ~500 output tokens = $0.01–0.05 depending on mode.

---

## Architecture Overview

### 6-Stage Pipeline

1. **DataPrepAgent** ← Loads, validates, type-infers data
2. **EDAAnalysisAgent** ← Computes 60+ statistics, correlations, missing pattern
3. **VisualizationAgent** ← Generates 6 plots (histograms, heatmaps)
4. **CriticAgent** ← Flags 13 data quality rules (outliers, multicollinearity, skew)
5. **FindingsGeneratorAgent** ← LLM synthesizes 3-lens insights from metadata (deterministic facts + LLM reasoning)
6. **ReportExporterAgent** ← Renders PDF/IPYNB with plots + commentary

### Metadata-First Hybrid (Lesson 26)

**Problem:** Raw data overflows LLM context. Vision hallucinations on exact values.

**Solution:** Intermediate fact blocks delivered to FindingsGeneratorAgent:
- Histogram bin counts + edges (complete shape DNA)
- 5-number summaries per column
- Correlation matrix (all cell values)
- Missing percentages per column
- Critic flags

**LLM receives:** ~6.5K tokens of deterministic facts → synthesizes 3-lens insights without hallucination.

**Result:** 100% data coverage + expert-quality interpretation.

### Two-State Separation (Lesson 16)

| Aspect | Conversation State | Pipeline State |
|--------|------------------|-----------------|
| **Lives in** | AG2 message history (ephemeral) | Artifact store on disk (persistent) |
| **Size** | Few KB (tokens) | Up to 50MB (DataFrames, plots) |
| **LLM role** | Author/consumer (reasoning) | Neither (infrastructure) |
| **Example** | "I'll call describe_stats next" | 17KB DataFrame JSON, 898B stats dict |

**Why this matters:** Small LLMs (gpt-5-nano) cannot copy large JSON from messages into tool parameters. Instead, tools save to disk and return `STATE_REF:key` references. Downstream tools load from disk. LLM only handles 30-char references, not 15KB blobs.

See [lessons_learned.md](lessons_learned.md) Lessons 16–23 for full details.

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

#### 2. **findings.json** (Structured Data)
- **Location:** `outputs/findings.json`
- **Format:** Pydantic model serialized to JSON
- **Contents:** 7 main sections + per-plot 3-lens commentary
- **Audience:** Developers, downstream systems, APIs

#### 3. **Visualizations** (Plots)
- **Location:** `outputs/plots/`
- **Files:**
  - `histogram_<column>.png` (1 per numeric column)
  - `correlation_heatmap.png` (1, if N>1 numeric columns)
  - `missing_heatmap.png` (1, if any missing data)
- **Size:** ~50–100 KB total (embedded in PDF, also standalone)
- **Audience:** Reports, presentations, documentation

#### 4. **Session Artifacts** (Developer Access)
- **Location:** `outputs/.pipeline_state/<uuid>/`
- **Contents:**
  - `data_json.json` — DataFrame as records JSON
  - `schema_json.json` — Column metadata
  - `describe_stats.json` — 13 statistics per column
  - `missing_analysis.json` — Null/NaN percentages
  - `correlation_matrix.json` — N×N correlation matrix
  - `critic_report.json` — Quality flags
  - `findings.json` — Assembled findings
- **Access:** Via `tools._pipeline_state.load_state(key)`
- **Audience:** Developers, automated workflows

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
python main.py data/analysis.csv
```

✅ Simple, no infrastructure required  
✅ Fast feedback loop for prototyping  
⚠️ No concurrent requests, state file cleanup manual  

### Web UI (Streamlit/Gradio)

**Recommended architecture** (no code included, architectural guidance):

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
conda env create -f environment.yml
conda activate ag2_env

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
- **[tests/](tests/)** — 625 unit tests
- **[lessons_learned.md](lessons_learned.md)** — 26 architectural lessons
- **[architecture.md](architecture.md)** — System design (13 sections)

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

**Current state:** 625 tests passing, ruff clean, zero linting errors.

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

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Ensure `.env` file exists in project root with valid key
- Or set env var: `export OPENAI_API_KEY="sk-..."`

**"File not found: iris.csv"**
- Use full path or place file in project root
- Or copy to `test_data/`: `cp my_data.csv test_data/`

**"gpt-5-nano not available" / HTTP 400 error**
- Check OpenAI account has access to latest models
- Verify API key is correct: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

**Tests fail with "PipelineStateError"**
- Session cleanup may have left stale state dirs
- `rm -rf outputs/.pipeline_state/` and retry

**PDF not generated**
- Check `outputs/plots/` directory exists and is writable
- Verify `report.pdf` isn't open in another process (Windows)

---

## License & Attribution

**AG2 Framework:** [Apache 2.0](https://github.com/ag2ai/ag2)  
**Project:** MIT (your choice)

**Design Principles Reference:**  
- StateFlow controller pattern (Lesson 1)
- Dedicated executor architecture (Lesson 2)
- Artifact store pattern (Lessons 16–23)
- Metadata-First Hybrid (Lesson 26)

See [lessons_learned.md](lessons_learned.md) for full architectural documentation.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) (if available) or:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-agent`)
3. Add tests (`pytest tests/test_my_agent.py`)
4. Run full test suite (`pytest tests/ -v`)
5. Commit & push
6. Open a PR with reference to specific Lessons if architectural changes

---

## Questions?

- **Architecture:** See [architecture.md](architecture.md)
- **Engineering details:** See [lessons_learned.md](lessons_learned.md) (26 lessons)
- **Agent development:** Lesson #11 (tool registration) + Lesson #1 (agent decision flow)
- **Performance tuning:** Config EDA_MODE, adjust model selection

Happy analyzing! 🚀
