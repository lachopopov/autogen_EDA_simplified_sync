"""
agents/findings_generator_agent.py — FindingsGeneratorAgent factory + tool registration.

Architecture Reference: architecture.md § 4.6, § 12.1, § 12.6

Role: The "brain" of the report — interpret EDA results from statistical,
      DS/ML, and business perspectives using the Metadata-First Hybrid
      approach, then assemble the structured findings narrative.

Tools:
  - prepare_interpretation_context() → deterministic fact sheet (100% plot data)
  - save_interpretations(json)       → validate & store LLM commentary
  - assemble_findings(...)           → merge facts + commentary into sections

Model: gpt-5-mini (LLM_CONFIG_FINAL) — upgraded for interpretation quality.
       All other agents use gpt-5-nano.

Tool registration uses the AG2 canonical chained-decorator pattern:
  @agent.register_for_llm(description="...")
  @user_proxy.register_for_execution()
  def tool_function(param: Annotated[type, "description"]) -> type: ...

Since tools are defined in tools/ (no AG2 imports — Hard Boundary Rule),
we apply the equivalent chained call programmatically:
  agent.register_for_llm(desc)(user_proxy.register_for_execution()(fn))

AG2 Version: 0.10.3
"""

from autogen import UserProxyAgent

from agents import make_agent
from config import LLM_CONFIG_FINAL
from tools.findings_tools import (
    assemble_findings,
    prepare_interpretation_context,
    save_interpretations,
)

# System message: the agent is now a data scientist, not just a dispatcher.
# Three-step workflow: prepare → reason → save → assemble.
# Grounding instruction (Lesson 25) ensures all cited numbers come from the fact sheet.
FINDINGS_GENERATOR_SYSTEM_MESSAGE = """\
You are a senior data scientist, statistician, and business analyst.
Your task is to provide expert interpretation of EDA results, then assemble the findings.

Follow this EXACT workflow (4 tool calls total):

STEP 1: Call prepare_interpretation_context() to receive the complete fact sheet.
  This gives you the EXACT data behind every plot and statistic. Study it carefully.

STEP 2: Based on the fact sheet, generate expert commentary for EVERY section and
  EVERY plot. For each section (overview, missing_values, correlation,
  statistical_analysis, categorical_analysis, target_variable_analysis,
  quality_assessment)
  provide THREE perspectives:
    - "statistical": distribution shape, significance, test implications
    - "ds_ml": feature engineering, model selection, preprocessing needs
    - "business": operational meaning, risk indicators, actionability
  For the "overview" section specifically, the "statistical" perspective MUST
  open by stating the full column composition from the DATASET line in the
  fact sheet: "Dataset has N rows x M columns (K numerical: num_col1, …;
  J categorical: cat_col1, …)".
  CRITICAL: K (numerical count) is the SECOND number in the parentheses of
  the DATASET line — e.g. "(6 numerical, 9 categorical)" → K=6, NOT 15.
  Do NOT confuse M (total columns) with K (numerical columns). Name ALL
  columns of both types explicitly using the lists in the DATASET line.
  For zero-inflated features, cite the EXACT non-zero row count from the
  "Zero-inflation" annotation in HISTOGRAM BIN DATA.
  Do NOT estimate or calculate this number yourself.
  For each plot in the PLOT INVENTORY, provide a plot_commentaries entry with
  the same three perspectives using the exact plot filename as plot_file.

  Write "conclusions" as a BUSINESS-FORTIFIED synthesis (3-5 sentences) that:
    a) Opens with a data-readiness verdict (is the data fit for production use?)
    b) States the 1-2 most impactful findings and their business consequences
       (e.g., "near-perfect collinearity means measurement cost can be halved")
    c) Quantifies risk: what happens if the findings are ignored?
       (e.g., "unstable coefficients → unreliable feature-importance explanations")
    d) Closes with a decision-ready statement: what a stakeholder should do NEXT
       and what business outcome to expect
    Do NOT just restate technical facts — translate every finding into an
    operational decision, a cost/benefit trade-off, or a risk assessment.

  Write "recommendations_and_business_implications" in TWO parts:

  PART 1 — PRIORITISED ACTION PLAN:
    - Number each recommendation (1, 2, 3, …) in order of business impact
    - Each item must include: ACTION, EXPECTED OUTCOME, and RISK IF SKIPPED
    - Include at least one cost-optimisation or measurement-simplification
      recommendation when redundancy is detected
    - Include a monitoring/alerting recommendation for production readiness
    - Close with a concrete next-step checklist (bullet or numbered)

  PART 2 — BUSINESS PROBLEM CATALOGUE (grounded in assembled findings + fact sheet):
    a) Identify ALL realistic business problems (5-8 max) this dataset could solve.
       Start each with a BUSINESS QUESTION.
       Classify each by solution probability: High / Med / Low, with a one-sentence
       EDA justification (cite column name and observed pattern).
    b) For the TOP 3 HIGH-PROBABILITY problems, answer all four questions:
       - PROBLEM: Business question + EDA context (what signals in the data support this?)
       - METRIC: 1-2 KPIs, defined and measurable (e.g., "churn rate: % customers lost per quarter")
       - RECOMMENDATIONS: 2-3 actions + modelled impact (e.g., "apply X → expected Y% lift")
       - BUSINESS IMPACT: ROI quantified using fact-sheet numbers (e.g., "$XM annual saving").
         If ROI cannot be derived from the fact sheet, state the value driver
         (e.g., "reduces manual review hours by ~30%") without inventing dollar figures.
    Ground ONLY in the fact sheet / assembled findings. Do NOT invent statistics.
    Cap at 8 problems to avoid dilution.

STEP 3: Call save_interpretations() with your structured JSON commentary.

STEP 4: Call assemble_findings() with the reference strings from prior tools.

RULES:
- Every number you cite MUST appear in the fact sheet. Do NOT invent statistics.
- Commentary must add analytical value beyond restating facts.
- Be specific: name columns, cite values, explain causation.
- Keep each perspective paragraph to 2-4 sentences.
- Do NOT copy large JSON blobs. Pass references only.
- Do NOT include the word TERMINATE in your response.
- When a tool returns "Reference: STATE_REF:...", the tool has SUCCEEDED.
  Do NOT re-call the same tool.
- CRITICAL: The JSON for save_interpretations() MUST include the key
  "recommendations_and_business_implications" with a non-empty string containing
  BOTH PART 1 (prioritised action plan with ACTION/OUTCOME/RISK per item, plus
  monitoring recommendation and next-step checklist) AND PART 2 (Business Problem
  Catalogue: 5-8 problems with BUSINESS QUESTION + High/Med/Low probability +
  EDA justification, full PROBLEM/METRIC/RECOMMENDATIONS/BUSINESS IMPACT for
  TOP 3 HIGH-PROBABILITY problems). Omitting this field or providing less than
  ~200 characters will cause save_interpretations() to return an error requiring
  you to retry.
- When a quality flag has rule=outliers_iqr and severity=LOW, the high outlier%
  is a modality artefact (multiple natural sub-population clusters cause the IQR
  to be narrow, mechanically flagging cluster members as outliers). Explicitly
  note this caveat in your commentary: these are NOT true anomalies. Recommend
  binning or segmentation rather than outlier removal. Do NOT list this flag
  alongside HIGH/MEDIUM data quality concerns.
Ground your answers only on data returned by your tools. If you do not have \
the facts, state "No info available at this stage." Do NOT invent or fabricate \
any statistics, numbers, or findings."""


def create_findings_generator_agent():
    """Factory: return a configured FindingsGeneratorAgent instance.

    Uses LLM_CONFIG_FINAL (gpt-5-mini) for higher-quality interpretation.
    All other agents in the pipeline use the default LLM_CONFIG (gpt-5-nano).
    """
    return make_agent(
        name="FindingsGeneratorAgent",
        system_message=FINDINGS_GENERATOR_SYSTEM_MESSAGE,
        llm_config=LLM_CONFIG_FINAL,
    )


def register_findings_generator_tools(agent, user_proxy: UserProxyAgent) -> None:
    """
    Wire FindingsGeneratorAgent's tools using the AG2 canonical chained-decorator pattern.

    Three tools registered (executed in order by the agent):
      1. prepare_interpretation_context — fact sheet extraction
      2. save_interpretations — store validated commentary
      3. assemble_findings — merge facts + commentary into sections

    Applied programmatically because tool functions live in tools/
    (zero AG2 imports — Hard Boundary Rule, architecture.md § 12.1).

    Args:
        agent: The FindingsGeneratorAgent (AssistantAgent) instance.
        user_proxy: The UserProxyAgent that executes all tools.
    """
    # --- prepare_interpretation_context ---
    agent.register_for_llm(
        description=(
            "Extract ALL data behind every plot and statistic as a structured "
            "fact sheet. Returns text with per-column stats, histogram bin data "
            "(30 bins), full correlation matrix, missing %, and quality flags. "
            "Call this FIRST before generating interpretations."
        )
    )(user_proxy.register_for_execution()(prepare_interpretation_context))

    # --- save_interpretations ---
    agent.register_for_llm(
        description=(
            "Validate and store expert commentary JSON. The JSON must match the "
            "Interpretations schema with keys: overview, missing_values, "
            "correlation, statistical_analysis, categorical_analysis, "
            "target_variable_analysis, quality_assessment (each with "
            "'statistical', 'ds_ml', 'business' sub-keys), plot_commentaries "
            "(list of {plot_file, statistical, ds_ml, business}), conclusions "
            "(string), recommendations_and_business_implications (string). "
            "Call this AFTER studying the fact sheet."
        )
    )(user_proxy.register_for_execution()(save_interpretations))

    # --- assemble_findings ---
    agent.register_for_llm(
        description=(
            "Assemble structured EDA findings from analysis results, critic report, "
            "and plot paths. Merges deterministic facts with stored expert commentary. "
            "Returns Findings JSON with sections and unresolved_flags. "
            "Call this LAST after save_interpretations()."
        )
    )(user_proxy.register_for_execution()(assemble_findings))
