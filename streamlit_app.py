"""
streamlit_app.py — Streamlit web UI for the EDA Multi-Agent pipeline.

Usage:
    streamlit run streamlit_app.py

Phases:
  1. Upload   — user uploads CSV / Parquet / XLSX
  2. Configure — confirm target variable + encoded categorical subtypes
  3. Execute  — run_pipeline() under st.spinner()
  4. Results  — display plots, reports, cost; offer downloads
"""

from __future__ import annotations

# Enable all 3 report formats BEFORE any project imports read the env var.
import os

os.environ["IPYNB_EXPORT"] = "true"

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import get_outputs_dir, get_plots_dir
from eda_state import EncodedCategoricalSuspect, TargetInfo
from main import run_pipeline
from tools.data_loader import _get_loader, detect_encoded_categoricals, detect_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_md_images(md_text: str, base_dir: Path) -> str:
    """Replace relative ``![alt](path)`` image refs with base64 data URIs.

    This allows ``st.markdown()`` to render images that live on disk
    without needing Streamlit's static-file serving.
    """
    import base64
    import re

    def _replace(m: re.Match) -> str:
        alt, rel_path = m.group(1), m.group(2)
        if rel_path.startswith(("data:", "http://", "https://")):
            return m.group(0)  # already absolute or embedded
        img_path = base_dir / rel_path
        if not img_path.is_file():
            return m.group(0)
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"![{alt}](data:image/png;base64,{b64})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, md_text)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EDA Multi-Agent AG2 Report Generator For Classification Tasks",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EDA Multi-Agent AG2 Report Generator For Classification Tasks")
st.caption("Upload a dataset, configure target & categoricals, then run the pipeline.")


# ---------------------------------------------------------------------------
# Phase A — Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload dataset",
    type=["csv", "parquet", "xlsx"],
    help="Max 50 MB. Supported formats: CSV, Parquet, XLSX.",
)

st.caption(
    "⚠️ **Required format:** The first row of the dataset must contain feature names (column headers). "
    "Headerless files are not supported."
)

if uploaded is None:
    st.info("Upload a dataset to get started.")
    st.stop()

# Persist uploaded file to a temp path (Streamlit doesn't expose a real path)
if "file_path" not in st.session_state or st.session_state.get("file_name") != uploaded.name:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp.close()
    st.session_state["file_path"] = Path(tmp.name)
    st.session_state["file_name"] = uploaded.name
    # Clear stale results when a new file is uploaded
    st.session_state.pop("session_id", None)

file_path: Path = st.session_state["file_path"]

# ---------------------------------------------------------------------------
# Load DataFrame for configuration widgets
# ---------------------------------------------------------------------------
if "df" not in st.session_state or st.session_state.get("file_name_loaded") != uploaded.name:
    loader = _get_loader(str(file_path))
    df = loader.load(str(file_path))
    df = df.drop_duplicates().reset_index(drop=True)
    st.session_state["df"] = df
    st.session_state["file_name_loaded"] = uploaded.name

df: pd.DataFrame = st.session_state["df"]

st.success(f"**{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# Phase B — Configure: Target Detection
# ---------------------------------------------------------------------------
st.header("1. Target Variable")

# Run heuristic detection (cached per file)
if "target_candidate" not in st.session_state or st.session_state.get("file_name_target") != uploaded.name:
    candidate_json = detect_target(df.to_json(orient="records"))
    st.session_state["target_candidate"] = TargetInfo.model_validate_json(candidate_json)
    st.session_state["file_name_target"] = uploaded.name

candidate: TargetInfo = st.session_state["target_candidate"]

target_mode = st.radio(
    "Target detection mode",
    ["Auto-detected", "Select manually", "No target (unsupervised)"],
    index=0,
    horizontal=True,
)

target_flag: str | None = None
no_target_flag: bool = False

if target_mode == "Auto-detected":
    if candidate.column:
        st.info(
            f"Detected: **{candidate.column}** "
            f"({candidate.problem_type}, method: {candidate.detection_method})"
        )
        if candidate.problem_type == "classification" and candidate.class_counts:
            class_str = ", ".join(
                f"{k} ({v})" for k, v in candidate.class_counts.items()
            )
            st.caption(f"Classes ({candidate.n_classes}): {class_str}")
        target_flag = candidate.column
    else:
        st.warning("No target candidate detected — running unsupervised.")
        no_target_flag = True
elif target_mode == "Select manually":
    target_flag = st.selectbox("Choose target column", df.columns.tolist())
else:
    no_target_flag = True

# ---------------------------------------------------------------------------
# Phase B — Configure: Encoded Categorical Detection
# ---------------------------------------------------------------------------
st.header("2. Encoded Categorical Columns")

if "suspects" not in st.session_state or st.session_state.get("file_name_suspects") != uploaded.name:
    st.session_state["suspects"] = detect_encoded_categoricals(
        df, target_column=target_flag
    )
    st.session_state["file_name_suspects"] = uploaded.name

suspects: list[EncodedCategoricalSuspect] = st.session_state["suspects"]

confirmed_cols: list[str] = []
confirmed_subtypes: dict[str, str] = {}

if suspects:
    st.caption(
        "These numeric columns may be encoded categoricals. "
        "Ordinal columns preserve value ordering (e.g. education level 1→4). "
        "Nominal columns have no meaningful order (e.g. SEX encoded as 1/2)."
    )
    for s in suspects:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                sample_str = ", ".join(str(v) for v in s.sample_values[:10])
                accepted = st.checkbox(
                    f"**{s.column}** — nunique={s.nunique}, values: [{sample_str}]",
                    value=True,
                    key=f"cat_{s.column}",
                )
                st.caption(f"Reason: {s.reason}")
            with right:
                if accepted:
                    default_idx = 0 if (s.subtype or "nominal") == "nominal" else 1
                    subtype = st.selectbox(
                        "Type",
                        ["nominal", "ordinal"],
                        index=default_idx,
                        key=f"subtype_{s.column}",
                    )
                    confirmed_cols.append(s.column)
                    confirmed_subtypes[s.column] = subtype
else:
    st.info("No encoded categorical suspects detected.")

categoricals_flag: str | None = ",".join(confirmed_cols) if confirmed_cols else None
subtypes_flag: dict[str, str] | None = confirmed_subtypes if confirmed_subtypes else None

# ---------------------------------------------------------------------------
# Phase C — Execute Pipeline
# ---------------------------------------------------------------------------
st.header("3. Run Pipeline")

# Show a summary of configuration before running
with st.expander("Configuration summary", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Target**: {target_flag or '(unsupervised)'}")
        st.markdown(f"**File**: {uploaded.name}")
    with col2:
        if confirmed_cols:
            st.markdown(f"**Categoricals**: {', '.join(confirmed_cols)}")
            subtype_str = ", ".join(f"{c}={t}" for c, t in confirmed_subtypes.items())
            st.markdown(f"**Subtypes**: {subtype_str}")
        else:
            st.markdown("**Categoricals**: (none confirmed)")

if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
    with st.spinner("Running EDA pipeline — this may take a few minutes..."):
        session_id = run_pipeline(
            file_path=file_path,
            target_flag=target_flag,
            no_target_flag=no_target_flag,
            enable_openlit=False,
            categoricals_flag=categoricals_flag,
            subtypes_flag=subtypes_flag,
            no_reclassify_flag=(categoricals_flag is None and not suspects),
        )
        st.session_state["session_id"] = session_id
    st.rerun()

# ---------------------------------------------------------------------------
# Phase D — Display Results
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.stop()

session_id: str = st.session_state["session_id"]
out_dir = get_outputs_dir(session_id)
plots_dir = get_plots_dir(session_id)

if not out_dir.exists():
    st.warning("Output directory not found — the run may have been cleaned up.")
    st.stop()

st.header("4. Results")

report_md = out_dir / "report.md"
report_pdf = out_dir / "report.pdf"
report_ipynb = out_dir / "report.ipynb"
cost_path = out_dir / "cost_summary.txt"

tab_plots, tab_md, tab_pdf, tab_ipynb, tab_cost = st.tabs(
    ["📊 Plots", "📄 Markdown Report", "📕 PDF Report", "📓 Notebook", "💰 Cost Summary"]
)

with tab_plots:
    plot_files = sorted(plots_dir.glob("*.png"))
    if plot_files:
        cols = st.columns(2)
        for i, pf in enumerate(plot_files):
            with cols[i % 2]:
                st.image(str(pf), caption=pf.stem)
    else:
        st.info("No plots generated.")

with tab_md:
    if report_md.exists():
        md_text = report_md.read_text(encoding="utf-8")
        # Resolve relative image paths to base64 for Streamlit rendering
        md_text = _resolve_md_images(md_text, out_dir)
        st.markdown(md_text, unsafe_allow_html=True)
    else:
        st.info("Markdown report not found.")

with tab_pdf:
    if report_pdf.exists():
        st.info("PDF report generated. Use the download button below.")
    else:
        st.info("PDF report not found.")

with tab_ipynb:
    if report_ipynb.exists():
        st.info("Jupyter notebook generated. Use the download button below.")
    else:
        st.info("Notebook not found.")

with tab_cost:
    st.markdown(
        "_Model pricing used for cost calculation (April 2026):_\n\n"
        "| Model | Input | Cached input | Output |\n"
        "|---|---|---|---|\n"
        "| `gpt-5` | \\$1.25 / 1M tokens | \\$0.125 / 1M tokens | \\$10.00 / 1M tokens |\n"
        "| `gpt-5-mini` | \\$0.25 / 1M tokens | \\$0.025 / 1M tokens | \\$2.00 / 1M tokens |"
    )
    if cost_path.exists():
        st.code(cost_path.read_text(encoding="utf-8"))
    else:
        st.info("Cost summary not found.")

# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------
st.subheader("Downloads")
dl_cols = st.columns(4)

with dl_cols[0]:
    if report_md.exists():
        st.download_button(
            "📄 Markdown Report",
            report_md.read_text(encoding="utf-8"),
            file_name="eda_report.md",
            mime="text/markdown",
        )

with dl_cols[1]:
    if report_pdf.exists():
        st.download_button(
            "📕 PDF Report",
            report_pdf.read_bytes(),
            file_name="eda_report.pdf",
            mime="application/pdf",
        )

with dl_cols[2]:
    if report_ipynb.exists():
        st.download_button(
            "📓 Jupyter Notebook",
            report_ipynb.read_text(encoding="utf-8"),
            file_name="eda_report.ipynb",
            mime="application/x-ipynb+json",
        )

with dl_cols[3]:
    if cost_path.exists():
        st.download_button(
            "💰 Cost Summary",
            cost_path.read_text(encoding="utf-8"),
            file_name="cost_summary.txt",
            mime="text/plain",
        )
