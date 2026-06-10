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
import contextlib
import os

os.environ["IPYNB_EXPORT"] = "true"

# ---------------------------------------------------------------------------
# Streamlit Cloud secret injection — MUST run before any project imports.
#
# On Streamlit Cloud the .env file is absent (it is gitignored).
# Environment variables must be configured via App Settings → Secrets:
#   EDA_MODE = "final"
#   OPENAI_API_KEY = "sk-..."
#   (etc.)
#
# We inject each secret into os.environ here, before config.py is imported,
# so that load_dotenv() in config.py is a no-op and the secrets take effect.
# Secrets that are already present (e.g. from a local .env) are NOT
# overwritten, preserving local developer workflow.
# ---------------------------------------------------------------------------
try:
    import streamlit as _st_early
    for _key, _val in _st_early.secrets.items():
        if isinstance(_val, str) and _key not in os.environ:
            os.environ[_key] = _val
except Exception:
    pass  # running locally without secrets — .env covers it

import datetime
import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st

from config import get_outputs_dir, get_plots_dir
from eda_state import EncodedCategoricalSuspect, TargetInfo
from ui_backend_adapter import (
    COOLDOWN_SECONDS,
    SystemBusy,
    _get_loader,
    detect_encoded_categoricals,
    detect_target,
    get_cache_status,
    get_submission_key,
    has_cached_result,
    submit_eda_job,
)

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


def _reset_uploaded_workflow() -> None:
    """Clear upload/results state so the user can start a fresh run."""
    tmpdir = st.session_state.pop("tmpdir", None)
    if tmpdir is not None:
        with contextlib.suppress(Exception):
            tmpdir.cleanup()

    for key in (
        "file_path",
        "file_name",
        "file_sig",
        "df",
        "file_sig_loaded",
        "target_candidate",
        "file_sig_target",
        "suspects",
        "file_sig_suspects",
        "session_id",
        "last_run_at",
        "last_submission_key",
        "running",
    ):
        st.session_state.pop(key, None)

    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1


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
# Sidebar — cache status indicator
# ---------------------------------------------------------------------------
with st.sidebar:
    _cache_on, _eda_mode, _cache_dir = get_cache_status()
    if _cache_on:
        st.success(f"Cache **ON** (`EDA_MODE={_eda_mode}`)\n\n`{_cache_dir}`")
    else:
        st.warning(
            f"Cache **OFF** (`EDA_MODE={_eda_mode}`)\n\n"
            "Set `EDA_MODE = \"final\"` in Streamlit Cloud Secrets to enable the cache."
        )


# ---------------------------------------------------------------------------
# Phase A — Upload
# ---------------------------------------------------------------------------
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0

uploaded = st.file_uploader(
    "Upload dataset",
    type=["csv", "parquet", "xlsx"],
    help="Max 50 MB. Supported formats: CSV, Parquet, XLSX.",
    key=f"dataset_uploader_{st.session_state['uploader_nonce']}",
)

st.caption(
    "⚠️ **Required format:** The first row of the dataset must contain feature names (column headers). "
    "Headerless files are not supported."
)

if uploaded is None:
    st.info("Upload a dataset to get started.")
    st.stop()

if st.button("Upload a different dataset", use_container_width=False):
    _reset_uploaded_workflow()
    st.rerun()

# Persist uploaded file in a session-scoped temp directory
if "tmpdir" not in st.session_state:
    st.session_state["tmpdir"] = tempfile.TemporaryDirectory()

tmpdir_path = Path(st.session_state["tmpdir"].name)
uploaded_bytes = uploaded.getvalue()
uploaded_sig = hashlib.sha256(uploaded_bytes).hexdigest()

if "file_path" not in st.session_state or st.session_state.get("file_sig") != uploaded_sig:
    # Clear previous uploaded files from this session temp dir
    for p in tmpdir_path.iterdir():
        if p.is_file():
            p.unlink(missing_ok=True)
    upload_path = tmpdir_path / uploaded.name
    upload_path.write_bytes(uploaded_bytes)
    st.session_state["file_path"] = upload_path
    st.session_state["file_name"] = uploaded.name
    st.session_state["file_sig"] = uploaded_sig
    # Clear stale results when a new file is uploaded
    st.session_state.pop("session_id", None)

file_path = st.session_state["file_path"]

# ---------------------------------------------------------------------------
# Load DataFrame for configuration widgets
# ---------------------------------------------------------------------------
if "df" not in st.session_state or st.session_state.get("file_sig_loaded") != uploaded_sig:
    loader = _get_loader(str(file_path))
    loaded_df = loader.load(str(file_path))
    loaded_df = loaded_df.drop_duplicates().reset_index(drop=True)
    st.session_state["df"] = loaded_df
    st.session_state["file_sig_loaded"] = uploaded_sig

df = st.session_state["df"]

st.success(f"**{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# Phase B — Configure: Target Detection
# ---------------------------------------------------------------------------
st.header("1. Target Variable")

# Run heuristic detection (cached per file)
if "target_candidate" not in st.session_state or st.session_state.get("file_sig_target") != uploaded_sig:
    candidate_json = detect_target(df.to_json(orient="records"))
    st.session_state["target_candidate"] = TargetInfo.model_validate_json(candidate_json)
    st.session_state["file_sig_target"] = uploaded_sig

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

if "suspects" not in st.session_state or st.session_state.get("file_sig_suspects") != uploaded_sig:
    st.session_state["suspects"] = detect_encoded_categoricals(
        df, target_column=target_flag
    )
    st.session_state["file_sig_suspects"] = uploaded_sig

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

if "running" not in st.session_state:
    st.session_state["running"] = False

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

# Per-session cooldown: prevent accidental double-submissions in the same tab.
_now = datetime.datetime.now()
_last_run: datetime.datetime | None = st.session_state.get("last_run_at")
_cooldown_remaining: int = (
    max(0, int(COOLDOWN_SECONDS - (_now - _last_run).total_seconds()))
    if _last_run is not None
    else 0
)
_current_submission_key = get_submission_key(
    file_path=file_path,
    target_flag=target_flag,
    no_target_flag=no_target_flag,
    categoricals_flag=categoricals_flag,
    no_reclassify_flag=(categoricals_flag is None and not suspects),
)
_same_submission_as_last = st.session_state.get("last_submission_key") == _current_submission_key
_cached_result_available = has_cached_result(
    file_path=file_path,
    target_flag=target_flag,
    no_target_flag=no_target_flag,
    categoricals_flag=categoricals_flag,
    no_reclassify_flag=(categoricals_flag is None and not suspects),
)
_in_cooldown: bool = (
    _cooldown_remaining > 0
    and _same_submission_as_last
    and not _cached_result_available
)

run_clicked = st.button(
    "▶ Run Pipeline",
    type="primary",
    use_container_width=True,
    disabled=st.session_state["running"] or _in_cooldown,
)
if _in_cooldown:
    st.caption(
        f"Per-session cooldown active — {_cooldown_remaining}s remaining before next submission."
    )
elif _cooldown_remaining > 0 and _same_submission_as_last and _cached_result_available:
    st.caption("Cached result available — cooldown bypassed for identical inputs.")

if run_clicked:
    st.session_state["running"] = True
    run_succeeded = False
    try:
        with st.spinner(
            "Running EDA pipeline — typically 4–6 minutes for small datasets; larger datasets may take longer."
        ):
            session_id = submit_eda_job(
                file_path=file_path,
                target_flag=target_flag,
                no_target_flag=no_target_flag,
                enable_openlit=False,
                categoricals_flag=categoricals_flag,
                no_reclassify_flag=(categoricals_flag is None and not suspects),
            )
        st.session_state["session_id"] = session_id
        st.session_state["last_run_at"] = datetime.datetime.now()
        st.session_state["last_submission_key"] = _current_submission_key
        run_succeeded = True
    except SystemBusy:
        st.warning("Another user is currently running the pipeline. Please retry in a minute.")
    finally:
        st.session_state["running"] = False

    if run_succeeded:
        st.rerun()

# ---------------------------------------------------------------------------
# Phase D — Display Results
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.stop()

session_id = st.session_state["session_id"]
out_dir = get_outputs_dir(session_id)
plots_dir = get_plots_dir(session_id)

if not out_dir.exists():
    st.warning(
        f"Output directory not found for session `{session_id}` — the app may have cold-started, "
        "the cached path may be stale, or the artifacts may have been cleaned up."
    )
    st.stop()

st.header("4. Results")

hit_summary_path = out_dir / "hit_summary.txt"
if hit_summary_path.exists():
    st.info(hit_summary_path.read_text(encoding="utf-8"))

report_md = out_dir / "report.md"
report_pdf = out_dir / "report.pdf"
report_ipynb = out_dir / "report.ipynb"
cost_path = out_dir / "cost_summary.txt"
timings_path = out_dir / "timings.jsonl"

debug = st.query_params.get("debug") == "1"

tabs_labels = [
    "📊 Plots",
    "📄 Markdown Report",
    "📕 PDF Report",
    "📓 Notebook",
    "💰 Cost Summary",
]
if debug:
    tabs_labels.append("⏱ Timings")

tabs = st.tabs(tabs_labels)
tab_plots, tab_md, tab_pdf, tab_ipynb, tab_cost = tabs[:5]

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

    if hit_summary_path.exists():
        st.markdown("---")
        st.caption("Most recent cache hit summary")
        st.code(hit_summary_path.read_text(encoding="utf-8"))

if debug:
    with tabs[-1]:
        if timings_path.exists():
            records = [
                json.loads(line)
                for line in timings_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            st.dataframe(records, use_container_width=True)
        else:
            st.info("No timings recorded.")

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
