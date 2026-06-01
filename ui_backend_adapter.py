"""ui_backend_adapter.py — streamlit-deploy branch ONLY.

Thin adapter that isolates streamlit_app.py from the pipeline implementation.
streamlit_app.py must import exclusively from this module — never from
pipeline, core.*, agents.*, or tools.* directly.

Phase 3 migration: only this file changes — swap submit_eda_job body for an
HTTP client call to FastAPI. streamlit_app.py remains unchanged.
"""
from __future__ import annotations

from pathlib import Path

from core.concurrency import SystemBusy  # noqa: F401 — re-exported for UI
from pipeline import run_pipeline
from tools.data_loader import (  # noqa: F401 — re-exported for UI
    _get_loader,
    detect_encoded_categoricals,
    detect_target,
)

# Per-session cooldown: seconds between successful submissions in the same
# browser tab.  Prevents accidental double-submissions.
# Cross-tab and cross-user protection is handled by BoundedSemaphore(1).
COOLDOWN_SECONDS: int = 300  # 5 minutes


def submit_eda_job(
    file_path: Path,
    *,
    target_flag: str | None,
    no_target_flag: bool,
    categoricals_flag: str | None,
    no_reclassify_flag: bool,
    enable_openlit: bool = False,
) -> str:
    """Submit an EDA job and return the session_id (or cache key on a hit).

    Raises SystemBusy when another job is already in progress.

    Phase 3: replace body with ``POST /jobs`` + return job_id string.
    """
    return run_pipeline(
        file_path=file_path,
        target_flag=target_flag,
        no_target_flag=no_target_flag,
        categoricals_flag=categoricals_flag,
        no_reclassify_flag=no_reclassify_flag,
        enable_openlit=enable_openlit,
    )
