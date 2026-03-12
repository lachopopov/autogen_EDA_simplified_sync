"""
tools/report_tools.py — Template Method pattern for report rendering.

Architecture Reference: architecture.md § 4.7, § 12.5

Public AG2-facing functions:
  - render_pdf(findings_json, output_dir) -> str
  - render_ipynb(findings_json, output_dir) -> str

Design:
  - Template Method: ReportRenderer ABC
    - render():            invariant algorithm — never overridden
    - _build_sections():   shared — extracts ordered sections from Findings
    - _format_sections():  shared — structures content + embeds plot references
    - _write_output():     **varies** — PDF (reportlab) vs IPYNB (nbformat)
  - PDFRenderer:  reportlab.platypus high-level layout (SimpleDocTemplate)
  - IPYNBRenderer: nbformat v4 notebook construction
  - render_pdf() and render_ipynb() are the only AG2-facing symbols
  - Zero AG2 imports. Pure Python.

Report structure (Option A — plots inline in parent sections):
  1. EDA Report (title)
  2. Dataset Overview             + expert commentary (3 perspectives)
  3. Missing Values               + expert commentary + missing_heatmap.png
  4. Correlation Analysis          + expert commentary + correlation_heatmap.png
  5. Statistical Analysis          + expert commentary + hist_*.png per column
  6. Data Quality Assessment       + expert commentary
  7. Conclusions                   (LLM-synthesized or deterministic fallback)
  8. Recommendations & Business    (LLM-synthesized or deterministic fallback)
  9. Unresolved Data Quality Issues (only if present)

  Each section may contain:
    - content:            deterministic facts (always present)
    - expert_commentary:  LLM 3-lens analysis (if interpretations available)
    - plot_paths:         list of PNG paths to render inline
    - plot_commentaries:  per-plot 3-lens commentary from LLM

Public AG2-facing functions (complete list):
  - render_pdf(findings_json, output_dir)      -> str  (always produced)
  - render_markdown(findings_json, output_dir) -> str  (always produced, LLM-readable)
  - render_ipynb(findings_json, output_dir)    -> str  (optional, IPYNB_EXPORT=true)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any
from xml.sax.saxutils import escape as _xml_escape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template Method ABC
# ---------------------------------------------------------------------------


class ReportRenderer(ABC):
    """
    Template Method: defines the invariant section ordering.
    Subclasses implement only the output-format-specific step (_write_output).
    """

    def render(
        self,
        findings: dict[str, Any],
        plot_paths: list[str],
        output_path: str,
    ) -> str:
        """Invariant algorithm — never overridden."""
        sections = self._build_sections(findings)
        formatted = self._format_sections(sections, plot_paths, findings)
        return self._write_output(formatted, output_path)

    def _build_sections(self, findings: dict[str, Any]) -> list[dict[str, Any]]:
        """Shared: extract ordered sections from Findings."""
        return findings.get("sections", [])

    def _format_sections(
        self,
        sections: list[dict[str, Any]],
        plot_paths: list[str],
        findings: dict[str, Any],
    ) -> dict[str, Any]:
        """Shared: structure content + embed plot references.

        Plots are now paired with their parent sections (Option A),
        so the top-level 'plots' key is kept only for backward
        compatibility — renderers use section-level plot_paths.
        """
        return {
            "sections": sections,
            "plots": plot_paths,
            "unresolved": findings.get("unresolved_flags", []),
        }

    @abstractmethod
    def _write_output(self, content: dict[str, Any], output_path: str) -> str:
        """Varies: PDF vs IPYNB rendering."""
        ...


# ---------------------------------------------------------------------------
# PDFRenderer — reportlab.platypus
# ---------------------------------------------------------------------------

# Lazy imports: reportlab is only loaded when _write_output is actually called.
# This keeps the module importable even if reportlab is missing (graceful error
# at call time rather than import time).

_REPORT_TITLE = "EDA Report"
_MAX_IMAGE_WIDTH_INCHES = 5.5
_MAX_IMAGE_HEIGHT_INCHES = 3.8


def _safe_xml(text: str) -> str:
    """Escape text for reportlab Paragraph (XML-based rendering)."""
    return _xml_escape(str(text)).replace("\n", "<br/>")


class PDFRenderer(ReportRenderer):
    """Concrete renderer: produces PDF via reportlab.platypus."""

    def _write_output(self, content: dict[str, Any], output_path: str) -> str:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "EDABody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=4,
        )

        # --- Styles for expert commentary ---
        commentary_style = ParagraphStyle(
            "EDACommentary",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=4,
            leftIndent=12,
            textColor="#333333",
        )
        perspective_label_style = ParagraphStyle(
            "PerspectiveLabel",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=2,
            leftIndent=12,
        )
        caption_style = ParagraphStyle(
            "Caption",
            parent=styles["Italic"],
            fontSize=9,
            alignment=1,  # TA_CENTER
            spaceAfter=8,
        )

        story: list = []

        # --- Title ---
        story.append(Paragraph(_safe_xml(_REPORT_TITLE), styles["Title"]))
        story.append(Spacer(1, 12))

        # --- Sections (with inline plots + expert commentary) ---
        for section in content.get("sections", []):
            title = section.get("title", "")
            body = section.get("content", "")

            story.append(Paragraph(_safe_xml(title), styles["Heading2"]))
            story.append(Spacer(1, 6))

            # Deterministic content
            for line in body.split("\n"):
                stripped = line.strip()
                if stripped:
                    story.append(Paragraph(_safe_xml(stripped), body_style))

            # Expert commentary (3-lens: statistical, DS/ML, business)
            expert = section.get("expert_commentary", "")
            if expert:
                story.append(Spacer(1, 6))
                for line in expert.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Bold the perspective labels
                    if stripped.startswith(("Statistical", "Data Science", "Business")):
                        colon_idx = stripped.find(":")
                        if colon_idx > 0:
                            label = stripped[:colon_idx + 1]
                            rest = stripped[colon_idx + 1:].strip()
                            story.append(Paragraph(
                                f"<b>{_safe_xml(label)}</b> {_safe_xml(rest)}",
                                perspective_label_style,
                            ))
                            continue
                    story.append(Paragraph(_safe_xml(stripped), commentary_style))

            # Inline plot images + per-plot commentary
            section_plots = section.get("plot_paths", [])
            plot_comms = section.get("plot_commentaries", [])
            if section_plots:
                story.append(Spacer(1, 8))
                for plot_path in section_plots:
                    p = Path(plot_path)
                    if p.exists() and p.stat().st_size > 0:
                        img = Image(
                            str(p),
                            width=_MAX_IMAGE_WIDTH_INCHES * inch,
                            height=_MAX_IMAGE_HEIGHT_INCHES * inch,
                        )
                        img.hAlign = "CENTER"
                        story.append(img)
                        story.append(Paragraph(
                            _safe_xml(p.stem.replace("_", " ").title()),
                            caption_style,
                        ))
                    else:
                        story.append(Paragraph(
                            _safe_xml(f"[Plot not found: {p.name}]"),
                            body_style,
                        ))
                    # Per-plot commentary (matched by filename)
                    fname = p.name
                    for pc in plot_comms:
                        if pc.get("plot_file") == fname:
                            for key, label in (
                                ("statistical", "Statistical Perspective"),
                                ("ds_ml", "Data Science &amp; ML Perspective"),
                                ("business", "Business Perspective"),
                            ):
                                text = pc.get(key, "")
                                if text:
                                    story.append(Paragraph(
                                        f"<b>{label}:</b> {_safe_xml(text)}",
                                        perspective_label_style,
                                    ))
                            story.append(Spacer(1, 8))
                            break

            story.append(Spacer(1, 12))

        # --- Unresolved flags ---
        unresolved = content.get("unresolved", [])
        if unresolved:
            story.append(
                Paragraph(
                    _safe_xml("Unresolved Data Quality Issues"),
                    styles["Heading2"],
                )
            )
            story.append(Spacer(1, 6))
            for flag_text in unresolved:
                # Replace → with -> for PDF font compatibility
                safe = flag_text.replace("\u2192", "->")
                story.append(
                    Paragraph(f"\u2022 {_safe_xml(safe)}", body_style)
                )
            story.append(Spacer(1, 12))

        doc.build(story)

        logger.info("PDF report generated: %s", output_path)
        return output_path


# ---------------------------------------------------------------------------
# IPYNBRenderer — nbformat v4
# ---------------------------------------------------------------------------


class IPYNBRenderer(ReportRenderer):
    """Concrete renderer: produces Jupyter .ipynb via nbformat."""

    def _write_output(self, content: dict[str, Any], output_path: str) -> str:
        import nbformat

        nb = nbformat.v4.new_notebook()

        # --- Title cell ---
        nb.cells.append(nbformat.v4.new_markdown_cell(f"# {_REPORT_TITLE}"))

        # --- Section cells (with inline plots + expert commentary) ---
        for section in content.get("sections", []):
            title = section.get("title", "")
            body = section.get("content", "")
            cell_parts = [f"## {title}\n\n{body}"]

            # Expert commentary (3-lens)
            expert = section.get("expert_commentary", "")
            if expert:
                cell_parts.append(f"\n\n{expert}")

            nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(cell_parts)))

            # Inline plot images + per-plot commentary (separate cell per plot)
            section_plots = section.get("plot_paths", [])
            plot_comms = section.get("plot_commentaries", [])
            for plot_path in section_plots:
                p = Path(plot_path)
                caption = p.stem.replace("_", " ").title()
                plot_lines = [f"![{caption}]({plot_path})\n"]
                fname = p.name
                for pc in plot_comms:
                    if pc.get("plot_file") == fname:
                        for key, label in (
                            ("statistical", "**Statistical Perspective**"),
                            ("ds_ml", "**Data Science & ML Perspective**"),
                            ("business", "**Business Perspective**"),
                        ):
                            text = pc.get(key, "")
                            if text:
                                plot_lines.append(f"{label}: {text}\n")
                        break
                nb.cells.append(
                    nbformat.v4.new_markdown_cell("\n".join(plot_lines))
                )

        # --- Unresolved flags cell ---
        unresolved = content.get("unresolved", [])
        if unresolved:
            warning_lines = ["## Unresolved Data Quality Issues\n"]
            for flag_text in unresolved:
                warning_lines.append(f"- {flag_text}")
            nb.cells.append(
                nbformat.v4.new_markdown_cell("\n".join(warning_lines))
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        logger.info("IPYNB report generated: %s", output_path)
        return output_path


# ---------------------------------------------------------------------------
# MarkdownRenderer — plain UTF-8 Markdown (LLM-readable)
# ---------------------------------------------------------------------------


class MarkdownRenderer(ReportRenderer):
    """Concrete renderer: produces GitHub-Flavored Markdown report.

    Produces plain UTF-8 text that is directly consumable by LLMs,
    diff tools, and version control, while remaining human-readable
    when rendered in any Markdown viewer.
    """

    def _write_output(self, content: dict[str, Any], output_path: str) -> str:
        lines: list[str] = []

        lines.append(f"# {_REPORT_TITLE}")
        lines.append("")

        for section in content.get("sections", []):
            title = section.get("title", "")
            body = section.get("content", "")

            lines.append(f"## {title}")
            lines.append("")
            lines.append(body)
            lines.append("")

            # Expert commentary (3-lens: statistical, DS/ML, business)
            expert = section.get("expert_commentary", "")
            if expert:
                for line in expert.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        lines.append("")
                        continue
                    # Bold the perspective labels
                    if stripped.startswith(("Statistical", "Data Science", "Business")):
                        colon_idx = stripped.find(":")
                        if colon_idx > 0:
                            label = stripped[: colon_idx + 1]
                            rest = stripped[colon_idx + 1 :].strip()
                            lines.append(f"**{label}** {rest}")
                            continue
                    lines.append(stripped)
                lines.append("")

            # Inline plot references + per-plot commentary
            section_plots = section.get("plot_paths", [])
            plot_comms = section.get("plot_commentaries", [])
            for plot_path in section_plots:
                p = Path(plot_path)
                caption = p.stem.replace("_", " ").title()
                lines.append(f"![{caption}]({plot_path})")
                fname = p.name
                for pc in plot_comms:
                    if pc.get("plot_file") == fname:
                        lines.append("")
                        for key, label in (
                            ("statistical", "Statistical Perspective"),
                            ("ds_ml", "Data Science & ML Perspective"),
                            ("business", "Business Perspective"),
                        ):
                            text = pc.get(key, "")
                            if text:
                                lines.append(f"**{label}:** {text}")
                        lines.append("")
                        break
            if section_plots:
                lines.append("")

        # Unresolved flags
        unresolved = content.get("unresolved", [])
        if unresolved:
            lines.append("## Unresolved Data Quality Issues")
            lines.append("")
            for flag_text in unresolved:
                lines.append(f"- {flag_text}")
            lines.append("")

        markdown_text = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        logger.info("Markdown report generated: %s", output_path)
        return output_path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_plot_paths(findings: dict[str, Any]) -> list[str]:
    """Extract all plot_paths from every section of Findings.

    With Option A (inline plots), plot_paths are distributed across
    their parent sections (Missing, Correlation, Statistical Analysis)
    rather than grouped in a single Visualizations section.
    """
    all_paths: list[str] = []
    for section in findings.get("sections", []):
        all_paths.extend(section.get("plot_paths", []))
    return all_paths


# ---------------------------------------------------------------------------
# AG2-facing public functions (flat callables, no OOP visible to AG2)
# ---------------------------------------------------------------------------


def render_pdf(
    findings_json: Annotated[
        str,
        "JSON string of Findings from assemble_findings()",
    ],
    output_dir: Annotated[
        str,
        "Output directory path — must be exactly 'outputs/' (report.pdf is created inside it)",
    ],
) -> str:
    """
    AG2 tool entry point. Renders Findings as a PDF report.

    Produces: {output_dir}/report.pdf
    Uses the Template Method pattern: PDFRenderer via reportlab.

    Returns:
        The absolute path to the generated PDF file.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve
    if is_active():
        findings_json = resolve(findings_json, "findings")

    findings = json.loads(findings_json)
    plot_paths = _extract_plot_paths(findings)
    output_path = str(Path(output_dir) / "report.pdf")

    result = PDFRenderer().render(findings, plot_paths, output_path)

    logger.info(
        "render_pdf: %d sections, %d plots -> %s",
        len(findings.get("sections", [])),
        len(plot_paths),
        result,
    )
    return result


def render_ipynb(
    findings_json: Annotated[
        str,
        "JSON string of Findings from assemble_findings()",
    ],
    output_dir: Annotated[
        str,
        "Output directory path — must be exactly 'outputs/' (report.ipynb is created inside it)",
    ],
) -> str:
    """
    AG2 tool entry point. Renders Findings as a Jupyter notebook.

    Produces: {output_dir}/report.ipynb
    Uses the Template Method pattern: IPYNBRenderer via nbformat.

    Only called when IPYNB_EXPORT=true (controlled by agent, not by this tool).

    Returns:
        The absolute path to the generated IPYNB file.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve
    if is_active():
        findings_json = resolve(findings_json, "findings")

    findings = json.loads(findings_json)
    plot_paths = _extract_plot_paths(findings)
    output_path = str(Path(output_dir) / "report.ipynb")

    result = IPYNBRenderer().render(findings, plot_paths, output_path)

    logger.info(
        "render_ipynb: %d sections, %d plots -> %s",
        len(findings.get("sections", [])),
        len(plot_paths),
        result,
    )
    return result


def render_markdown(
    findings_json: Annotated[
        str,
        "JSON string of Findings from assemble_findings()",
    ],
    output_dir: Annotated[
        str,
        "Output directory path — must be exactly 'outputs/' (report.md is created inside it)",
    ],
) -> str:
    """
    AG2 tool entry point. Renders Findings as a plain Markdown report.

    Produces: {output_dir}/report.md
    Uses the Template Method pattern: MarkdownRenderer.

    The Markdown output is LLM-readable (plain UTF-8 text with structure
    preserved via headings and bold labels) and human-readable when rendered.
    Call this unconditionally alongside render_pdf().

    Returns:
        The absolute path to the generated Markdown file.
    """
    # Artifact store: resolve input
    from tools._pipeline_state import is_active, resolve
    if is_active():
        findings_json = resolve(findings_json, "findings")

    findings = json.loads(findings_json)
    plot_paths = _extract_plot_paths(findings)
    output_path = str(Path(output_dir) / "report.md")

    result = MarkdownRenderer().render(findings, plot_paths, output_path)

    logger.info(
        "render_markdown: %d sections, %d plots -> %s",
        len(findings.get("sections", [])),
        len(plot_paths),
        result,
    )
    return result
