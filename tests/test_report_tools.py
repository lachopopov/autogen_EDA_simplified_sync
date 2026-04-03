"""
tests/test_report_tools.py — Unit tests for tools/report_tools.py

Tests the Template Method pattern (ReportRenderer ABC, PDFRenderer, IPYNBRenderer),
the AG2-facing render_pdf() and render_ipynb() functions, and the _extract_plot_paths helper.
No LLM calls — pure function tests.
"""

import inspect
import json
from pathlib import Path

import nbformat
import pytest

from tools.report_tools import (
    IPYNBRenderer,
    MarkdownRenderer,
    PDFRenderer,
    ReportRenderer,
    _extract_plot_paths,
    render_ipynb,
    render_markdown,
    render_pdf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_findings() -> dict:
    """A realistic Findings dict matching assemble_findings() output."""
    return {
        "sections": [
            {
                "title": "Dataset Overview",
                "content": "100 rows, 5 columns.\nMemory: 0.01 MB.",
            },
            {
                "title": "Missing Values",
                "content": "income: 12.0% missing\nage: 5.0% missing",
            },
            {
                "title": "Correlation Analysis",
                "content": "age vs income: r=0.45",
            },
            {
                "title": "Statistical Analysis",
                "content": "Distribution analysis was performed on 3 numerical feature(s). Potential outliers detected (values beyond 1.5×IQR fences) in: income.",
            },
            {
                "title": "Data Quality Assessment",
                "content": "1 quality flag(s) raised:\n[HIGH] income: |skew|=2.50 (rule: skewness)",
            },
            {
                "title": "Conclusions",
                "content": "Significant data quality concern: 8.5% overall missingness detected.",
            },
            {
                "title": "Recommendations & Business Implications",
                "content": "NEXT STEPS — After addressing the above items: (1) Re-run EDA.",
            },
            {
                "title": "Limitations & Caveats",
                "content": "This analysis is primarily univariate and bivariate.",
            },
        ],
        "unresolved_flags": [],
    }


@pytest.fixture()
def findings_with_plots(sample_findings, tmp_path) -> tuple[dict, list[Path]]:
    """Findings dict with actual plot PNG files on disk."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    plot_files = []
    for name in ["hist_age", "hist_income", "correlation_heatmap"]:
        p = plots_dir / f"{name}.png"
        # Create a minimal valid PNG (1x1 pixel red)
        p.write_bytes(_minimal_png())
        plot_files.append(p)

    # Pair plots with their parent sections (Option A: inline)
    hist_paths = [str(p) for p in plot_files if p.stem.startswith("hist_")]
    corr_paths = [str(p) for p in plot_files if p.stem == "correlation_heatmap"]
    for section in sample_findings["sections"]:
        if section["title"] == "Statistical Analysis":
            section["plot_paths"] = hist_paths
        elif section["title"] == "Correlation Analysis":
            section["plot_paths"] = corr_paths

    return sample_findings, plot_files


@pytest.fixture()
def findings_with_unresolved(sample_findings) -> dict:
    """Findings dict with unresolved flags."""
    sample_findings["unresolved_flags"] = [
        "[UNRESOLVED] [HIGH] income: |skew|=2.50 (rule: skewness) → log transform recommended",
        "[UNRESOLVED] [HIGH] dataset-level: 35% total cells missing (rule: dataset_missingness)",
    ]
    return sample_findings


@pytest.fixture()
def findings_json(sample_findings) -> str:
    """Findings as a JSON string (matches AG2 tool input)."""
    return json.dumps(sample_findings)


@pytest.fixture()
def output_dir(tmp_path) -> Path:
    """Temporary output directory."""
    d = tmp_path / "outputs"
    d.mkdir()
    return d




@pytest.fixture()
def mock_report_output_dir(monkeypatch, output_dir):
    """Mock path resolution so AG2 render_*() tools write to tmp output_dir."""
    monkeypatch.setattr(
        "tools.report_tools.get_outputs_dir",
        lambda session_id=None: output_dir,
    )
    return output_dir


def _minimal_png() -> bytes:
    """Return bytes for a minimal valid 1×1 red PNG."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_data = b"\x00\xff\x00\x00"  # filter=None, R=255, G=0, B=0
    idat = zlib.compress(raw_data)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# ReportRenderer ABC
# ---------------------------------------------------------------------------

class TestReportRendererABC:
    """Test the Template Method ABC contract."""

    def test_cannot_instantiate(self):
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ReportRenderer()

    def test_subclass_must_implement_write_output(self):
        """Concrete subclass without _write_output raises TypeError."""
        class Incomplete(ReportRenderer):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_build_sections_extracts_sections(self, sample_findings):
        """_build_sections returns the sections list from findings."""
        renderer = PDFRenderer()  # use concrete subclass to test shared method
        sections = renderer._build_sections(sample_findings)
        assert len(sections) == 8
        assert sections[0]["title"] == "Dataset Overview"

    def test_build_sections_empty_findings(self):
        """_build_sections returns [] for empty findings."""
        renderer = PDFRenderer()
        assert renderer._build_sections({}) == []

    def test_format_sections_includes_all_fields(self, sample_findings):
        """_format_sections returns sections + plots + unresolved."""
        renderer = PDFRenderer()
        sections = renderer._build_sections(sample_findings)
        formatted = renderer._format_sections(
            sections, ["/tmp/a.png"], sample_findings
        )
        assert "sections" in formatted
        assert "plots" in formatted
        assert "unresolved" in formatted
        assert formatted["plots"] == ["/tmp/a.png"]

    def test_format_sections_unresolved(self, findings_with_unresolved):
        """_format_sections picks up unresolved_flags from findings."""
        renderer = PDFRenderer()
        sections = renderer._build_sections(findings_with_unresolved)
        formatted = renderer._format_sections(
            sections, [], findings_with_unresolved
        )
        assert len(formatted["unresolved"]) == 2


# ---------------------------------------------------------------------------
# _extract_plot_paths helper
# ---------------------------------------------------------------------------

class TestExtractPlotPaths:
    """Test the _extract_plot_paths helper function."""

    def test_extracts_from_visualizations_section(self):
        """Finds plot_paths inside the Visualizations section."""
        findings = {
            "sections": [
                {"title": "Overview", "content": "..."},
                {"title": "Viz", "content": "...", "plot_paths": ["/a.png", "/b.png"]},
            ],
        }
        assert _extract_plot_paths(findings) == ["/a.png", "/b.png"]

    def test_no_plot_paths_returns_empty(self):
        """No section with plot_paths → empty list."""
        findings = {"sections": [{"title": "Overview", "content": "..."}]}
        assert _extract_plot_paths(findings) == []

    def test_empty_findings_returns_empty(self):
        """Empty findings dict → empty list."""
        assert _extract_plot_paths({}) == []

    def test_empty_plot_paths_list(self):
        """Section with empty plot_paths list → returns empty list."""
        findings = {"sections": [{"title": "Viz", "content": "...", "plot_paths": []}]}
        assert _extract_plot_paths(findings) == []


# ---------------------------------------------------------------------------
# PDFRenderer
# ---------------------------------------------------------------------------

class TestPDFRenderer:
    """Test PDF rendering via reportlab."""

    def test_produces_pdf_file(self, sample_findings, output_dir):
        """render() creates a PDF file on disk."""
        path = str(output_dir / "report.pdf")
        result = PDFRenderer().render(sample_findings, [], path)
        assert Path(result).exists()
        assert result == path

    def test_pdf_nonzero_size(self, sample_findings, output_dir):
        """Generated PDF has non-zero size."""
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(sample_findings, [], path)
        assert Path(path).stat().st_size > 0

    def test_pdf_magic_bytes(self, sample_findings, output_dir):
        """Generated file starts with %PDF- magic bytes."""
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(sample_findings, [], path)
        with open(path, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-"

    def test_creates_output_dir(self, sample_findings, tmp_path):
        """Creates parent directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested" / "dir"
        path = str(nested / "report.pdf")
        PDFRenderer().render(sample_findings, [], path)
        assert Path(path).exists()

    def test_handles_empty_sections(self, output_dir):
        """Empty findings → still produces valid PDF."""
        findings = {"sections": [], "unresolved_flags": []}
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings, [], path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_with_actual_plots(self, findings_with_plots, output_dir):
        """Plots are embedded when files exist on disk."""
        findings, plot_files = findings_with_plots
        plot_paths = [str(p) for p in plot_files]
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings, plot_paths, path)
        # PDF with images should be larger than without
        assert Path(path).stat().st_size > 100

    def test_missing_plot_handled(self, sample_findings, output_dir):
        """Non-existent plot path → placeholder text, no crash."""
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(sample_findings, ["/nonexistent/plot.png"], path)
        assert Path(path).exists()

    def test_with_unresolved_flags(self, findings_with_unresolved, output_dir):
        """Unresolved flags section is rendered without error."""
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings_with_unresolved, [], path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_special_characters_in_content(self, output_dir):
        """XML special characters (<, >, &) don't crash the renderer."""
        findings = {
            "sections": [
                {
                    "title": "Test <Special> Section",
                    "content": "Values: x < 5 & y > 10 with 'quotes' and \"doubles\"",
                },
            ],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings, [], path)
        assert Path(path).exists()

    def test_multiline_content(self, output_dir):
        """Multi-line section content is rendered properly."""
        findings = {
            "sections": [
                {
                    "title": "Multi-Line",
                    "content": "Line 1\nLine 2\nLine 3\n\nLine 5 after blank",
                },
            ],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings, [], path)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# IPYNBRenderer
# ---------------------------------------------------------------------------

class TestIPYNBRenderer:
    """Test IPYNB rendering via nbformat."""

    def test_produces_ipynb_file(self, sample_findings, output_dir):
        """render() creates an IPYNB file on disk."""
        path = str(output_dir / "report.ipynb")
        result = IPYNBRenderer().render(sample_findings, [], path)
        assert Path(result).exists()
        assert result == path

    def test_ipynb_valid_json(self, sample_findings, output_dir):
        """Generated file is valid JSON."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        with open(path) as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)

    def test_ipynb_valid_notebook(self, sample_findings, output_dir):
        """Generated file is a valid nbformat notebook."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        nb = nbformat.read(path, as_version=4)
        assert nb.nbformat == 4

    def test_title_cell_present(self, sample_findings, output_dir):
        """First cell is the title markdown cell."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        nb = nbformat.read(path, as_version=4)
        assert nb.cells[0].cell_type == "markdown"
        assert "# EDA Report" in nb.cells[0].source

    def test_section_cells_count(self, sample_findings, output_dir):
        """One markdown cell per section + title cell."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        nb = nbformat.read(path, as_version=4)
        # 1 title + 8 sections = 9 cells (no plots, no unresolved)
        assert len(nb.cells) == 9

    def test_section_headings_present(self, sample_findings, output_dir):
        """Each section heading appears in the notebook."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        nb = nbformat.read(path, as_version=4)
        all_sources = " ".join(c.source for c in nb.cells)
        assert "Dataset Overview" in all_sources
        assert "Missing Values" in all_sources
        assert "Correlation Analysis" in all_sources

    def test_creates_output_dir(self, sample_findings, tmp_path):
        """Creates parent directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested" / "dir"
        path = str(nested / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        assert Path(path).exists()

    def test_handles_empty_sections(self, output_dir):
        """Empty findings → still produces valid notebook."""
        findings = {"sections": [], "unresolved_flags": []}
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings, [], path)
        nb = nbformat.read(path, as_version=4)
        assert nb.nbformat == 4
        # Just the title cell
        assert len(nb.cells) == 1

    def test_plot_references_in_notebook(self, output_dir):
        """Plot paths in sections appear as markdown image references."""
        findings = {
            "sections": [
                {
                    "title": "Sec",
                    "content": "text",
                    "plot_paths": ["/tmp/a.png", "/tmp/b.png"],
                },
            ],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings, [], path)
        nb = nbformat.read(path, as_version=4)
        all_sources = " ".join(c.source for c in nb.cells)
        assert "![" in all_sources  # markdown image syntax
        assert "/tmp/a.png" in all_sources
        assert "/tmp/b.png" in all_sources

    def test_unresolved_in_notebook(self, findings_with_unresolved, output_dir):
        """Unresolved flags appear in a dedicated cell."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings_with_unresolved, [], path)
        nb = nbformat.read(path, as_version=4)
        all_sources = " ".join(c.source for c in nb.cells)
        assert "Unresolved" in all_sources
        assert "[UNRESOLVED]" in all_sources

    def test_all_cells_are_markdown(self, sample_findings, output_dir):
        """All generated cells are markdown (no code cells)."""
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(sample_findings, [], path)
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            assert cell.cell_type == "markdown"


# ---------------------------------------------------------------------------
# Expert Commentary + Per-Plot Commentary (Option A inline rendering)
# ---------------------------------------------------------------------------

class TestExpertCommentaryPDF:
    """Test that expert_commentary and plot_commentaries render in PDF."""

    def test_expert_commentary_renders(self, output_dir):
        """Section with expert_commentary produces a larger PDF."""
        findings_plain = {
            "sections": [{"title": "Overview", "content": "100 rows."}],
            "unresolved_flags": [],
        }
        findings_enriched = {
            "sections": [{
                "title": "Overview",
                "content": "100 rows.",
                "expert_commentary": (
                    "Statistical Perspective: Normal distribution.\n\n"
                    "Data Science & ML Perspective: Good for modeling.\n\n"
                    "Business Perspective: Revenue growth indicator."
                ),
            }],
            "unresolved_flags": [],
        }
        plain_path = str(output_dir / "plain.pdf")
        enriched_path = str(output_dir / "enriched.pdf")
        PDFRenderer().render(findings_plain, [], plain_path)
        PDFRenderer().render(findings_enriched, [], enriched_path)
        # Enriched PDF should be larger due to commentary text
        assert Path(enriched_path).stat().st_size > Path(plain_path).stat().st_size

    def test_inline_plots_render(self, tmp_path, output_dir):
        """Plots in section plot_paths are rendered inline (not in separate block)."""
        plots_dir = tmp_path / "p"
        plots_dir.mkdir()
        img = plots_dir / "hist_age.png"
        img.write_bytes(_minimal_png())
        findings = {
            "sections": [{
                "title": "Statistical Analysis",
                "content": "Distributions analyzed.",
                "plot_paths": [str(img)],
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings, [], path)
        assert Path(path).stat().st_size > 100

    def test_per_plot_commentary_renders(self, tmp_path, output_dir):
        """Per-plot commentary beneath images doesn't crash and produces PDF."""
        plots_dir = tmp_path / "p"
        plots_dir.mkdir()
        img = plots_dir / "hist_age.png"
        img.write_bytes(_minimal_png())
        findings = {
            "sections": [{
                "title": "Statistical Analysis",
                "content": "Distributions analyzed.",
                "plot_paths": [str(img)],
                "plot_commentaries": [{
                    "plot_file": "hist_age.png",
                    "statistical": "Right-skewed distribution.",
                    "ds_ml": "Log transform recommended.",
                    "business": "Revenue concentration in lower range.",
                }],
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.pdf")
        PDFRenderer().render(findings, [], path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 100


class TestExpertCommentaryIPYNB:
    """Test that expert_commentary and plot_commentaries render in IPYNB."""

    def test_expert_commentary_in_notebook(self, output_dir):
        """Expert commentary text appears in notebook cell source."""
        findings = {
            "sections": [{
                "title": "Overview",
                "content": "100 rows.",
                "expert_commentary": "Statistical Perspective: Normal distribution.",
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings, [], path)
        nb = nbformat.read(path, as_version=4)
        all_src = " ".join(c.source for c in nb.cells)
        assert "Statistical Perspective" in all_src

    def test_inline_plots_in_notebook(self, tmp_path, output_dir):
        """Section-level plot_paths produce separate image cells."""
        plots_dir = tmp_path / "p"
        plots_dir.mkdir()
        img = plots_dir / "hist_age.png"
        img.write_bytes(_minimal_png())
        findings = {
            "sections": [{
                "title": "Stats",
                "content": "text",
                "plot_paths": [str(img)],
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings, [], path)
        nb = nbformat.read(path, as_version=4)
        # 1 title + 1 section + 1 plot cell = 3
        assert len(nb.cells) == 3
        all_src = " ".join(c.source for c in nb.cells)
        assert "![" in all_src
        # Image is embedded as base64 data URI (portable)
        assert "data:image/png;base64," in all_src

    def test_per_plot_commentary_in_notebook(self, tmp_path, output_dir):
        """Per-plot 3-lens commentary appears in plot cells."""
        plots_dir = tmp_path / "p"
        plots_dir.mkdir()
        img = plots_dir / "hist_age.png"
        img.write_bytes(_minimal_png())
        findings = {
            "sections": [{
                "title": "Stats",
                "content": "text",
                "plot_paths": [str(img)],
                "plot_commentaries": [{
                    "plot_file": "hist_age.png",
                    "statistical": "Right-skewed.",
                    "ds_ml": "Transform needed.",
                    "business": "Concentration risk.",
                }],
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings, [], path)
        nb = nbformat.read(path, as_version=4)
        all_src = " ".join(c.source for c in nb.cells)
        assert "Right-skewed." in all_src
        assert "Transform needed." in all_src
        assert "Concentration risk." in all_src

    def test_multiple_sections_with_different_plots(self, tmp_path, output_dir):
        """Different sections get their own plot cells."""
        plots_dir = tmp_path / "p"
        plots_dir.mkdir()
        for name in ["hist_age.png", "correlation_heatmap.png"]:
            (plots_dir / name).write_bytes(_minimal_png())
        findings = {
            "sections": [
                {
                    "title": "Statistical Analysis",
                    "content": "Distributions.",
                    "plot_paths": [str(plots_dir / "hist_age.png")],
                },
                {
                    "title": "Correlation Analysis",
                    "content": "Correlations.",
                    "plot_paths": [str(plots_dir / "correlation_heatmap.png")],
                },
            ],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.ipynb")
        IPYNBRenderer().render(findings, [], path)
        nb = nbformat.read(path, as_version=4)
        # 1 title + 2 sections + 2 plot cells = 5
        assert len(nb.cells) == 5
        all_src = " ".join(c.source for c in nb.cells)
        # Captions derived from filenames appear in ![caption](data:...)
        assert "Hist Age" in all_src
        assert "Correlation Heatmap" in all_src
        assert "data:image/png;base64," in all_src


# ---------------------------------------------------------------------------
# render_pdf() — AG2-facing function
# ---------------------------------------------------------------------------

class TestRenderPdf:
    """Test the render_pdf AG2 entry point."""

    def test_returns_path_string(self, findings_json, mock_report_output_dir):
        """Function returns the output file path as a string."""
        result = render_pdf(findings_json)
        assert isinstance(result, str)
        assert result.endswith("report.pdf")

    def test_creates_pdf_file(self, findings_json, mock_report_output_dir):
        """PDF file is created on disk."""
        result = render_pdf(findings_json)
        assert Path(result).exists()

    def test_pdf_is_valid(self, findings_json, mock_report_output_dir):
        """Generated PDF starts with magic bytes."""
        result = render_pdf(findings_json)
        with open(result, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_creates_output_dir_if_missing(self, monkeypatch, findings_json, tmp_path):
        """Auto-creates output directory if it doesn't exist."""
        nonexistent = tmp_path / "new_output_dir"
        monkeypatch.setattr(
            "tools.report_tools.get_outputs_dir",
            lambda session_id=None: nonexistent,
        )
        result = render_pdf(findings_json)
        assert Path(result).exists()

    def test_extracts_plots_from_findings(self, findings_with_plots, mock_report_output_dir):
        """Plots embedded in findings are extracted and rendered."""
        findings, _ = findings_with_plots
        fj = json.dumps(findings)
        result = render_pdf(fj)
        assert Path(result).stat().st_size > 0

    def test_empty_findings(self, mock_report_output_dir):
        """Empty findings → still produces valid PDF."""
        fj = json.dumps({"sections": [], "unresolved_flags": []})
        result = render_pdf(fj)
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# render_ipynb() — AG2-facing function
# ---------------------------------------------------------------------------

class TestRenderIpynb:
    """Test the render_ipynb AG2 entry point."""

    def test_returns_path_string(self, findings_json, mock_report_output_dir):
        """Function returns the output file path as a string."""
        result = render_ipynb(findings_json)
        assert isinstance(result, str)
        assert result.endswith("report.ipynb")

    def test_creates_ipynb_file(self, findings_json, mock_report_output_dir):
        """IPYNB file is created on disk."""
        result = render_ipynb(findings_json)
        assert Path(result).exists()

    def test_valid_notebook(self, findings_json, mock_report_output_dir):
        """Generated file is a valid nbformat v4 notebook."""
        result = render_ipynb(findings_json)
        nb = nbformat.read(result, as_version=4)
        assert nb.nbformat == 4

    def test_creates_output_dir_if_missing(self, monkeypatch, findings_json, tmp_path):
        """Auto-creates output directory if it doesn't exist."""
        nonexistent = tmp_path / "new_output_dir"
        monkeypatch.setattr(
            "tools.report_tools.get_outputs_dir",
            lambda session_id=None: nonexistent,
        )
        result = render_ipynb(findings_json)
        assert Path(result).exists()

    def test_empty_findings(self, mock_report_output_dir):
        """Empty findings → still produces valid notebook."""
        fj = json.dumps({"sections": [], "unresolved_flags": []})
        result = render_ipynb(fj)
        nb = nbformat.read(result, as_version=4)
        assert nb.nbformat == 4


# ---------------------------------------------------------------------------
# End-to-end: assemble_findings → render
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Test the full pipeline: Findings → PDF/IPYNB."""

    def _make_findings_json(self) -> str:
        """Create a Findings JSON from the real assemble_findings structure."""
        return json.dumps({
            "sections": [
                {"title": "Dataset Overview", "content": "50 rows, 3 columns."},
                {"title": "Missing Values", "content": "col_a: 2.0% missing"},
                {"title": "Correlation Analysis", "content": "col_a vs col_a: r=1.00"},
                {"title": "Statistical Analysis", "content": "Distribution analysis was performed on 2 numerical feature(s)."},
                {"title": "Visualizations", "content": "1 plot(s) generated.", "plot_paths": []},
                {"title": "Data Quality Assessment", "content": "All quality checks passed."},
                {"title": "Conclusions", "content": "The dataset is fully complete with no missing values."},
                {"title": "Recommendations & Business Implications", "content": "The dataset shows good overall quality."},
            ],
            "unresolved_flags": [],
        })

    def test_findings_to_pdf(self, mock_report_output_dir):
        """Complete Findings JSON → PDF file."""
        result = render_pdf(self._make_findings_json())
        assert Path(result).exists()
        with open(result, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_findings_to_ipynb(self, mock_report_output_dir):
        """Complete Findings JSON → IPYNB file."""
        result = render_ipynb(self._make_findings_json())
        nb = nbformat.read(result, as_version=4)
        assert nb.nbformat == 4
        assert len(nb.cells) >= 8  # title + 7 sections

    def test_pdf_and_ipynb_coexist(self, mock_report_output_dir):
        """Both PDF and IPYNB can be generated in the same directory."""
        fj = self._make_findings_json()
        pdf_path = render_pdf(fj)
        ipynb_path = render_ipynb(fj)
        assert Path(pdf_path).exists()
        assert Path(ipynb_path).exists()
        assert pdf_path != ipynb_path


# ---------------------------------------------------------------------------
# Hard Boundary Rule
# ---------------------------------------------------------------------------

class TestHardBoundaryRule:
    """Verify zero AG2 imports in tools/report_tools.py (P7)."""

    def test_no_ag2_imports(self):
        import tools.report_tools as module
        source = inspect.getsource(module)
        assert "import autogen" not in source
        assert "from autogen" not in source


# ---------------------------------------------------------------------------
# MarkdownRenderer
# ---------------------------------------------------------------------------

class TestMarkdownRenderer:
    """Test Markdown rendering."""

    def test_produces_md_file(self, sample_findings, output_dir):
        """render() creates a .md file on disk."""
        path = str(output_dir / "report.md")
        result = MarkdownRenderer().render(sample_findings, [], path)
        assert Path(result).exists()
        assert result == path

    def test_valid_utf8(self, sample_findings, output_dir):
        """Generated file is valid UTF-8 text."""
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(sample_findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_title_heading(self, sample_findings, output_dir):
        """Document starts with # EDA Report."""
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(sample_findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "# EDA Report" in text

    def test_section_headings_present(self, sample_findings, output_dir):
        """Each section title appears as a ## heading."""
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(sample_findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "## Dataset Overview" in text
        assert "## Missing Values" in text
        assert "## Conclusions" in text

    def test_content_present(self, sample_findings, output_dir):
        """Deterministic section content appears in the output."""
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(sample_findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "100 rows, 5 columns" in text
        assert "income: 12.0% missing" in text

    def test_expert_commentary_bold_labels(self, output_dir):
        """3-lens perspective labels are bolded in output."""
        findings = {
            "sections": [{
                "title": "Overview",
                "content": "100 rows.",
                "expert_commentary": (
                    "Statistical Perspective: Normal distribution.\n\n"
                    "Data Science & ML Perspective: Good for modeling.\n\n"
                    "Business Perspective: Revenue growth indicator."
                ),
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "**Statistical Perspective:**" in text
        assert "**Business Perspective:**" in text
        assert "Normal distribution." in text

    def test_plot_references(self, output_dir):
        """Plot paths appear as Markdown image references."""
        findings = {
            "sections": [{
                "title": "Stats",
                "content": "text",
                "plot_paths": ["/tmp/hist_age.png", "/tmp/corr.png"],
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "![" in text
        assert "/tmp/hist_age.png" in text
        assert "/tmp/corr.png" in text

    def test_per_plot_commentary(self, output_dir):
        """Per-plot 3-lens commentary appears beneath plot reference."""
        findings = {
            "sections": [{
                "title": "Stats",
                "content": "text",
                "plot_paths": ["/tmp/hist_age.png"],
                "plot_commentaries": [{
                    "plot_file": "hist_age.png",
                    "statistical": "Right-skewed.",
                    "ds_ml": "Log transform needed.",
                    "business": "Concentration risk.",
                }],
            }],
            "unresolved_flags": [],
        }
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "Right-skewed." in text
        assert "Log transform needed." in text
        assert "Concentration risk." in text

    def test_unresolved_flags(self, findings_with_unresolved, output_dir):
        """Unresolved flags appear as bullet list items."""
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(findings_with_unresolved, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "## Unresolved Data Quality Issues" in text
        assert "- [UNRESOLVED]" in text

    def test_creates_output_dir(self, sample_findings, tmp_path):
        """Creates parent directory if it doesn't exist."""
        nested = tmp_path / "deep" / "nested" / "dir"
        path = str(nested / "report.md")
        MarkdownRenderer().render(sample_findings, [], path)
        assert Path(path).exists()

    def test_handles_empty_sections(self, output_dir):
        """Empty findings → still produces valid Markdown file."""
        findings = {"sections": [], "unresolved_flags": []}
        path = str(output_dir / "report.md")
        MarkdownRenderer().render(findings, [], path)
        text = Path(path).read_text(encoding="utf-8")
        assert "# EDA Report" in text


# ---------------------------------------------------------------------------
# render_markdown() — AG2-facing function
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    """Test the render_markdown AG2 entry point."""

    def test_returns_path_string(self, findings_json, mock_report_output_dir):
        """Function returns the output file path as a string."""
        result = render_markdown(findings_json)
        assert isinstance(result, str)
        assert result.endswith("report.md")

    def test_creates_md_file(self, findings_json, mock_report_output_dir):
        """Markdown file is created on disk."""
        result = render_markdown(findings_json)
        assert Path(result).exists()

    def test_valid_utf8_content(self, findings_json, mock_report_output_dir):
        """Generated file is valid UTF-8 plain text."""
        result = render_markdown(findings_json)
        text = Path(result).read_text(encoding="utf-8")
        assert "# EDA Report" in text

    def test_creates_output_dir_if_missing(self, monkeypatch, findings_json, tmp_path):
        """Auto-creates output directory if it doesn't exist."""
        nonexistent = tmp_path / "new_output_dir"
        monkeypatch.setattr(
            "tools.report_tools.get_outputs_dir",
            lambda session_id=None: nonexistent,
        )
        result = render_markdown(findings_json)
        assert Path(result).exists()

    def test_empty_findings(self, mock_report_output_dir):
        """Empty findings → still produces valid Markdown."""
        fj = json.dumps({"sections": [], "unresolved_flags": []})
        result = render_markdown(fj)
        text = Path(result).read_text(encoding="utf-8")
        assert "# EDA Report" in text

    def test_hard_boundary_rule(self):
        """render_markdown lives in tools/ with zero AG2 imports."""
        import tools.report_tools as module
        source = inspect.getsource(module)
        assert "import autogen" not in source
        assert "from autogen" not in source
