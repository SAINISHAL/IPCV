"""Generates a publication-grade PDF document from COMPLETE_PROJECT_REPORT.md.

Uses Python-Markdown with MathJax for LaTeX mathematical equations, Mermaid.js for
system architecture diagrams, and headless Chrome/Edge for vector PDF rendering.
"""
import sys
import re
import shutil
import subprocess
from pathlib import Path
import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_MD = PROJECT_ROOT / "reports" / "COMPLETE_PROJECT_REPORT.md"
OUTPUT_PDF = PROJECT_ROOT / "reports" / "Classroom_Face_Recognition_Report.pdf"
INTERMEDIATE_HTML = PROJECT_ROOT / "reports" / "_temp_report.html"

BRAIN_DIR = Path(r"C:\Users\saini\.gemini\antigravity\brain\87162f59-03a8-4fb6-b3b0-79fc477bef8f")


def find_browser_executable() -> str:
    """Finds Google Chrome or Microsoft Edge headless executable."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge executable could be found.")


def preprocess_markdown(text: str) -> str:
    """Convert special markdown elements such as Mermaid and Callouts to HTML."""
    # Convert Mermaid code blocks to <div class="mermaid">...</div>
    def replace_mermaid(match):
        content = match.group(1).strip()
        return f'<div class="mermaid">\n{content}\n</div>'

    text = re.sub(r'```mermaid\s*\n(.*?)\n```', replace_mermaid, text, flags=re.DOTALL)

    # Convert GitHub alert callouts > [!NOTE]
    def replace_callout(match):
        c_type = match.group(1).lower()
        c_body = match.group(2).strip()
        icon = "💡" if c_type == "note" else "ℹ️"
        return f'<div class="callout callout-{c_type}"><div class="callout-title">{icon} {c_type.upper()}</div><p>{c_body}</p></div>'

    text = re.sub(r'>\s*\[!(NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*\n((?:>.*(?:\n|$))*)',
                  lambda m: replace_callout(re.match(r'>\s*\[!(NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*\n((?:>.*(?:\n|$))*)', m.group(0))),
                  text)

    # Clean remaining blockquote markers inside callouts if any
    text = re.sub(r'<p>>\s*(.*?)</p>', r'<p>\1</p>', text)

    return text


def build_styled_html(body_html: str) -> str:
    """Wrap body HTML in an executive, publication-grade CSS stylesheet."""
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Classroom Face Recognition System — Engineering Report</title>

<!-- MathJax for vector LaTeX mathematical equations -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Mermaid.js for system architecture flowcharts -->
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({{
  startOnLoad: true,
  theme: 'neutral',
  flowchart: {{
    useMaxWidth: true,
    htmlLabels: true,
    curve: 'basis'
  }}
}});
</script>

<style>
@page {{
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-right {{
    content: "Page " counter(page);
    font-size: 8pt;
    color: #64748B;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  @bottom-left {{
    content: "Classroom Face Recognition System (YOLO + Dual PCA Eigenfaces)";
    font-size: 8pt;
    color: #64748B;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
}}

*, *:before, *:after {{
  box-sizing: border-box;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 10.2pt;
  line-height: 1.55;
  color: #1E293B;
  background-color: #FFFFFF;
  margin: 0;
  padding: 0;
}}

/* Cover / Header section */
.report-header {{
  border-bottom: 3px solid #2563EB;
  padding-bottom: 12px;
  margin-bottom: 24px;
}}

.report-title {{
  font-size: 22pt;
  font-weight: 800;
  color: #0F172A;
  margin: 0 0 6px 0;
  line-height: 1.2;
}}

.report-subtitle {{
  font-size: 12pt;
  font-weight: 500;
  color: #475569;
  margin: 0 0 14px 0;
}}

.meta-grid {{
  display: flex;
  gap: 20px;
  background-color: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 9pt;
  margin-bottom: 20px;
}}

.meta-item strong {{
  color: #334155;
}}

/* Headings */
h1 {{
  font-size: 18pt;
  font-weight: 700;
  color: #0F172A;
  border-bottom: 1.5px solid #E2E8F0;
  padding-bottom: 6px;
  margin-top: 28px;
  margin-bottom: 12px;
  page-break-after: avoid;
}}

h2 {{
  font-size: 13.5pt;
  font-weight: 700;
  color: #1E3A8A;
  margin-top: 22px;
  margin-bottom: 10px;
  page-break-after: avoid;
}}

h3 {{
  font-size: 11pt;
  font-weight: 600;
  color: #1E293B;
  margin-top: 16px;
  margin-bottom: 8px;
  page-break-after: avoid;
}}

p, ul, ol {{
  margin-top: 0;
  margin-bottom: 10px;
}}

ul, ol {{
  padding-left: 20px;
}}

li {{
  margin-bottom: 4px;
}}

/* Tables */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 18px 0;
  font-size: 9pt;
  page-break-inside: avoid;
}}

th, td {{
  padding: 7px 10px;
  text-align: left;
  border: 1px solid #CBD5E1;
}}

th {{
  background-color: #F1F5F9;
  font-weight: 700;
  color: #0F172A;
}}

tr:nth-child(even) td {{
  background-color: #F8FAFC;
}}

/* Callouts / Alerts */
.callout {{
  border-left: 4px solid #2563EB;
  background-color: #EFF6FF;
  padding: 10px 14px;
  border-radius: 0 6px 6px 0;
  margin: 14px 0;
  page-break-inside: avoid;
}}

.callout-title {{
  font-weight: 700;
  font-size: 9.5pt;
  color: #1D4ED8;
  margin-bottom: 4px;
}}

.callout p {{
  margin: 0;
  font-size: 9.5pt;
  color: #1E3A8A;
}}

/* Code Blocks */
pre {{
  background-color: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-radius: 6px;
  padding: 10px 12px;
  font-family: "Cascadia Code", Consolas, "Courier New", monospace;
  font-size: 8.5pt;
  line-height: 1.45;
  overflow-x: auto;
  margin: 12px 0 16px 0;
  page-break-inside: avoid;
}}

code {{
  font-family: "Cascadia Code", Consolas, "Courier New", monospace;
  font-size: 8.8pt;
  background-color: #F1F5F9;
  padding: 1px 4px;
  border-radius: 3px;
  color: #0F172A;
}}

pre code {{
  background-color: transparent;
  padding: 0;
  font-size: 8.5pt;
}}

/* Mermaid diagram container */
.mermaid {{
  display: flex;
  justify-content: center;
  margin: 16px 0 20px 0;
  page-break-inside: avoid;
  background-color: #FFFFFF;
}}

/* Math formatting */
.mjx-chtml {{
  font-size: 105% !important;
}}

hr {{
  border: none;
  border-top: 1px solid #E2E8F0;
  margin: 22px 0;
}}
</style>
</head>
<body>

<div class="report-header">
  <div class="report-title">Classroom Face Recognition System</div>
  <div class="report-subtitle">Production Engineering & Research Report — YOLO + Pure NumPy Dual PCA (Eigenfaces)</div>
  <div class="meta-grid">
    <div class="meta-item"><strong>Architecture:</strong> YOLOv8-Face + Dual PCA + SVM</div>
    <div class="meta-item"><strong>Dataset:</strong> 15 Student Archives (71 Classroom Photos)</div>
    <div class="meta-item"><strong>Verified Classes:</strong> 12 Enrolled + UNKNOWN</div>
    <div class="meta-item"><strong>Status:</strong> All Tests A-H Passed (100%)</div>
  </div>
</div>

{body_html}

</body>
</html>"""
    return html_template


def main():
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-GRADE PDF REPORT")
    print("=" * 70)

    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Markdown report not found at: {REPORT_MD}")

    browser_bin = find_browser_executable()
    print(f"Using Browser Engine: {browser_bin}")

    # 1. Read Markdown
    raw_md = REPORT_MD.read_text(encoding="utf-8")

    # 2. Preprocess custom blocks (Mermaid, Callouts)
    clean_md = preprocess_markdown(raw_md)

    # 3. Convert Markdown to HTML
    body_html = markdown.markdown(
        clean_md,
        extensions=["tables", "fenced_code", "nl2br"]
    )

    # 4. Build Full Styled HTML Document
    full_html = build_styled_html(body_html)
    INTERMEDIATE_HTML.write_text(full_html, encoding="utf-8")
    print(f"Generated intermediate HTML: {INTERMEDIATE_HTML}")

    # 5. Convert HTML to Vector PDF via Headless Chrome/Edge
    print("Rendering vector PDF with MathJax equations & Mermaid diagrams...")
    cmd = [
        browser_bin,
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={OUTPUT_PDF.resolve()}",
        str(INTERMEDIATE_HTML.resolve())
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Browser stderr: {res.stderr}")
        raise RuntimeError(f"Browser PDF generation failed with code {res.returncode}")

    if not OUTPUT_PDF.exists() or OUTPUT_PDF.stat().st_size == 0:
        raise RuntimeError("PDF file was not created or is 0 bytes.")

    pdf_size_kb = OUTPUT_PDF.stat().st_size / 1024.0
    print(f"\nSUCCESS: Generated PDF Report ({pdf_size_kb:.1f} KB)")
    print(f"Target: {OUTPUT_PDF}")

    # Also copy to Artifacts directory if it exists
    if BRAIN_DIR.exists():
        artifact_pdf = BRAIN_DIR / "Classroom_Face_Recognition_Report.pdf"
        shutil.copy2(OUTPUT_PDF, artifact_pdf)
        print(f"Copied to Artifacts directory: {artifact_pdf}")

    # Clean intermediate HTML
    INTERMEDIATE_HTML.unlink(missing_ok=True)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

