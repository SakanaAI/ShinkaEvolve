import json
import shutil
import subprocess
from pathlib import Path

import pytest

VIZ_TREE_HTML = Path(__file__).parents[1] / "shinka" / "webui" / "viz_tree.html"


def _javascript_between(html: str, start: str, end: str) -> str:
    start_index = html.index(start)
    end_index = html.index(end, start_index)
    return html[start_index:end_index]


def test_prompt_program_rows_escape_content_and_use_data_event_binding():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "onclick=\"selectProgramFromPromptView('${p.id}')\"" not in html
    assert 'data-program-id="${escapeHtml(p.id)}"' in html
    assert 'title="${escapeHtml(name)}">${escapeHtml(name)}' in html
    assert "bindPromptProgramRowHandlers();" in html


def test_full_prompt_escapes_prompt_and_parent_names():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "${escapeHtml(prompt.name)}" in html
    assert "${escapeHtml(parentPrompt.name ||" in html


def test_viz_tree_defines_one_shared_escape_helper():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert html.count("function escapeHtml(text)") == 1


def test_program_summary_fields_escape_html_before_formatting():
    if not shutil.which("node"):
        pytest.skip("Node.js is required to execute the browser formatter")

    html = VIZ_TREE_HTML.read_text(encoding="utf-8")
    formatter_source = _javascript_between(
        html,
        "function formatProgramSummaries(content)",
        "function formatStructuredEntries(content)",
    )
    escape_source = _javascript_between(
        html,
        "function escapeHtml(text)",
        "function renderMarkdown(text)",
    )
    payload = """intro <img src=x onerror=alert(1)>
# INDIVIDUAL PROGRAM SUMMARIES
Program Name: <svg onload=alert(2)>
Implementation: **bold** <script>alert(3)</script>
Performance: *italic* <iframe srcdoc=x>
Feedback: <details open ontoggle=alert(4)>
- <a href=javascript:alert(5)>link</a>
continuation <math href=javascript:alert(6)>"""
    script = "\n".join(
        (
            escape_source,
            formatter_source,
            f"const output = formatProgramSummaries({json.dumps(payload)});",
            "console.log(JSON.stringify({output}));",
        )
    )

    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)["output"]

    assert "<img" not in output
    assert "<svg" not in output
    assert "<script" not in output
    assert "<iframe" not in output
    assert "<details" not in output
    assert "<a " not in output
    assert "<math" not in output
    assert "<strong>bold</strong>" in output
    assert "<em>italic</em>" in output
