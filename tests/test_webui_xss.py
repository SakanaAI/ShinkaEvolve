"""Regression tests for script injection through run data in the web UI.

Patch names, descriptions and meta notes are model output, and metric names,
ids and file names come from the run directory, so every one of them must be
escaped or sanitised before it reaches ``innerHTML``.
"""

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from shinka.webui import visualization

WEBUI_DIR = Path(__file__).resolve().parents[1] / "shinka" / "webui"
PAGE_NAMES = ["viz_tree.html", "index.html", "compare.html"]
PAGES = {name: (WEBUI_DIR / name).read_text(encoding="utf-8") for name in PAGE_NAMES}
VIZ_TREE = PAGES["viz_tree.html"]

PAYLOAD = "<img src=x onerror=\"alert('x')\">"


def _extract_js_function_source(html: str, function_name: str) -> str:
    match = re.search(rf"function {function_name}\([^)]*\)\s*\{{", html)
    assert match, f"Could not find function {function_name}"
    depth = 0
    for idx in range(html.find("{", match.start()), len(html)):
        if html[idx] == "{":
            depth += 1
        elif html[idx] == "}":
            depth -= 1
            if depth == 0:
                return html[match.start() : idx + 1]
    raise AssertionError(f"Unbalanced braces in {function_name}")


def _run_js(sources: List[str], expression: str):
    script = "\n".join([*sources, f"console.log(JSON.stringify({expression}));"])
    completed = subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def _viz_tree_functions(*names: str) -> List[str]:
    return [_extract_js_function_source(VIZ_TREE, name) for name in names]


def _markdown_helper_sources() -> List[str]:
    options = re.search(
        r"const MARKDOWN_SANITIZE_OPTIONS = \{.*?\n        \};", VIZ_TREE, re.DOTALL
    )
    assert options, "MARKDOWN_SANITIZE_OPTIONS not found"
    return [
        options.group(0),
        *_viz_tree_functions(
            "escapeHtml",
            "sanitizeRenderedMarkdown",
            "renderMarkdown",
            "renderMarkdownInline",
        ),
    ]


# --- escaping ---------------------------------------------------------------


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_escape_html_covers_element_and_attribute_context(page: str):
    source = _extract_js_function_source(PAGES[page], "escapeHtml")

    escaped, empty, number = _run_js(
        [source],
        f"[escapeHtml({json.dumps(PAYLOAD + ' & more')}), escapeHtml(null), escapeHtml(0)]",
    )

    assert escaped == (
        "&lt;img src=x onerror=&quot;alert(&#39;x&#39;)&quot;&gt; &amp; more"
    )
    assert empty == ""
    assert number == "0"


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_each_page_defines_escape_html_once(page: str):
    # A later duplicate silently replaces an earlier one, which is how the
    # quote-escaping version used to be shadowed by one that left quotes alone.
    assert PAGES[page].count("function escapeHtml(") == 1


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_inline_handlers_never_interpolate_run_data(page: str):
    # onclick="f('${id}')" lets a quote in the value end the JS string.
    # Handlers read the value from a data- attribute instead.
    assert re.search(r"\son\w+=\"[^\"\n]*\$\{", PAGES[page]) is None


def test_model_output_is_escaped_at_the_reported_sinks():
    unescaped_sinks = [
        "${patchName}",
        "${patchDescription}",
        "${patchType}<",
        "</strong> ${modelName}",
        "${agentName} (Gen",
        "${d.metadata.patch_description}</p>",
        "${d.metadata.patch_name || 'unnamed'}</span>",
        "${program.metadata.patch_name || 'unnamed'}</strong>",
        "${cellData.metadata?.patch_name || 'Unnamed'}</strong>",
        "<strong>${key}:</strong>",
        "${formatScore(value)}</p>",
        "${data.error}</span>",
        "${promptData.name}",
        "${promptData.description}",
        "${prompt.name}",
        "${message}</p>",
        "<p>Error: ${error.message}",
        "</strong> ${error.message}",
        "${err.message}",
        "${trimmedLine}",
        "Program Name: ${programName}",
        "File: ${metaData.filename}",
    ]

    assert [sink for sink in unescaped_sinks if sink in VIZ_TREE] == []


# --- meta notes ---------------------------------------------------------------


def test_program_summaries_escape_markup_and_keep_bold():
    sources = _viz_tree_functions(
        "escapeHtml", "processSimpleMarkdown", "formatProgramSummaries"
    )
    content = (
        f"**Program Name:** fast {PAYLOAD}\n"
        f"**Implementation:** uses **bold** and {PAYLOAD}\n"
        f"- bullet {PAYLOAD}\n"
    )

    html = _run_js(sources, f"formatProgramSummaries({json.dumps(content)})")

    assert "<img" not in html
    assert "&lt;img src=x" in html
    assert "<strong>bold</strong>" in html


def test_structured_entries_escape_markup():
    sources = _viz_tree_functions("escapeHtml", "formatStructuredEntries")
    content = f"Name: {PAYLOAD}\nSummary: {PAYLOAD}\nPerformance: {PAYLOAD}\n{PAYLOAD}"

    html = _run_js(sources, f"formatStructuredEntries({json.dumps(content)})")

    assert "<img" not in html
    assert html.count("&lt;img src=x") == 4


def test_markdown_is_sanitised_after_rendering():
    stubs = [
        "globalThis.marked = {parse: s => 'BLOCK:' + s, parseInline: s => 'INLINE:' + s};",
        "globalThis.DOMPurify = {sanitize: (html, options) => ({html, options})};",
    ]

    block, inline = _run_js(
        stubs + _markdown_helper_sources(),
        f"[renderMarkdown({json.dumps(PAYLOAD)}), renderMarkdownInline('x')]",
    )

    assert block["html"] == "BLOCK:" + PAYLOAD
    assert inline["html"] == "INLINE:x"
    for tag in ("img", "style", "form"):
        assert tag in block["options"]["FORBID_TAGS"]
    for attribute in ("style", "background", "id", "name"):
        assert attribute in block["options"]["FORBID_ATTR"]


def test_markdown_falls_back_to_escaped_text_without_sanitiser():
    # A blocked CDN must not turn into unsanitised HTML.
    html = _run_js(_markdown_helper_sources(), f"renderMarkdown({json.dumps(PAYLOAD)})")

    assert "<img" not in html
    assert "&lt;img src=x" in html


def test_marked_is_only_called_through_the_sanitising_helpers():
    assert VIZ_TREE.count("marked.parse(") == 1
    assert VIZ_TREE.count("marked.parseInline(") == 1
    assert "marked.parse(" in _extract_js_function_source(VIZ_TREE, "renderMarkdown")
    assert "marked.parseInline(" in _extract_js_function_source(
        VIZ_TREE, "renderMarkdownInline"
    )


# --- third-party scripts ------------------------------------------------------


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_cdn_resources_are_pinned_with_sri(page: str):
    html = re.sub(r"<!--.*?-->", "", PAGES[page], flags=re.DOTALL)
    tags = re.findall(
        r"<(?:script|link)\b[^>]*\b(?:src|href)=\"https?://[^\"]+\"[^>]*>", html
    )
    tags = [tag for tag in tags if "<script" in tag or 'rel="stylesheet"' in tag]
    if page != "index.html":
        assert tags, "expected CDN resources"

    for tag in tags:
        url_match = re.search(r"(?:src|href)=\"([^\"]+)\"", tag)
        assert url_match is not None
        url = url_match.group(1)
        assert re.search(r"[@/-]\d+\.\d+\.\d+[/.]", url), f"unpinned: {url}"
        assert "latest" not in url
        assert 'integrity="sha384-' in tag, f"no SRI: {url}"
        assert 'crossorigin="anonymous"' in tag, f"no crossorigin: {url}"


# --- meta PDF export ----------------------------------------------------------


def test_meta_pdf_renderer_runs_without_scripts_or_local_file_access(monkeypatch):
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="stub")

    monkeypatch.setattr(visualization.subprocess, "run", fake_run)
    monkeypatch.setattr(
        visualization,
        "markdown",
        SimpleNamespace(markdown=lambda text, extensions=None: text),
    )
    handler = visualization.DatabaseRequestHandler.__new__(
        visualization.DatabaseRequestHandler
    )

    handler._generate_pdf(f"# notes\n{PAYLOAD}", "1")

    wkhtmltopdf = next(cmd for cmd in commands if cmd[0] == "wkhtmltopdf")
    assert "--disable-javascript" in wkhtmltopdf
    assert "--disable-local-file-access" in wkhtmltopdf


# --- downloads ----------------------------------------------------------------


def test_download_names_cannot_carry_path_separators():
    source = _extract_js_function_source(VIZ_TREE, "safeFilenamePart")

    assert _run_js([source], "safeFilenamePart('../../x.sh')") == "______x_sh"
    assert _run_js([source], "safeFilenamePart(7)") == "7"
    assert "gen${gen}" not in VIZ_TREE
    assert ".${extension}" not in VIZ_TREE
