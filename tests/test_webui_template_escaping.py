"""Every value the web UI writes into markup is escaped, or reviewed by hand.

The pages build HTML from JavaScript template literals. Run data in them is
model output or chosen by evaluated programs, so each ``${...}`` in a
markup-building template must be a call to an escaping function. The rest
(numbers, fixed strings, fragments built from escaped parts, and text handed
to Plotly, which runs no script but still honours a few tags) is listed with
its count in ``fixtures/webui_reviewed_interpolations.json``.

A new or changed interpolation that is not an escaping call fails here until
someone has looked at it and regenerated the list with
``python tests/webui_template_scan.py``. HTML built by string concatenation
is outside what the scanner sees.
"""

import json
import re
from typing import Dict, List

import pytest

from webui_template_scan import (
    PAGE_NAMES,
    REVIEWED_PATH,
    extract_js_function,
    is_escaping_call,
    markup_interpolations,
    read_page,
    run_js,
    unescaped_interpolations,
)

PAGES = {name: read_page(name) for name in PAGE_NAMES}
REVIEWED: Dict[str, Dict[str, int]] = json.loads(
    REVIEWED_PATH.read_text(encoding="utf-8")
)


def _page(script: str) -> str:
    return f"<script>{script}</script>"


# --- the scanner ----------------------------------------------------------------


def test_scanner_reads_only_markup_templates():
    page = _page(
        "const a = `<td>${name}</td>`;\n"
        "const b = `${count} programs`;\n"
        "// a comment with `<b>${ignored}</b>`\n"
        "const c = '<i>${not_a_template}</i>';\n"
        'el.innerHTML = `<p style="x">${escapeHtml(text)}</p>`;'
    )

    assert markup_interpolations(page) == ["name", "escapeHtml(text)"]
    assert unescaped_interpolations(page) == {"name": 1}


def test_scanner_reads_templates_nested_in_an_expression():
    page = _page("x = `${ok ? `<b>${raw}</b>` : ''}${n}`;")

    # The outer template holds no markup of its own; the nested one does.
    assert markup_interpolations(page) == ["raw"]


def test_scanner_keeps_a_nested_template_in_the_outer_expression():
    page = _page("x = `<div>${ok ? `<b>${escapeHtml(raw)}</b>` : ''}</div>`;")

    assert unescaped_interpolations(page) == {
        "ok ? `<b>${escapeHtml(raw)}</b>` : ''": 1
    }


@pytest.mark.parametrize(
    "expression, escaped",
    [
        ("escapeHtml(name)", True),
        ("escapeHtml(a.b(c) || 'x')", True),
        ("encodeURIComponent(id)", True),
        ("renderMarkdownInline(note)", True),
        ("escapeHtml(a) + b", False),
        ("escapeHtml(a) + escapeHtml(b)", False),
        ("name", False),
        ("formatScore(value)", False),
        ("unescapeHtml(name)", False),
    ],
)
def test_only_a_whole_escaping_call_counts_as_escaped(expression: str, escaped: bool):
    assert is_escaping_call(expression) is escaped


# --- the pages ------------------------------------------------------------------


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_markup_interpolations_are_escaped_or_reviewed(page: str):
    found = unescaped_interpolations(PAGES[page])
    reviewed = REVIEWED[page]

    not_reviewed = {
        expression: count - reviewed.get(expression, 0)
        for expression, count in found.items()
        if count > reviewed.get(expression, 0)
    }
    no_longer_present = {
        expression: count - found.get(expression, 0)
        for expression, count in reviewed.items()
        if count > found.get(expression, 0)
    }

    assert not_reviewed == {}, (
        f"{page}: interpolations in markup that are neither escaped nor reviewed. "
        "Wrap run data in escapeHtml(); see this module's docstring for the rest."
    )
    assert no_longer_present == {}, (
        f"{page}: reviewed interpolations that are gone; regenerate the list "
        "so they cannot come back unnoticed."
    )


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_event_handlers_never_interpolate_values(page: str):
    # onclick="f('${id}')" lets a quote in the value end the JS string, in
    # either quote style and however the attribute is set. Handlers read the
    # value from a data- attribute instead.
    source = PAGES[page]

    assert re.search(r"\son\w+\s*=\s*\"[^\"]*\$\{", source) is None
    assert re.search(r"\son\w+\s*=\s*'[^']*\$\{", source) is None
    assert re.search(r"\son\w+\s*=\s*\\?[\"'][^\"'\n]*[\"']\s*\+", source) is None
    assert re.search(r"(?:setAttribute|\.attr)\(\s*[\"']on\w+[\"']", source) is None


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_pages_do_not_load_or_build_script_at_run_time(page: str):
    # The SRI check reads static tags; these would get around it.
    source = PAGES[page]

    for pattern in (
        r"createElement\(\s*[\"']script[\"']",
        r"document\.write",
        r"\bimport\s*\(",
        r"\beval\s*\(",
        r"new\s+Function\s*\(",
        r"\.(?:outerHTML|srcdoc)\s*=",
        r"insertAdjacentHTML",
    ):
        assert re.search(pattern, source) is None, pattern


@pytest.mark.parametrize("page", PAGE_NAMES)
def test_every_external_resource_is_pinned_with_sri(page: str):
    html = re.sub(r"<!--.*?-->", "", PAGES[page], flags=re.DOTALL)
    external = re.findall(
        r"<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*[\"']?(?:https?:)?//[^>]*>",
        html,
    )

    for tag in external:
        assert re.search(r"[@/-]\d+\.\d+\.\d+[/.]", tag), f"unpinned: {tag}"
        assert 'integrity="sha384-' in tag, f"no SRI: {tag}"
        assert 'crossorigin="anonymous"' in tag, f"no crossorigin: {tag}"


def test_markdown_libraries_are_reached_only_through_the_sanitising_helper():
    # Any other spelling (window["marked"], an alias) would add a mention.
    script = "".join(
        re.findall(r"<script>(.*?)</script>", PAGES["viz_tree.html"], re.DOTALL)
    )
    code = re.sub(r"//[^\n]*", "", script)

    assert re.findall(r"\bmarked\b[^\n;]*", code) == [
        "marked === 'undefined' || typeof DOMPurify === 'undefined') {",
        "marked.parse(source), text)",
        "marked.parseInline(source), text)",
    ]
    assert len(re.findall(r"\bDOMPurify\b", code)) == 2
    for page in ("index.html", "compare.html"):
        assert re.search(r"\b(?:marked|DOMPurify)\b", PAGES[page]) is None


# --- fragments escaped before they reach a template -----------------------------
# The scan above sees `${escapedLine}`, not how `escapedLine` was built, so
# the pure helpers that build such fragments are run instead.

PAYLOAD = "<img src=x onerror=\"alert('x')\">"


def _run_viz_tree_js(function_names: List[str], expression: str) -> str:
    sources = [
        extract_js_function(PAGES["viz_tree.html"], name) for name in function_names
    ]
    return run_js(sources, expression)


@pytest.mark.parametrize("prefix", ["+", "-", "@@ ", " "])
def test_diff_lines_are_escaped(prefix: str):
    html = _run_viz_tree_js(
        ["escapeHtml", "formatDiff"], f"formatDiff({json.dumps(prefix + PAYLOAD)})"
    )

    assert "<img" not in html
    assert "&lt;img src=x" in html


def test_prompt_diff_lines_are_escaped():
    html = _run_viz_tree_js(
        ["escapeHtml", "computeLCS", "computeLineDiff", "generatePromptDiff"],
        f"generatePromptDiff('kept\\nold', {json.dumps('kept' + chr(10) + PAYLOAD)})",
    )

    assert "<img" not in html
    assert "&lt;img src=x" in html
