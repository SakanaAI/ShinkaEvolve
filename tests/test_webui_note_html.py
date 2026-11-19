"""The meta PDF export renders a note without loading anything the note names.

wkhtmltopdf and pandoc fetch every URL the HTML they are given refers to, and
a note is model output. So the HTML made from it carries text, structure and
links only.
"""

from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import markdown  # type: ignore[import-untyped]
import pytest

from shinka.webui import visualization
from shinka.webui.note_html import render_note_html

ATTACKER = "http://attacker.example"

HOSTILE_NOTE = f"""# Notes

Raw <img src="{ATTACKER}/raw-img"> and <b onmouseover="x()">inline</b> markup.

<iframe src="{ATTACKER}/iframe"></iframe>
<link rel="stylesheet" href="{ATTACKER}/link-css">
<style>@import url({ATTACKER}/style-import);</style>
<div style="background:url({ATTACKER}/style-attr)">x</div>
<object data="{ATTACKER}/object"></object>
<embed src="{ATTACKER}/embed">
<script src="{ATTACKER}/script"></script>

![inline image]({ATTACKER}/md-img)
![reference image][ref]

[ref]: {ATTACKER}/md-ref-img

A paragraph with attributes
{{: style="background-image:url({ATTACKER}/attr-list)" onclick="x()" }}

A [link]({ATTACKER}/link) stays a link.
"""

# What Markdown itself produces for text formatting; none of it loads a URL.
TEXT_TAGS = {
    *("h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "hr", "blockquote"),
    *("strong", "em", "code", "pre", "a", "ul", "ol", "li", "div", "span"),
    *("table", "thead", "tbody", "tr", "th", "td", "dl", "dt", "dd", "sup", "abbr"),
}
LOADING_ATTRIBUTES = {"src", "srcset", "data", "background", "poster", "action"}

Element = Tuple[str, List[Tuple[str, Optional[str]]]]


class _Elements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: List[Element] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        self.elements.append((tag, attrs))


def _elements(html: str) -> List[Element]:
    parser = _Elements()
    parser.feed(html)
    return parser.elements


def test_note_html_holds_nothing_that_loads_a_url():
    html = render_note_html(HOSTILE_NOTE)

    for tag, attrs in _elements(html):
        assert tag in TEXT_TAGS, tag
        for name, value in attrs:
            assert name not in LOADING_ATTRIBUTES, (tag, name)
            assert not name.startswith("on"), (tag, name)
            assert "url(" not in (value or ""), (tag, name, value)


def test_note_html_shows_raw_markup_as_text_and_keeps_links():
    html = render_note_html(HOSTILE_NOTE)

    assert "&lt;iframe" in html
    assert "&lt;b onmouseover" in html
    assert "inline image" in html
    assert "reference image" in html
    assert f'<a href="{ATTACKER}/link">link</a>' in html


def test_note_html_keeps_markdown_formatting():
    html = render_note_html(
        "# Title\n\n**bold** and `code`\n\n- one\n- two\n\n"
        "| left | right |\n|:-----|------:|\n| a | b |\n\n"
        "```python\nx = 1 < 2\n```\n\nline one\nline two\n"
    )

    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<li>one</li>" in html
    assert 'style="text-align: right;"' in html
    assert "x = 1 &lt; 2" in html
    assert "line one<br" in html


def test_meta_pdf_tools_are_given_html_that_loads_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    rendered: List[str] = []

    def fake_run(cmd, **kwargs):
        html_path = next(arg for arg in cmd if str(arg).endswith(".html"))
        rendered.append(Path(html_path).read_text(encoding="utf-8"))
        return SimpleNamespace(returncode=1, stdout="", stderr="stub")

    monkeypatch.setattr(visualization.subprocess, "run", fake_run)
    handler = visualization.DatabaseRequestHandler.__new__(
        visualization.DatabaseRequestHandler
    )

    handler._generate_pdf(HOSTILE_NOTE, "1")

    assert len(rendered) == 2, "expected wkhtmltopdf and the pandoc fallback"
    for html in rendered:
        elements = _elements(html)
        assert not {tag for tag, _ in elements} & {
            *("iframe", "script", "object", "embed", "link")
        }
        for _, attrs in elements:
            for name, value in attrs:
                # The page's own logo, embedded as a data URI, is the one image.
                if name in LOADING_ATTRIBUTES:
                    assert (value or "").startswith("data:image/png;base64,")
                assert "url(" not in (value or "")
        # One stylesheet, the page's own; the note's is shown as text.
        assert html.count("<style") == 1


def test_note_without_markup_or_images_renders_as_plain_markdown_does():
    note = (
        "# INDIVIDUAL PROGRAM SUMMARIES\n\n"
        "**Program Name: fast_sort**\n"
        "- **Implementation**: uses `heapq`, 2 < 3 & so on\n"
        "- **Performance**: 0.93\n\n"
        "> a quote\n\n"
        "1. first\n2. second\n\n"
        "| a | b |\n|:--|--:|\n| 1 | 2 |\n\n"
        "See [the docs](https://example.org/docs).\n"
    )

    assert render_note_html(note) == markdown.markdown(
        note, extensions=["extra", "nl2br"]
    )
