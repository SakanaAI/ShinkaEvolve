"""Sanitization helpers for PDF renderers that process model-generated text."""

import html
from html.parser import HTMLParser


class _TagOnlyHtmlParser(HTMLParser):
    ALLOWED_TAGS = frozenset(
        {
            "a",
            "blockquote",
            "br",
            "code",
            "dd",
            "del",
            "div",
            "dl",
            "dt",
            "em",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "hr",
            "li",
            "ol",
            "p",
            "pre",
            "span",
            "strong",
            "sub",
            "sup",
            "table",
            "tbody",
            "td",
            "th",
            "thead",
            "tr",
            "ul",
        }
    )
    VOID_TAGS = frozenset({"br", "hr"})

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []

    def handle_starttag(self, tag, _attrs):
        if tag in self.ALLOWED_TAGS:
            self._parts.append(f"<{tag}>")

    def handle_startendtag(self, tag, _attrs):
        if tag in self.ALLOWED_TAGS:
            self._parts.append(f"<{tag}>")

    def handle_endtag(self, tag):
        if tag in self.ALLOWED_TAGS and tag not in self.VOID_TAGS:
            self._parts.append(f"</{tag}>")

    def handle_data(self, data):
        self._parts.append(html.escape(data))

    def handle_entityref(self, name):
        self._parts.append(f"&{name};")

    def handle_charref(self, name):
        self._parts.append(f"&#{name};")

    def sanitized_html(self) -> str:
        return "".join(self._parts)


def sanitize_pdf_html(html_content: str) -> str:
    """Keep formatting tags while removing every resource-bearing attribute."""
    parser = _TagOnlyHtmlParser()
    parser.feed(html_content)
    parser.close()
    return parser.sanitized_html()


def escape_pdf_text(content: str) -> str:
    """Escape untrusted text interpolated into the PDF document shell."""
    return html.escape(content)
