"""Render a meta note to HTML for the PDF export.

The PDF tools fetch every URL the HTML they are given refers to, and a note
is model output. The HTML made here therefore holds text, structure and
links only: raw HTML in the note is shown as text, images are replaced by
their alternative text, and attributes are limited to ones that load nothing.
"""

import re
import xml.etree.ElementTree as etree

import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor

# Attributes Markdown's own syntax produces that never load a resource.
_PLAIN_ATTRIBUTES = frozenset(
    {"href", "title", "class", "id", "colspan", "rowspan", "align", "start", "rel"}
)
# The one inline style Markdown writes itself: table column alignment.
_TEXT_ALIGN_STYLE = re.compile(r"text-align: (?:left|right|center);")


def _loads_nothing(name: str, value: str) -> bool:
    if name == "style":
        return _TEXT_ALIGN_STYLE.fullmatch(value) is not None
    return name in _PLAIN_ATTRIBUTES


class _DropLoadingMarkup(Treeprocessor):
    """Replace images by their alt text and drop attributes that can load."""

    def run(self, root: etree.Element) -> None:
        for parent in list(root.iter()):
            for image in parent.findall("img"):
                self._replace_by_alt_text(parent, image)
        for element in root.iter():
            for name, value in list(element.attrib.items()):
                if not _loads_nothing(name, value):
                    del element.attrib[name]

    @staticmethod
    def _replace_by_alt_text(parent: etree.Element, image: etree.Element) -> None:
        text = image.get("alt", "") + (image.tail or "")
        index = list(parent).index(image)
        if index == 0:
            parent.text = (parent.text or "") + text
        else:
            previous = parent[index - 1]
            previous.tail = (previous.tail or "") + text
        parent.remove(image)


class _TextOnly(Extension):
    """Markdown without raw HTML, images or free-form attributes."""

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        # Without these two, HTML written in the note is escaped like any
        # other text instead of being passed through.
        md.preprocessors.deregister("html_block")
        md.inlinePatterns.deregister("html")
        # After attr_list (priority 8), which lets a note set any attribute.
        md.treeprocessors.register(_DropLoadingMarkup(md), "drop_loading_markup", 0)


def render_note_html(text: str) -> str:
    """HTML for a note: formatting and links, nothing that loads a URL."""
    # `extra` first: it installs its own raw-HTML handling, which _TextOnly
    # then removes. nl2br keeps the note's single line breaks.
    return markdown.markdown(text, extensions=["extra", "nl2br", _TextOnly()])
