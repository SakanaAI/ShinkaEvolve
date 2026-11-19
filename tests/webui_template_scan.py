"""Find the ``${...}`` interpolations in the web UI's markup-building templates.

The pages build HTML with JavaScript template literals. This scanner lists
every interpolation in a template literal that looks like markup, so a test
can require each one to be escaped or reviewed. It knows just enough
JavaScript lexing for that: comments, quoted strings, and template literals
with nested ``${}``. Regular-expression literals are not recognised; one
holding a backtick or a quote would need handling here first.
"""

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# Calls whose result is safe to place in markup or in a URL.
ESCAPING_CALLS = (
    "escapeHtml",
    "renderMarkdown",
    "renderMarkdownInline",
    "encodeURIComponent",
)

_MARKUP = re.compile(r"<|\bstyle=|\bon\w+=")
_SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def _skip_quoted(src: str, start: int) -> int:
    """Index just past the quoted string that opens at ``start``."""
    quote = src[start]
    i = start + 1
    while i < len(src) and src[i] != quote and src[i] != "\n":
        i += 2 if src[i] == "\\" else 1
    return i + 1


def _scan_template(src: str, start: int, found: List[Tuple[str, List[str]]]) -> int:
    """Record the template literal whose body starts at ``start``.

    Appends ``(text, expressions)`` to ``found`` for it and for every template
    nested in its expressions, and returns the index past the closing backtick.
    """
    text: List[str] = []
    expressions: List[str] = []
    i = start
    while i < len(src):
        char = src[i]
        if char == "\\":
            text.append(src[i : i + 2])
            i += 2
        elif char == "`":
            i += 1
            break
        elif src.startswith("${", i):
            end = _scan_expression(src, i + 2, found)
            expressions.append(" ".join(src[i + 2 : end - 1].split()))
            text.append("${}")
            i = end
        else:
            text.append(char)
            i += 1
    found.append(("".join(text), expressions))
    return i


def _scan_expression(src: str, start: int, found: List[Tuple[str, List[str]]]) -> int:
    """Index past the ``}`` that closes the ``${`` whose body starts at ``start``."""
    depth = 1
    i = start
    while i < len(src) and depth:
        char = src[i]
        if char == "`":
            i = _scan_template(src, i + 1, found)
        elif char in "'\"":
            i = _skip_quoted(src, i)
        else:
            depth += {"{": 1, "}": -1}.get(char, 0)
            i += 1
    return i


def _templates(script: str) -> Iterator[Tuple[str, List[str]]]:
    found: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(script):
        char = script[i]
        if script.startswith("//", i):
            newline = script.find("\n", i)
            i = len(script) if newline == -1 else newline
        elif script.startswith("/*", i):
            close = script.find("*/", i)
            i = len(script) if close == -1 else close + 2
        elif char in "'\"":
            i = _skip_quoted(script, i)
        elif char == "`":
            i = _scan_template(script, i + 1, found)
        else:
            i += 1
    yield from found


def is_escaping_call(expression: str) -> bool:
    """Whether the whole expression is one call to an escaping function."""
    match = re.match(r"(\w+)\(", expression)
    if match is None or match.group(1) not in ESCAPING_CALLS:
        return False
    depth = 0
    for index in range(match.end() - 1, len(expression)):
        depth += {"(": 1, ")": -1}.get(expression[index], 0)
        if depth == 0:
            return index == len(expression) - 1
    return False


def markup_interpolations(page: str) -> List[str]:
    """Every interpolation in a markup-building template literal of ``page``."""
    return [
        expression
        for script in _SCRIPT.findall(page)
        for text, expressions in _templates(script)
        if _MARKUP.search(text)
        for expression in expressions
    ]


def unescaped_interpolations(page: str) -> Dict[str, int]:
    """How often each interpolation that is not an escaping call occurs."""
    counts = Counter(
        expression
        for expression in markup_interpolations(page)
        if not is_escaping_call(expression)
    )
    return dict(sorted(counts.items()))


WEBUI_DIR = Path(__file__).resolve().parents[1] / "shinka" / "webui"
PAGE_NAMES = ["viz_tree.html", "index.html", "compare.html"]
REVIEWED_PATH = (
    Path(__file__).parent / "fixtures" / "webui_reviewed_interpolations.json"
)


def read_page(name: str) -> str:
    return (WEBUI_DIR / name).read_text(encoding="utf-8")


def extract_js_function(page: str, function_name: str) -> str:
    """Source of the named top-level ``function`` declaration in ``page``."""
    match = re.search(rf"function {function_name}\([^)]*\)\s*\{{", page)
    assert match, f"Could not find function {function_name}"
    depth = 0
    for idx in range(page.find("{", match.start()), len(page)):
        if page[idx] == "{":
            depth += 1
        elif page[idx] == "}":
            depth -= 1
            if depth == 0:
                return page[match.start() : idx + 1]
    raise AssertionError(f"Unbalanced braces in {function_name}")


def run_js(sources: List[str], expression: str) -> Any:
    """Evaluate ``expression`` in node after the given sources; return its JSON."""
    script = "\n".join([*sources, f"console.log(JSON.stringify({expression}));"])
    completed = subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    # Rewrites the reviewed list from the pages as they are now. Run it only
    # after checking that every interpolation it adds holds no run data.
    current = {name: unescaped_interpolations(read_page(name)) for name in PAGE_NAMES}
    REVIEWED_PATH.write_text(
        json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {REVIEWED_PATH}")
