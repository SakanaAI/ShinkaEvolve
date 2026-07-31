import subprocess
from pathlib import Path

from shinka.webui.visualization import DatabaseRequestHandler


def test_pandoc_fallback_parses_llm_content_as_plain_text(monkeypatch):
    handler = DatabaseRequestHandler.__new__(DatabaseRequestHandler)
    monkeypatch.setattr(handler, "_fix_line_breaks", lambda content: content)
    monkeypatch.setattr(handler, "_add_program_boxes_html", lambda content: content)
    monkeypatch.setattr(handler, "_get_logo_base64", lambda: None)

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "wkhtmltopdf":
            raise FileNotFoundError
        Path(command[-1]).write_bytes(b"pdf")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    payload = '<img src="file:///etc/passwd"><img src="https://example.test/x">'
    assert handler._generate_pdf(payload, "1") == b"pdf"

    pandoc_command = calls[-1]
    assert pandoc_command[0] == "pandoc"
    assert "--from=plain" in pandoc_command
    source_path = Path(pandoc_command[1])
    assert source_path.suffix == ".txt"


def test_wkhtmltopdf_disables_active_and_external_content(monkeypatch):
    handler = DatabaseRequestHandler.__new__(DatabaseRequestHandler)
    monkeypatch.setattr(handler, "_fix_line_breaks", lambda content: content)
    monkeypatch.setattr(handler, "_add_program_boxes_html", lambda content: content)
    monkeypatch.setattr(handler, "_get_logo_base64", lambda: None)

    commands = []
    rendered_html = None

    def run(command, **kwargs):
        nonlocal rendered_html
        commands.append(command)
        rendered_html = Path(command[-2]).read_text(encoding="utf-8")
        Path(command[-1]).write_bytes(b"pdf")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    payload = """**safe**
> quoted
`<img src="code">`
<img src="https://example.test/raw">
![remote](https://example.test/markdown)
[link](https://example.test/link)
<style>body { background: url(https://example.test/css); }</style>
<script>fetch('https://example.test/script')</script>"""
    generation = '1</title><img src="https://example.test/generation">'
    assert handler._generate_pdf(payload, generation) == b"pdf"
    assert "--disable-javascript" in commands[0]
    assert "--disable-local-file-access" in commands[0]
    assert "--disable-external-links" in commands[0]
    assert rendered_html is not None
    assert "<img" not in rendered_html
    assert " href=" not in rendered_html
    assert "<style>body" not in rendered_html
    assert "<script" not in rendered_html
    assert "<strong>safe</strong>" in rendered_html
    assert "<blockquote>" in rendered_html
    assert "<code>&lt;img src=&quot;code&quot;&gt;</code>" in rendered_html
