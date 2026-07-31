import subprocess
from io import BytesIO
from pathlib import Path

from shinka.webui.visualization import DatabaseRequestHandler


def _make_handler(search_root: Path) -> DatabaseRequestHandler:
    handler = DatabaseRequestHandler.__new__(DatabaseRequestHandler)
    handler.search_root = str(search_root)
    handler._get_actual_db_path = lambda db_path: db_path
    handler.send_header = lambda *_args, **_kwargs: None
    handler.end_headers = lambda: None
    handler.wfile = BytesIO()
    return handler


def test_pandoc_fallback_reads_sanitized_html_in_sandbox(monkeypatch):
    handler = DatabaseRequestHandler.__new__(DatabaseRequestHandler)
    monkeypatch.setattr(handler, "_fix_line_breaks", lambda content: content)
    monkeypatch.setattr(handler, "_add_program_boxes_html", lambda content: content)
    monkeypatch.setattr(handler, "_get_logo_base64", lambda: None)

    calls = []
    pandoc_html = None

    def run(command, **kwargs):
        nonlocal pandoc_html
        calls.append(command)
        if command[0] == "wkhtmltopdf":
            raise FileNotFoundError
        pandoc_html = Path(command[1]).read_text(encoding="utf-8")
        Path(command[-1]).write_bytes(b"pdf")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    payload = """**safe**
<img src="file:///etc/passwd">
![remote](https://example.test/x)
<script>fetch('https://example.test/script')</script>"""
    assert handler._generate_pdf(payload, "1") == b"pdf"

    pandoc_command = calls[-1]
    assert pandoc_command[0] == "pandoc"
    assert "--from=html" in pandoc_command
    assert "--sandbox" in pandoc_command
    source_path = Path(pandoc_command[1])
    assert source_path.suffix == ".html"
    assert pandoc_html is not None
    assert "<img" not in pandoc_html
    assert " href=" not in pandoc_html
    assert "<script" not in pandoc_html
    assert "<strong>safe</strong>" in pandoc_html


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


def test_pdf_endpoint_reports_conversion_failure(tmp_path):
    results_dir = tmp_path / "results"
    meta_dir = results_dir / "meta"
    meta_dir.mkdir(parents=True)
    (results_dir / "programs.sqlite").write_text("", encoding="utf-8")
    (meta_dir / "meta_1.txt").write_text("safe", encoding="utf-8")

    handler = _make_handler(tmp_path)
    handler._generate_pdf = lambda _content, _generation: None
    errors = []
    responses = []
    sent_headers = []
    handler.send_error = lambda code, message: errors.append((code, message))
    handler.send_response = lambda code: responses.append(code)
    handler.send_header = lambda name, value: sent_headers.append((name, value))

    handler.handle_download_meta_pdf("results/programs.sqlite", "1")

    assert errors == [(500, "PDF generation failed")]
    assert responses == []
    assert ("Content-Type", "application/pdf") not in sent_headers
