from pathlib import Path
import subprocess

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

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"pdf")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    assert handler._generate_pdf("safe", "1") == b"pdf"
    assert "--disable-javascript" in commands[0]
    assert "--disable-local-file-access" in commands[0]
    assert "--disable-external-links" in commands[0]
