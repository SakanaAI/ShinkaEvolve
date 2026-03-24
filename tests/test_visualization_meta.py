import sys
import types
from pathlib import Path

sys.modules.setdefault("markdown", types.SimpleNamespace(markdown=lambda text: text))

from shinka.webui.visualization import DatabaseRequestHandler


def _make_handler(search_root: Path):
    handler = DatabaseRequestHandler.__new__(DatabaseRequestHandler)
    handler.search_root = str(search_root)
    handler._get_actual_db_path = lambda db_path: db_path
    handler.send_response = lambda code: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None
    handler.wfile = None
    return handler


def test_handle_get_meta_files_returns_processed_counts(tmp_path):
    results_dir = tmp_path / "results"
    meta_dir = results_dir / "meta"
    meta_dir.mkdir(parents=True)
    db_path = results_dir / "programs.sqlite"
    db_path.write_text("", encoding="utf-8")
    (meta_dir / "meta_5.txt").write_text("first", encoding="utf-8")
    (meta_dir / "meta_60.txt").write_text("latest", encoding="utf-8")

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_meta_files("results/programs.sqlite")

    assert "error" not in sent
    assert sent["data"] == [
        {
            "processed_count": 5,
            "generation": 5,
            "filename": "meta_5.txt",
            "path": str(meta_dir / "meta_5.txt"),
        },
        {
            "processed_count": 60,
            "generation": 60,
            "filename": "meta_60.txt",
            "path": str(meta_dir / "meta_60.txt"),
        },
    ]


def test_handle_get_meta_content_returns_processed_count(tmp_path):
    results_dir = tmp_path / "results"
    meta_dir = results_dir / "meta"
    meta_dir.mkdir(parents=True)
    db_path = results_dir / "programs.sqlite"
    db_path.write_text("", encoding="utf-8")
    meta_path = meta_dir / "meta_60.txt"
    meta_path.write_text("# META RECOMMENDATIONS", encoding="utf-8")

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_meta_content("results/programs.sqlite", "60")

    assert "error" not in sent
    assert sent["data"] == {
        "processed_count": 60,
        "generation": 60,
        "filename": "meta_60.txt",
        "content": "# META RECOMMENDATIONS",
    }
