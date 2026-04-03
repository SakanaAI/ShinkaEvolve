import sys
import sqlite3
from pathlib import Path
from types import ModuleType

markdown_stub = ModuleType("markdown")
setattr(markdown_stub, "markdown", lambda text: text)
sys.modules.setdefault("markdown", markdown_stub)


def _handler_cls():
    from shinka.webui.visualization import DatabaseRequestHandler

    return DatabaseRequestHandler


def _make_handler(search_root: Path):
    handler_cls = _handler_cls()
    handler = handler_cls.__new__(handler_cls)
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


def test_handle_get_database_stats_uses_best_correct_program(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    db_path = results_dir / "programs.sqlite"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE programs (
            id TEXT PRIMARY KEY,
            code TEXT,
            generation INTEGER,
            correct INTEGER,
            combined_score REAL,
            timestamp REAL,
            metadata TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "incorrect-best",
                "print('bad')",
                5,
                0,
                10.0,
                200.0,
                '{"pipeline_started_at": 100.0, "postprocess_finished_at": 200.0}',
            ),
            (
                "correct-best",
                "print('good')",
                2,
                1,
                3.5,
                150.0,
                '{"pipeline_started_at": 110.0, "postprocess_finished_at": 150.0}',
            ),
        ],
    )
    conn.commit()
    conn.close()

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_database_stats("results/programs.sqlite")

    assert "error" not in sent
    assert sent["data"]["generation_count"] == 2
    assert sent["data"]["best_generation"] == 2
    assert sent["data"]["max_generation"] == 5
    assert sent["data"]["correct_count"] == 1
    assert sent["data"]["best_score"] == 3.5
    assert sent["data"]["gens_since_improvement"] == 3


def test_handle_get_database_stats_returns_no_best_when_no_correct_programs(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    db_path = results_dir / "programs.sqlite"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE programs (
            id TEXT PRIMARY KEY,
            code TEXT,
            generation INTEGER,
            correct INTEGER,
            combined_score REAL,
            timestamp REAL,
            metadata TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("p1", "print('a')", 1, 0, 2.0, 100.0, "{}"),
            ("p2", "print('b')", 4, 0, 9.0, 130.0, "{}"),
        ],
    )
    conn.commit()
    conn.close()

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_database_stats("results/programs.sqlite")

    assert "error" not in sent
    assert sent["data"]["generation_count"] == 2
    assert sent["data"]["best_generation"] is None
    assert sent["data"]["max_generation"] == 4
    assert sent["data"]["correct_count"] == 0
    assert sent["data"]["best_score"] is None
    assert sent["data"]["gens_since_improvement"] == 4
