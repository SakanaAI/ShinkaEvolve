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


def test_handle_get_programs_summary_merges_failed_attempt_nodes(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    db_path = results_dir / "programs.sqlite"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE programs (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            generation INTEGER,
            timestamp REAL,
            combined_score REAL,
            correct INTEGER,
            complexity REAL,
            island_idx INTEGER,
            children_count INTEGER,
            public_metrics TEXT,
            private_metrics TEXT,
            metadata TEXT,
            embedding_pca_2d TEXT,
            embedding_pca_3d TEXT,
            embedding_cluster_id INTEGER,
            language TEXT,
            text_feedback TEXT,
            top_k_inspiration_ids TEXT,
            archive_inspiration_ids TEXT,
            migration_history TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE archive (
            program_id TEXT PRIMARY KEY
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE metadata_store (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attempt_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    failure_json = results_dir / "gen_7" / "failure.json"
    failure_json.parent.mkdir(parents=True)
    failure_json.write_text(
        '{"failure_reason":"proposal failed","failure_json_path":"results/gen_7/failure.json"}',
        encoding="utf-8",
    )
    conn.execute(
        "INSERT INTO attempt_log (generation, stage, status, details, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            7,
            "proposal",
            "failed",
            '{"node_kind":"failed_proposal","failure_stage":"proposal","failure_class":"patch_apply_failed","failure_reason":"proposal failed","parent_id":"parent-1","failure_json_path":"results/gen_7/failure.json","pipeline_started_at":100.0,"sampling_started_at":100.0,"sampling_finished_at":105.0,"evaluation_started_at":105.0,"evaluation_finished_at":105.0,"postprocess_started_at":105.0,"postprocess_finished_at":105.0}',
            123.0,
        ),
    )
    conn.commit()
    conn.close()

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_programs_summary("results/programs.sqlite")

    assert "error" not in sent
    assert len(sent["data"]) == 1
    failed_node = sent["data"][0]
    assert failed_node["id"] == "failed:proposal:7"
    assert failed_node["parent_id"] == "parent-1"
    assert failed_node["metadata"]["node_kind"] == "failed_proposal"
    assert failed_node["text_feedback"] == "proposal failed"
    assert failed_node["metadata"]["sampling_started_at"] == 100.0
    assert failed_node["metadata"]["postprocess_finished_at"] == 105.0


def test_handle_get_program_details_returns_failed_attempt_node(tmp_path):
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
    conn.execute(
        """
        CREATE TABLE metadata_store (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attempt_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    gen_dir = results_dir / "gen_7"
    gen_dir.mkdir(parents=True)
    (gen_dir / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    (gen_dir / "failure.json").write_text(
        '{"artifacts":{"generated_code_path":"results/gen_7/main.py"},"failure_reason":"proposal failed","failure_json_path":"results/gen_7/failure.json"}',
        encoding="utf-8",
    )
    conn.execute(
        "INSERT INTO attempt_log (generation, stage, status, details, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            7,
            "proposal",
            "failed",
            '{"node_kind":"failed_proposal","failure_stage":"proposal","failure_class":"llm_output_invalid","failure_reason":"proposal failed","failure_json_path":"results/gen_7/failure.json","pipeline_started_at":100.0,"sampling_started_at":100.0,"sampling_finished_at":105.0,"evaluation_started_at":105.0,"evaluation_finished_at":105.0,"postprocess_started_at":105.0,"postprocess_finished_at":105.0}',
            123.0,
        ),
    )
    conn.commit()
    conn.close()

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_program_details(
        "results/programs.sqlite", "failed:proposal:7"
    )

    assert "error" not in sent
    assert sent["data"]["id"] == "failed:proposal:7"
    assert sent["data"]["code"] == "print('candidate')\n"
    assert sent["data"]["metadata"]["failure_class"] == "llm_output_invalid"
    assert sent["data"]["metadata"]["pipeline_started_at"] == 100.0
    assert sent["data"]["metadata"]["postprocess_finished_at"] == 105.0


def test_handle_get_program_details_loads_failed_non_python_node_with_language_fallback(
    tmp_path,
):
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
    conn.execute(
        """
        CREATE TABLE metadata_store (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attempt_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    gen_dir = results_dir / "gen_8"
    gen_dir.mkdir(parents=True)
    (gen_dir / "main.js").write_text("console.log('candidate');\n", encoding="utf-8")
    (gen_dir / "failure.json").write_text(
        '{"language":"javascript","failure_reason":"proposal failed","failure_json_path":"results/gen_8/failure.json"}',
        encoding="utf-8",
    )
    conn.execute(
        "INSERT INTO attempt_log (generation, stage, status, details, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            8,
            "proposal",
            "failed",
            '{"node_kind":"failed_proposal","failure_stage":"proposal","failure_class":"llm_output_invalid","failure_reason":"proposal failed","failure_json_path":"results/gen_8/failure.json"}',
            130.0,
        ),
    )
    conn.commit()
    conn.close()

    handler = _make_handler(tmp_path)
    sent = {}
    handler.send_json_response = lambda data: sent.setdefault("data", data)
    handler.send_error = lambda code, msg: sent.setdefault("error", (code, msg))

    handler.handle_get_program_details(
        "results/programs.sqlite", "failed:proposal:8"
    )

    assert "error" not in sent
    assert sent["data"]["id"] == "failed:proposal:8"
    assert sent["data"]["language"] == "javascript"
    assert sent["data"]["code"] == "console.log('candidate');\n"


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
