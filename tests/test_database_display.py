import sqlite3

from rich.console import Console

from shinka.database.display import DatabaseDisplay


def test_print_summary_shows_total_programs_as_stored_over_target():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE programs (
            id TEXT PRIMARY KEY,
            code TEXT,
            language TEXT,
            parent_id TEXT,
            generation INTEGER,
            timestamp REAL,
            metadata TEXT,
            combined_score REAL,
            public_metrics TEXT,
            private_metrics TEXT,
            complexity REAL,
            embedding TEXT,
            correct BOOLEAN,
            island_idx INTEGER,
            children_count INTEGER
        )
        """
    )
    cursor.execute("CREATE TABLE archive (program_id TEXT PRIMARY KEY)")
    cursor.execute(
        """
        INSERT INTO programs (
            id, code, language, parent_id, generation, timestamp, metadata,
            combined_score, public_metrics, private_metrics, complexity,
            embedding, correct, island_idx, children_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "p1",
            "print('hi')",
            "python",
            None,
            0,
            0.0,
            "{}",
            1.0,
            "{}",
            "{}",
            1.0,
            "[]",
            1,
            0,
            0,
        ),
    )
    conn.commit()

    console = Console(force_terminal=False, color_system=None, width=120)
    island_manager = type(
        "Islands",
        (),
        {
            "format_island_display": lambda self: "",
            "get_migration_info": lambda self: "",
        },
    )()
    config = type("Cfg", (), {"archive_size": 40, "num_islands": 0})()
    display = DatabaseDisplay(
        cursor=cursor,
        conn=conn,
        config=config,
        island_manager=island_manager,
        count_programs_func=lambda: 1,
        get_best_program_func=lambda: None,
        default_console=console,
    )

    with console.capture() as capture:
        display.print_summary(total_program_target=100)

    output = capture.get()
    assert "1 / 100" in output

    conn.close()
