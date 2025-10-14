import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
import random
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import math
from .complexity import analyze_code_metrics
from .parents import CombinedParentSelector
from .inspirations import CombinedContextSelector
from .islands import CombinedIslandManager
from .display import DatabaseDisplay
from shinka.llm.embedding import EmbeddingClient

logger = logging.getLogger(__name__)


def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values from a data structure, replacing them with
    None. This ensures JSON serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        # Handle numpy arrays and scalars
        if np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            else:
                return float(obj)
        else:
            # For numpy arrays, convert to list and clean recursively
            return clean_nan_values(obj.tolist())
    else:
        return obj


@dataclass
class DatabaseConfig:
    db_path: Optional[str] = None
    num_islands: int = 4
    archive_size: int = 100

    # Inspiration parameters
    elite_selection_ratio: float = 0.3  # Prop of elites inspirations
    num_archive_inspirations: int = 5  # No. inspiration programs
    num_top_k_inspirations: int = 2  # No. top-k inspiration programs

    # Island model/migration parameters
    migration_interval: int = 10  # Migrate every N generations
    migration_rate: float = 0.1  # Prop. of island pop. to migrate
    island_elitism: bool = True  # Keep best prog on their islands
    enforce_island_separation: bool = (
        True  # Enforce full island separation for inspirations
    )

    # Parent selection parameters
    parent_selection_strategy: str = (
        "power_law"  # "weighted"/"power_law" / "beam_search"
    )

    # Power-law parent selection parameters
    exploitation_alpha: float = 1.0  # 0=uniform, 1=power-law
    exploitation_ratio: float = 0.2  # Chance to pick from archive

    # Weighted tree parent selection parameters
    parent_selection_lambda: float = 10.0  # >0 sharpness of sigmoid

    # Beam search parent selection parameters
    num_beams: int = 5


def db_retry(max_retries=5, initial_delay=0.1, backoff_factor=2):
    """
    A decorator to retry database operations on specific SQLite errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    sqlite3.OperationalError,
                    sqlite3.DatabaseError,
                    sqlite3.IntegrityError,
                ) as e:
                    if i == max_retries - 1:
                        logger.error(
                            f"DB operation {func.__name__} failed after "
                            f"{max_retries} retries: {e}"
                        )
                        raise
                    logger.warning(
                        f"DB operation {func.__name__} failed with "
                        f"{type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            # This part should not be reachable if max_retries > 0
            raise RuntimeError(
                f"DB retry logic failed for function {func.__name__} without "
                "raising an exception."
            )

        return wrapper

    return decorator


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    archive_inspiration_ids: List[str] = field(
        default_factory=list
    )  # IDs of programs used as archive inspiration
    top_k_inspiration_ids: List[str] = field(
        default_factory=list
    )  # IDs of programs used as top-k inspiration
    island_idx: Optional[int] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    code_diff: Optional[str] = None

    # Performance metrics
    combined_score: float = 0.0
    public_metrics: Dict[str, Any] = field(default_factory=dict)
    private_metrics: Dict[str, Any] = field(default_factory=dict)
    text_feedback: Union[str, List[str]] = ""
    correct: bool = False  # Whether the program is functionally correct
    children_count: int = 0

    # Derived features
    complexity: float = 0.0  # Calculated based on code or other features
    embedding: List[float] = field(default_factory=list)
    embedding_pca_2d: List[float] = field(default_factory=list)
    embedding_pca_3d: List[float] = field(default_factory=list)
    embedding_cluster_id: Optional[int] = None

    # Migration history
    migration_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Archive status
    in_archive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation, cleaning NaN values for JSON."""
        data = asdict(self)
        return clean_nan_values(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation, ensuring correct types for
        nested dicts."""
        # Ensure metrics and metadata are dictionaries, even if None/empty from
        # DB or input
        data["public_metrics"] = (
            data.get("public_metrics")
            if isinstance(data.get("public_metrics"), dict)
            else {}
        )
        data["private_metrics"] = (
            data.get("private_metrics")
            if isinstance(data.get("private_metrics"), dict)
            else {}
        )
        data["metadata"] = (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        )
        # Ensure inspiration_ids is a list
        archive_ids_val = data.get("archive_inspiration_ids")
        if isinstance(archive_ids_val, list):
            data["archive_inspiration_ids"] = archive_ids_val
        else:
            data["archive_inspiration_ids"] = []

        top_k_ids_val = data.get("top_k_inspiration_ids")
        if isinstance(top_k_ids_val, list):
            data["top_k_inspiration_ids"] = top_k_ids_val
        else:
            data["top_k_inspiration_ids"] = []

        # Ensure embedding is a list
        embedding_val = data.get("embedding")
        if isinstance(embedding_val, list):
            data["embedding"] = embedding_val
        else:
            data["embedding"] = []

        embedding_pca_2d_val = data.get("embedding_pca_2d")
        if isinstance(embedding_pca_2d_val, list):
            data["embedding_pca_2d"] = embedding_pca_2d_val
        else:
            data["embedding_pca_2d"] = []

        embedding_pca_3d_val = data.get("embedding_pca_3d")
        if isinstance(embedding_pca_3d_val, list):
            data["embedding_pca_3d"] = embedding_pca_3d_val
        else:
            data["embedding_pca_3d"] = []

        # Ensure migration_history is a list
        migration_history_val = data.get("migration_history")
        if isinstance(migration_history_val, list):
            data["migration_history"] = migration_history_val
        else:
            data["migration_history"] = []

        # Filter out keys not in Program fields to avoid TypeError with **data
        program_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in program_fields}

        return cls(**filtered_data)


class ProgramDatabase:
    """
    SQLite-backed database for storing and managing programs during an
    evolutionary process.
    Supports MAP-Elites style feature-based organization, island
    populations, and an archive of elites.
    """

    def __init__(self, config: DatabaseConfig, read_only: bool = False):
        self.config = config
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.read_only = read_only
        self.embedding_client = EmbeddingClient()

        self.last_iteration: int = 0
        self.best_program_id: Optional[str] = None
        self.beam_search_parent_id: Optional[str] = None
        # For deferring expensive operations
        self._schedule_migration: bool = False

        # Initialize island manager (will be set after db connection)
        self.island_manager: Optional[CombinedIslandManager] = None

        db_path_str = getattr(self.config, "db_path", None)

        if db_path_str:
            db_file = Path(db_path_str).resolve()
            if not read_only:
                # Robustness check for unclean shutdown with WAL
                db_wal_file = Path(f"{db_file}-wal")
                db_shm_file = Path(f"{db_file}-shm")
                if (
                    db_file.exists()
                    and db_file.stat().st_size == 0
                    and (db_wal_file.exists() or db_shm_file.exists())
                ):
                    logger.warning(
                        f"Database file {db_file} is empty but WAL/SHM files "
                        "exist. This may indicate an unclean shutdown. "
                        "Removing WAL/SHM files to attempt recovery."
                    )
                    if db_wal_file.exists():
                        db_wal_file.unlink()
                    if db_shm_file.exists():
                        db_shm_file.unlink()
                db_file.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(db_file), timeout=30.0)
                logger.debug(f"Connected to SQLite database: {db_file}")
            else:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                db_uri = f"file:{db_file}?mode=ro"
                self.conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)
                logger.debug(
                    "Connected to SQLite database in read-only mode: %s",
                    db_file,
                )
        else:
            self.conn = sqlite3.connect(":memory:")
            logger.info("Initialized in-memory SQLite database.")

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        if not self.read_only:
            self._create_tables()
        self._load_metadata_from_db()

        # Initialize island manager now that database is ready
        self.island_manager = CombinedIslandManager(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
        )
        self.island_manager.attach_database(self)

        count = self._count_programs_in_db()
        logger.debug(f"DB initialized with {count} programs.")
        logger.debug(
            f"Last iter: {self.last_iteration}. Best ID: {self.best_program_id}"
        )

    def _create_tables(self):
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        # Set SQLite pragmas for better performance and stability
        # Use WAL mode for better concurrency support and reduced locking
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA busy_timeout = 30000;")  # 30 second busy timeout
        self.cursor.execute(
            "PRAGMA wal_autocheckpoint = 1000;"
        )  # Checkpoint every 1000 pages
        self.cursor.execute("PRAGMA synchronous = NORMAL;")  # Safer, faster
        self.cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys = ON;")  # For data integrity

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                parent_id TEXT,
                generation INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                code_diff TEXT,
                combined_score REAL,
                text_feedback TEXT,
                complexity REAL,
                embedding_cluster_id INTEGER,
                correct BOOLEAN DEFAULT 0,
                children_count INTEGER NOT NULL DEFAULT 0,
                island_idx INTEGER
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS program_metrics (
                program_id TEXT PRIMARY KEY,
                public_metrics TEXT,
                private_metrics TEXT,
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS program_embeddings (
                program_id TEXT PRIMARY KEY,
                embedding TEXT,
                embedding_pca_2d TEXT,
                embedding_pca_3d TEXT,
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS program_metadata (
                program_id TEXT PRIMARY KEY,
                metadata TEXT,
                migration_history TEXT,
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS program_inspirations (
                program_id TEXT NOT NULL,
                inspiration_id TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('archive', 'top_k')),
                PRIMARY KEY (program_id, inspiration_id, source),
                FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE,
                FOREIGN KEY (inspiration_id) REFERENCES programs(id) ON DELETE CASCADE
            )
            """
        )

        # Add indices for common query patterns
        self._ensure_component_indexes()

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archive (
                program_id TEXT PRIMARY KEY,
                FOREIGN KEY (program_id) REFERENCES programs(id)
                    ON DELETE CASCADE
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_store (
                key TEXT PRIMARY KEY, value TEXT
            )
            """
        )

        self.conn.commit()

        # Run any necessary migrations
        self._run_migrations()

        logger.debug("Database tables and indices ensured to exist.")

    def _run_migrations(self):
        """Run database migrations for schema changes."""
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        try:
            self._migrate_legacy_program_table()
            self._ensure_component_indexes()
        except sqlite3.Error as exc:
            logger.error("Database migration failed: %s", exc)
            raise

    def _ensure_program_indexes(self) -> None:
        """Ensure indexes on programs table for common access patterns."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        idx_cmds = [
            "CREATE INDEX IF NOT EXISTS idx_programs_generation ON "
            "programs(generation)",
            "CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_programs_complexity ON "
            "programs(complexity)",
            "CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_programs_children_count ON "
            "programs(children_count)",
            "CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON "
            "programs(island_idx)",
        ]
        for cmd in idx_cmds:
            self.cursor.execute(cmd)

    def _ensure_component_indexes(self) -> None:
        """Ensure indexes for the new component tables."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        self._ensure_program_indexes()

        component_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_program_embeddings_program_id "
            "ON program_embeddings(program_id)",
            "CREATE INDEX IF NOT EXISTS idx_program_inspirations_program_id "
            "ON program_inspirations(program_id)",
            "CREATE INDEX IF NOT EXISTS idx_program_inspirations_source "
            "ON program_inspirations(source)",
            "CREATE INDEX IF NOT EXISTS idx_program_inspirations_inspiration "
            "ON program_inspirations(inspiration_id)",
        ]
        for cmd in component_indexes:
            self.cursor.execute(cmd)

    def _migrate_legacy_program_table(self) -> None:
        """Normalize legacy programs table storing heavy JSON content inline."""
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        self.cursor.execute("PRAGMA table_info(programs)")
        table_info = self.cursor.fetchall()
        column_names: Set[str] = {row["name"] for row in table_info}

        legacy_columns = {
            "archive_inspiration_ids",
            "top_k_inspiration_ids",
            "public_metrics",
            "private_metrics",
            "metadata",
            "migration_history",
            "embedding",
            "embedding_pca_2d",
            "embedding_pca_3d",
        }

        if not column_names.intersection(legacy_columns):
            # Already migrated
            return

        logger.info("Detected legacy programs table; migrating to normalized schema.")

        # Backfill component tables before removing columns
        self._backfill_component_tables_from_legacy(column_names)

        # Temporarily disable FK checks while rebuilding the table.
        self.cursor.execute("PRAGMA foreign_keys = OFF;")
        self.conn.commit()

        try:
            self.cursor.execute("ALTER TABLE programs RENAME TO programs_legacy")

            self.cursor.execute(
                """
                CREATE TABLE programs (
                    id TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    language TEXT NOT NULL,
                    parent_id TEXT,
                    generation INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    code_diff TEXT,
                    combined_score REAL,
                    text_feedback TEXT,
                    complexity REAL,
                    embedding_cluster_id INTEGER,
                    correct BOOLEAN DEFAULT 0,
                    children_count INTEGER NOT NULL DEFAULT 0,
                    island_idx INTEGER
                )
                """
            )

            def col_expr(name: str, default_sql: str) -> str:
                return name if name in column_names else f"{default_sql} AS {name}"

            text_feedback_expr = col_expr("text_feedback", "''")
            complexity_expr = col_expr("complexity", "NULL")
            cluster_expr = col_expr("embedding_cluster_id", "NULL")
            children_expr = col_expr("children_count", "0")
            island_expr = col_expr("island_idx", "NULL")

            self.cursor.execute(
                f"""
                INSERT INTO programs (
                    id, code, language, parent_id, generation, timestamp,
                    code_diff, combined_score, text_feedback, complexity,
                    embedding_cluster_id, correct, children_count, island_idx
                )
                SELECT
                    id,
                    code,
                    language,
                    parent_id,
                    generation,
                    timestamp,
                    code_diff,
                    combined_score,
                    {text_feedback_expr},
                    {complexity_expr},
                    {cluster_expr},
                    correct,
                    {children_expr},
                    {island_expr}
                FROM programs_legacy
                """
            )

            self.cursor.execute("DROP TABLE programs_legacy")
            self.conn.commit()
            logger.info("Programs table migrated successfully.")
        finally:
            self.cursor.execute("PRAGMA foreign_keys = ON;")
            self.conn.commit()

    def _backfill_component_tables_from_legacy(self, column_names: Set[str]) -> None:
        """Populate new component tables using legacy columns."""
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        # Determine which columns we can extract from.
        selectable_columns = []
        for name in [
            "archive_inspiration_ids",
            "top_k_inspiration_ids",
            "public_metrics",
            "private_metrics",
            "metadata",
            "migration_history",
            "embedding",
            "embedding_pca_2d",
            "embedding_pca_3d",
        ]:
            if name in column_names:
                selectable_columns.append(name)

        if not selectable_columns:
            return

        logger.info("Backfilling component tables from legacy program columns.")

        select_clause = ", ".join(["id"] + selectable_columns)
        self.cursor.execute(f"SELECT {select_clause} FROM programs")
        rows = self.cursor.fetchall()

        for row in rows:
            program_id = row["id"]
            row_keys = set(row.keys())

            def _safe_load(value: Optional[str], fallback):
                if value is None or value == "":
                    return fallback
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return fallback

            # Metrics
            public_metrics_raw = (
                row["public_metrics"] if "public_metrics" in row_keys else None
            )
            private_metrics_raw = (
                row["private_metrics"] if "private_metrics" in row_keys else None
            )
            if public_metrics_raw is not None or private_metrics_raw is not None:
                self.cursor.execute(
                    """
                    INSERT OR REPLACE INTO program_metrics
                        (program_id, public_metrics, private_metrics)
                    VALUES (?, ?, ?)
                    """,
                    (
                        program_id,
                        public_metrics_raw or json.dumps({}),
                        private_metrics_raw or json.dumps({}),
                    ),
                )

            # Metadata and migration history
            metadata_raw = row["metadata"] if "metadata" in row_keys else None
            migration_raw = (
                row["migration_history"]
                if "migration_history" in row_keys
                else None
            )
            if metadata_raw is not None or migration_raw is not None:
                self.cursor.execute(
                    """
                    INSERT OR REPLACE INTO program_metadata
                        (program_id, metadata, migration_history)
                    VALUES (?, ?, ?)
                    """,
                    (
                        program_id,
                        metadata_raw or json.dumps({}),
                        migration_raw or json.dumps([]),
                    ),
                )

            # Embeddings
            embedding_raw = row["embedding"] if "embedding" in row_keys else None
            embedding_2d_raw = (
                row["embedding_pca_2d"] if "embedding_pca_2d" in row_keys else None
            )
            embedding_3d_raw = (
                row["embedding_pca_3d"] if "embedding_pca_3d" in row_keys else None
            )
            if (
                embedding_raw is not None
                or embedding_2d_raw is not None
                or embedding_3d_raw is not None
            ):
                self.cursor.execute(
                    """
                    INSERT OR REPLACE INTO program_embeddings
                        (program_id, embedding, embedding_pca_2d, embedding_pca_3d)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        program_id,
                        embedding_raw or json.dumps([]),
                        embedding_2d_raw or json.dumps([]),
                        embedding_3d_raw or json.dumps([]),
                    ),
                )

            # Inspirations
            archive_ids = _safe_load(
                row["archive_inspiration_ids"]
                if "archive_inspiration_ids" in row_keys
                else None,
                [],
            )
            top_k_ids = _safe_load(
                row["top_k_inspiration_ids"]
                if "top_k_inspiration_ids" in row_keys
                else None,
                [],
            )

            if archive_ids or top_k_ids:
                self.cursor.execute(
                    "DELETE FROM program_inspirations WHERE program_id = ?",
                    (program_id,),
                )
                for insp_id in archive_ids:
                    self.cursor.execute(
                        """
                        INSERT OR IGNORE INTO program_inspirations
                            (program_id, inspiration_id, source)
                        VALUES (?, ?, 'archive')
                        """,
                        (program_id, insp_id),
                    )
                for insp_id in top_k_ids:
                    self.cursor.execute(
                        """
                        INSERT OR IGNORE INTO program_inspirations
                            (program_id, inspiration_id, source)
                        VALUES (?, ?, 'top_k')
                        """,
                        (program_id, insp_id),
                    )

        self.conn.commit()

    def _persist_program_components(self, program: Program) -> None:
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        public_metrics_json = json.dumps(program.public_metrics or {})
        private_metrics_json = json.dumps(program.private_metrics or {})
        metadata_json = json.dumps(program.metadata or {})
        migration_history_json = json.dumps(program.migration_history or [])
        embedding_json = json.dumps(program.embedding or [])
        embedding_pca_2d_json = json.dumps(program.embedding_pca_2d or [])
        embedding_pca_3d_json = json.dumps(program.embedding_pca_3d or [])

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO program_metrics
                (program_id, public_metrics, private_metrics)
            VALUES (?, ?, ?)
            """,
            (program.id, public_metrics_json, private_metrics_json),
        )

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO program_metadata
                (program_id, metadata, migration_history)
            VALUES (?, ?, ?)
            """,
            (program.id, metadata_json, migration_history_json),
        )

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO program_embeddings
                (program_id, embedding, embedding_pca_2d, embedding_pca_3d)
            VALUES (?, ?, ?, ?)
            """,
            (program.id, embedding_json, embedding_pca_2d_json, embedding_pca_3d_json),
        )

        self.cursor.execute(
            "DELETE FROM program_inspirations WHERE program_id = ?", (program.id,)
        )
        for inspiration_id in program.archive_inspiration_ids or []:
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO program_inspirations
                    (program_id, inspiration_id, source)
                VALUES (?, ?, 'archive')
                """,
                (program.id, inspiration_id),
            )
        for inspiration_id in program.top_k_inspiration_ids or []:
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO program_inspirations
                    (program_id, inspiration_id, source)
                VALUES (?, ?, 'top_k')
                """,
                (program.id, inspiration_id),
            )

    def _insert_base_program_record(self, program: Program) -> None:
        """Insert the core program row and associated component data."""
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        self.cursor.execute(
            """
            INSERT INTO programs
               (id, code, language, parent_id, generation, timestamp,
                code_diff, combined_score, text_feedback, complexity,
                embedding_cluster_id, correct, children_count, island_idx)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                program.id,
                program.code,
                program.language,
                program.parent_id,
                program.generation,
                program.timestamp,
                program.code_diff,
                program.combined_score,
                program.text_feedback,
                program.complexity,
                program.embedding_cluster_id,
                program.correct,
                program.children_count,
                program.island_idx,
            ),
        )

        self._persist_program_components(program)

    def _clone_program_for_island(
        self,
        source_program: Program,
        new_program_id: str,
        island_idx: int,
        metadata_override: Dict[str, Any],
    ) -> None:
        """Create a cloned program record for another island."""
        clone_metadata = metadata_override.copy()

        if isinstance(source_program.text_feedback, list):
            text_feedback = "\n".join(str(item) for item in source_program.text_feedback)
        elif source_program.text_feedback is None:
            text_feedback = ""
        else:
            text_feedback = str(source_program.text_feedback)

        clone_program = Program(
            id=new_program_id,
            code=source_program.code,
            language=source_program.language,
            parent_id=source_program.parent_id,
            archive_inspiration_ids=list(source_program.archive_inspiration_ids or []),
            top_k_inspiration_ids=list(source_program.top_k_inspiration_ids or []),
            generation=source_program.generation,
            timestamp=source_program.timestamp,
            code_diff=source_program.code_diff,
            combined_score=source_program.combined_score,
            public_metrics=source_program.public_metrics.copy(),
            private_metrics=source_program.private_metrics.copy(),
            text_feedback=text_feedback,
            complexity=source_program.complexity,
            embedding=list(source_program.embedding or []),
            embedding_pca_2d=list(source_program.embedding_pca_2d or []),
            embedding_pca_3d=list(source_program.embedding_pca_3d or []),
            embedding_cluster_id=source_program.embedding_cluster_id,
            correct=source_program.correct,
            children_count=source_program.children_count,
            metadata=clone_metadata,
            migration_history=list(source_program.migration_history or []),
            island_idx=island_idx,
        )

        self._insert_base_program_record(clone_program)

    def _persist_program_components(self, program: Program) -> None:
        """Persist program components in the normalized tables."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        public_metrics_json = json.dumps(program.public_metrics or {})
        private_metrics_json = json.dumps(program.private_metrics or {})
        metadata_json = json.dumps(program.metadata or {})
        migration_history_json = json.dumps(program.migration_history or [])
        embedding_json = json.dumps(program.embedding or [])
        embedding_pca_2d_json = json.dumps(program.embedding_pca_2d or [])
        embedding_pca_3d_json = json.dumps(program.embedding_pca_3d or [])

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO program_metrics
                (program_id, public_metrics, private_metrics)
            VALUES (?, ?, ?)
            """,
            (program.id, public_metrics_json, private_metrics_json),
        )

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO program_metadata
                (program_id, metadata, migration_history)
            VALUES (?, ?, ?)
            """,
            (program.id, metadata_json, migration_history_json),
        )

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO program_embeddings
                (program_id, embedding, embedding_pca_2d, embedding_pca_3d)
            VALUES (?, ?, ?, ?)
            """,
            (program.id, embedding_json, embedding_pca_2d_json, embedding_pca_3d_json),
        )

        self.cursor.execute(
            "DELETE FROM program_inspirations WHERE program_id = ?", (program.id,)
        )
        for inspiration_id in program.archive_inspiration_ids or []:
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO program_inspirations
                    (program_id, inspiration_id, source)
                VALUES (?, ?, 'archive')
                """,
                (program.id, inspiration_id),
            )
        for inspiration_id in program.top_k_inspiration_ids or []:
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO program_inspirations
                    (program_id, inspiration_id, source)
                VALUES (?, ?, 'top_k')
                """,
                (program.id, inspiration_id),
            )

    def _load_program_components(
        self, program_id: str, cursor: Optional[sqlite3.Cursor] = None
    ) -> Dict[str, Any]:
        """Load program component data from normalized tables."""
        cursor = cursor or self.cursor
        if not cursor:
            raise ConnectionError("DB not connected.")

        result: Dict[str, Any] = {
            "public_metrics": {},
            "private_metrics": {},
            "metadata": {},
            "migration_history": [],
            "archive_inspiration_ids": [],
            "top_k_inspiration_ids": [],
            "embedding": [],
            "embedding_pca_2d": [],
            "embedding_pca_3d": [],
        }

        cursor.execute(
            "SELECT public_metrics, private_metrics "
            "FROM program_metrics WHERE program_id = ?",
            (program_id,),
        )
        row = cursor.fetchone()
        if row:
            for key in ["public_metrics", "private_metrics"]:
                value = row[key]
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to decode %s for program %s; defaulting to empty.",
                            key,
                            program_id,
                        )

        cursor.execute(
            "SELECT metadata, migration_history "
            "FROM program_metadata WHERE program_id = ?",
            (program_id,),
        )
        row = cursor.fetchone()
        if row:
            metadata_val = row["metadata"]
            migration_val = row["migration_history"]
            if metadata_val:
                try:
                    result["metadata"] = json.loads(metadata_val)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to decode metadata for program %s; using empty dict.",
                        program_id,
                    )
            if migration_val:
                try:
                    result["migration_history"] = json.loads(migration_val)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to decode migration history for program %s; using empty list.",
                        program_id,
                    )

        cursor.execute(
            "SELECT embedding, embedding_pca_2d, embedding_pca_3d "
            "FROM program_embeddings WHERE program_id = ?",
            (program_id,),
        )
        row = cursor.fetchone()
        if row:
            for key in ["embedding", "embedding_pca_2d", "embedding_pca_3d"]:
                value = row[key]
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to decode %s for program %s; defaulting to empty list.",
                            key,
                            program_id,
                        )

        cursor.execute(
            """
            SELECT inspiration_id, source
            FROM program_inspirations
            WHERE program_id = ?
            """,
            (program_id,),
        )
        rows = cursor.fetchall()
        for insp in rows:
            source = insp["source"]
            if source == "archive":
                result["archive_inspiration_ids"].append(insp["inspiration_id"])
            elif source == "top_k":
                result["top_k_inspiration_ids"].append(insp["inspiration_id"])

        return result

    @db_retry()
    def _load_metadata_from_db(self):
        if not self.cursor:
            raise ConnectionError("DB cursor not available.")

        self.cursor.execute(
            "SELECT value FROM metadata_store WHERE key = 'last_iteration'"
        )
        row = self.cursor.fetchone()
        self.last_iteration = (
            int(row["value"]) if row and row["value"] is not None else 0
        )
        if not row or row["value"] is not None:  # Initialize in DB if first time
            if not self.read_only:
                self._update_metadata_in_db("last_iteration", str(self.last_iteration))

        self.cursor.execute(
            "SELECT value FROM metadata_store WHERE key = 'best_program_id'"
        )
        row = self.cursor.fetchone()
        self.best_program_id = (
            str(row["value"])
            if row and row["value"] is not None and row["value"] != "None"
            else None
        )
        if (
            not row or row["value"] is None or row["value"] == "None"
        ):  # Initialize or clear if stored as 'None' string
            if not self.read_only:
                self._update_metadata_in_db("best_program_id", None)

        self.cursor.execute(
            "SELECT value FROM metadata_store WHERE key = 'beam_search_parent_id'"
        )
        row = self.cursor.fetchone()
        self.beam_search_parent_id = (
            str(row["value"])
            if row and row["value"] is not None and row["value"] != "None"
            else None
        )
        if not row or row["value"] is None or row["value"] == "None":
            if not self.read_only:
                self._update_metadata_in_db("beam_search_parent_id", None)

    @db_retry()
    def _update_metadata_in_db(self, key: str, value: Optional[str]):
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
            (key, value),  # SQLite handles None as NULL
        )
        self.conn.commit()

    @db_retry()
    def _count_programs_in_db(self) -> int:
        if not self.cursor:
            return 0
        self.cursor.execute("SELECT COUNT(*) FROM programs")
        return (self.cursor.fetchone() or {"COUNT(*)": 0})["COUNT(*)"]

    @db_retry()
    def add(self, program: Program, verbose: bool = False) -> str:
        """
        Add a program to the database with optimized performance.

        This method uses batched transactions and defers expensive operations
        to improve performance with large databases. After adding a program,
        you should call check_scheduled_operations() to run any deferred
        operations like migrations.

        Example:
            db.add(program)  # Fast add
            db.check_scheduled_operations()  # Run deferred operations

        Args:
            program: The Program object to add

        Returns:
            str: The ID of the added program
        """
        if self.read_only:
            raise PermissionError("Cannot add program in read-only mode.")
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        self.island_manager.assign_island(program)

        # Calculate complexity if not pre-set (or if default 0.0)
        if program.complexity == 0.0:
            try:
                code_metrics = analyze_code_metrics(program.code, program.language)
                program.complexity = code_metrics.get("complexity_score", 0.0)
                if program.metadata is None:
                    program.metadata = {}
                program.metadata["code_analysis_metrics"] = code_metrics
            except Exception as e:
                logger.warning(
                    f"Could not calculate complexity for program {program.id}: {e}"
                )
                program.complexity = float(len(program.code))  # Fallback to length

        # Embedding is expected to be provided by the user.
        # Ensure program.embedding is a list, even if empty.
        if not isinstance(program.embedding, list):
            logger.warning(
                f"Program {program.id} embedding is not a list, "
                "defaulting to empty list."
            )
            program.embedding = []

        # Handle text_feedback - normalize to string
        if isinstance(program.text_feedback, list):
            # Join list items with newlines for readability
            program.text_feedback = "\n".join(
                str(item) for item in program.text_feedback
            )
        elif program.text_feedback is None:
            program.text_feedback = ""
        else:
            program.text_feedback = str(program.text_feedback)

        # Begin transaction - this improves performance by batching operations
        self.conn.execute("BEGIN TRANSACTION")

        try:
            # Insert the program in a single operation
            self._insert_base_program_record(program)

            # Increment parent's children_count
            if program.parent_id:
                self.cursor.execute(
                    "UPDATE programs SET children_count = children_count + 1 "
                    "WHERE id = ?",
                    (program.parent_id,),
                )

            # Commit the main program insertion and related operations
            self.conn.commit()
            logger.info(
                "Program %s added to DB - score: %s.",
                program.id,
                program.combined_score,
            )

        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            logger.error(f"IntegrityError for program {program.id}: {e}")
            raise
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding program {program.id}: {e}")
            raise

        self._update_archive(program)

        # Update best program tracking
        self._update_best_program(program)

        # Recompute embeddings and clusters for all programs
        self._recompute_embeddings_and_clusters()

        # Update generation tracking
        if program.generation > self.last_iteration:
            self.last_iteration = program.generation
            self._update_metadata_in_db("last_iteration", str(self.last_iteration))

        # Print verbose summary if requested
        if verbose:
            self._print_program_summary(program)

        # Check if this program needs to be copied to other islands
        if self.island_manager.needs_island_copies(program):
            logger.info(
                f"Creating copies of initial program {program.id} for all islands"
            )
            self.island_manager.copy_program_to_islands(program)
            # Remove the flag from the original program's metadata
            if program.metadata:
                program.metadata.pop("_needs_island_copies", None)
                metadata_json = json.dumps(program.metadata)
                migration_history_json = json.dumps(program.migration_history or [])
                self.cursor.execute(
                    """
                    INSERT OR REPLACE INTO program_metadata
                        (program_id, metadata, migration_history)
                    VALUES (?, ?, ?)
                    """,
                    (program.id, metadata_json, migration_history_json),
                )
                self.conn.commit()

        # Check if migration should be scheduled
        if self.island_manager.should_schedule_migration(program):
            self._schedule_migration = True

        self.check_scheduled_operations()
        return program.id

    def _program_from_row(
        self, row: sqlite3.Row, cursor: Optional[sqlite3.Cursor] = None
    ) -> Optional[Program]:
        """Helper to create a Program object from a database row."""
        if not row:
            return None

        program_data = dict(row)  # type: ignore[arg-type]

        # Load components from normalized tables
        try:
            component_data = self._load_program_components(
                program_data["id"], cursor=cursor
            )
            program_data.update(component_data)
        except sqlite3.Error as exc:
            logger.error(
                "Failed to load component data for program %s: %s",
                program_data.get("id"),
                exc,
            )

        if "text_feedback" not in program_data or program_data["text_feedback"] is None:
            program_data["text_feedback"] = ""

        # Handle archive status
        program_data["in_archive"] = bool(program_data.get("in_archive", 0))

        return Program.from_dict(program_data)

    @db_retry()
    def get(self, program_id: str) -> Optional[Program]:
        """Get a program by its ID with optimized JSON operations."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
        row = self.cursor.fetchone()
        return self._program_from_row(row)

    @db_retry()
    def sample(
        self,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ) -> Tuple[Program, List[Program], List[Program]]:
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        # Check if all islands are initialized
        if not self.island_manager.are_all_islands_initialized():
            # Get initial program (first program in database)
            self.cursor.execute("SELECT * FROM programs ORDER BY timestamp ASC LIMIT 1")
            row = self.cursor.fetchone()
            if not row:
                raise RuntimeError("No programs found in database")

            parent = self._program_from_row(row)
            if not parent:
                raise RuntimeError("Failed to load initial program")

            logger.info(
                f"Not all islands initialized. Using initial program {parent.id} "
                "without inspirations."
            )

            # Print sampling summary
            self._print_sampling_summary_helper(
                parent,
                [],
                [],
                target_generation,
                novelty_attempt,
                max_novelty_attempts,
                resample_attempt,
                max_resample_attempts,
            )

            return parent, [], []

        # All islands initialized - sample island + constrain parents
        initialized_islands = self.island_manager.get_initialized_islands()
        sampled_island = random.choice(initialized_islands)

        logger.debug(f"Sampling from island {sampled_island}")

        # Use CombinedParentSelector with island constraint
        parent_selector = CombinedParentSelector(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
            get_program_func=self.get,
            best_program_id=self.best_program_id,
            beam_search_parent_id=self.beam_search_parent_id,
            last_iteration=self.last_iteration,
            update_metadata_func=self._update_metadata_in_db,
            get_best_program_func=self.get_best_program,
        )

        parent = parent_selector.sample_parent(island_idx=sampled_island)
        if not parent:
            raise RuntimeError(f"Failed to sample parent from island {sampled_island}")

        num_archive_insp = (
            self.config.num_archive_inspirations
            if hasattr(self.config, "num_archive_inspirations")
            else 5
        )
        num_top_k_insp = (
            self.config.num_top_k_inspirations
            if hasattr(self.config, "num_top_k_inspirations")
            else 2
        )

        # Use the combined context selector
        context_selector = CombinedContextSelector(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
            get_program_func=self.get,
            best_program_id=self.best_program_id,
            get_island_idx_func=self.island_manager.get_island_idx,
            program_from_row_func=self._program_from_row,
        )

        archive_inspirations, top_k_inspirations = context_selector.sample_context(
            parent, num_archive_insp, num_top_k_insp
        )

        logger.debug(
            f"Sampled parent {parent.id} from island {sampled_island}, "
            f"{len(archive_inspirations)} archive inspirations, "
            f"{len(top_k_inspirations)} top-k inspirations."
        )

        # Print sampling summary
        self._print_sampling_summary_helper(
            parent,
            archive_inspirations,
            top_k_inspirations,
            target_generation,
            novelty_attempt,
            max_novelty_attempts,
            resample_attempt,
            max_resample_attempts,
        )

        return parent, archive_inspirations, top_k_inspirations

    def _print_sampling_summary_helper(
        self,
        parent,
        archive_inspirations,
        top_k_inspirations,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ):
        """Helper method to print sampling summary."""
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self.get_best_program,
            )

        self._database_display.print_sampling_summary(
            parent,
            archive_inspirations,
            top_k_inspirations,
            target_generation,
            novelty_attempt,
            max_novelty_attempts,
            resample_attempt,
            max_resample_attempts,
        )

    @db_retry()
    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        # Attempt to use tracked best_program_id first if no specific metric
        if metric is None and self.best_program_id:
            program = self.get(self.best_program_id)
            if program and program.correct:  # Ensure best program is correct
                return program
            else:  # Stale ID or incorrect program
                logger.warning(
                    f"Tracked best_program_id '{self.best_program_id}' "
                    "not found or incorrect. Re-evaluating."
                )
                if not self.read_only:
                    self._update_metadata_in_db("best_program_id", None)
                self.best_program_id = None

        # Fetch only correct programs and sort in Python.
        self.cursor.execute("SELECT * FROM programs WHERE correct = 1")
        all_rows = self.cursor.fetchall()
        if not all_rows:
            logger.debug("No correct programs found in database.")
            return None

        programs = [
            prog for prog in (self._program_from_row(row) for row in all_rows) if prog
        ]

        if not programs:
            return None

        sorted_p: List[Program] = []
        log_key = "average metrics"

        if metric:
            progs_with_metric = [
                p for p in programs if p.public_metrics and metric in p.public_metrics
            ]
            sorted_p = sorted(
                progs_with_metric,
                key=lambda p_item: p_item.public_metrics.get(metric, -float("inf")),
                reverse=True,
            )
            log_key = f"metric '{metric}'"
        elif any(p.combined_score is not None for p in programs):
            progs_with_cs = [p for p in programs if p.combined_score is not None]
            sorted_p = sorted(
                progs_with_cs,
                key=lambda p_item: p_item.combined_score or -float("inf"),
                reverse=True,
            )
            log_key = "combined_score"
        else:
            progs_with_metrics = [p for p in programs if p.public_metrics]
            sorted_p = sorted(
                progs_with_metrics,
                key=lambda p_item: sum(p_item.public_metrics.values())
                / len(p_item.public_metrics)
                if p_item.public_metrics
                else -float("inf"),
                reverse=True,
            )

        if not sorted_p:
            logger.debug("No correct programs matched criteria for get_best_program.")
            return None

        best_overall = sorted_p[0]
        logger.debug(f"Best correct program by {log_key}: {best_overall.id}")

        if self.best_program_id != best_overall.id:  # Update ID if different
            logger.info(
                "Updating tracked best program from "
                f"'{self.best_program_id}' to '{best_overall.id}'."
            )
            self.best_program_id = best_overall.id
            if not self.read_only:
                self._update_metadata_in_db("best_program_id", self.best_program_id)
        return best_overall

    @db_retry()
    def get_all_programs(self) -> List[Program]:
        """Get all programs from the database."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute(
            """
            SELECT p.*,
                   CASE WHEN a.program_id IS NOT NULL THEN 1 ELSE 0 END as in_archive
            FROM programs p
            LEFT JOIN archive a ON p.id = a.program_id
            """
        )
        rows = self.cursor.fetchall()
        programs = [self._program_from_row(row) for row in rows]
        # Filter out any None values that might result from row processing errors
        return [p for p in programs if p is not None]

    @db_retry()
    def get_programs_by_generation(self, generation: int) -> List[Program]:
        """Get all programs from a specific generation."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute(
            "SELECT * FROM programs WHERE generation = ?", (generation,)
        )
        rows = self.cursor.fetchall()
        programs = [self._program_from_row(row) for row in rows]
        return [p for p in programs if p is not None]

    @db_retry()
    def get_top_programs(
        self,
        n: int = 10,
        metric: Optional[str] = "combined_score",
        correct_only: bool = False,
    ) -> List[Program]:
        """Get top programs, using SQL for sorting when possible."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        # Add correctness filter to WHERE clause if requested
        correctness_filter = "WHERE correct = 1" if correct_only else ""

        # Try to use SQL for sorting when possible for better performance
        if metric == "combined_score":
            # Use SQLite's json_extract for better performance
            base_query = """
                SELECT * FROM programs
                WHERE combined_score IS NOT NULL
            """
            if correct_only:
                base_query += " AND correct = 1"
            base_query += " ORDER BY combined_score DESC LIMIT ?"

            self.cursor.execute(base_query, (n,))
            all_rows = self.cursor.fetchall()
        elif metric == "timestamp":
            # Direct timestamp sorting
            query = (
                f"SELECT * FROM programs {correctness_filter} "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            self.cursor.execute(query, (n,))
            all_rows = self.cursor.fetchall()
        else:
            # Fall back to Python sorting for complex cases
            query = f"SELECT * FROM programs {correctness_filter}"
            self.cursor.execute(query)
            all_rows = self.cursor.fetchall()

        if not all_rows:
            return []

        # Process results
        programs = [
            prog for prog in (self._program_from_row(row) for row in all_rows) if prog
        ]

        # If we already have the sorted programs from SQL, just return them
        if metric in ["combined_score", "timestamp"] and programs:
            return programs[:n]

        # Otherwise, sort in Python
        if programs:
            if metric:
                progs_with_metric = [
                    p
                    for p in programs
                    if p.public_metrics and metric in p.public_metrics
                ]
                sorted_p = sorted(
                    progs_with_metric,
                    key=lambda p_item: p_item.public_metrics.get(metric, -float("inf")),
                    reverse=True,
                )
            else:  # Default: average metrics
                progs_with_metrics = [p for p in programs if p.public_metrics]
                sorted_p = sorted(
                    progs_with_metrics,
                    key=lambda p_item: sum(p_item.public_metrics.values())
                    / len(p_item.public_metrics)
                    if p_item.public_metrics
                    else -float("inf"),
                    reverse=True,
                )

            return sorted_p[:n]

        return []

    def save(self, path: Optional[str] = None) -> None:
        if not self.conn or not self.cursor:
            logger.warning("No DB connection, skipping save.")
            return

        # Main purpose here is to save/commit metadata like last_iteration.
        current_db_file_path_str = self.config.db_path
        if path and current_db_file_path_str:
            if Path(path).resolve() != Path(current_db_file_path_str).resolve():
                logger.warning(
                    f"Save path '{path}' differs from connected DB "
                    f"'{current_db_file_path_str}'. Metadata saved to "
                    "connected DB."
                )
        elif path and not current_db_file_path_str:
            logger.warning(
                f"Attempting to save with path '{path}' but current "
                "database is in-memory. Metadata will be committed to the "
                "in-memory instance."
            )

        self._update_metadata_in_db("last_iteration", str(self.last_iteration))

        self.conn.commit()  # Commit any pending transactions
        logger.info(
            f"Database state committed. Last iteration: "
            f"{self.last_iteration}. Best: {self.best_program_id}"
        )

    def load(self, path: str) -> None:
        logger.info(f"Loading database from '{path}'...")
        if self.conn:
            db_display_name = self.config.db_path or ":memory:"
            logger.info(f"Closing existing connection to '{db_display_name}'.")
            self.conn.close()

        db_path_obj = Path(path).resolve()
        # Robustness check for unclean shutdown with WAL
        db_wal_file = Path(f"{db_path_obj}-wal")
        db_shm_file = Path(f"{db_path_obj}-shm")
        if (
            db_path_obj.exists()
            and db_path_obj.stat().st_size == 0
            and (db_wal_file.exists() or db_shm_file.exists())
        ):
            logger.warning(
                f"Database file {db_path_obj} is empty but WAL/SHM files "
                "exist. This may indicate an unclean shutdown. Removing "
                "WAL/SHM files to attempt recovery.",
                db_path_obj,
            )
            if db_wal_file.exists():
                db_wal_file.unlink()
            if db_shm_file.exists():
                db_shm_file.unlink()

        self.config.db_path = str(db_path_obj)  # Update config

        if not db_path_obj.exists():
            logger.warning(
                f"DB file '{db_path_obj}' not found. New DB created if writes occur."
            )
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path_obj), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._load_metadata_from_db()

        count = self._count_programs_in_db()
        logger.info(
            f"Loaded DB from '{db_path_obj}'. {count} programs. "
            f"Last iter: {self.last_iteration}."
        )

    def _is_better(self, program1: Program, program2: Program) -> bool:
        # First prioritize correctness
        if program1.correct and not program2.correct:
            return True
        if program2.correct and not program1.correct:
            return False

        # If both have same correctness status, compare scores
        s1 = program1.combined_score
        s2 = program2.combined_score

        if s1 is not None and s2 is not None:
            if s1 != s2:
                return s1 > s2
        elif s1 is not None:
            return True  # p1 has score, p2 doesn't
        elif s2 is not None:
            return False  # p2 has score, p1 doesn't

        try:
            avg1 = (
                sum(program1.public_metrics.values()) / len(program1.public_metrics)
                if program1.public_metrics
                else -float("inf")
            )
            avg2 = (
                sum(program2.public_metrics.values()) / len(program2.public_metrics)
                if program2.public_metrics
                else -float("inf")
            )
            if avg1 != avg2:
                return avg1 > avg2
        except Exception:
            return False
        return program1.timestamp > program2.timestamp  # Tie-breaker

    @db_retry()
    def _update_archive(self, program: Program) -> None:
        if (
            not self.cursor
            or not self.conn
            or not hasattr(self.config, "archive_size")
            or self.config.archive_size <= 0
        ):
            logger.debug("Archive update skipped (config/DB issue or size <= 0).")
            return

        # Only add correct programs to the archive
        if not program.correct:
            logger.debug(f"Program {program.id} not added to archive (not correct).")
            return

        self.cursor.execute("SELECT COUNT(*) FROM archive")
        count = (self.cursor.fetchone() or [0])[0]

        if count < self.config.archive_size:
            self.cursor.execute(
                "INSERT OR IGNORE INTO archive (program_id) VALUES (?)",
                (program.id,),
            )
        else:  # Archive is full, find worst to replace
            self.cursor.execute(
                "SELECT a.program_id, p.combined_score, p.timestamp, p.correct "
                "FROM archive a JOIN programs p ON a.program_id = p.id"
            )
            archived_rows = self.cursor.fetchall()
            if not archived_rows:  # Should not happen if count was > 0
                self.cursor.execute(
                    "INSERT OR IGNORE INTO archive (program_id) VALUES (?)",
                    (program.id,),
                )
                self.conn.commit()
                return

            archive_programs_for_cmp = []
            for r_data in archived_rows:
                # Create minimal Program-like dict for _is_better
                combined_score_val = r_data["combined_score"]
                # This is a simplified way, _is_better needs Program objects
                # For full Program object: self.get(r_data["program_id"]) but could be slow
                archive_programs_for_cmp.append(
                    Program(
                        id=r_data["program_id"],
                        code="",
                        combined_score=combined_score_val,
                        timestamp=r_data["timestamp"],
                        correct=bool(r_data["correct"]),
                    )
                )

            if (
                not archive_programs_for_cmp
            ):  # Should be populated if archived_rows existed
                self.cursor.execute(
                    "INSERT OR IGNORE INTO archive (program_id) VALUES (?)",
                    (program.id,),
                )
                self.conn.commit()
                return

            worst_in_archive = archive_programs_for_cmp[0]
            for p_archived in archive_programs_for_cmp[1:]:
                if self._is_better(worst_in_archive, p_archived):
                    worst_in_archive = p_archived

            if self._is_better(program, worst_in_archive):
                self.cursor.execute(
                    "DELETE FROM archive WHERE program_id = ?",
                    (worst_in_archive.id,),
                )
                self.cursor.execute(
                    "INSERT INTO archive (program_id) VALUES (?)", (program.id,)
                )
                logger.info(
                    f"Program {program.id} replaced {worst_in_archive.id} in archive."
                )
        self.conn.commit()

    @db_retry()
    def _update_best_program(self, program: Program) -> None:
        # Only consider correct programs for best program tracking
        if not program.correct:
            logger.debug(f"Program {program.id} not considered for best (not correct).")
            return

        current_best_p = None
        if self.best_program_id:
            current_best_p = self.get(self.best_program_id)

        if current_best_p is None or self._is_better(program, current_best_p):
            self.best_program_id = program.id
            self._update_metadata_in_db("best_program_id", self.best_program_id)

            log_msg = f"New best program: {program.id}"
            if current_best_p:
                p1_score = program.combined_score or 0.0
                p2_score = current_best_p.combined_score or 0.0
                log_msg += (
                    f" (gen: {current_best_p.generation} → {program.generation}, "
                    f"score: {p2_score:.4f} → {p1_score:.4f}, "
                    f"island: {current_best_p.island_idx} → {program.island_idx})"
                )
            else:
                score = program.combined_score or 0.0
                log_msg += (
                    f" (gen: {program.generation}, score: {score:.4f}, initialized "
                    f"island: {program.island_idx})."
                )
            logger.info(log_msg)

    def print_summary(self, console=None) -> None:
        """Print a summary of the database contents using DatabaseDisplay."""
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self.get_best_program,
            )
            self._database_display.set_last_iteration(self.last_iteration)

        self._database_display.print_summary(console)

    def _print_program_summary(self, program) -> None:
        """Print a rich summary of a newly added program using DatabaseDisplay."""
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self.get_best_program,
            )

        self._database_display.print_program_summary(program)

    def check_scheduled_operations(self):
        """Run any operations that were scheduled during add but deferred for performance."""
        if self._schedule_migration:
            logger.info("Running scheduled migration operation")
            self.island_manager.perform_migration(self.last_iteration)
            self._schedule_migration = False

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)

        norm_a = np.linalg.norm(arr1)
        norm_b = np.linalg.norm(arr2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(arr1, arr2) / (norm_a * norm_b)
        return float(similarity)

    @db_retry()
    def compute_similarity_thread_safe(
        self, vec: List[float], island_idx: int
    ) -> List[float]:
        """
        Thread-safe version of similarity computation. Creates its own DB connection.
        """
        conn = None
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT pe.embedding
                FROM program_embeddings pe
                JOIN programs p ON p.id = pe.program_id
                WHERE p.island_idx = ?
                  AND pe.embedding IS NOT NULL
                  AND pe.embedding != '[]'
                """,
                (island_idx,),
            )
            rows = cursor.fetchall()

            if not rows:
                return []

            similarities = []
            for row in rows:
                db_embedding = json.loads(row["embedding"])
                if db_embedding:
                    sim = self._cosine_similarity(vec, db_embedding)
                    similarities.append(sim)
            return similarities

        except Exception as e:
            logger.error(f"Thread-safe similarity computation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @db_retry()
    def compute_similarity(
        self, code_embedding: List[float], island_idx: int
    ) -> List[float]:
        """
        Compute similarity scores between the given embedding and all programs
        in the specified island.

        Args:
            code_embedding: The embedding to compare against
            island_idx: The island index to constrain the search to

        Returns:
            List of similarity scores (cosine similarity between 0 and 1)
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        if not code_embedding:
            logger.warning("Empty code embedding provided to compute_similarity")
            return []

        # Get all programs in the specified island that have embeddings
        self.cursor.execute(
            """
            SELECT p.id, pe.embedding
            FROM programs p
            JOIN program_embeddings pe ON p.id = pe.program_id
            WHERE p.island_idx = ?
              AND pe.embedding IS NOT NULL
              AND pe.embedding != '[]'
            """,
            (island_idx,),
        )
        rows = self.cursor.fetchall()

        if not rows:
            logger.debug(f"No programs with embeddings found in island {island_idx}")
            return []

        # Extract embeddings and compute similarities
        similarity_scores = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding"])
                if embedding:  # Skip empty embeddings
                    similarity = self._cosine_similarity(code_embedding, embedding)
                    similarity_scores.append(similarity)
                else:
                    similarity_scores.append(0.0)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode embedding for program {row['id']}")
                similarity_scores.append(0.0)
                continue

        logger.debug(
            f"Computed {len(similarity_scores)} similarity scores for "
            f"island {island_idx}"
        )
        return similarity_scores

    @db_retry()
    def get_most_similar_program(
        self, code_embedding: List[float], island_idx: int
    ) -> Optional[Program]:
        """
        Get the most similar program to the given embedding in the specified island.

        Args:
            code_embedding: The embedding to compare against
            island_idx: The island index to constrain the search to

        Returns:
            The most similar Program object, or None if no programs found
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        if not code_embedding:
            logger.warning("Empty code embedding provided to get_most_similar_program")
            return None

        # Get all programs in the specified island that have embeddings
        self.cursor.execute(
            """
            SELECT p.id, pe.embedding
            FROM programs p
            JOIN program_embeddings pe ON p.id = pe.program_id
            WHERE p.island_idx = ?
              AND pe.embedding IS NOT NULL
              AND pe.embedding != '[]'
            """,
            (island_idx,),
        )
        rows = self.cursor.fetchall()

        if not rows:
            logger.debug(f"No programs with embeddings found in island {island_idx}")
            return None

        # Find the program with highest similarity
        max_similarity = -1.0
        most_similar_id = None

        for row in rows:
            try:
                embedding = json.loads(row["embedding"])
                if embedding:  # Skip empty embeddings
                    similarity = self._cosine_similarity(code_embedding, embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_id = row["id"]
            except json.JSONDecodeError:
                logger.warning(f"Could not decode embedding for program {row['id']}")
                continue

        if most_similar_id:
            return self.get(most_similar_id)
        return None

    @db_retry()
    def get_most_similar_program_thread_safe(
        self, code_embedding: List[float], island_idx: int
    ) -> Optional[Program]:
        """
        Thread-safe version of get_most_similar_program that creates its own DB connection.

        Args:
            code_embedding: The embedding to compare against
            island_idx: The island index to constrain the search to

        Returns:
            The most similar Program object, or None if not found
        """
        if not code_embedding:
            logger.warning(
                "Empty code embedding provided to get_most_similar_program_thread_safe"
            )
            return None

        conn = None
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all programs in the specified island that have embeddings
            cursor.execute(
                """
                SELECT p.id, pe.embedding
                FROM programs p
                JOIN program_embeddings pe ON p.id = pe.program_id
                WHERE p.island_idx = ?
                  AND pe.embedding IS NOT NULL
                  AND pe.embedding != '[]'
                """,
                (island_idx,),
            )

            rows = cursor.fetchall()
            if not rows:
                return None

            # Compute similarities
            import numpy as np

            similarities = []
            program_ids = []

            for row in rows:
                try:
                    embedding = json.loads(row["embedding"])
                    if embedding:  # Check if embedding is not empty
                        similarity = np.dot(code_embedding, embedding) / (
                            np.linalg.norm(code_embedding) * np.linalg.norm(embedding)
                        )
                        similarities.append(similarity)
                        program_ids.append(row["id"])
                except (json.JSONDecodeError, ValueError, ZeroDivisionError) as e:
                    logger.warning(
                        f"Error computing similarity for program {row['id']}: {e}"
                    )
                    continue

            if not similarities:
                return None

            # Find the most similar program
            max_similarity_idx = np.argmax(similarities)
            most_similar_id = program_ids[max_similarity_idx]

            # Get the full program data
            cursor.execute("SELECT * FROM programs WHERE id = ?", (most_similar_id,))
            row = cursor.fetchone()

            if row:
                return self._program_from_row(row, cursor=cursor)
            return None

        except Exception as e:
            logger.error(f"Error in get_most_similar_program_thread_safe: {e}")
            return None
        finally:
            if conn:
                conn.close()

    @db_retry()
    def _recompute_embeddings_and_clusters(self, num_clusters: int = 4):
        if self.read_only:
            return
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        self.cursor.execute(
            """
            SELECT p.id, pe.embedding
            FROM programs p
            JOIN program_embeddings pe ON p.id = pe.program_id
            WHERE pe.embedding IS NOT NULL
              AND pe.embedding != '[]'
            """
        )
        rows = self.cursor.fetchall()

        if len(rows) < num_clusters:
            logger.info(
                f"Not enough programs with embeddings ({len(rows)}) to "
                f"perform clustering. Need at least {num_clusters}."
            )
            return

        program_ids = [row["id"] for row in rows]
        embeddings = [json.loads(row["embedding"]) for row in rows]

        # Use EmbeddingClient for dim reduction and clustering
        try:
            logger.info(
                "Recomputing PCA-reduced embedding features for %s programs.",
                len(program_ids),
            )
            reduced_2d = self.embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=2
            )
            reduced_3d = self.embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=3
            )
            cluster_ids = self.embedding_client.get_embedding_clusters(
                embeddings, num_clusters=num_clusters
            )
        except Exception as e:
            logger.error(f"Failed to recompute embedding features: {e}")
            return

        # Update all programs in a single transaction
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for i, program_id in enumerate(program_ids):
                embedding_pca_2d_json = json.dumps(reduced_2d[i].tolist())
                embedding_pca_3d_json = json.dumps(reduced_3d[i].tolist())
                cluster_id = int(cluster_ids[i])

                self.cursor.execute(
                    """
                    UPDATE program_embeddings
                    SET embedding_pca_2d = ?,
                        embedding_pca_3d = ?
                    WHERE program_id = ?
                    """,
                    (
                        embedding_pca_2d_json,
                        embedding_pca_3d_json,
                        program_id,
                    ),
                )
                self.cursor.execute(
                    """
                    UPDATE programs
                    SET embedding_cluster_id = ?
                    WHERE id = ?
                    """,
                    (
                        cluster_id,
                        program_id,
                    ),
                )
            self.conn.commit()
            logger.info(
                "Successfully updated embedding features for %s programs.",
                len(program_ids),
            )
        except Exception as e:
            self.conn.rollback()
            logger.error("Failed to update programs with new embedding features: %s", e)

    @db_retry()
    def _recompute_embeddings_and_clusters_thread_safe(self, num_clusters: int = 4):
        """
        Thread-safe version of embedding recomputation. Creates its own DB connection.
        """
        if self.read_only:
            return

        conn = None
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT p.id, pe.embedding
                FROM programs p
                JOIN program_embeddings pe ON p.id = pe.program_id
                WHERE pe.embedding IS NOT NULL
                  AND pe.embedding != '[]'
                """
            )
            rows = cursor.fetchall()

            if len(rows) < num_clusters:
                if len(rows) > 0:
                    logger.info(
                        f"Not enough programs with embeddings ({len(rows)}) to "
                        f"perform clustering. Need at least {num_clusters}."
                    )
                return

            program_ids = [row["id"] for row in rows]
            embeddings = [json.loads(row["embedding"]) for row in rows]

            # Use EmbeddingClient for dim reduction and clustering
            try:
                logger.info(
                    "Recomputing PCA-reduced embedding features for %s programs.",
                    len(program_ids),
                )

                logger.info("Computing 2D PCA reduction...")
                reduced_2d = self.embedding_client.get_dim_reduction(
                    embeddings, method="pca", dims=2
                )
                logger.info("2D PCA reduction completed")

                logger.info("Computing 3D PCA reduction...")
                reduced_3d = self.embedding_client.get_dim_reduction(
                    embeddings, method="pca", dims=3
                )
                logger.info("3D PCA reduction completed")

                logger.info(f"Computing GMM clustering with {num_clusters} clusters...")
                cluster_ids = self.embedding_client.get_embedding_clusters(
                    embeddings, num_clusters=num_clusters
                )
                logger.info("GMM clustering completed")
            except Exception as e:
                logger.error(f"Failed to recompute embedding features: {e}")
                return

            # Update all programs in a single transaction
            conn.execute("BEGIN TRANSACTION")
            try:
                for i, program_id in enumerate(program_ids):
                    embedding_pca_2d_json = json.dumps(reduced_2d[i].tolist())
                    embedding_pca_3d_json = json.dumps(reduced_3d[i].tolist())
                    cluster_id = int(cluster_ids[i])

                    cursor.execute(
                        """
                        UPDATE program_embeddings
                        SET embedding_pca_2d = ?,
                            embedding_pca_3d = ?
                        WHERE program_id = ?
                        """,
                        (
                            embedding_pca_2d_json,
                            embedding_pca_3d_json,
                            program_id,
                        ),
                    )
                    cursor.execute(
                        """
                        UPDATE programs
                        SET embedding_cluster_id = ?
                        WHERE id = ?
                        """,
                        (
                            cluster_id,
                            program_id,
                        ),
                    )
                conn.commit()
                logger.info(
                    "Successfully updated embedding features for %s programs.",
                    len(program_ids),
                )
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Failed to update programs with new embedding features: %s", e
                )
                raise  # Re-raise exception

        except Exception as e:
            logger.error(f"Thread-safe embedding recomputation failed: {e}")
            raise  # Re-raise exception

        finally:
            if conn:
                conn.close()

    @db_retry()
    def get_programs_by_generation_thread_safe(self, generation: int) -> List[Program]:
        """Thread-safe version of get_programs_by_generation."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM programs WHERE generation = ?", (generation,))
            rows = cursor.fetchall()

            programs = [
                prog
                for prog in (
                    self._program_from_row(row, cursor=cursor) for row in rows
                )
                if prog
            ]
            return programs
        finally:
            if conn:
                conn.close()

    @db_retry()
    def get_top_programs_thread_safe(
        self,
        n: int = 10,
        correct_only: bool = True,
    ) -> List[Program]:
        """Thread-safe version of get_top_programs."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Use combined_score for sorting
            base_query = """
                SELECT * FROM programs
                WHERE combined_score IS NOT NULL
            """
            if correct_only:
                base_query += " AND correct = 1"
            base_query += " ORDER BY combined_score DESC LIMIT ?"

            cursor.execute(base_query, (n,))
            all_rows = cursor.fetchall()

            if not all_rows:
                return []

            # Process results
            programs = [
                prog
                for prog in (
                    self._program_from_row(row, cursor=cursor) for row in all_rows
                )
                if prog
            ]

            return programs

        finally:
            if conn:
                conn.close()

    def _get_programs_for_island(self, island_idx: int) -> List[Program]:
        """
        Get all programs for a specific island.
        """
