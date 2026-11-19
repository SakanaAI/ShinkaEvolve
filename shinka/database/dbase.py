import json
import logging
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from pathlib import Path
import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import math
from .attempt_log import insert_attempt_event
from .complexity import analyze_code_metrics
from .parents import CombinedParentSelector
from .inspirations import CombinedContextSelector
from .islands import CombinedIslandManager
from .island_sampler import create_island_sampler, IslandSampler
from .display import DatabaseDisplay
from shinka.embed import EmbeddingClient
from shinka.defaults import default_archive_criteria
from .write_transaction import write_transaction

logger = logging.getLogger(__name__)
_PREVIEW_RUN_STATE_REPAIR_KEYS = ("run_state_reconciled_v1", "run_state_reconciled_v2")
_RUN_STATE_REPAIR_KEY = "run_state_reconciled_v3"
_MAINTENANCE_APPLIED_KEY = "postprocess_db_maintenance_applied"
_COMPUTE_TIME_MIGRATION_KEY = "compute_time_migrated_v1"


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
    db_path: Optional[str] = None  # Path to SQLite database file
    num_islands: int = 2
    archive_size: int = 40
    # Max chars of stdout_log persisted to program metadata (tail kept).
    # None = no truncation (full stdout stored). Full log always stays on disk.
    max_stdout_log_chars: Optional[int] = None

    # Inspiration parameters
    elite_selection_ratio: float = 0.3  # Prop of elites inspirations
    num_archive_inspirations: int = 1  # No. inspiration programs
    num_top_k_inspirations: int = 1  # No. top-k inspiration programs

    # Island model/migration parameters
    migration_interval: int = 10  # Migrate every N generations
    migration_rate: float = 0.0  # Prop. of island pop. to migrate
    island_elitism: bool = True  # Keep best prog on their islands
    enforce_island_separation: bool = (
        True  # Enforce full island separation for inspirations
    )
    island_selection_strategy: str = (
        "uniform"  # Island sampling strategy: "uniform"/"equal"/"proportional"/"weighted"
    )

    # Dynamic island spawning parameters (stagnation-based)
    enable_dynamic_islands: bool = False  # Enable stagnation-based island spawning
    stagnation_threshold: int = 100  # Gens without improvement to trigger spawn
    island_spawn_strategy: str = (
        "initial"  # How to seed new islands: "initial", "best", "archive_random"
    )
    island_spawn_subtree_size: int = 1  # Max programs to copy (1=single, >1=subtree)

    # Parent selection parameters
    parent_selection_strategy: str = (
        "weighted"  # "weighted"/"power_law" / "beam_search"
    )

    # Power-law parent selection parameters
    exploitation_alpha: float = 1.0  # 0=uniform, 1=power-law
    exploitation_ratio: float = 0.2  # Chance to pick from archive

    # Weighted tree parent selection parameters
    parent_selection_lambda: float = 10.0  # >0 sharpness of sigmoid

    # Beam search parent selection parameters
    num_beams: int = 5

    tournament_top_k: int = 8
    tournament_num_draws: int = 3

    # Archive selection parameters
    archive_selection_strategy: str = "fitness"  # "fitness" or "crowding"
    # Criteria weights for archive selection (sign indicates direction):
    #   Positive weight = higher is better (e.g., combined_score)
    #   Negative weight = lower is better (e.g., loc, complexity)
    # Weights represent relative importance after rank normalization
    archive_criteria: Dict[str, float] = field(default_factory=default_archive_criteria)

    def __post_init__(self):
        if self.parent_selection_strategy != "tournament":
            return
        for name in ("tournament_top_k", "tournament_num_draws"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tournament_num_draws > self.tournament_top_k:
            raise ValueError("tournament_num_draws must not exceed tournament_top_k")


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

    # Meta-prompt evolution: track which system prompt generated this program
    system_prompt_id: Optional[str] = None

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


@dataclass
class _RunState:
    last_iteration: int
    best_program_id: Optional[str]
    best_score_generation: int
    best_score_ever: Optional[float]


@dataclass
class _PendingPreviewRecord:
    best: Optional[Program] = None
    score: Optional[float] = None
    watermark_matches_pending: bool = False


class ProgramDatabase:
    """
    SQLite-backed database for storing and managing programs during an
    evolutionary process.
    Supports MAP-Elites style feature-based organization, island
    populations, and an archive of elites.
    """

    def __init__(
        self,
        config: DatabaseConfig,
        embedding_model: str = "text-embedding-3-small",
        read_only: bool = False,
    ):
        self.config = config
        self.embedding_model = embedding_model
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.read_only = read_only
        self.display_console: Optional[Any] = None

        # Lazy-init embedding client to avoid requiring API credentials for
        # database-only operations and tests that do not compute embeddings.
        self.embedding_client: Optional[EmbeddingClient] = None
        self._embedding_client_init_failed = False

        self.last_iteration: int = 0
        self.best_program_id: Optional[str] = None
        self._run_state_lock = threading.Lock()
        self.beam_search_parent_id: Optional[str] = None
        # For deferring expensive operations
        self._schedule_migration: bool = False

        # Stagnation tracking for dynamic island spawning
        self.best_score_generation: int = 0  # Generation when best score was found
        self.best_score_ever: Optional[float] = None  # Track best score for comparison

        # Initialize island manager (will be set after db connection)
        self.island_manager: Optional[CombinedIslandManager] = None
        # Initialize island sampler (will be set after db connection)
        self.island_sampler: Optional[IslandSampler] = None

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
                self.conn = sqlite3.connect(str(db_file), timeout=60.0)
                logger.debug(f"Connected to SQLite database: {db_file}")
            else:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                db_uri = f"{db_file.as_uri()}?mode=ro"
                self.conn = sqlite3.connect(db_uri, uri=True, timeout=60.0)
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

        # Initialize island sampler with configured strategy
        island_selection_strategy = getattr(
            self.config, "island_selection_strategy", "uniform"
        )
        self.island_sampler = create_island_sampler(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
            strategy=island_selection_strategy,
        )

        count = self._count_programs_in_db()
        logger.debug(f"DB initialized with {count} programs.")
        logger.debug(
            f"Last iter: {self.last_iteration}. Best ID: {self.best_program_id}"
        )

    def _ensure_embedding_client(self) -> Optional[EmbeddingClient]:
        """Create embedding client on demand.

        Returns:
            EmbeddingClient if available, otherwise None.
        """
        if self.read_only or not self.embedding_model:
            return None
        if self.embedding_client is not None:
            return self.embedding_client
        if self._embedding_client_init_failed:
            return None

        try:
            self.embedding_client = EmbeddingClient(model_name=self.embedding_model)
        except Exception as e:
            self._embedding_client_init_failed = True
            logger.warning(
                "Embedding client init failed for model '%s'; "
                "continuing without embedding recomputation: %s",
                self.embedding_model,
                e,
            )
            return None

        return self.embedding_client

    def _create_tables(self):
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        # Set SQLite pragmas for better performance and stability
        # Use WAL mode for better concurrency support and reduced locking
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA busy_timeout = 60000;")  # 60 second busy timeout
        self.cursor.execute(
            "PRAGMA wal_autocheckpoint = 1000;"
        )  # Checkpoint every 1000 pages
        self.cursor.execute("PRAGMA synchronous = NORMAL;")  # Safer, faster
        self.cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys = ON;")  # For data integrity

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                parent_id TEXT,
                archive_inspiration_ids TEXT,  -- JSON serialized List[str]
                top_k_inspiration_ids TEXT,    -- JSON serialized List[str]
                generation INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                code_diff TEXT,     -- Stores edit difference
                combined_score REAL,
                public_metrics TEXT, -- JSON serialized Dict[str, Any]
                private_metrics TEXT, -- JSON serialized Dict[str, Any]
                text_feedback TEXT, -- Text feedback for the program
                complexity REAL,   -- Calculated complexity metric
                embedding TEXT,    -- JSON serialized List[float]
                embedding_pca_2d TEXT, -- JSON serialized List[float]
                embedding_pca_3d TEXT, -- JSON serialized List[float]
                embedding_cluster_id INTEGER,
                correct BOOLEAN DEFAULT 0,  -- Correct (0=False, 1=True)
                children_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,      -- JSON serialized Dict[str, Any]
                migration_history TEXT, -- JSON of migration events
                island_idx INTEGER,  -- Add island_idx to the schema
                system_prompt_id TEXT  -- ID of system prompt that generated this program
            )
            """)

        # Add indices for common query patterns
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

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS archive (
                program_id TEXT PRIMARY KEY,
                FOREIGN KEY (program_id) REFERENCES programs(id)
                    ON DELETE CASCADE
            )
            """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_store (
                key TEXT PRIMARY KEY, value TEXT
            )
            """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                source_job_id TEXT,
                details TEXT,
                created_at REAL NOT NULL
            )
            """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_generation_event_log_generation
            ON generation_event_log(generation)
            """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attempt_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT,
                created_at REAL NOT NULL
            )
            """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempt_log_generation
            ON attempt_log(generation)
            """)

        self.conn.commit()

        # Run any necessary migrations
        self._run_migrations()
        # Migration 2 adds this column to databases created before it existed,
        # so its index can only be created once the migrations have run.
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_programs_system_prompt_id ON "
            "programs(system_prompt_id)"
        )
        self.conn.commit()
        self._repair_legacy_run_state()

        logger.debug("Database tables and indices ensured to exist.")

    def _run_migrations(self):
        """Run database migrations for schema changes."""
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        # Get current columns
        self.cursor.execute("PRAGMA table_info(programs)")
        columns = [row[1] for row in self.cursor.fetchall()]

        # Migration 1: Add text_feedback column if it doesn't exist
        try:
            if "text_feedback" not in columns:
                logger.info("Adding text_feedback column to programs table")
                self.cursor.execute(
                    "ALTER TABLE programs ADD COLUMN text_feedback TEXT DEFAULT ''"
                )
                self.conn.commit()
                logger.info("Successfully added text_feedback column")
        except sqlite3.Error as e:
            logger.error(f"Error during text_feedback migration: {e}")
            # Don't raise - this is not critical for existing functionality

        # Migration 2: Add system_prompt_id column if it doesn't exist
        try:
            if "system_prompt_id" not in columns:
                logger.info("Adding system_prompt_id column to programs table")
                self.cursor.execute(
                    "ALTER TABLE programs ADD COLUMN system_prompt_id TEXT"
                )
                self.conn.commit()
                logger.info("Successfully added system_prompt_id column")
        except sqlite3.Error as e:
            logger.error(f"Error during system_prompt_id migration: {e}")

        # Migration 3: Restore legacy compute_time semantics when detailed
        # pipeline timing is present. compute_time should mirror evaluation
        # runtime, while pipeline_seconds stores end-to-end wall time.
        # Rows written since already satisfy this, so the full-table rewrite
        # runs once per database instead of taking the write lock on every open.
        try:
            if not self._has_run_state_repair(_COMPUTE_TIME_MIGRATION_KEY):
                self.cursor.execute("""
                    UPDATE programs
                    SET metadata = json_set(
                        metadata,
                        '$.compute_time',
                        json_extract(metadata, '$.evaluation_seconds')
                    )
                    WHERE json_valid(metadata)
                      AND json_type(metadata, '$.evaluation_seconds')
                          IN ('real', 'integer')
                      AND (
                          json_type(metadata, '$.compute_time') IS NULL
                          OR ABS(
                              COALESCE(json_extract(metadata, '$.compute_time'), 0.0) -
                              COALESCE(
                                  json_extract(metadata, '$.evaluation_seconds'), 0.0
                              )
                          ) > 1e-9
                      )
                    """)
                self.cursor.execute(
                    "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, '1')",
                    (_COMPUTE_TIME_MIGRATION_KEY,),
                )
                self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error during compute_time timing migration: {e}")

        # Migration 4: Ensure attempt_log exists for proposal-failure accounting.
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS attempt_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    created_at REAL NOT NULL
                )
                """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_attempt_log_generation
                ON attempt_log(generation)
                """)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error during attempt_log migration: {e}")

    @db_retry()
    def _load_metadata_from_db(self):
        self._refresh_run_state()
        self.cursor.execute(
            "SELECT value FROM metadata_store WHERE key = 'beam_search_parent_id'"
        )
        row = self.cursor.fetchone()
        value = row["value"] if row else None
        self.beam_search_parent_id = value if value != "None" else None

    def _read_run_state(self, cursor: Optional[sqlite3.Cursor] = None) -> _RunState:
        cursor = cursor or self.cursor
        if not cursor:
            raise ConnectionError("DB cursor not available.")

        values = dict(
            cursor.execute("""
                SELECT key, value FROM metadata_store
                WHERE key IN ('last_iteration', 'best_program_id',
                              'best_score_generation', 'best_score_ever')
                """).fetchall()
        )
        best_id = values.get("best_program_id")
        best_score = values.get("best_score_ever")
        return _RunState(
            last_iteration=int(values.get("last_iteration") or 0),
            best_program_id=best_id if best_id != "None" else None,
            best_score_generation=int(values.get("best_score_generation") or 0),
            best_score_ever=float(best_score) if best_score is not None else None,
        )

    def _cache_run_state(self, state: _RunState) -> None:
        self.last_iteration = state.last_iteration
        self.best_program_id = state.best_program_id
        self.best_score_generation = state.best_score_generation
        self.best_score_ever = state.best_score_ever

    def _refresh_run_state(self, reader: Optional["ProgramDatabase"] = None) -> None:
        with self._run_state_lock:
            self._cache_run_state((reader or self)._read_run_state())

    @contextmanager
    def _run_state_transaction(self) -> Iterator[_RunState]:
        if not self.conn:
            raise ConnectionError("DB not connected.")
        # Publish the committed snapshot before another thread refreshes this cache.
        with self._run_state_lock:
            self.conn.execute("BEGIN IMMEDIATE")
            with self.conn:
                state = self._read_run_state()
                yield state
            self._cache_run_state(state)

    def _advance_last_iteration(self, state: _RunState, generation: int) -> None:
        """Advance the latest iteration inside the caller's write transaction."""
        if generation > state.last_iteration:
            self._write_metadata("last_iteration", str(generation))
            state.last_iteration = generation

    def _advance_best_score(self, state: _RunState, program: Program) -> None:
        score = program.combined_score or 0.0
        if state.best_score_ever is None or score > state.best_score_ever:
            state.best_score_ever = score
            # An improvement can arrive after a later generation was processed.
            self._advance_best_score_generation(state, program.generation)
            self._write_metadata("best_score_ever", str(score))

    def _advance_best_score_generation(self, state: _RunState, generation: int) -> None:
        state.best_score_generation = max(state.best_score_generation, generation)
        self._write_metadata("best_score_generation", str(state.best_score_generation))

    def _has_run_state_repair(self, key: str) -> bool:
        row = self.cursor.execute(
            "SELECT 1 FROM metadata_store WHERE key = ?", (key,)
        ).fetchone()
        return row is not None

    @db_retry()
    def _repair_legacy_run_state(self) -> None:
        if self._has_run_state_repair(_RUN_STATE_REPAIR_KEY):
            return
        with self._run_state_transaction() as state:
            # A different connection may have repaired the database while we waited.
            if self._has_run_state_repair(_RUN_STATE_REPAIR_KEY):
                return
            preview_repaired = any(
                self._has_run_state_repair(key)
                for key in _PREVIEW_RUN_STATE_REPAIR_KEYS
            )
            pending = (
                self._pending_preview_record(state)
                if preview_repaired
                else _PendingPreviewRecord()
            )
            trusted = (
                replace(state, best_score_ever=None)
                if pending.score is not None
                else state
            )
            programs = self._run_state_repair_candidates(
                trusted if preview_repaired else None
            )
            original, has_spawn_history = self._scan_original_run_programs(
                trusted, programs
            )
            if has_spawn_history:
                # Older copies record the source generation, so they cannot rule
                # out a real spawn reset at the same generation as the pending row.
                pending.watermark_matches_pending = False
            self._repair_legacy_best_score(state, original, pending)
            self._advance_last_iteration(state, original.last_iteration)
            if (
                not preview_repaired or pending.best is not None
            ) and original.best_program_id != state.best_program_id:
                self._write_metadata("best_program_id", original.best_program_id)
                state.best_program_id = original.best_program_id
            self._write_metadata(_RUN_STATE_REPAIR_KEY, "1")

    def _pending_preview_record(self, state: _RunState) -> _PendingPreviewRecord:
        pending = _PendingPreviewRecord()
        best = self._get_tracked_best(state)
        if best is not None and self._is_original_program(best):
            if self._has_pending_run_maintenance(best):
                pending.best = best
            elif (best.combined_score or 0.0) == state.best_score_ever:
                # A supported score can coincide with a pending tie at a reset.
                return pending
        if state.best_score_ever is None:
            return pending

        generation = state.best_score_generation
        # A tied or unscored row could set the score watermark while a different
        # candidate won the program ranking.
        source = self._pending_preview_score_source(state, pending.best)
        if source is not None:
            pending.watermark_matches_pending = source == generation
            pending.score = state.best_score_ever
        elif (
            pending.best is not None
            and generation >= pending.best.generation
            and (pending.best.combined_score or 0.0) == state.best_score_ever
        ):
            pending.score = state.best_score_ever
        return pending

    def _pending_preview_score_source(
        self, state: _RunState, best: Optional[Program]
    ) -> Optional[int]:
        watermark = state.best_score_generation
        earliest = watermark
        # A later reset can hide the generation of an unscored pending row
        # when the preview ranked a different, negative-scored pending row.
        if (
            state.best_score_ever == 0.0
            and best is not None
            and best.combined_score is not None
            and best.combined_score < 0.0
        ):
            earliest = 0
        for row in self.cursor.execute(
            """
            SELECT id, '' AS code, generation, combined_score, correct, metadata
            FROM programs
            WHERE generation BETWEEN ? AND ? AND correct = 1
              AND COALESCE(combined_score, 0.0) = ?
              AND (generation = ? OR combined_score IS NULL)
            ORDER BY generation DESC
            """,
            (earliest, watermark, state.best_score_ever, watermark),
        ):
            program = self._program_from_row(row)
            if (
                program is not None
                and self._is_original_program(program)
                and self._has_pending_run_maintenance(program)
            ):
                return program.generation
        return None

    def _repair_legacy_best_score(
        self, state: _RunState, original: _RunState, pending: _PendingPreviewRecord
    ) -> None:
        score = original.best_score_ever
        published_score = state.best_score_ever
        if score is None or (published_score is not None and score < published_score):
            if pending.score is None:
                return
            if pending.watermark_matches_pending:
                state.best_score_generation = original.best_score_generation
        # Ties do not reset stagnation. Without maintenance order, only advance
        # to the earliest supporting generation and preserve any later reset.
        self._advance_best_score_generation(state, original.best_score_generation)
        if score != published_score:
            state.best_score_ever = score
            self._write_metadata(
                "best_score_ever", str(score) if score is not None else None
            )

    def _run_state_repair_candidates(
        self, preview: Optional[_RunState]
    ) -> Iterator[tuple[Program, bool]]:
        for row in self.cursor.execute("""
            SELECT p.id, '' AS code, p.generation, p.timestamp, p.combined_score,
                   p.public_metrics, p.correct, p.metadata,
                   a.program_id IS NOT NULL AS archived
            FROM programs AS p LEFT JOIN archive AS a ON a.program_id = p.id
            ORDER BY p.rowid
            """):
            program = self._program_from_row(row)
            if program is None:
                continue
            eligible = not self._has_pending_run_maintenance(program) and (
                preview is None
                or row["archived"]
                or program.metadata.get(_MAINTENANCE_APPLIED_KEY) is True
                or program.metadata.get("postprocess_side_effects_applied") is True
                or (
                    (program.combined_score or 0.0) == preview.best_score_ever
                    and (
                        program.id == preview.best_program_id
                        or program.generation <= preview.best_score_generation
                    )
                )
            )
            yield program, eligible

    @staticmethod
    def _has_pending_run_maintenance(program: Program) -> bool:
        metadata = program.metadata
        if metadata.get("postprocess_side_effects_applied") is True:
            return False
        if _MAINTENANCE_APPLIED_KEY in metadata:
            return metadata[_MAINTENANCE_APPLIED_KEY] is False
        # Older deferred runner writes have timing and scheduler provenance.
        job_id = metadata.get("source_job_id")
        return (
            program.generation > 0
            and metadata.get("safe_processing") is True
            and metadata.get("source_generation") == program.generation
            and isinstance(job_id, str)
            and bool(job_id)
            and type(metadata.get("pipeline_accounted_seconds")) in (int, float)
        )

    @staticmethod
    def _is_original_program(program: Program) -> bool:
        return not (
            program.metadata.get("_is_island_copy")
            or program.metadata.get("_spawned_island")
        )

    def _scan_original_run_programs(
        self, published: _RunState, programs: Iterator[tuple[Program, bool]]
    ) -> tuple[_RunState, bool]:
        state = _RunState(
            last_iteration=0,
            best_program_id=None,
            best_score_generation=0,
            best_score_ever=None,
        )
        best = None
        earliest_unscored = None
        unscored_seed_finished_at = None
        earliest_scored_started_at = math.inf
        has_spawn_history = False
        for program, eligible in programs:
            if not self._is_original_program(program):
                if program.metadata.get("_spawned_island"):
                    has_spawn_history = True
                continue
            state.last_iteration = max(state.last_iteration, program.generation)
            if not eligible or not program.correct:
                continue
            if program.combined_score is None:
                generation = program.generation
                earliest_unscored = (
                    generation
                    if earliest_unscored is None
                    else min(generation, earliest_unscored)
                )
                if published.best_score_ever == 0.0 and (
                    best is None or generation <= published.best_score_generation
                ):
                    self._record_repair_score(state, 0.0, generation)
                if generation == 0 and best is None:
                    unscored_seed_finished_at = self._maintenance_time(
                        program, "finished"
                    )
            else:
                started_at = self._maintenance_time(program, "started")
                earliest_scored_started_at = min(
                    earliest_scored_started_at,
                    started_at if started_at is not None else -math.inf,
                )
                self._record_repair_score(
                    state, program.combined_score, program.generation
                )
            if best is None or self._is_better(program, best):
                best = program
            elif program.id == published.best_program_id and not self._is_better(
                best, program
            ):
                # Keep the incumbent if the original ranking cannot break the tie.
                best = program
        # Insertion order can differ from maintenance order, and replay rewrites
        # maintenance timestamps. A completed original seed that precedes all
        # known scored maintenance can also explain a zero record lost in preview.
        if (
            published.best_score_ever is None
            and unscored_seed_finished_at is not None
            and unscored_seed_finished_at <= earliest_scored_started_at
        ):
            self._record_repair_score(state, 0.0, 0)
        elif earliest_unscored is not None and (
            state.best_score_ever is None
            or (published.best_score_ever == 0.0 and state.best_score_ever < 0.0)
        ):
            self._record_repair_score(state, 0.0, earliest_unscored)
        state.best_program_id = best.id if best else None
        return state, has_spawn_history

    @staticmethod
    def _maintenance_time(program: Program, stage: str) -> Optional[float]:
        value = program.metadata.get(f"postprocess_db_maintenance_{stage}_at")
        if (
            program.metadata.get(_MAINTENANCE_APPLIED_KEY) is True
            and type(value) in (int, float)
            and math.isfinite(value)
        ):
            return float(value)
        return None

    @staticmethod
    def _record_repair_score(state: _RunState, score: float, generation: int) -> None:
        if state.best_score_ever is None or score > state.best_score_ever:
            state.best_score_ever = score
            state.best_score_generation = generation
        elif score == state.best_score_ever:
            state.best_score_generation = min(state.best_score_generation, generation)

    @db_retry()
    def _update_metadata_in_db(self, key: str, value: Optional[str]):
        if not self.conn:
            raise ConnectionError("DB not connected.")
        with self.conn:
            self._write_metadata(key, value)

    def _write_metadata(self, key: str, value: Optional[str]) -> None:
        if not self.cursor:
            raise ConnectionError("DB cursor not available.")
        self.cursor.execute(
            "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
            (key, value),
        )

    @db_retry()
    def record_generation_event(
        self,
        generation: int,
        status: str,
        source_job_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        payload = None
        if details is not None:
            payload = json.dumps(clean_nan_values(details), sort_keys=True)

        self.cursor.execute(
            """
            INSERT INTO generation_event_log (
                generation, status, source_job_id, details, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (generation, status, source_job_id, payload, time.time()),
        )
        self.conn.commit()

    @db_retry()
    def record_attempt_event(
        self,
        generation: int,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        insert_attempt_event(
            self.conn, generation, stage, status, clean_nan_values(details)
        )

    @db_retry()
    def _count_programs_in_db(self) -> int:
        if not self.cursor:
            return 0
        self.cursor.execute("SELECT COUNT(*) FROM programs")
        return (self.cursor.fetchone() or {"COUNT(*)": 0})["COUNT(*)"]

    @db_retry()
    def has_program_with_source_job_id(self, source_job_id: str) -> bool:
        """Return True if a program row already exists for the given job id."""
        if not self.cursor:
            return False
        self.cursor.execute(
            """
            SELECT 1
            FROM programs
            WHERE json_valid(metadata)
              AND json_extract(metadata, '$.source_job_id') = ?
            LIMIT 1
            """,
            (source_job_id,),
        )
        return self.cursor.fetchone() is not None

    @db_retry()
    def get_program_by_source_job_id(self, source_job_id: str) -> Optional[Program]:
        """Return the persisted program row for a completed scheduler job."""
        if not self.cursor:
            return None
        self.cursor.execute(
            """
            SELECT *
            FROM programs
            WHERE json_valid(metadata)
              AND json_extract(metadata, '$.source_job_id') = ?
            LIMIT 1
            """,
            (source_job_id,),
        )
        row = self.cursor.fetchone()
        return self._program_from_row(row) if row else None

    def add(
        self,
        program: Program,
        verbose: bool = False,
        defer_maintenance: bool = False,
    ) -> str:
        """
        Add a program to the database with optimized performance.

        This method uses batched transactions and defers expensive operations
        to improve performance with large databases. After adding a program,
        you should call check_scheduled_operations() to run any deferred
        operations like migrations.

        The insert commits before maintenance. If maintenance fails, use
        run_post_add_maintenance() to retry it for the inserted program.

        Example:
            db.add(program)  # Fast add
            db.check_scheduled_operations()  # Run deferred operations

        Args:
            program: The Program object to add
            verbose: Whether to print the per-program summary.
            defer_maintenance: When true, skip archive / best / migration
                follow-up work so callers can replay it later off the insert
                hot path.

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

        self._insert_program(program)
        if not defer_maintenance:
            self.run_post_add_maintenance(
                program,
                verbose=verbose,
                recompute_embeddings=True,
            )
        return program.id

    @db_retry()
    def _insert_program(self, program: Program) -> None:
        if not self.conn or not self.cursor:
            raise ConnectionError("DB not connected.")

        # Pre-serialize all JSON data once
        public_metrics_json = json.dumps(program.public_metrics or {})
        private_metrics_json = json.dumps(program.private_metrics or {})
        metadata_json = json.dumps(program.metadata or {})
        archive_insp_ids_json = json.dumps(program.archive_inspiration_ids or [])
        top_k_insp_ids_json = json.dumps(program.top_k_inspiration_ids or [])
        embedding_json = json.dumps(program.embedding)  # Serialize embedding
        embedding_pca_2d_json = json.dumps(program.embedding_pca_2d or [])
        embedding_pca_3d_json = json.dumps(program.embedding_pca_3d or [])
        migration_history_json = json.dumps(program.migration_history or [])

        # Handle text_feedback - convert to string if it's a list
        text_feedback_str = program.text_feedback
        if isinstance(text_feedback_str, list):
            # Join list items with newlines for readability
            text_feedback_str = "\n".join(str(item) for item in text_feedback_str)
        elif text_feedback_str is None:
            text_feedback_str = ""
        else:
            text_feedback_str = str(text_feedback_str)

        with self._run_state_transaction() as state:
            # Insert the program in a single operation
            self.cursor.execute(
                """
                INSERT INTO programs
                   (id, code, language, parent_id, archive_inspiration_ids,
                    top_k_inspiration_ids, generation, timestamp, code_diff,
                    combined_score, public_metrics, private_metrics,
                    text_feedback, complexity, embedding, embedding_pca_2d,
                    embedding_pca_3d, embedding_cluster_id, correct,
                    children_count, metadata, island_idx, migration_history,
                    system_prompt_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    program.id,
                    program.code,
                    program.language,
                    program.parent_id,
                    archive_insp_ids_json,
                    top_k_insp_ids_json,
                    program.generation,
                    program.timestamp,
                    program.code_diff,
                    program.combined_score,
                    public_metrics_json,
                    private_metrics_json,
                    text_feedback_str,
                    program.complexity,
                    embedding_json,  # Use serialized embedding
                    embedding_pca_2d_json,
                    embedding_pca_3d_json,
                    program.embedding_cluster_id,
                    program.correct,
                    program.children_count,
                    metadata_json,
                    program.island_idx,
                    migration_history_json,
                    program.system_prompt_id,
                ),
            )

            # Increment parent's children_count
            if program.parent_id:
                self.cursor.execute(
                    "UPDATE programs SET children_count = children_count + 1 "
                    "WHERE id = ?",
                    (program.parent_id,),
                )

            self._advance_last_iteration(state, program.generation)

        logger.info(
            "Program %s added to DB - score: %s.",
            program.id,
            program.combined_score,
        )

    def run_post_add_maintenance(
        self,
        program: Program,
        verbose: bool = False,
        recompute_embeddings: bool = False,
    ) -> None:
        """Replay deferred maintenance for a program that is already inserted."""
        self.run_post_add_maintenance_batch(
            [program],
            verbose=verbose,
            recompute_embeddings=recompute_embeddings,
        )

    def run_post_add_maintenance_batch(
        self,
        programs: List[Program],
        verbose: bool = False,
        recompute_embeddings: bool = False,
    ) -> None:
        """Replay deferred maintenance for already-inserted programs."""
        if not programs:
            return

        previous_migration = self._schedule_migration
        updated_metadata = []
        try:
            for program in sorted(programs, key=lambda item: item.generation):
                stored = self._stored_program_metadata(program.id)
                if stored.get("postprocess_db_maintenance_applied") is True:
                    program.metadata = stored
                    continue
                metadata = self._apply_pending_program_maintenance(program)
                updated_metadata.append((program, metadata))
            if updated_metadata:
                updated_metadata = self._finish_program_maintenance(updated_metadata)
        except BaseException:
            self._schedule_migration = previous_migration
            raise

        for program, metadata in updated_metadata:
            program.metadata = metadata
            if verbose:
                self._print_program_summary(program)

        if recompute_embeddings:
            self._recompute_embeddings_and_clusters()

    def _stored_program_metadata(self, program_id: str) -> Dict[str, Any]:
        row = self.cursor.execute(
            "SELECT metadata FROM programs WHERE id = ?", (program_id,)
        ).fetchone()
        stored = json.loads(row["metadata"] or "{}") if row else {}
        return stored if isinstance(stored, dict) else {}

    def _apply_pending_program_maintenance(self, program: Program) -> Dict[str, Any]:
        started_at = time.time()
        metadata = dict(program.metadata or {})
        self._update_archive(program)
        self._update_best_program(program)
        self._finish_tracked_best_score(program)
        if self.island_manager.needs_island_copies(program):
            self.island_manager.copy_program_to_islands(program)
            metadata.pop("_needs_island_copies", None)
        if program.correct:
            # An incorrect seed leaves its copies' islands without a correct
            # program; start them from the first one that works instead.
            self.island_manager.seed_uninitialized_islands(program)
        self.check_and_spawn_island_if_stagnant(
            program.generation, maintenance_program_id=program.id
        )

        finished_at = time.time()
        metadata.update(
            postprocess_db_maintenance_applied=True,
            postprocess_db_maintenance_started_at=started_at,
            postprocess_db_maintenance_finished_at=finished_at,
            postprocess_db_maintenance_seconds=max(0.0, finished_at - started_at),
        )
        return metadata

    def _finish_tracked_best_score(self, program: Program) -> None:
        # An older worker could commit the best ID before the stagnation fields.
        if not program.correct or self.best_program_id != program.id:
            return
        score = program.combined_score or 0.0
        if not (self.best_score_ever is None or score > self.best_score_ever):
            return
        generation = max(self.best_score_generation, program.generation)
        with self.conn:
            self.cursor.executemany(
                "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
                [("best_score_generation", str(generation)), ("best_score_ever", str(score))],
            )
        self.best_score_generation = generation
        self.best_score_ever = score

    def _finish_program_maintenance(
        self, updates: List[Tuple[Program, Dict[str, Any]]]
    ) -> List[Tuple[Program, Dict[str, Any]]]:
        # Migration can move a program twice if it commits before its receipt.
        with write_transaction(self.conn):
            pending = []
            already_applied = []
            for program, metadata in updates:
                stored = self._stored_program_metadata(program.id)
                if stored.get("postprocess_db_maintenance_applied") is True:
                    already_applied.append((program, stored))
                    continue
                pending.append((program, metadata))
                if self.island_manager.should_schedule_migration(program):
                    self._schedule_migration = True
            self.check_scheduled_operations()
            self.cursor.executemany(
                "UPDATE programs SET metadata = ? WHERE id = ?",
                [(json.dumps(metadata), program.id) for program, metadata in pending],
            )
            self.cursor.executemany(
                "DELETE FROM metadata_store WHERE key = ?",
                [
                    (self._maintenance_receipt_key(program.id, step),)
                    for program, _ in pending
                    for step in ("archive", "spawn")
                ],
            )
        return pending + already_applied

    @staticmethod
    def _maintenance_receipt_key(program_id: str, step: str) -> str:
        return f"program_maintenance:{step}:{program_id}"

    def _claim_maintenance_step(self, program_id: str, step: str) -> bool:
        """Record the decision in the same write transaction as its effects."""
        stored = self._stored_program_metadata(program_id)
        if stored.get("postprocess_db_maintenance_applied") is True:
            return False
        self.cursor.execute(
            "INSERT OR IGNORE INTO metadata_store (key, value) VALUES (?, NULL)",
            (self._maintenance_receipt_key(program_id, step),),
        )
        return self.cursor.rowcount == 1

    def recover_pending_child_maintenance(self) -> None:
        """Finish the database work of already evaluated children on resume."""
        pending_ids = []
        self.cursor.execute(
            "SELECT id, generation, metadata FROM programs WHERE generation > 0 "
            "ORDER BY generation, rowid"
        )
        for row in self.cursor:
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except ValueError:
                continue
            if isinstance(metadata, dict) and self._has_pending_child_maintenance(
                row["generation"], metadata
            ):
                pending_ids.append(row["id"])

        for program_id in pending_ids:
            program = self.get(program_id)
            if program is not None:
                self.run_post_add_maintenance(program)

    @staticmethod
    def _has_pending_child_maintenance(generation: int, metadata: dict) -> bool:
        if (
            metadata.get("_is_island_copy")
            or metadata.get("_spawned_island")
            or metadata.get("postprocess_side_effects_applied") is True
        ):
            return False
        if "postprocess_db_maintenance_applied" in metadata:
            return metadata["postprocess_db_maintenance_applied"] is False
        # The timing breakdown arrived with the deferred maintenance marker.
        # Older inline writes also have scheduler IDs, even when fully maintained.
        job_id = metadata.get("source_job_id")
        return (
            metadata.get("safe_processing") is True
            and metadata.get("source_generation") == generation
            and isinstance(job_id, str)
            and bool(job_id)
            and type(metadata.get("pipeline_accounted_seconds")) in (int, float)
        )

    def _persist_program_metadata(self, program: Program) -> None:
        with self.conn:
            self.cursor.execute(
                "UPDATE programs SET metadata = ? WHERE id = ?",
                (json.dumps(program.metadata), program.id),
            )

    def _program_from_row(self, row: sqlite3.Row) -> Optional[Program]:
        """Helper to create a Program object from a database row."""
        if not row:
            return None

        program_data = dict(row)

        # Use faster json loads
        public_metrics_text = program_data.get("public_metrics")
        if public_metrics_text:
            try:
                program_data["public_metrics"] = json.loads(public_metrics_text)
            except json.JSONDecodeError:
                program_data["public_metrics"] = {}
        else:
            program_data["public_metrics"] = {}

        private_metrics_text = program_data.get("private_metrics")
        if private_metrics_text:
            try:
                program_data["private_metrics"] = json.loads(private_metrics_text)
            except json.JSONDecodeError:
                program_data["private_metrics"] = {}
        else:
            program_data["private_metrics"] = {}

        # Same for metadata
        metadata_text = program_data.get("metadata")
        if metadata_text:
            try:
                program_data["metadata"] = json.loads(metadata_text)
            except json.JSONDecodeError:
                program_data["metadata"] = {}
        else:
            program_data["metadata"] = {}

        # Handle text_feedback (simple string field)
        if "text_feedback" not in program_data or program_data["text_feedback"] is None:
            program_data["text_feedback"] = ""

        # Handle inspiration_ids
        archive_insp_ids_text = program_data.get("archive_inspiration_ids")
        if archive_insp_ids_text:
            try:
                program_data["archive_inspiration_ids"] = json.loads(
                    archive_insp_ids_text
                )
            except json.JSONDecodeError:
                program_data["archive_inspiration_ids"] = []
        else:
            program_data["archive_inspiration_ids"] = []

        top_k_insp_ids_text = program_data.get("top_k_inspiration_ids")
        if top_k_insp_ids_text:
            try:
                program_data["top_k_inspiration_ids"] = json.loads(top_k_insp_ids_text)
            except json.JSONDecodeError:
                logger.warning(
                    "Could not decode top_k_inspiration_ids for "
                    f"program {program_data.get('id')}. "
                    "Defaulting to empty list."
                )
                program_data["top_k_inspiration_ids"] = []
        else:
            program_data["top_k_inspiration_ids"] = []

        # Handle embedding
        embedding_text = program_data.get("embedding")
        if embedding_text:
            try:
                program_data["embedding"] = json.loads(embedding_text)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode embedding for program "
                    f"{program_data.get('id')}. Defaulting to empty list."
                )
                program_data["embedding"] = []
        else:
            program_data["embedding"] = []

        embedding_pca_2d_text = program_data.get("embedding_pca_2d")
        if embedding_pca_2d_text:
            try:
                program_data["embedding_pca_2d"] = json.loads(embedding_pca_2d_text)
            except json.JSONDecodeError:
                program_data["embedding_pca_2d"] = []
        else:
            program_data["embedding_pca_2d"] = []

        embedding_pca_3d_text = program_data.get("embedding_pca_3d")
        if embedding_pca_3d_text:
            try:
                program_data["embedding_pca_3d"] = json.loads(embedding_pca_3d_text)
            except json.JSONDecodeError:
                program_data["embedding_pca_3d"] = []
        else:
            program_data["embedding_pca_3d"] = []

        # Handle migration_history
        migration_history_text = program_data.get("migration_history")
        if migration_history_text:
            try:
                program_data["migration_history"] = json.loads(migration_history_text)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode migration_history for program "
                    f"{program_data.get('id')}. Defaulting to empty list."
                )
                program_data["migration_history"] = []
        else:
            program_data["migration_history"] = []

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
    def get_ancestry(self, program_id: str, max_ancestors: int = 10) -> List[Program]:
        """
        Get the ancestry (lineage) of a program by walking up the parent chain.

        Args:
            program_id: ID of the program to get ancestry for
            max_ancestors: Maximum number of ancestors to retrieve

        Returns:
            List of ancestor programs, sorted chronologically (oldest first)
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        ancestors: List[Program] = []
        current_id = program_id

        # Walk up the parent chain
        for _ in range(max_ancestors):
            self.cursor.execute(
                "SELECT parent_id FROM programs WHERE id = ?", (current_id,)
            )
            row = self.cursor.fetchone()
            if not row or not row["parent_id"]:
                break

            parent_id = row["parent_id"]
            parent = self.get(parent_id)
            if parent:
                ancestors.append(parent)
                current_id = parent_id
            else:
                break

        # Reverse to get chronological order (oldest ancestor first)
        ancestors.reverse()

        if ancestors:
            logger.info(
                f"Retrieved {len(ancestors)} ancestors for program {program_id} "
                f"(generations: {[p.generation for p in ancestors]})"
            )

        return ancestors

    def _initialized_islands_for_sampling(self) -> Optional[List[int]]:
        """Islands a parent can be sampled from, or None while bootstrapping."""
        initialized_islands = self.island_manager.get_initialized_islands()
        if initialized_islands or self.island_manager.are_all_islands_initialized():
            return initialized_islands
        return None

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

        # Until an island holds a correct program there is nothing to select
        # from, so start from the initial program.
        initialized_islands = self._initialized_islands_for_sampling()
        if initialized_islands is None:
            # Get initial program (first program in database)
            self.cursor.execute("SELECT * FROM programs ORDER BY timestamp ASC LIMIT 1")
            row = self.cursor.fetchone()
            if not row:
                raise RuntimeError("No programs found in database")

            parent = self._program_from_row(row)
            if not parent:
                raise RuntimeError("Failed to load initial program")

            logger.info(
                f"No island initialized. Using initial program {parent.id} "
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

        # Sample among the islands that hold a correct program + constrain parents
        sampled_island = self.island_sampler.sample_island(initialized_islands)

        logger.debug(f"Sampling from island {sampled_island}")

        # Use CombinedParentSelector with island constraint
        # Don't pass update_metadata_func in read-only mode to avoid retry noise
        parent_selector = CombinedParentSelector(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
            get_program_func=self.get,
            best_program_id=self.best_program_id,
            beam_search_parent_id=self.beam_search_parent_id,
            last_iteration=self.last_iteration,
            update_metadata_func=self._set_beam_search_parent_id,
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

    @db_retry()
    def sample_with_fix_mode(
        self,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ) -> Tuple[Program, List[Program], List[Program], bool]:
        """
        Sample a parent program, returning fix mode indicator if no correct
        programs exist.

        Returns:
            Tuple of (parent, archive_inspirations, top_k_inspirations, needs_fix)
            where needs_fix is True if no correct programs exist and fix mode
            should be used.
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        # Until an island holds a correct program there is nothing to select
        # from. Islands that are still empty must not hold the run back: their
        # only program can be an incorrect seed copy that nothing ever replaces.
        initialized_islands = self._initialized_islands_for_sampling()
        if initialized_islands is None:
            # Check if there are any correct programs at all
            self.cursor.execute(
                "SELECT COUNT(*) as cnt FROM programs WHERE correct = 1"
            )
            correct_count = self.cursor.fetchone()["cnt"]

            if correct_count > 0:
                # There are correct programs, just not in all islands yet
                # Use initial program (first program in database)
                self.cursor.execute(
                    "SELECT * FROM programs ORDER BY timestamp ASC LIMIT 1"
                )
                row = self.cursor.fetchone()
                if not row:
                    raise RuntimeError("No programs found in database")
                parent = self._program_from_row(row)
                if not parent:
                    raise RuntimeError("Failed to load initial program")
                needs_fix = not parent.correct
                logger.info(
                    f"Not all islands initialized. "
                    f"Using initial program {parent.id} (needs_fix={needs_fix})."
                )
            else:
                # No correct programs exist - randomly sample from incorrect
                self.cursor.execute("SELECT * FROM programs WHERE correct = 0")
                rows = self.cursor.fetchall()
                if rows:
                    selected_row = rows[np.random.randint(len(rows))]
                    parent = self._program_from_row(selected_row)
                    if not parent:
                        raise RuntimeError("Failed to load sampled program")
                    needs_fix = True
                    logger.info(
                        f"No correct programs. Randomly sampled incorrect program "
                        f"{parent.id} (Gen: {parent.generation}) "
                        f"[from {len(rows)} incorrect programs]."
                    )
                else:
                    # Fallback to initial program if no incorrect programs either
                    self.cursor.execute(
                        "SELECT * FROM programs ORDER BY timestamp ASC LIMIT 1"
                    )
                    row = self.cursor.fetchone()
                    if not row:
                        raise RuntimeError("No programs found in database")
                    parent = self._program_from_row(row)
                    if not parent:
                        raise RuntimeError("Failed to load initial program")
                    needs_fix = not parent.correct
                    logger.info(
                        f"Not all islands initialized. "
                        f"Using initial program {parent.id} (needs_fix={needs_fix})."
                    )

            # For fix mode, get ancestors as inspirations
            if needs_fix:
                num_ancestors = (
                    self.config.num_archive_inspirations
                    + self.config.num_top_k_inspirations
                )
                ancestor_inspirations = self.get_ancestry(
                    parent.id, max_ancestors=num_ancestors
                )
                self._print_sampling_summary_helper(
                    parent,
                    [],
                    [],
                    target_generation,
                    novelty_attempt,
                    max_novelty_attempts,
                    resample_attempt,
                    max_resample_attempts,
                    ancestor_inspirations=ancestor_inspirations,
                    is_fix_mode=True,
                )
                # Return ancestors as archive_inspirations for fix mode
                return parent, ancestor_inspirations, [], needs_fix
            else:
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
                return parent, [], [], needs_fix

        # Sample among the islands that hold a correct program + constrain parents
        sampled_island = self.island_sampler.sample_island(initialized_islands)

        logger.debug(f"Sampling from island {sampled_island}")

        # Use CombinedParentSelector with island constraint
        # Don't pass update_metadata_func in read-only mode to avoid retry noise
        parent_selector = CombinedParentSelector(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
            get_program_func=self.get,
            best_program_id=self.best_program_id,
            beam_search_parent_id=self.beam_search_parent_id,
            last_iteration=self.last_iteration,
            update_metadata_func=self._set_beam_search_parent_id,
            get_best_program_func=self.get_best_program,
        )

        # Use the new method that returns fix mode
        parent, needs_fix = parent_selector.sample_parent_with_fix_mode(
            island_idx=sampled_island
        )
        if not parent:
            raise RuntimeError(f"Failed to sample parent from island {sampled_island}")

        # If in fix mode, don't sample inspirations (they'd all be incorrect too)
        if needs_fix:
            logger.info(
                f"FIX MODE: Using incorrect program {parent.id} "
                f"(Gen: {parent.generation}, Score: {parent.combined_score})"
            )
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
            return parent, [], [], True

        # Normal mode - sample inspirations
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

        return parent, archive_inspirations, top_k_inspirations, False

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
        ancestor_inspirations=None,
        is_fix_mode=False,
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
                default_console=self.display_console,
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
            ancestor_inspirations,
            is_fix_mode,
            console=self.display_console,
        )

    @db_retry()
    def get_best_program(
        self,
        metric: Optional[str] = None,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        if metric is not None or island_idx is not None:
            return self._find_best_program(metric, island_idx)

        with self._run_state_lock:
            program = self._get_tracked_best(self._read_run_state())
            if program is not None:
                return program
            if self.read_only:
                program = self._find_best_program()
                self.best_program_id = self._maintained_best_id(program)
                return program
        return self._repair_tracked_best()

    def _get_tracked_best(self, state: _RunState) -> Optional[Program]:
        program = self.get(state.best_program_id) if state.best_program_id else None
        return program if program and program.correct else None

    def _repair_tracked_best(self) -> Optional[Program]:
        with self._run_state_transaction() as state:
            # Another worker may have published a best while we waited to write.
            program = self._get_tracked_best(state)
            if program is None:
                program = self._find_best_program()
                best_id = self._maintained_best_id(program)
                if best_id != state.best_program_id:
                    self._write_metadata("best_program_id", best_id)
                    state.best_program_id = best_id
        return program

    def _maintained_best_id(self, program: Optional[Program]) -> Optional[str]:
        if (
            program is not None
            and program.metadata.get(_MAINTENANCE_APPLIED_KEY)
            and self._is_original_program(program)
        ):
            return program.id
        return None

    def _find_best_program(
        self,
        metric: Optional[str] = None,
        island_idx: Optional[int] = None,
    ) -> Optional[Program]:
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        # Fetch only correct programs and sort in Python.
        if island_idx is not None:
            self.cursor.execute(
                "SELECT * FROM programs WHERE correct = 1 AND island_idx = ?",
                (island_idx,),
            )
        else:
            self.cursor.execute("SELECT * FROM programs WHERE correct = 1")
        all_rows = self.cursor.fetchall()
        if not all_rows:
            logger.debug("No correct programs found in database.")
            return None

        programs = []
        for row_data in all_rows:
            program = self._program_from_row(row_data)
            if program is not None:
                programs.append(program)

        if not programs:
            return None

        if metric:
            return max(
                (
                    program
                    for program in programs
                    if program.public_metrics and metric in program.public_metrics
                ),
                key=lambda program: program.public_metrics[metric],
                default=None,
            )
        if island_idx is None and metric is None:
            originals = [
                program for program in programs if self._is_original_program(program)
            ]
            maintained = [
                program
                for program in originals
                if program.metadata.get(_MAINTENANCE_APPLIED_KEY)
            ]
            # A run with no maintained best can still return a provisional lookup.
            programs = maintained or originals or programs
        best_program = programs[0]
        for candidate in programs[1:]:
            if self._is_better(candidate, best_program):
                best_program = candidate
        return best_program

    @db_retry()
    def get_population_telemetry(self) -> List[Dict[str, Any]]:
        """Return compact per-island aggregates for progress telemetry.

        The query never selects program code, embeddings, evaluation metrics, or
        full metadata. Final reporting can still use :meth:`get_all_programs`.
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute("""
            WITH metadata_source AS (
                SELECT
                    island_idx,
                    correct,
                    combined_score,
                    COALESCE(metadata, '{}') AS metadata,
                    REPLACE(
                        REPLACE(
                            REPLACE(COALESCE(metadata, '{}'), '-Infinity', 'null'),
                            'Infinity', 'null'
                        ),
                        'NaN', 'null'
                    ) AS finite_metadata
                FROM programs
            ), source AS (
                SELECT
                    island_idx,
                    correct,
                    combined_score,
                    CASE
                        WHEN json_valid(metadata) THEN metadata
                        WHEN json_valid(finite_metadata) THEN finite_metadata
                        ELSE '{}'
                    END AS metadata
                FROM metadata_source
            ), raw AS (
                SELECT
                    island_idx,
                    correct,
                    CASE WHEN combined_score > -1e999 AND combined_score < 1e999
                        THEN combined_score END AS combined_score,
                    CASE WHEN
                        COALESCE(json_extract(metadata, '$._is_island_copy'), 0) = 1
                        OR COALESCE(json_extract(metadata, '$._spawned_island'), 0) = 1
                    THEN 1 ELSE 0 END AS is_copy,
                    COALESCE(
                        NULLIF(json_extract(metadata, '$.model_name'), ''),
                        NULLIF(json_extract(metadata, '$.model'), ''),
                        NULLIF(json_extract(metadata, '$.llm_result.model_name'), ''),
                        NULLIF(json_extract(metadata, '$.llm_result.model'), '')
                    ) AS model_name,
                    COALESCE(
                        json_extract(metadata, '$.headless_pricing_unknown'), 0
                    ) AS pricing_unknown,
                    CASE WHEN json_type(metadata, '$.api_costs') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.api_costs') ELSE 0 END AS api_cost,
                    CASE WHEN json_type(metadata, '$.embed_cost') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.embed_cost') ELSE 0 END AS embed_cost,
                    CASE WHEN json_type(metadata, '$.novelty_cost') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.novelty_cost') ELSE 0 END AS novelty_cost,
                    CASE WHEN json_type(metadata, '$.meta_cost') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.meta_cost') ELSE 0 END AS meta_cost,
                    CASE WHEN json_type(metadata, '$.sampling_seconds') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.sampling_seconds') ELSE 0 END AS sampling_seconds,
                    CASE WHEN json_type(metadata, '$.evaluation_seconds') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.evaluation_seconds') ELSE 0 END AS evaluation_seconds,
                    CASE WHEN json_type(metadata, '$.postprocess_seconds') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.postprocess_seconds') ELSE 0 END AS postprocess_seconds,
                    CASE WHEN json_type(metadata, '$.pipeline_seconds') IN ('integer', 'real')
                        THEN json_extract(metadata, '$.pipeline_seconds') ELSE 0 END AS pipeline_seconds
                FROM source
            ), telemetry AS (
                SELECT
                    island_idx,
                    correct,
                    combined_score,
                    is_copy,
                    model_name,
                    CASE WHEN api_cost > -1e999 AND api_cost < 1e999
                        THEN api_cost ELSE 0 END AS api_cost,
                    CASE WHEN embed_cost > -1e999 AND embed_cost < 1e999
                        THEN embed_cost ELSE 0 END AS embed_cost,
                    CASE WHEN novelty_cost > -1e999 AND novelty_cost < 1e999
                        THEN novelty_cost ELSE 0 END AS novelty_cost,
                    CASE WHEN meta_cost > -1e999 AND meta_cost < 1e999
                        THEN meta_cost ELSE 0 END AS meta_cost,
                    CASE WHEN sampling_seconds > -1e999 AND sampling_seconds < 1e999
                        THEN sampling_seconds ELSE 0 END AS sampling_seconds,
                    CASE WHEN evaluation_seconds > -1e999 AND evaluation_seconds < 1e999
                        THEN evaluation_seconds ELSE 0 END AS evaluation_seconds,
                    CASE WHEN postprocess_seconds > -1e999 AND postprocess_seconds < 1e999
                        THEN postprocess_seconds ELSE 0 END AS postprocess_seconds,
                    CASE WHEN pipeline_seconds > -1e999 AND pipeline_seconds < 1e999
                        THEN pipeline_seconds ELSE 0 END AS pipeline_seconds,
                    CASE WHEN model_name GLOB 'headless/*' AND pricing_unknown = 1
                        THEN 1 ELSE 0 END AS api_cost_unknown
                FROM raw
            )
            SELECT
                island_idx,
                COUNT(*) AS count,
                SUM(CASE WHEN is_copy = 0 THEN 1 ELSE 0 END) AS evaluated_count,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                COUNT(combined_score) AS score_count,
                COALESCE(SUM(combined_score), 0) AS score_sum,
                MAX(combined_score) AS best_score,
                MAX(CASE WHEN correct = 1 THEN combined_score END) AS best_correct_score,
                SUM(CASE WHEN is_copy = 0 AND api_cost_unknown = 0
                    THEN api_cost ELSE 0 END) AS api_cost,
                SUM(CASE WHEN is_copy = 0 THEN embed_cost ELSE 0 END) AS embed_cost,
                SUM(CASE WHEN is_copy = 0 THEN novelty_cost ELSE 0 END) AS novelty_cost,
                SUM(CASE WHEN is_copy = 0 THEN meta_cost ELSE 0 END) AS meta_cost,
                SUM(CASE WHEN is_copy = 0 AND api_cost_unknown = 1
                    THEN 1 ELSE 0 END) AS pricing_unknown_count,
                SUM(CASE WHEN is_copy = 0 THEN sampling_seconds ELSE 0 END)
                    AS sampling_seconds,
                SUM(CASE WHEN is_copy = 0 THEN evaluation_seconds ELSE 0 END)
                    AS evaluation_seconds,
                SUM(CASE WHEN is_copy = 0 THEN postprocess_seconds ELSE 0 END)
                    AS postprocess_seconds,
                SUM(CASE WHEN is_copy = 0 THEN pipeline_seconds ELSE 0 END)
                    AS pipeline_seconds
            FROM telemetry
            GROUP BY island_idx
            ORDER BY island_idx
            """)
        return [dict(row) for row in self.cursor.fetchall()]

    @db_retry()
    def get_all_programs(self) -> List[Program]:
        """Get all programs from the database."""
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute("""
            SELECT p.*,
                   CASE WHEN a.program_id IS NOT NULL THEN 1 ELSE 0 END as in_archive
            FROM programs p
            LEFT JOIN archive a ON p.id = a.program_id
            """)
        rows = self.cursor.fetchall()
        programs = [self._program_from_row(row) for row in rows]
        # Filter out any None values that might result from row processing errors
        return [p for p in programs if p is not None]

    @db_retry()
    def get_programs_summary(self) -> List[Dict[str, Any]]:
        """
        Get lightweight summary of all programs for visualization.
        Excludes heavy fields like code, embeddings, and large metadata.
        Returns raw dicts instead of Program objects for efficiency.
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute("""
            SELECT
                p.id,
                p.parent_id,
                p.generation,
                p.timestamp,
                p.combined_score,
                p.correct,
                p.complexity,
                p.island_idx,
                p.children_count,
                p.public_metrics,
                p.private_metrics,
                p.metadata,
                p.embedding_pca_2d,
                p.embedding_pca_3d,
                p.embedding_cluster_id,
                p.language,
                p.text_feedback,
                p.top_k_inspiration_ids,
                p.archive_inspiration_ids,
                p.migration_history,
                CASE WHEN a.program_id IS NOT NULL THEN 1 ELSE 0 END as in_archive
            FROM programs p
            LEFT JOIN archive a ON p.id = a.program_id
            """)
        rows = self.cursor.fetchall()
        summaries = []
        for row in rows:
            row_dict = dict(row)
            # Parse only the lightweight JSON fields
            for json_field in ["public_metrics", "private_metrics", "metadata"]:
                if row_dict.get(json_field):
                    try:
                        row_dict[json_field] = json.loads(row_dict[json_field])
                    except json.JSONDecodeError:
                        row_dict[json_field] = {}
                else:
                    row_dict[json_field] = {}
            # Parse PCA embeddings (small arrays)
            for pca_field in ["embedding_pca_2d", "embedding_pca_3d"]:
                if row_dict.get(pca_field):
                    try:
                        row_dict[pca_field] = json.loads(row_dict[pca_field])
                    except json.JSONDecodeError:
                        row_dict[pca_field] = []
                else:
                    row_dict[pca_field] = []
            # Parse inspiration IDs and migration history (small arrays)
            for list_field in [
                "top_k_inspiration_ids",
                "archive_inspiration_ids",
                "migration_history",
            ]:
                if row_dict.get(list_field):
                    try:
                        row_dict[list_field] = json.loads(row_dict[list_field])
                    except json.JSONDecodeError:
                        row_dict[list_field] = []
                else:
                    row_dict[list_field] = []
            row_dict["in_archive"] = bool(row_dict.get("in_archive", 0))
            summaries.append(row_dict)
        return summaries

    @db_retry()
    def get_program_count_and_timestamp(self) -> Dict[str, Any]:
        """
        Get program count and max timestamp for efficient change detection.
        Used by auto-refresh to check if data has changed without loading all programs.
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute(
            "SELECT COUNT(*) as count, MAX(timestamp) as max_timestamp FROM programs"
        )
        row = self.cursor.fetchone()
        return {
            "count": row["count"] if row else 0,
            "max_timestamp": row["max_timestamp"] if row else None,
        }

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
    def repair_legacy_failed_seeds(self) -> None:
        """Repair generation-zero rows promoted by the old evaluation fallback."""
        seeds = self.get_programs_by_generation(0)
        failed_ids = {
            seed.id
            for seed in seeds
            if (seed.metadata or {}).get("evaluation_failed") is True
        }
        if not failed_ids:
            return
        while True:
            copies = {
                seed.id
                for seed in seeds
                if any(
                    (seed.metadata or {}).get(key) in failed_ids
                    for key in ("_original_program_id", "_spawned_from_program_id")
                    if isinstance((seed.metadata or {}).get(key), str)
                )
            }
            if copies <= failed_ids:
                break
            failed_ids.update(copies)

        invalid_seeds = [seed for seed in seeds if seed.correct and seed.id in failed_ids]
        if not invalid_seeds:
            return
        with self.conn:
            rows = [(seed.id,) for seed in invalid_seeds]
            self.cursor.executemany("UPDATE programs SET correct = 0 WHERE id = ?", rows)
            self.cursor.executemany("DELETE FROM archive WHERE program_id = ?", rows)
            self._repair_legacy_seed_tracking(invalid_seeds)
        self._load_metadata_from_db()

    def _repair_legacy_seed_tracking(self, invalid_seeds: List[Program]) -> None:
        invalid_ids = {seed.id for seed in invalid_seeds}
        repaired_metadata: dict[str, Optional[str]] = {}
        if self.beam_search_parent_id in invalid_ids:
            repaired_metadata["beam_search_parent_id"] = None
        repair_best = self.best_program_id in invalid_ids
        repair_stagnation = (repair_best or self.best_score_generation == 0) and any(
            (seed.combined_score or 0.0) == self.best_score_ever for seed in invalid_seeds
        )
        if repair_best or repair_stagnation:
            self.cursor.execute("SELECT * FROM programs WHERE correct = 1")
            best = None
            highest_score = None
            for row in self.cursor:
                candidate = self._program_from_row(row)
                if candidate is None:
                    continue
                metadata = candidate.metadata or {}
                if metadata.get("_is_island_copy") or metadata.get("_spawned_island"):
                    continue
                if best is None or self._is_better(candidate, best):
                    best = candidate
                if highest_score is None or (candidate.combined_score or 0.0) > (
                    highest_score.combined_score or 0.0
                ):
                    highest_score = candidate
            if repair_best:
                repaired_metadata["best_program_id"] = best.id if best else None
            if repair_stagnation:
                repaired_metadata["best_score_generation"] = str(
                    max(self.best_score_generation, highest_score.generation)
                    if highest_score else self.best_score_generation
                )
                repaired_metadata["best_score_ever"] = (
                    str(highest_score.combined_score or 0.0) if highest_score else None
                )
        self.cursor.executemany(
            "INSERT OR REPLACE INTO metadata_store (key, value) VALUES (?, ?)",
            repaired_metadata.items(),
        )

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
        programs = []
        for row_data in all_rows:
            p_dict = dict(row_data)

            # Optimize JSON parsing
            public_metrics_text = p_dict.get("public_metrics")
            if public_metrics_text:
                try:
                    p_dict["public_metrics"] = json.loads(public_metrics_text)
                except json.JSONDecodeError:
                    p_dict["public_metrics"] = {}
            else:
                p_dict["public_metrics"] = {}

            private_metrics_text = p_dict.get("private_metrics")
            if private_metrics_text:
                try:
                    p_dict["private_metrics"] = json.loads(private_metrics_text)
                except json.JSONDecodeError:
                    p_dict["private_metrics"] = {}
            else:
                p_dict["private_metrics"] = {}

            metadata_text = p_dict.get("metadata")
            if metadata_text:
                try:
                    p_dict["metadata"] = json.loads(metadata_text)
                except json.JSONDecodeError:
                    p_dict["metadata"] = {}
            else:
                p_dict["metadata"] = {}

            # Create program object
            programs.append(Program.from_dict(p_dict))

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
                    key=lambda p_item: (
                        sum(p_item.public_metrics.values()) / len(p_item.public_metrics)
                        if p_item.public_metrics
                        else -float("inf")
                    ),
                    reverse=True,
                )

            return sorted_p[:n]

        return []

    @db_retry()
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

        with self._run_state_lock, self._committed_run_state_cursor() as reader:
            committed = self._read_run_state(reader)
            if self.last_iteration > committed.last_iteration:
                if not self.conn.in_transaction and not self.read_only:
                    self.conn.execute("BEGIN IMMEDIATE")
                with self.conn:
                    state = self._read_run_state()
                    self._advance_last_iteration(state, self.last_iteration)
            else:
                self.conn.commit()
            self._cache_run_state(self._read_run_state(reader))
        logger.info(
            f"Database state committed. Last iteration: "
            f"{self.last_iteration}. Best: {self.best_program_id}"
        )

    @contextmanager
    def _committed_run_state_cursor(self) -> Iterator[sqlite3.Cursor]:
        # Caller cursors can keep the connection on an older WAL snapshot even
        # after commit. Read the active file through an independent connection.
        main_file = self.conn.execute(
            "SELECT file FROM pragma_database_list WHERE name = 'main'"
        ).fetchone()["file"]
        if not main_file:
            yield self.cursor
            return
        with closing(
            sqlite3.connect(
                f"{Path(main_file).as_uri()}?mode=ro", uri=True, timeout=60.0
            )
        ) as reader:
            yield reader.cursor()

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

        self.conn = sqlite3.connect(str(db_path_obj), timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._load_metadata_from_db()

        count = self._count_programs_in_db()
        logger.info(
            f"Loaded DB from '{db_path_obj}'. {count} programs. "
            f"Last iter: {self.last_iteration}."
        )

    def _get_criterion_value(self, program: Program, criterion: str) -> float:
        """
        Get the value of a specific criterion for a program.

        Supported criteria:
            - combined_score: The program's combined fitness score
            - loc: Lines of code
            - lloc: Logical lines of code
            - complexity: Cyclomatic complexity
            - maintainability: Maintainability index
            - nesting: Maximum nesting depth
        """
        if criterion == "combined_score":
            return program.combined_score or 0.0

        # Get code analysis metrics from metadata
        metrics = {}
        if program.metadata:
            metrics = program.metadata.get("code_analysis_metrics", {})

        if criterion == "loc":
            return metrics.get(
                "lines_of_code", len(program.code.split("\n")) if program.code else 1
            )
        elif criterion == "lloc":
            return metrics.get(
                "logical_lines_of_code",
                len(program.code.split("\n")) if program.code else 1,
            )
        elif criterion == "complexity":
            return metrics.get("cyclomatic_complexity", 1.0)
        elif criterion == "maintainability":
            return metrics.get("maintainability_index", 100.0)
        elif criterion == "nesting":
            return metrics.get("max_nesting_depth", 1)

        # Unknown criterion - return 0
        logger.warning(f"Unknown archive criterion: {criterion}")
        return 0.0

    def _compute_archive_score_ranked(
        self, program: Program, archive_programs: List[Program]
    ) -> float:
        """
        Compute score using rank-based normalization for scale-invariant comparison.

        Each criterion is converted to a percentile rank (0-1) based on the
        archive population, then weighted according to archive_criteria config.

        Args:
            program: The program to score
            archive_programs: Current archive programs for rank computation

        Returns:
            Combined score where higher is always better
        """
        criteria = getattr(self.config, "archive_criteria", {"combined_score": 1.0})

        if not archive_programs:
            # No archive yet - just use the primary criterion
            primary_criterion = next(iter(criteria.keys()), "combined_score")
            primary_weight = criteria.get(primary_criterion, 1.0)
            value = self._get_criterion_value(program, primary_criterion)
            return value if primary_weight > 0 else -value

        all_programs = archive_programs + [program]

        score = 0.0
        for criterion, weight in criteria.items():
            # Get values for all programs
            values = [self._get_criterion_value(p, criterion) for p in all_programs]
            program_value = values[-1]  # The new program's value

            # Compute percentile rank (0 = worst, 1 = best)
            if weight > 0:
                # Higher is better: count how many are strictly worse
                rank = sum(1 for v in values if v < program_value) / len(values)
            else:
                # Lower is better: count how many are strictly worse (i.e., higher)
                rank = sum(1 for v in values if v > program_value) / len(values)
                weight = abs(weight)  # Use absolute weight after handling direction

            score += weight * rank

        return score

    def _get_archive_programs(self) -> List[Program]:
        """Fetch all programs currently in the archive."""
        if not self.cursor:
            return []

        self.cursor.execute(
            "SELECT p.* FROM programs p JOIN archive a ON p.id = a.program_id"
        )
        rows = self.cursor.fetchall()

        programs = []
        for row in rows:
            prog = self._program_from_row(row)
            if prog:
                programs.append(prog)

        return programs

    def _find_most_similar_in_archive(
        self, embedding: List[float]
    ) -> Optional[Program]:
        """
        Find the most similar program in the archive by embedding cosine similarity.
        Used for crowding-based archive selection.

        Args:
            embedding: The embedding vector to compare against

        Returns:
            The most similar program, or None if no valid comparisons possible
        """
        if not embedding or not self.cursor:
            return None

        self.cursor.execute(
            "SELECT p.* FROM programs p JOIN archive a ON p.id = a.program_id"
        )
        rows = self.cursor.fetchall()

        if not rows:
            return None

        best_similarity = -float("inf")
        most_similar = None

        embedding_arr = np.array(embedding)
        embedding_norm = np.linalg.norm(embedding_arr)

        if embedding_norm < 1e-8:
            return None

        for row in rows:
            prog = self._program_from_row(row)
            if not prog or not prog.embedding:
                continue

            prog_embedding = np.array(prog.embedding)
            prog_norm = np.linalg.norm(prog_embedding)

            if prog_norm < 1e-8:
                continue

            # Cosine similarity
            similarity = np.dot(embedding_arr, prog_embedding) / (
                embedding_norm * prog_norm
            )

            if similarity > best_similarity:
                best_similarity = similarity
                most_similar = prog

        return most_similar

    def _is_better(
        self,
        program1: Program,
        program2: Program,
        archive_programs: Optional[List[Program]] = None,
    ) -> bool:
        """
        Compare two programs to determine if program1 is better than program2.

        Args:
            program1: First program to compare
            program2: Second program to compare
            archive_programs: Optional archive context for rank-based scoring.
                If provided and archive_criteria has multiple criteria,
                uses rank-based normalization for scale-invariant comparison.

        Returns:
            True if program1 is better than program2
        """
        # First prioritize correctness (always)
        if program1.correct and not program2.correct:
            return True
        if program2.correct and not program1.correct:
            return False

        # Check if we should use multi-criteria ranked scoring
        criteria = getattr(self.config, "archive_criteria", {"combined_score": 1.0})
        use_ranked = archive_programs is not None and len(criteria) > 1

        if use_ranked:
            # Use rank-based scoring with archive context
            # Rank each program against the archive and the other program, so
            # both are ranked in the same population and their own comparison
            # counts
            context = [
                p for p in archive_programs if p.id not in (program1.id, program2.id)
            ]
            s1 = self._compute_archive_score_ranked(program1, context + [program2])
            s2 = self._compute_archive_score_ranked(program2, context + [program1])

            if s1 != s2:
                return s1 > s2
        else:
            # Simple single-criterion comparison (original behavior)
            s1 = program1.combined_score
            s2 = program2.combined_score

            if s1 is not None and s2 is not None:
                if s1 != s2:
                    return s1 > s2
            elif s1 is not None:
                return True  # p1 has score, p2 doesn't
            elif s2 is not None:
                return False  # p2 has score, p1 doesn't

            # Fallback to average public metrics
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
                pass

        # Tie-breaker: prefer newer programs
        return program1.timestamp > program2.timestamp

    @db_retry()
    def _update_archive(self, program: Program) -> None:
        """
        Update the archive with a new program using the configured selection strategy.

        Strategies:
            - "fitness": Replace the worst program globally (classic approach)
            - "crowding": Replace the most similar program if better (maintains diversity)
        """
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

        self.conn.execute("BEGIN IMMEDIATE")
        with self.conn:
            if not self._claim_maintenance_step(program.id, "archive"):
                return
            # Maintenance can be replayed after the archive update committed.
            self.cursor.execute(
                "SELECT 1 FROM archive WHERE program_id = ?",
                (program.id,),
            )
            if self.cursor.fetchone():
                return

            self.cursor.execute("SELECT COUNT(*) FROM archive")
            count = (self.cursor.fetchone() or [0])[0]
            if count < self.config.archive_size:
                self.cursor.execute(
                    "INSERT INTO archive (program_id) VALUES (?)", (program.id,)
                )
                logger.debug(
                    f"Program {program.id} added to archive (space available)."
                )
                return

            strategy = getattr(self.config, "archive_selection_strategy", "fitness")
            if strategy == "crowding":
                self._update_archive_crowding(program)
            else:
                self._update_archive_fitness(program)

    def _update_archive_fitness(self, program: Program) -> None:
        """
        Fitness-based archive update: replace the worst program globally.

        Uses rank-based scoring if multiple criteria are configured.
        """
        # Fetch full archive programs for multi-criteria comparison
        archive_programs = self._get_archive_programs()

        if not archive_programs:
            self.cursor.execute(
                "INSERT OR IGNORE INTO archive (program_id) VALUES (?)",
                (program.id,),
            )
            return

        # Find the worst program using ranked scoring
        criteria = getattr(self.config, "archive_criteria", {"combined_score": 1.0})

        if len(criteria) > 1:
            # Multi-criteria: use ranked scoring to find worst
            scores = [
                (p, self._compute_archive_score_ranked(p, archive_programs))
                for p in archive_programs
            ]
            worst_in_archive = min(scores, key=lambda x: x[1])[0]
        else:
            # Single criterion: find worst by pairwise comparison
            worst_in_archive = archive_programs[0]
            for p_archived in archive_programs[1:]:
                if self._is_better(worst_in_archive, p_archived):
                    worst_in_archive = p_archived

        # Check if new program is better than the worst
        if self._is_better(program, worst_in_archive, archive_programs):
            self.cursor.execute(
                "DELETE FROM archive WHERE program_id = ?",
                (worst_in_archive.id,),
            )
            self.cursor.execute(
                "INSERT INTO archive (program_id) VALUES (?)", (program.id,)
            )

            # Log with score information
            p_score = program.combined_score or 0.0
            w_score = worst_in_archive.combined_score or 0.0
            logger.info(
                f"Program {program.id} (score={p_score:.4f}) replaced "
                f"{worst_in_archive.id} (score={w_score:.4f}) in archive [fitness]."
            )

    def _update_archive_crowding(self, program: Program) -> None:
        """
        Crowding-based archive update: replace the most similar program if better.

        This maintains diversity by making new programs compete with their
        "neighbors" in solution space rather than globally worst programs.
        Falls back to fitness-based if no embedding is available.
        """
        # Check if program has embedding for similarity computation
        if not program.embedding:
            logger.debug(
                f"Program {program.id} has no embedding, falling back to fitness-based."
            )
            return self._update_archive_fitness(program)

        # Find most similar program in archive
        most_similar = self._find_most_similar_in_archive(program.embedding)

        if most_similar is None:
            logger.debug(
                "No similar programs found in archive, falling back to fitness-based."
            )
            return self._update_archive_fitness(program)

        # Get archive for ranked comparison
        archive_programs = self._get_archive_programs()

        # Only replace if better than the similar program (niching)
        if self._is_better(program, most_similar, archive_programs):
            self.cursor.execute(
                "DELETE FROM archive WHERE program_id = ?",
                (most_similar.id,),
            )
            self.cursor.execute(
                "INSERT INTO archive (program_id) VALUES (?)", (program.id,)
            )

            p_score = program.combined_score or 0.0
            s_score = most_similar.combined_score or 0.0
            logger.info(
                f"Program {program.id} (score={p_score:.4f}) replaced similar program "
                f"{most_similar.id} (score={s_score:.4f}) in archive [crowding]."
            )

    @db_retry()
    def _update_best_program(self, program: Program) -> None:
        if not program.correct:
            self._refresh_run_state()
            return

        with self._run_state_transaction() as state:
            current_best = self._get_tracked_best(state)
            improved = current_best is None or self._is_better(program, current_best)
            if improved:
                self._write_metadata("best_program_id", program.id)
                state.best_program_id = program.id

            # A lookup may already have recovered this ID before maintenance ran.
            if state.best_program_id == program.id:
                self._advance_best_score(state, program)

        if improved:
            logger.info(
                "New best program: %s (gen: %s, score: %s, island: %s)",
                program.id,
                program.generation,
                program.combined_score,
                program.island_idx,
            )

    def print_summary(
        self, console=None, total_program_target: Optional[int] = None
    ) -> None:
        """Print a summary of the database contents using DatabaseDisplay."""
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self.get_best_program,
                default_console=self.display_console,
            )
            self._database_display.set_last_iteration(self.last_iteration)

        if hasattr(self._database_display, "set_default_console"):
            self._database_display.set_default_console(self.display_console)
        self._database_display.print_summary(
            console,
            total_program_target=total_program_target,
        )

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
                default_console=self.display_console,
            )

        if hasattr(self._database_display, "set_default_console"):
            self._database_display.set_default_console(self.display_console)
        self._database_display.print_program_summary(
            program, console=self.display_console
        )

    def set_display_console(self, console: Optional[Any]) -> None:
        """Set shared console used for rich DB summaries."""
        self.display_console = console
        if hasattr(self, "_database_display") and hasattr(
            self._database_display, "set_default_console"
        ):
            self._database_display.set_default_console(console)

    def check_scheduled_operations(self):
        """Run any operations that were scheduled during add but deferred for performance."""
        if self._schedule_migration:
            logger.info("Running scheduled migration operation")
            self.island_manager.perform_migration(self.last_iteration)
            self._schedule_migration = False

    @db_retry()
    def is_stagnant(self, current_generation: int) -> bool:
        """Check if evolution is stagnant based on generations without improvement.

        Args:
            current_generation: The current generation number

        Returns:
            True if stagnant (no improvement for stagnation_threshold generations)
        """
        if not getattr(self.config, "enable_dynamic_islands", False):
            return False

        threshold = getattr(self.config, "stagnation_threshold", 100)
        gens_since_improvement = (
            current_generation - self._read_run_state().best_score_generation
        )

        return gens_since_improvement >= threshold

    def _set_beam_search_parent_id(
        self,
        key: str,
        value: Optional[str],
    ) -> None:
        """Update beam-search state in memory and persist it when writable."""
        if key != "beam_search_parent_id":
            raise ValueError(f"Unexpected beam-search metadata key: {key}")

        self.beam_search_parent_id = value

        if not self.read_only:
            self._update_metadata_in_db(key, value)

    @db_retry()
    def check_and_spawn_island_if_stagnant(
        self, current_generation: int, *, maintenance_program_id: Optional[str] = None
    ) -> bool:
        """Check for stagnation and spawn a new island if needed.

        Args:
            current_generation: The current generation number

        Returns:
            True if a new island was spawned, False otherwise
        """
        if not getattr(self.config, "enable_dynamic_islands", False):
            return False

        if not self.island_manager:
            logger.warning("Cannot spawn island: no island manager configured")
            return False

        with self._run_state_transaction() as state:
            if maintenance_program_id is not None and not self._claim_maintenance_step(
                maintenance_program_id, "spawn"
            ):
                return False
            threshold = getattr(self.config, "stagnation_threshold", 100)
            gens_stagnant = current_generation - state.best_score_generation
            spawned = (
                gens_stagnant >= threshold
                and self.island_manager.insert_spawned_island()
            )
            if spawned:
                state.best_score_generation = max(
                    state.best_score_generation, current_generation
                )
                self._write_metadata(
                    "best_score_generation", str(state.best_score_generation)
                )

        if spawned:
            logger.info(
                f"🏝️ Spawned new island due to stagnation "
                f"(no improvement for {gens_stagnant} generations)"
            )

        return spawned

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
                "SELECT embedding FROM programs WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'",
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
            SELECT id, embedding FROM programs 
            WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'
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
            SELECT id, embedding FROM programs 
            WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'
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
                SELECT id, embedding FROM programs 
                WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'
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
                return self._program_from_row(row)
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
            "SELECT id, embedding FROM programs "
            "WHERE embedding IS NOT NULL AND embedding != '[]'"
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
        embedding_client = self._ensure_embedding_client()
        if embedding_client is None:
            return

        # Use EmbeddingClient for dim reduction and clustering
        try:
            logger.info(
                "Recomputing PCA-reduced embedding features for %s programs.",
                len(program_ids),
            )
            reduced_2d = embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=2
            )
            reduced_3d = embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=3
            )
            cluster_ids = embedding_client.get_embedding_clusters(
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
                    UPDATE programs
                    SET embedding_pca_2d = ?,
                        embedding_pca_3d = ?,
                        embedding_cluster_id = ?
                    WHERE id = ?
                    """,
                    (
                        embedding_pca_2d_json,
                        embedding_pca_3d_json,
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
                "SELECT id, embedding FROM programs "
                "WHERE embedding IS NOT NULL AND embedding != '[]'"
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
            embedding_client = self._ensure_embedding_client()
            if embedding_client is None:
                return

            # Use EmbeddingClient for dim reduction and clustering
            try:
                logger.info(
                    "Recomputing PCA-reduced embedding features for %s programs.",
                    len(program_ids),
                )

                logger.info("Computing 2D PCA reduction...")
                reduced_2d = embedding_client.get_dim_reduction(
                    embeddings, method="pca", dims=2
                )
                logger.info("2D PCA reduction completed")

                logger.info("Computing 3D PCA reduction...")
                reduced_3d = embedding_client.get_dim_reduction(
                    embeddings, method="pca", dims=3
                )
                logger.info("3D PCA reduction completed")

                logger.info(f"Computing GMM clustering with {num_clusters} clusters...")
                cluster_ids = embedding_client.get_embedding_clusters(
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
                        UPDATE programs
                        SET embedding_pca_2d = ?,
                            embedding_pca_3d = ?,
                            embedding_cluster_id = ?
                        WHERE id = ?
                        """,
                        (
                            embedding_pca_2d_json,
                            embedding_pca_3d_json,
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

            programs = []
            for row in rows:
                if not row:
                    continue
                program_data = dict(row)
                # Manually handle JSON deserialization for thread safety
                for key, value in program_data.items():
                    if key in [
                        "public_metrics",
                        "private_metrics",
                        "metadata",
                        "archive_inspiration_ids",
                        "top_k_inspiration_ids",
                        "embedding",
                        "embedding_pca_2d",
                        "embedding_pca_3d",
                        "migration_history",
                    ] and isinstance(value, str):
                        try:
                            program_data[key] = json.loads(value)
                        except json.JSONDecodeError:
                            program_data[key] = {} if key.endswith("_metrics") else []
                programs.append(Program(**program_data))
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
            programs = []
            for row_data in all_rows:
                program_data = dict(row_data)

                # Manually handle JSON deserialization for thread safety
                json_fields = [
                    "public_metrics",
                    "private_metrics",
                    "metadata",
                    "archive_inspiration_ids",
                    "top_k_inspiration_ids",
                    "embedding",
                    "embedding_pca_2d",
                    "embedding_pca_3d",
                    "migration_history",
                ]
                for key, value in program_data.items():
                    if key in json_fields and isinstance(value, str):
                        try:
                            program_data[key] = json.loads(value)
                        except json.JSONDecodeError:
                            is_dict_field = (
                                key.endswith("_metrics") or key == "metadata"
                            )
                            program_data[key] = {} if is_dict_field else []

                # Handle text_feedback
                if (
                    "text_feedback" not in program_data
                    or program_data["text_feedback"] is None
                ):
                    program_data["text_feedback"] = ""

                programs.append(Program.from_dict(program_data))

            return programs

        finally:
            if conn:
                conn.close()

    def _get_programs_for_island(self, island_idx: int) -> List[Program]:
        """
        Get all programs for a specific island.
        """
