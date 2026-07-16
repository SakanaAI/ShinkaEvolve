"""
Async Evolution Runner for concurrent proposal generation and job management.
Provides fully asynchronous evolution pipeline with concurrent LLM sampling.
"""

import json
import asyncio
import logging
import shutil
import time
import uuid
import os
import math
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Union, Iterable
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
import rich.box

from shinka.database import ProgramDatabase, DatabaseConfig, Program
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.prompt_dbase import (
    SystemPromptDatabase,
    SystemPromptConfig,
    create_system_prompt,
)
from shinka.llm import (
    AsyncLLMClient,
    extract_between,
    BanditBase,
    FixedSampler,
    AsymmetricUCB,
    ThompsonSampler,
    RouteHealthCircuitBreaker,
    AsyncProviderRateLimiter,
    validate_daily_quota_feasibility,
)
from shinka.llm.providers import LLMRouteError
from shinka.embed import AsyncEmbeddingClient
from shinka.launch import JobScheduler, JobConfig, LocalJobConfig
from shinka.run_manifest import ensure_wandb_run_id, write_run_manifest
from shinka.edit.async_apply import (
    apply_patch_async,
    get_text_embedding_async,
    write_file_async,
)
from shinka.edit import summarize_diff
from shinka.core.sampler import PromptSampler
from shinka.core.summarizer import MetaSummarizer
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.core.novelty_judge import NoveltyJudge
from shinka.core.config import EvolutionConfig, FOLDER_PREFIX
from shinka.core.pipeline_timing import (
    summarize_timing_metadata,
    with_pipeline_timing,
    with_side_effect_timing,
)
from shinka.core.prompt_evolver import (
    SystemPromptSampler,
    AsyncSystemPromptEvolver,
)
from shinka.core.runtime_slots import LogicalSlotPool
from shinka.logo import BannerStyle, get_logo_ascii, print_gradient_logo
from shinka.model_availability import validate_model_env_access
from shinka.wandb_logging import (
    LOGGING_METHOD_WANDB,
    LOGGING_METHOD_WEBUI,
    ShinkaWandbLogger,
    resolve_logging_methods,
)
from shinka.utils import (
    get_language_extension,
    parse_time_to_seconds,
    truncate_log_tail,
)
from shinka.utils.languages import get_evolve_comment_prefix
from shinka.repo import (
    RepoWorktree,
    WorktreeManager,
    build_initial_summary,
    build_summary_template,
    validate_summary,
)

logger = logging.getLogger(__name__)


def _print_gradient_logo_and_mirror(
    log_path: Optional[Path] = None,
    banner_style: BannerStyle = "full",
) -> None:
    """Print gradient logo to terminal and mirror plain ASCII to log."""
    logo_ascii = get_logo_ascii(banner_style)
    print_gradient_logo((255, 0, 0), (255, 255, 255), logo_ascii=logo_ascii)
    if log_path is None:
        return

    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(logo_ascii if logo_ascii.endswith("\n") else f"{logo_ascii}\n")
    except Exception:
        # Never break startup output if log write fails.
        pass


class RichTeeConsole:
    """Mirror rich console output to terminal and a plain-text log file."""

    def __init__(self, console: Console, log_path: Optional[Path] = None):
        self._console = console
        self._log_path = log_path
        self._capture_console = Console(
            force_terminal=False,
            no_color=True,
            highlight=False,
            emoji=False,
            width=120,
        )
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self._console, name)

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self._console.print(*objects, **kwargs)
        if self._log_path is None:
            return

        try:
            with self._lock:
                with self._capture_console.capture() as capture:
                    self._capture_console.print(*objects, **kwargs)
                rendered = capture.get()
                if not rendered:
                    return
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        rendered if rendered.endswith("\n") else f"{rendered}\n"
                    )
        except Exception as e:
            logger.debug(f"Failed to mirror rich output to log file: {e}")


@dataclass
class AsyncRunningJob:
    """Async version of RunningJob with additional async metadata."""

    job_id: Union[str, Any]
    exec_fname: str
    results_dir: str
    start_time: float
    proposal_started_at: float
    evaluation_submitted_at: float
    generation: int
    individual_id: Optional[str] = None
    evaluation_started_at: Optional[float] = None
    sampling_worker_id: Optional[int] = None
    evaluation_worker_id: Optional[int] = None
    active_proposals_at_start: int = 0
    running_eval_jobs_at_submit: int = 0
    parent_id: Optional[str] = None
    archive_insp_ids: List[str] = field(default_factory=list)
    top_k_insp_ids: List[str] = field(default_factory=list)
    code_diff: Optional[str] = None
    meta_patch_data: Dict[str, Any] = field(default_factory=dict)
    code_embedding: Optional[List[float]] = None
    embed_cost: float = 0.0
    novelty_cost: float = 0.0  # Track novelty checking cost
    proposal_task_id: Optional[str] = None  # Track which proposal task created this job
    repo_path: Optional[str] = None
    repo_commit: Optional[str] = None
    repo_parent_commit: Optional[str] = None
    repo_summary: Optional[str] = None
    summary_version: Optional[str] = None
    changed_files: List[str] = field(default_factory=list)
    agent_session_id: Optional[str] = None
    agent_session_name: Optional[str] = None
    agent_provider: Optional[str] = None
    agent_model: Optional[str] = None
    db_retry_count: int = 0  # Track number of DB write retry attempts
    results_retrieved_at: Optional[float] = None
    completion_detected_at: Optional[float] = None
    discard_if_completed: bool = False
    evaluation_slot_released: bool = False


@dataclass
class PersistedProgramEvent:
    """Durably persisted program ready for slower follow-up side effects."""

    job: AsyncRunningJob
    program: Program
    evaluation_finished_at: float
    postprocess_started_at: float
    postprocess_finished_at: float


@dataclass
class CompletedJobPersistResult:
    """Result of the hot persistence path for one completed evaluation job."""

    job: AsyncRunningJob
    success: bool
    persisted_event: Optional[PersistedProgramEvent] = None


def _dedupe_model_names(model_names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


def _llm_kwargs_with_headless_work_dir(
    llm_kwargs: Dict[str, Any],
    results_dir: Path,
) -> Dict[str, Any]:
    return {"headless_work_dir": str(results_dir), **llm_kwargs}


def _validate_evo_config_model_env_access(evo_config: EvolutionConfig) -> None:
    # TODO: sample from different harnesses
    llm_models = list(evo_config.llm_models or [])

    if evo_config.meta_rec_interval and evo_config.meta_llm_models:
        llm_models.extend(evo_config.meta_llm_models)

    if evo_config.novelty_llm_models:
        llm_models.extend(evo_config.novelty_llm_models)

    if evo_config.evolve_prompts and evo_config.prompt_llm_models:
        llm_models.extend(evo_config.prompt_llm_models)

    embedding_models = (
        [evo_config.embedding_model] if evo_config.embedding_model else []
    )

    validate_model_env_access(
        llm_models=_dedupe_model_names(llm_models),
        embedding_models=_dedupe_model_names(embedding_models),
    )


class ShinkaEvolveRunner:
    """Fully async evolution runner with concurrent proposal generation."""

    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        banner_style: BannerStyle = "full",
        verbose: bool = True,
        max_evaluation_jobs: int = 4,
        max_proposal_jobs: int = 6,
        max_db_workers: int = 2,
        debug: bool = False,
        evaluate_str: Optional[str] = None,
    ):
        """Initialize async evolution runner.

        Args:
            evo_config: Evolution configuration
            job_config: Job configuration
            db_config: Database configuration
            verbose: Enable verbose logging
            max_evaluation_jobs: Maximum concurrent evaluation jobs
                (defaults to 4)
            max_proposal_jobs: Maximum concurrent proposal generation tasks
                (defaults to 6)
            max_db_workers: Maximum concurrent async DB worker threads
                (defaults to 2)
            evaluate_str: Optional string content for evaluate script
                (will be saved to results dir and path updated in job_config)
        """
        _validate_evo_config_model_env_access(evo_config)

        self.verbose = verbose
        # Setup results directory first
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results_{timestamp}"
        else:
            self.results_dir = Path(evo_config.results_dir)

        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.logging_methods = resolve_logging_methods(evo_config)
        self.webui_logging_enabled = LOGGING_METHOD_WEBUI in self.logging_methods
        self.wandb_logger = ShinkaWandbLogger(
            enabled=LOGGING_METHOD_WANDB in self.logging_methods
        )
        self.banner_style = banner_style
        self.enable_deadlock_debugging = debug
        log_filename = f"{self.results_dir}/evolution_run.log"

        if self.verbose:
            # Set up logging like the sync version
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

            # Configure logging with console output
            from rich.logging import RichHandler

            logging.basicConfig(
                level=logging.DEBUG if debug else logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    RichHandler(
                        show_time=False, show_level=False, show_path=False
                    ),  # Console output (clean)
                    logging.FileHandler(
                        log_filename, mode="a", encoding="utf-8"
                    ),  # File output (detailed)
                ],
                force=True,  # Override any existing logging config
            )
        else:
            # Ensure results directory exists even when not verbose
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        _print_gradient_logo_and_mirror(
            Path(log_filename), banner_style=self.banner_style
        )

        # Handle evaluate_str: write to file and update config path
        if evaluate_str is not None:
            evaluate_path = Path(self.results_dir) / "evaluate.py"
            evaluate_path.write_text(evaluate_str, encoding="utf-8")
            self.job_config.eval_program_path = str(evaluate_path)
            if self.verbose:
                logger.info(f"Saved evaluate_str to {evaluate_path}")

        # Validate and adjust concurrency settings based on available CPU cores
        cpu_count = os.cpu_count() or 4  # Default to 4 if can't detect

        # Apply intelligent constraints
        max_evaluation_jobs, max_proposal_jobs, max_db_workers = (
            self._validate_concurrency_settings(
                max_evaluation_jobs,
                max_proposal_jobs,
                max_db_workers,
                cpu_count,
            )
        )

        self.max_evaluation_jobs = max_evaluation_jobs
        self.max_proposal_jobs = max_proposal_jobs
        self.max_db_workers = max_db_workers
        self._configure_local_job_runtime(cpu_count)

        if self.evo_config.num_generations is None:
            assert self.evo_config.max_api_costs is not None, (
                "Max API costs must be specified if num_generations is not specified"
            )
            logger.info(
                f"No target generations specified, running indefinitely until cost limit of ${self.evo_config.max_api_costs:.2f} is reached"
            )
            self.evo_config.num_generations = int(1e6)

        logger.info("=" * 80)
        logger.info("ASYNC EVOLUTION RUN STARTED")
        logger.info("=" * 80)
        logger.info(f"Max evaluation jobs: {self.max_evaluation_jobs}")
        logger.info(f"Max proposal jobs: {self.max_proposal_jobs}")
        logger.info(f"Target generations: {self.evo_config.num_generations}")
        logger.info(f"Generation target mode: {self.evo_config.generation_target_mode}")
        logger.info(f"Language: {self.evo_config.language}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Log file: {log_filename}")
        logger.info(f"Logging methods: {sorted(self.logging_methods)}")
        if not self.webui_logging_enabled:
            logger.info(
                "WebUI logging disabled; SQLite persistence remains enabled "
                "because the evolutionary loop uses it as state."
            )
        if self.evo_config.max_api_costs is not None:
            logger.info(f"Max API costs: ${self.evo_config.max_api_costs:.2f}")
        logger.info("=" * 80)

        # Initialize rich console and mirror rich renderables into the run log.
        self.console = RichTeeConsole(Console(), Path(log_filename))

        # Initialize LLM selection strategy
        if evo_config.llm_dynamic_selection is None:
            self.llm_selection = None
        elif isinstance(evo_config.llm_dynamic_selection, BanditBase):
            self.llm_selection = evo_config.llm_dynamic_selection
        elif evo_config.llm_dynamic_selection.lower() == "fixed":
            self.llm_selection = FixedSampler(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        elif (evo_config.llm_dynamic_selection.lower() == "ucb") or (
            evo_config.llm_dynamic_selection.lower() == "ucb1"
        ):
            self.llm_selection = AsymmetricUCB(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        elif evo_config.llm_dynamic_selection.lower() == "thompson":
            self.llm_selection = ThompsonSampler(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        else:
            raise ValueError("Invalid llm_dynamic_selection")
        self.route_health = RouteHealthCircuitBreaker(
            failure_threshold=evo_config.route_failure_threshold,
            cooldown_seconds=evo_config.route_cooldown_seconds,
        )
        self.minimum_request_demand = validate_daily_quota_feasibility(evo_config)
        if evo_config.generation_target_mode not in {
            "evaluated_candidates",
            "proposal_ids",
        }:
            raise ValueError(
                "generation_target_mode must be 'evaluated_candidates' or "
                "'proposal_ids'"
            )
        self.llm_rate_limiter = AsyncProviderRateLimiter(
            limits=evo_config.llm_rate_limits,
            daily_quotas=evo_config.llm_daily_quotas,
        )

        # Store db_config for later initialization (after results_dir is set)
        # Database will be initialized in _setup_async()
        self.db = None
        self.async_db = None

        # LLM clients
        if not all(model.startswith("headless/") for model in evo_config.llm_models):
            raise ValueError(
                "llm_models must be a list of headless model strings, "
                f"got {evo_config.llm_models!r}"
            )
        self.llm = AsyncLLMClient(
            model_names=evo_config.llm_models,
            headless_timeout_seconds=evo_config.headless_proposal_timeout_seconds,
            headless_cleanup_grace_seconds=evo_config.headless_cleanup_grace_seconds,
            headless_output_mode=evo_config.headless_output_mode,
            headless_model_timeouts=evo_config.headless_model_timeouts,
            propagate_route_errors=True,
            rate_limiter=self.llm_rate_limiter,
            request_class="mutation",
            **_llm_kwargs_with_headless_work_dir(
                evo_config.llm_kwargs,
                Path(self.results_dir),
            ),
        )

        # Embedding client (use async version for async runner)
        if evo_config.embedding_model:
            self.embedding_client = AsyncEmbeddingClient(
                model_name=evo_config.embedding_model
            )
        else:
            self.embedding_client = None

        # Job scheduler
        self.scheduler = JobScheduler(
            job_type=evo_config.job_type, config=job_config, verbose=verbose
        )

        self.seed_repo_commit: Optional[str] = None

        seed_repo_path = evo_config.seed_repo_path
        if seed_repo_path is None:
            raise ValueError("seed_repo_path must be specified")
        worktree_root = evo_config.worktree_root or str(Path(self.results_dir) / "worktrees")
        self.repo_worktree_manager = WorktreeManager(
            seed_repo_path=seed_repo_path,
            worktree_root=worktree_root,
            mutable_paths=evo_config.mutable_paths,
            immutable_paths=evo_config.immutable_paths,
            hidden_paths=getattr(evo_config, "agent_hidden_paths", []),
            ignore_paths=evo_config.ignore_paths,
            base_ref=evo_config.base_ref,
            max_file_bytes=evo_config.max_file_bytes,
            allow_binary_files=evo_config.allow_binary_files,
            allow_deletions=evo_config.allow_deletions,
            allow_lockfile_changes=evo_config.allow_lockfile_changes,
        )

        # Prompt sampler
        self.prompt_sampler = PromptSampler(
            task_sys_msg=evo_config.task_sys_msg,
            language=evo_config.language,
            patch_types=evo_config.patch_types,
            patch_type_probs=evo_config.patch_type_probs,
            use_text_feedback=evo_config.use_text_feedback,
            inspiration_sort_order=evo_config.inspiration_sort_order,
        )

        # Meta summarizer (create both sync and async versions)
        if evo_config.meta_rec_interval and evo_config.meta_llm_models:
            # Create async LLM client for meta analysis
            async_meta_llm = AsyncLLMClient(
                model_names=evo_config.meta_llm_models or evo_config.llm_models,
                rate_limiter=self.llm_rate_limiter,
                request_class="meta",
                **_llm_kwargs_with_headless_work_dir(
                    evo_config.meta_llm_kwargs,
                    Path(self.results_dir),
                ),
            )
            # Create sync summarizer for state management
            sync_meta_summarizer = MetaSummarizer(
                meta_llm_client=None,  # Async version handles LLM calls
                language=evo_config.language,
                use_text_feedback=evo_config.use_text_feedback,
                max_recommendations=evo_config.meta_max_recommendations,
                async_mode=True,  # Enable async mode
            )
            # Create async wrapper
            self.meta_summarizer = AsyncMetaSummarizer(
                sync_meta_summarizer,
                async_meta_llm,
            )
        else:
            self.meta_summarizer = None

        # Novelty judge
        if evo_config.novelty_llm_models:
            novelty_llm = AsyncLLMClient(
                model_names=evo_config.novelty_llm_models,
                rate_limiter=self.llm_rate_limiter,
                request_class="novelty",
                **_llm_kwargs_with_headless_work_dir(
                    evo_config.novelty_llm_kwargs,
                    Path(self.results_dir),
                ),
            )
            sync_novelty_judge = NoveltyJudge(
                novelty_llm_client=None,  # We'll use async version
                language=evo_config.language,
                similarity_threshold=evo_config.code_embed_sim_threshold,
                max_novelty_attempts=evo_config.max_novelty_attempts,
            )
            self.novelty_judge = AsyncNoveltyJudge(
                sync_novelty_judge,
                novelty_llm,
            )
        else:
            self.novelty_judge = None

        # Meta-prompt evolution components
        # These will be initialized in _setup_async after results_dir is set
        self.prompt_db: Optional[SystemPromptDatabase] = None
        self.prompt_sampler_evo: Optional[SystemPromptSampler] = None
        self.prompt_evolver: Optional[AsyncSystemPromptEvolver] = None
        self.current_prompt_id: Optional[str] = None
        self.prompt_evolution_counter = 0  # Track programs since last prompt evolution
        self.prompt_percentile_recompute_counter = (
            0  # Track programs since last percentile recompute
        )
        self.prompt_api_cost = 0.0  # Track prompt evolution API costs separately

        # Initialize prompt evolution LLM client if enabled
        if evo_config.evolve_prompts:
            prompt_llm_models = evo_config.prompt_llm_models or evo_config.llm_models
            self.prompt_llm = AsyncLLMClient(
                model_names=prompt_llm_models,
                rate_limiter=self.llm_rate_limiter,
                request_class="prompt",
                **_llm_kwargs_with_headless_work_dir(
                    evo_config.prompt_llm_kwargs,
                    Path(self.results_dir),
                ),
            )
            logger.info(f"Prompt evolution enabled with models: {prompt_llm_models}")
        else:
            self.prompt_llm = None

        # Runtime state
        self.running_jobs: List[AsyncRunningJob] = []
        self.completed_generations = 0
        self.next_generation_to_submit = (
            1  # Start from generation 1 since 0 is handled in setup
        )
        self.assigned_generations: Set[int] = set()  # Track assigned gens
        self.best_program_id: Optional[str] = None
        self.lang_ext = get_language_extension(evo_config.language)
        # Async coordination
        self.slot_available = asyncio.Event()
        self.should_stop = asyncio.Event()
        self.finalization_complete = asyncio.Event()
        self.proposal_queue = asyncio.Queue()
        self.active_proposal_tasks: Dict[str, asyncio.Task] = {}

        # Performance tracking
        self.total_proposals_generated = 0
        self.total_api_cost = 0.0
        self.start_time = None

        # In-flight cost estimation for accurate budget enforcement
        self.completed_proposal_costs: List[
            float
        ] = []  # Track costs of completed proposals
        self.avg_proposal_cost = 0.0  # Running average cost per proposal
        self._sampling_seconds_ewma: Optional[float] = None
        self._evaluation_seconds_ewma: Optional[float] = None
        self._proposal_timing_samples = 0
        self._last_proposal_target_log: Optional[Tuple[int, float, float]] = None

        # Robust job tracking - ensure no jobs are lost
        self.submitted_jobs: Dict[str, AsyncRunningJob] = {}  # All jobs ever submitted
        self.processing_lock = asyncio.Lock()  # Prevent concurrent processing issues
        self.sampling_slot_pool = LogicalSlotPool(self.max_proposal_jobs, "sampling")
        self.evaluation_slot_pool = LogicalSlotPool(
            self.max_evaluation_jobs, "evaluation"
        )
        self.postprocess_slot_pool = LogicalSlotPool(self.max_db_workers, "postprocess")
        self.side_effect_event_queue: asyncio.Queue = asyncio.Queue()
        self._background_side_effect_task: Optional[asyncio.Task] = None
        self._background_side_effect_tasks: Set[asyncio.Task] = set()
        self._background_side_effects_pending = 0
        self._background_side_effects_busy = False
        self._background_side_effects_busy_count = 0
        self._completed_job_batch_tasks: Set[asyncio.Task] = set()
        self._completed_jobs_pending = 0
        self._meta_side_effect_lock = asyncio.Lock()
        self._prompt_side_effect_lock = asyncio.Lock()
        self._best_solution_lock = asyncio.Lock()
        self._prompt_percentile_recompute_task: Optional[asyncio.Task] = None
        self._prompt_percentile_recompute_pending = False

        # Database retry mechanism
        self.failed_jobs_for_retry: Dict[
            str, AsyncRunningJob
        ] = {}  # Jobs that failed DB write
        self.MAX_DB_RETRY_ATTEMPTS = (
            5  # Maximum number of retry attempts for DB operations
        )

        # Stuck detection and recovery
        self.last_progress_time = None
        self.stuck_detection_count = 0
        self.max_stuck_detections = 3  # Allow 3 stuck detections before giving up
        self.stuck_detection_timeout = 60.0  # 60 seconds without progress = stuck
        self.cost_limit_reached = False  # Track if we've hit the cost limit

        # Meta task logging state (to reduce verbosity)
        self._last_meta_log_state: dict | None = None
        self._last_meta_log_info_time: float | None = None

    def _save_bandit_state(self) -> None:
        """Save the LLM selection bandit state to disk."""
        if self.llm_selection is None:
            return
        try:
            bandit_path = Path(self.results_dir) / "bandit_state.pkl"
            self.llm_selection.save_state(bandit_path)
            logger.debug(f"Saved bandit state to {bandit_path}")
        except Exception as e:
            logger.warning(f"Failed to save bandit state: {e}")

    def _load_bandit_state(self) -> None:
        """Load the LLM selection bandit state from disk."""
        if self.llm_selection is None:
            return
        try:
            bandit_path = Path(self.results_dir) / "bandit_state.pkl"
            if bandit_path.exists():
                self.llm_selection.load_state(bandit_path)
                logger.info(f"Loaded bandit state from {bandit_path}")
                if hasattr(self.llm_selection, "print_summary"):
                    self.llm_selection.print_summary(console=self.console)
            else:
                logger.debug(
                    f"No bandit state file found at {bandit_path}, "
                    "starting with fresh bandit state"
                )
        except Exception as e:
            logger.warning(f"Failed to load bandit state: {e}")

    async def _record_generation_event(
        self,
        generation: int,
        status: str,
        source_job_id: Optional[Union[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Best-effort durable generation event logging."""
        if not getattr(self, "enable_deadlock_debugging", False):
            return
        if not hasattr(self.async_db, "record_generation_event_async"):
            return

        normalized_job_id = None
        if source_job_id is not None:
            normalized_job_id = str(source_job_id)

        try:
            await self.async_db.record_generation_event_async(
                generation=generation,
                status=status,
                source_job_id=normalized_job_id,
                details=details,
            )
        except Exception as e:
            logger.warning(
                "Failed to record generation event %s for gen %s: %s",
                status,
                generation,
                e,
            )

    async def _record_attempt_event(
        self,
        generation: int,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Best-effort durable attempt logging outside the programs table."""
        if not hasattr(self.async_db, "record_attempt_event_async"):
            return

        try:
            await self.async_db.record_attempt_event_async(
                generation=generation,
                stage=stage,
                status=status,
                details=details,
            )
        except Exception as e:
            logger.warning(
                "Failed to record attempt event %s/%s for gen %s: %s",
                stage,
                status,
                generation,
                e,
            )

    def _validate_concurrency_settings(
        self,
        max_evaluation_jobs: int,
        max_proposal_jobs: int,
        max_db_workers: int,
        cpu_count: int,
    ) -> Tuple[int, int, int]:
        """Validate and adjust concurrency settings based on available CPU cores."""

        # Get system memory info
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            memory_gb = 8  # Default assumption

        # Conservative approach: don't exceed CPU count for total active threads
        # Formula: evaluation_jobs + proposal_jobs + db_workers + overhead <= cpu_count * 1.5
        max_total_threads = int(cpu_count * 1.5)  # Allow some oversubscription

        # Memory-based constraints (each concurrent job can use ~200-500MB)
        memory_based_limit = max(1, int(memory_gb * 2))  # Conservative: 2 jobs per GB

        # Apply individual limits based on CPU and memory
        max_evaluation_jobs = min(max_evaluation_jobs, cpu_count, memory_based_limit)
        max_proposal_jobs = min(
            max_proposal_jobs, max(1, cpu_count // 2), memory_based_limit // 2
        )
        max_db_workers = min(
            max_db_workers, max(1, cpu_count // 2), 8
        )  # DB workers are less memory intensive

        # Check total thread usage
        total_threads = max_evaluation_jobs + max_proposal_jobs + max_db_workers

        if total_threads > max_total_threads:
            # Scale down proportionally while maintaining minimums
            scale_factor = max_total_threads / total_threads

            max_evaluation_jobs = max(1, int(max_evaluation_jobs * scale_factor))
            max_proposal_jobs = max(1, int(max_proposal_jobs * scale_factor))
            max_db_workers = max(1, int(max_db_workers * scale_factor))

            if self.verbose:
                logger.warning(
                    f"⚠️  Scaled down concurrency settings to fit {cpu_count} CPU cores:"
                )
                logger.warning(
                    f"   Total threads: {total_threads} → {max_evaluation_jobs + max_proposal_jobs + max_db_workers}"
                )

        if self.verbose:
            logger.info("🖥️  System resources detected:")
            logger.info(f"   • CPU cores: {cpu_count}")
            logger.info(f"   • Memory: {memory_gb:.1f} GB")
            logger.info("🔧 Concurrency settings:")
            logger.info(f"   • Evaluation jobs: {max_evaluation_jobs}")
            logger.info(f"   • Proposal jobs: {max_proposal_jobs}")
            logger.info(f"   • DB workers: {max_db_workers}")
            logger.info(
                f"   • Total threads: {max_evaluation_jobs + max_proposal_jobs + max_db_workers}"
            )

            # Warn if settings seem too high
            if max_evaluation_jobs + max_proposal_jobs > cpu_count:
                logger.warning(
                    "⚠️  High concurrency settings may cause CPU oversubscription"
                )
            if max_evaluation_jobs + max_proposal_jobs > memory_based_limit:
                logger.warning(
                    f"⚠️  High concurrency settings may cause memory pressure (limit: {memory_based_limit})"
                )

        return max_evaluation_jobs, max_proposal_jobs, max_db_workers

    def _configure_local_job_runtime(self, cpu_count: int) -> None:
        """Tune local evaluation subprocess runtime defaults for scaling."""
        if not isinstance(self.job_config, LocalJobConfig):
            return

        if self.job_config.numeric_threads_per_job is None:
            numeric_threads = max(1, cpu_count // max(1, self.max_evaluation_jobs))
            self.job_config.numeric_threads_per_job = numeric_threads
            if self.verbose:
                logger.info(
                    "Configured local numeric thread cap per eval process: %s",
                    numeric_threads,
                )

    async def _get_total_api_costs(self) -> float:
        """Calculate total API costs from all programs and prompt evolution."""

        def _compute_costs_thread_safe():
            """Thread-safe computation of total costs from database."""
            import sqlite3
            import json

            conn = None
            try:
                conn = sqlite3.connect(
                    self.db.config.db_path, check_same_thread=False, timeout=60.0
                )
                cursor = conn.cursor()

                # Get all metadata fields
                cursor.execute(
                    "SELECT metadata FROM programs WHERE metadata IS NOT NULL"
                )
                rows = cursor.fetchall()

                total_costs = 0.0
                for row in rows:
                    metadata_str = row[0]
                    if metadata_str:
                        try:
                            metadata = json.loads(metadata_str)
                            # Sum up all cost-related fields (handle None values)
                            api_cost = metadata.get("api_costs")
                            total_costs += api_cost if api_cost is not None else 0.0
                            embed_cost = metadata.get("embed_cost")
                            total_costs += embed_cost if embed_cost is not None else 0.0
                            novelty_cost = metadata.get("novelty_cost")
                            total_costs += (
                                novelty_cost if novelty_cost is not None else 0.0
                            )
                            meta_cost = metadata.get("meta_cost")
                            total_costs += meta_cost if meta_cost is not None else 0.0
                        except json.JSONDecodeError:
                            continue

                return total_costs
            finally:
                if conn:
                    conn.close()

        # Call thread-safe method through executor
        loop = asyncio.get_event_loop()
        total_costs = await loop.run_in_executor(None, _compute_costs_thread_safe)

        # Add prompt evolution costs if prompt evolution is enabled
        if self.prompt_db is not None:
            try:
                prompt_costs = self.prompt_db.get_total_evolution_costs()
                total_costs += prompt_costs
            except Exception as e:
                logger.warning(f"Failed to get prompt evolution costs: {e}")

        return total_costs

    def _update_avg_proposal_cost(self, proposal_cost: float) -> None:
        """Update the running average cost per proposal.

        Called when a proposal completes to track the average cost,
        which is used to estimate in-flight costs for budget enforcement.
        """
        self.completed_proposal_costs.append(proposal_cost)
        self.avg_proposal_cost = sum(self.completed_proposal_costs) / len(
            self.completed_proposal_costs
        )

    def _get_committed_cost(self) -> float:
        """Calculate the committed cost including estimated in-flight proposals.

        This provides a more accurate cost estimate for budget enforcement by
        accounting for proposals that are currently running but haven't reported
        their costs yet.

        Returns:
            Total committed cost = current cost + (active proposals * avg cost)
        """
        num_active_proposals = len(self.active_proposal_tasks)

        if num_active_proposals == 0:
            return self.total_api_cost

        # Use average cost if we have historical data, otherwise use a conservative estimate
        if self.avg_proposal_cost > 0:
            estimated_in_flight = num_active_proposals * self.avg_proposal_cost
        else:
            # No historical data yet - don't add estimates to avoid blocking early proposals
            estimated_in_flight = 0.0

        committed_cost = self.total_api_cost + estimated_in_flight
        return committed_cost

    def _record_oversubscription_timing_sample(self, metadata: Dict[str, Any]) -> None:
        """Update proposal/evaluation timing EWMAs for adaptive oversubscription."""
        sampling_seconds = metadata.get("sampling_seconds")
        evaluation_seconds = metadata.get("evaluation_seconds")
        if not isinstance(sampling_seconds, (int, float)) or not isinstance(
            evaluation_seconds, (int, float)
        ):
            return
        if sampling_seconds <= 0 or evaluation_seconds <= 0:
            return

        alpha = getattr(self.evo_config, "proposal_target_ewma_alpha", 0.3)
        if self._sampling_seconds_ewma is None:
            self._sampling_seconds_ewma = float(sampling_seconds)
        else:
            self._sampling_seconds_ewma = (
                alpha * float(sampling_seconds)
                + (1 - alpha) * self._sampling_seconds_ewma
            )

        if self._evaluation_seconds_ewma is None:
            self._evaluation_seconds_ewma = float(evaluation_seconds)
        else:
            self._evaluation_seconds_ewma = (
                alpha * float(evaluation_seconds)
                + (1 - alpha) * self._evaluation_seconds_ewma
            )

        self._proposal_timing_samples += 1

    def _compute_proposal_pipeline_target(self) -> int:
        """Compute the bounded proposal target for controlled oversubscription."""
        base_target = self.max_evaluation_jobs
        if not getattr(self.evo_config, "enable_controlled_oversubscription", False):
            return base_target

        buffer_max = max(0, getattr(self.evo_config, "proposal_buffer_max", 0))
        hard_cap = getattr(self.evo_config, "proposal_target_hard_cap", None)
        if hard_cap is not None and hard_cap < base_target:
            if not getattr(
                self, "_warned_invalid_proposal_target_hard_cap", False
            ):
                logger.warning(
                    "Ignoring proposal_target_hard_cap=%s because it is below "
                    "max_evaluation_jobs=%s and would disable oversubscription.",
                    hard_cap,
                    base_target,
                )
                self._warned_invalid_proposal_target_hard_cap = True
            hard_cap = None
        effective_hard_cap = (
            self.max_proposal_jobs
            if hard_cap is None
            else min(hard_cap, self.max_proposal_jobs)
        )

        mode = getattr(self.evo_config, "proposal_target_mode", "adaptive")
        if mode == "fixed":
            raw_target = base_target + buffer_max
        else:
            min_samples = max(
                1, getattr(self.evo_config, "proposal_target_min_samples", 5)
            )
            ratio_cap = max(
                1.0, getattr(self.evo_config, "proposal_target_ratio_cap", 2.0)
            )
            if (
                self._proposal_timing_samples < min_samples
                or self._sampling_seconds_ewma is None
                or self._evaluation_seconds_ewma is None
                or self._evaluation_seconds_ewma <= 0
            ):
                raw_target = base_target + min(1, buffer_max)
            else:
                observed_ratio = min(
                    self._sampling_seconds_ewma / self._evaluation_seconds_ewma,
                    ratio_cap,
                )
                raw_target = math.ceil(base_target * observed_ratio)

        final_target = max(
            base_target,
            min(
                raw_target,
                base_target + buffer_max,
                self.max_proposal_jobs,
                effective_hard_cap,
            ),
        )
        return final_target

    def _log_proposal_target_decision(self, pipeline_target: int) -> None:
        """Log proposal target changes with current timing stats."""
        sampling_ewma = self._sampling_seconds_ewma or 0.0
        evaluation_ewma = self._evaluation_seconds_ewma or 0.0
        log_state = (
            pipeline_target,
            round(sampling_ewma, 3),
            round(evaluation_ewma, 3),
        )
        if self._last_proposal_target_log == log_state:
            return
        self._last_proposal_target_log = log_state
        logger.info(
            "Proposal target=%s (sampling_ewma=%.2fs, evaluation_ewma=%.2fs, "
            "timing_samples=%s, active_proposals=%s, running_jobs=%s)",
            pipeline_target,
            sampling_ewma,
            evaluation_ewma,
            self._proposal_timing_samples,
            len(self.active_proposal_tasks),
            len(self.running_jobs),
        )

    def run(self):
        """Synchronous convenience wrapper for script/CLI usage."""
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Event loop already running. Use `await runner.run_async()` in async contexts."
                )
        except RuntimeError as exc:
            # asyncio.get_running_loop raises RuntimeError when no loop exists.
            if "no running event loop" not in str(exc):
                raise
        asyncio.run(self.run_async())

    async def run_async(self):
        """Main async evolution loop."""
        self.start_time = time.time()
        self.last_progress_time = self.start_time  # Initialize progress tracking
        tasks = []  # Initialize tasks list to avoid UnboundLocalError

        try:
            # Setup initial program (results_dir now set)
            await self._setup_async()

            # Ensure database is ready for sampling before starting proposal
            await self._verify_database_ready()

            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self._job_monitor_task(), name="job_monitor"),
                asyncio.create_task(
                    self._proposal_coordinator_task(), name="proposal_coordinator"
                ),
            ]

            # Add meta summarizer task if enabled
            if self.meta_summarizer:
                tasks.append(
                    asyncio.create_task(
                        self._meta_summarizer_task(), name="meta_summarizer"
                    )
                )

            # Wait for the finalization signal instead of gathering all tasks
            await self.finalization_complete.wait()

            await self._wait_for_completed_job_batches()

            if self._has_background_side_effect_work():
                if self.verbose:
                    logger.info(
                        "Draining %s queued background side effect(s) before finalization...",
                        self._get_background_side_effect_work_count(),
                    )
                await self._wait_for_background_side_effects()
            await self._shutdown_background_side_effect_worker()
            if self._prompt_percentile_recompute_task is not None:
                await asyncio.gather(
                    self._prompt_percentile_recompute_task,
                    return_exceptions=True,
                )

            # Perform final operations before cleanup
            if self.verbose:
                logger.info(
                    "🔄 Performing final embedding recomputation and meta summary..."
                )

            # Force final embedding recomputation before shutdown
            if self.embedding_client:
                try:
                    if self.verbose:
                        logger.info("Starting final PCA/embedding recomputation...")
                        logger.info("⚠️  This may take a while for large datasets...")

                    # Add timeout to prevent infinite blocking - reduced timeout for safety
                    try:
                        await asyncio.wait_for(
                            self.async_db.force_recompute_embeddings_async(),
                            timeout=120.0,  # 2 minute timeout (reduced from 5 minutes)
                        )
                        if self.verbose:
                            logger.info(
                                "Final PCA/embedding recomputation completed successfully"
                            )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "⚠️  Final embedding recomputation timed out after 2 minutes"
                        )
                        logger.warning(
                            "   This is often due to large dataset PCA/clustering computation"
                        )
                        logger.warning(
                            "   Evolution results are still valid, embeddings just not updated"
                        )
                        logger.warning(
                            "   Proceeding to finalization to avoid hanging..."
                        )

                except Exception as e:
                    logger.error(f"Error in final embedding recomputation: {e}")
                    logger.error(
                        "   Evolution results are still valid, embeddings just not updated"
                    )

            # Perform final meta summary for any remaining unprocessed programs
            if self.meta_summarizer:
                try:
                    if self.verbose:
                        logger.info("Starting final meta summary generation...")

                    # Add timeout to meta summary operations
                    try:
                        best_program = await asyncio.wait_for(
                            self.async_db.get_best_program_async(),
                            timeout=30.0,  # 30 second timeout for getting best program
                        )
                        if best_program:
                            # Run async meta summary with timeout
                            success, final_meta_cost = await asyncio.wait_for(
                                self.meta_summarizer.perform_final_summary_async(
                                    str(self.results_dir),
                                    best_program,
                                    self.db.config,
                                ),
                                timeout=600.0,  # 10 minute timeout for final meta summary
                            )
                            if success and final_meta_cost > 0:
                                self.total_api_cost += final_meta_cost
                            if self.verbose:
                                if success and final_meta_cost > 0:
                                    logger.info(
                                        f"Final meta summary completed successfully "
                                        f"(cost: ${final_meta_cost:.4f})"
                                    )
                                else:
                                    logger.info(
                                        "Final meta summary completed successfully"
                                    )
                        else:
                            logger.warning(
                                "No best program found for final meta summary"
                            )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "⚠️  Final meta summary timed out, proceeding to cleanup..."
                        )

                except Exception as e:
                    logger.error(f"❌ Error in final meta summary: {e}")

            # Small delay to ensure all final operations are fully complete
            await asyncio.sleep(0.5)

            # Save final bandit state
            self._save_bandit_state()

            if self.verbose:
                logger.info(
                    "🏁 All final operations completed, proceeding to cleanup..."
                )

        except Exception as e:
            logger.error(f"Error in async evolution run: {e}")
            raise
        finally:
            await self._cancel_completed_job_batches()
            await self._cancel_background_side_effect_worker()
            # Ensure all tasks are cancelled on exit
            for task in tasks:
                task.cancel()
            if tasks:  # Only gather if there are tasks
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._cleanup_async()

        # Print final summary
        await self._print_final_summary()

    async def _setup_async(self):
        """Setup initial program (results directory already created)."""
        # Update database path to be in results directory
        db_path = Path(f"{self.results_dir}/programs.sqlite")

        # Update database config with results directory path
        self.db_config.db_path = str(db_path)

        # Reinitialize database with updated path
        self.db = ProgramDatabase(
            self.db_config, embedding_model=self.evo_config.embedding_model
        )
        if hasattr(self.db, "set_display_console"):
            self.db.set_display_console(self.console)
        self.async_db = AsyncProgramDatabase(
            self.db,
            max_workers=self.max_db_workers,
            enable_deadlock_debugging=self.enable_deadlock_debugging,
        )

        # Initialize prompt evolution database if enabled
        if self.evo_config.evolve_prompts:
            await self._setup_prompt_evolution()

        self.evo_config.wandb_run_id = ensure_wandb_run_id(
            Path(self.results_dir),
            self.evo_config.wandb_run_id,
        )
        write_run_manifest(
            evo_config=self.evo_config,
            db_config=self.db_config,
            job_config=self.job_config,
            results_dir=Path(self.results_dir),
            effective_workers={
                "proposal": self.max_proposal_jobs,
                "evaluation": self.max_evaluation_jobs,
                "database": self.max_db_workers,
            },
            minimum_request_demand=self.minimum_request_demand,
        )
        self.wandb_logger.start(
            evo_config=self.evo_config,
            db_config=self.db_config,
            job_config=self.job_config,
            results_dir=Path(self.results_dir),
        )

        # Check if we're resuming from an existing database
        resuming_run = db_path.exists() and self.db.last_iteration > 0

        # Load bandit state if resuming
        if resuming_run:
            logger.info("=" * 80)
            logger.info("RESUMING PREVIOUS ASYNC EVOLUTION RUN")
            logger.info("=" * 80)
            logger.info(f"Resuming from generation {self.db.last_iteration}")
            program_count = await self.async_db.get_total_program_count_async()
            logger.info(f"Found {program_count} programs in database")

            # Load existing API costs from database
            existing_costs = await self._get_total_api_costs()
            self.total_api_cost = existing_costs
            logger.info(f"Loaded existing API costs: ${existing_costs:.4f}")

            logger.info("=" * 80)
            self._load_bandit_state()

            # Update state for resuming
            await self._restore_resume_progress()
        else:
            # Copy initial repo only if NOT resuming
            await self._setup_initial_repo()

    async def _setup_prompt_evolution(self):
        """Setup prompt evolution database and components."""
        # Create prompt database path
        prompt_db_path = Path(f"{self.results_dir}/prompts.sqlite")

        # Create prompt database config
        prompt_config = SystemPromptConfig(
            db_path=str(prompt_db_path),
            archive_size=self.evo_config.prompt_archive_size,
            ucb_exploration_constant=self.evo_config.prompt_ucb_exploration_constant,
            epsilon=self.evo_config.prompt_epsilon,
        )

        # Initialize prompt database
        self.prompt_db = SystemPromptDatabase(prompt_config)

        # Check if we're resuming from existing prompt database
        if prompt_db_path.exists() and self.prompt_db.last_generation > 0:
            logger.info(
                f"Resuming prompt evolution from generation "
                f"{self.prompt_db.last_generation}"
            )
            prompt_count = self.prompt_db._count_prompts_in_db()
            logger.info(f"Found {prompt_count} prompts in database")
        else:
            # Add initial prompt to database
            initial_prompt_text = (
                self.evo_config.task_sys_msg or "You are an expert software engineer."
            )
            initial_prompt = create_system_prompt(
                prompt_text=initial_prompt_text,
                generation=0,
                patch_type="init",
                metadata={"source": "initial_config"},
                name="initial_system_prompt",
                description="Initial system prompt provided by the user.",
            )
            self.prompt_db.add(initial_prompt, verbose=self.verbose)
            logger.info(f"Added initial prompt {initial_prompt.id[:8]}... to database")

        # Initialize prompt sampler
        self.prompt_sampler_evo = SystemPromptSampler(
            prompt_db=self.prompt_db,
            exploration_constant=self.evo_config.prompt_ucb_exploration_constant,
            epsilon=self.evo_config.prompt_epsilon,
        )

        # Initialize prompt evolver
        self.prompt_evolver = AsyncSystemPromptEvolver(
            llm_client=self.prompt_llm,
            patch_types=self.evo_config.prompt_patch_types,
            patch_type_probs=self.evo_config.prompt_patch_type_probs,
            llm_kwargs=self.evo_config.prompt_llm_kwargs,
        )

        logger.info(
            f"Prompt evolution initialized with archive size "
            f"{self.evo_config.prompt_archive_size}"
        )

    def _get_current_system_prompt(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the current system prompt text and ID.

        Returns:
            Tuple of (prompt_text, prompt_id)
            prompt_text may be None if no task_sys_msg is configured
        """
        if not self.evo_config.evolve_prompts or not self.prompt_sampler_evo:
            # Return fixed prompt from config
            return self.evo_config.task_sys_msg, None

        # Sample a prompt from the archive
        sampled_prompt = self.prompt_sampler_evo.sample()
        if sampled_prompt:
            logger.debug(
                f"Using prompt {sampled_prompt.id[:8]}... "
                f"(fitness={sampled_prompt.fitness:.4f})"
            )
            return sampled_prompt.prompt_text, sampled_prompt.id

        # Fallback to config prompt if no prompts in archive
        logger.warning("No prompts in archive, using config prompt")
        return self.evo_config.task_sys_msg, None

    async def _update_prompt_fitness(
        self,
        prompt_id: Optional[str],
        program_id: str,
        program_score: float,
        improvement: float,
        correct: bool = True,
    ):
        """
        Update the fitness of a prompt based on program performance.

        Uses percentile-based fitness which is scale-invariant and automatically
        adjusts for performance saturation. A prompt's fitness represents the
        average percentile rank of programs generated with it.

        Args:
            prompt_id: ID of the prompt used to generate the program
            program_id: ID of the generated program
            program_score: The absolute score of the program (combined_score)
            improvement: Score improvement (child_score - parent_score), kept for logging
            correct: Whether the program was correct. Only correct programs
                     contribute to fitness calculation.
        """
        if not prompt_id or not self.prompt_db:
            return

        try:
            # Compute the percentile rank of this program's score
            # Only compute percentile for correct programs to avoid noise
            if correct:
                percentile = await self.async_db.compute_percentile_async(
                    program_score, correct_only=True
                )
            else:
                percentile = 0.0  # Incorrect programs get 0 percentile

            self.prompt_db.update_fitness(
                prompt_id=prompt_id,
                percentile=percentile,
                program_id=program_id,
                correct=correct,
                improvement=improvement,  # Keep for backward compat/logging
                program_score=program_score,  # Store for percentile recomputation
            )
            logger.debug(
                f"Updated prompt {prompt_id[:8]}... fitness with "
                f"percentile={percentile:.4f} (score={program_score:.4f}, "
                f"improvement={improvement:.4f}, correct={correct})"
            )

            # Periodically recompute all prompt percentiles to avoid stale fitness values
            # As population grows, old percentiles become outdated
            if correct:
                self.prompt_percentile_recompute_counter += 1
            recompute_interval = self.evo_config.prompt_percentile_recompute_interval
            if (
                correct
                and recompute_interval > 0
                and self.prompt_percentile_recompute_counter >= recompute_interval
            ):
                self.prompt_percentile_recompute_counter = 0
                self._schedule_prompt_percentile_recompute(recompute_interval)

        except Exception as e:
            logger.error(f"Failed to update prompt fitness: {e}")

    def _schedule_prompt_percentile_recompute(self, recompute_interval: int) -> None:
        """Debounce global prompt percentile recomputation onto a background task."""
        if not self.prompt_db:
            return

        task = self._prompt_percentile_recompute_task
        if task is not None and not task.done():
            self._prompt_percentile_recompute_pending = True
            return

        self._prompt_percentile_recompute_pending = False
        self._prompt_percentile_recompute_task = asyncio.create_task(
            self._recompute_prompt_percentiles_async(recompute_interval),
            name="prompt_percentile_recompute",
        )

    async def _recompute_prompt_percentiles_async(self, recompute_interval: int) -> None:
        """Refresh prompt fitness percentiles without blocking side-effect workers."""
        try:
            loop = asyncio.get_event_loop()
            if hasattr(self.db, "config"):

                def load_program_scores_thread_safe() -> Tuple[List[float], Dict[str, float]]:
                    thread_db = None
                    try:
                        thread_db = ProgramDatabase(self.db.config, read_only=True)
                        all_programs = thread_db.get_all_programs()
                        all_correct_scores = [
                            p.combined_score
                            for p in all_programs
                            if p.correct and p.combined_score is not None
                        ]
                        program_id_to_score = {
                            p.id: p.combined_score
                            for p in all_programs
                            if p.correct and p.combined_score is not None
                        }
                        return all_correct_scores, program_id_to_score
                    finally:
                        if thread_db is not None:
                            thread_db.close()

                all_correct_scores, program_id_to_score = await loop.run_in_executor(
                    None, load_program_scores_thread_safe
                )
            else:
                all_programs = self.db.get_all_programs()
                all_correct_scores = [
                    p.combined_score
                    for p in all_programs
                    if p.correct and p.combined_score is not None
                ]
                program_id_to_score = {
                    p.id: p.combined_score
                    for p in all_programs
                    if p.correct and p.combined_score is not None
                }
            self.prompt_db.recompute_all_percentiles(
                all_correct_scores, program_id_to_score
            )
            logger.info(
                "Recomputed prompt fitness percentiles "
                "(every %s programs, using %s correct program scores)",
                recompute_interval,
                len(all_correct_scores),
            )
        except Exception as recompute_err:
            logger.warning(
                "Failed to recompute prompt percentiles: %s", recompute_err
            )
        finally:
            rerun_requested = self._prompt_percentile_recompute_pending
            self._prompt_percentile_recompute_pending = False
            self._prompt_percentile_recompute_task = None
            if rerun_requested:
                self._schedule_prompt_percentile_recompute(recompute_interval)

    async def _maybe_evolve_prompt(self):
        """
        Check if we should evolve a new prompt and do so if needed.

        This is triggered based on prompt_evolution_interval.
        """
        if not self.evo_config.evolve_prompts:
            return

        interval = self.evo_config.prompt_evolution_interval
        if interval is None:
            return

        self.prompt_evolution_counter += 1

        if self.prompt_evolution_counter < interval:
            return

        # Reset counter
        self.prompt_evolution_counter = 0

        logger.info("Triggering prompt evolution...")

        try:
            # Get parent prompt
            parent_prompt = self.prompt_sampler_evo.sample()
            if not parent_prompt:
                logger.warning("No parent prompt available for evolution")
                return

            # Get top-k programs for context
            top_k = self.evo_config.prompt_evo_top_k_programs
            top_programs = await self.async_db.get_top_programs_async(top_k)

            logger.info(
                f"Got {len(top_programs)} top programs for prompt evolution context"
            )

            # Get next prompt generation (chronological counter)
            next_prompt_generation = self.prompt_db.last_generation + 1

            # Get current program generation for tracking
            current_program_generation = self.completed_generations

            # Get global scratchpad from meta-summarizer if available
            global_scratchpad = None
            if self.meta_summarizer:
                _, _, global_scratchpad = self.meta_summarizer.get_current()
                if global_scratchpad:
                    logger.debug("Including global scratchpad in prompt evolution")

            # Evolve new prompt
            new_prompt, patch_type, cost = await self.prompt_evolver.evolve(
                parent_prompt=parent_prompt,
                next_generation=next_prompt_generation,
                program_generation=current_program_generation,
                top_programs=top_programs,
                language=self.evo_config.language,
                include_text_feedback=self.evo_config.use_text_feedback,
                global_scratchpad=global_scratchpad,
            )

            self.prompt_api_cost += cost
            self.total_api_cost += cost

            if new_prompt:
                self.prompt_db.add(new_prompt, verbose=self.verbose)
                logger.info(
                    f"Evolved new prompt {new_prompt.id[:8]}... "
                    f"(prompt_gen={new_prompt.generation}, prog_gen={current_program_generation}, "
                    f"patch={patch_type}, cost=${cost:.4f})"
                )
            else:
                logger.warning(f"Prompt evolution failed (patch_type={patch_type})")

        except Exception as e:
            logger.error(f"Error during prompt evolution: {e}")

    async def _get_text_embedding_async(
        self, text: str
    ) -> Tuple[Optional[List[float]], float]:
        """Get text embedding asynchronously."""
        if not self.embedding_client or not text:
            return None, 0.0

        async with self.llm_rate_limiter.limit(
            self.evo_config.embedding_model,
            "embedding",
        ):
            return await get_text_embedding_async(text, self.embedding_client)

    async def _setup_initial_repo(self):
        """Setup generation 0 for repo-backed evolution."""
        pipeline_started_at = time.time()
        parent_commit = self.repo_worktree_manager.initialize_seed_repo()
        self.seed_repo_commit = parent_commit
        individual_id = str(uuid.uuid4())
        worktree = self.repo_worktree_manager.create_child_worktree(
            parent_commit=parent_commit,
            generation=0,
            individual_id=individual_id,
        )
        repo_path = worktree.path
        summary_path = repo_path / self.evo_config.summary_filename
        if summary_path.exists():
            summary_text = summary_path.read_text(encoding="utf-8")
            summary_check = validate_summary(
                summary_text,
                max_chars=self.evo_config.summary_max_chars,
            )
            summary_version = summary_check.schema_version
        else:
            summary_text = build_initial_summary(
                individual_id="initial_program",
                generation=0,
                commit_sha=parent_commit,
            )
            summary_version = "repo-individual-v1"
            summary_path.parent.mkdir(parents=True, exist_ok=True) # TODO: should be able to remove
            summary_path.write_text(summary_text, encoding="utf-8")

        gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"
        results_dir = f"{gen_dir}/results"
        Path(gen_dir).mkdir(parents=True, exist_ok=True)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        summary_artifact_path = Path(gen_dir) / "individual.md"
        await write_file_async(str(summary_artifact_path), summary_text)

        evaluation_failed = False
        # Run initial repo evaluation to get proper metrics
        try:
            if self.verbose:
                logger.info(f"Starting initial repo evaluation: {worktree.path}")
            
            # Run the evaluation synchronously for generation 0
            loop = asyncio.get_event_loop()
            evaluation_started_at = time.time()
            results, rtime = await loop.run_in_executor(
                None,
                self.scheduler.run,
                str(worktree.path),
                results_dir,
                str(worktree.path),
            )
            evaluation_finished_at = time.time()
            postprocess_started_at = evaluation_finished_at

            if self.verbose:
                logger.info(f"Initial repo evaluation completed in {rtime:.2f}s")

        except Exception as e:
            logger.warning(f"Initial repo evaluation failed: {e}")
            evaluation_finished_at = time.time()
            postprocess_started_at = evaluation_finished_at
            results = {"correct": {"correct": False}, "metrics": {}}
            rtime = 0.0
            evaluation_failed = True

        code_embedding, e_cost = await self._get_text_embedding_async(summary_text)
        if self.verbose and code_embedding:
            logger.info(f"Initial repo embedding computed (cost: ${e_cost:.4f})")

        # Extract metrics properly like the sync version.
        correct_val = results.get("correct", {}).get("correct", False)
        metrics_val = results.get("metrics", {})
        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")
        stdout_log = truncate_log_tail(
            results.get("stdout_log", ""),
            self.db_config.max_stdout_log_chars,
        )
        stderr_log = results.get("stderr_log", "")

        metadata = with_pipeline_timing(
            {
                "api_costs": 0.0,
                "embed_cost": e_cost,
                "novelty_cost": 0.0,
                "evaluation_failed": evaluation_failed,
                "patch_type": "repo_init",
                "patch_name": "initial_program",
                "patch_description": "Initial repository setup",
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "repo_path": str(worktree.path),
                "worktree_path": str(worktree.path),
                "summary_artifact_path": str(summary_artifact_path),
                "timeline_lane_mode": "pool_slots",
                "sampling_worker_capacity": self.max_proposal_jobs,
                "evaluation_worker_capacity": self.max_evaluation_jobs,
                "postprocess_worker_capacity": self.max_db_workers,
            },
            pipeline_started_at=pipeline_started_at,
            sampling_started_at=pipeline_started_at,
            sampling_finished_at=evaluation_started_at,
            evaluation_started_at=evaluation_started_at,
            evaluation_finished_at=evaluation_finished_at,
            postprocess_started_at=postprocess_started_at,
            postprocess_finished_at=postprocess_started_at,
        )

        initial_program = Program(
            id=individual_id,
            parent_id=None,
            generation=0,
            repo_commit=parent_commit,
            repo_parent_commit=None,
            repo_diff="",
            repo_summary=summary_text,
            summary_version=summary_version,
            changed_files=[],
            artifact_uri=str(worktree.path),
            mutable_paths=self.evo_config.mutable_paths,
            immutable_paths=self.evo_config.immutable_paths,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            correct=correct_val if not evaluation_failed else True,
            text_feedback=text_feedback,
            timestamp=datetime.now().timestamp(),
            embedding=code_embedding or [],
            metadata=metadata,
        )

        if self.verbose and not evaluation_failed:
            logger.info(f"Initial repo evaluated - correct: {initial_program.correct}, "
                        f"combined_score: {initial_program.combined_score}")

        await self.async_db.add_program_async(initial_program, verbose=self.verbose)
        self.total_api_cost += e_cost

        # Add the initial repo to meta memory tracking
        if self.meta_summarizer:
            self.meta_summarizer.add_evaluated_program(initial_program)

            # Check if we should update meta memory after adding this repo
            if self.meta_summarizer.should_update_meta(
                self.evo_config.meta_rec_interval
            ):
                logger.info(
                    f"Updating meta memory after processing "
                    f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
                )
                best_program = await self.async_db.get_best_program_async()
                # Use async meta summarizer for non-blocking meta analysis
                (
                    updated_recs,
                    meta_cost,
                ) = await self.meta_summarizer.update_meta_memory_async(best_program)
                if updated_recs:
                    # Write meta output file asynchronously
                    await self.meta_summarizer.write_meta_output_async(
                        str(self.results_dir)
                    )
                    # Store meta cost for tracking
                    if meta_cost > 0:
                        logger.info(
                            f"Meta recommendation generation cost: ${meta_cost:.4f}"
                        )
                        # Add meta cost to in-memory total for accurate budget tracking
                        self.total_api_cost += meta_cost

                        # Add meta cost to this repo's metadata (the one that triggered the update)
                        if initial_program.metadata is None:
                            initial_program.metadata = {}
                        initial_program.metadata["meta_cost"] = meta_cost

        # Set baseline score for LLM selection
        if self.llm_selection is not None:
            self.llm_selection.set_baseline_score(
                initial_program.combined_score if initial_program.correct else 0.0,
            )

        # Mark generation 0 as completed
        self.completed_generations = 1

        # Record progress after initial setup
        self._record_progress()

        postprocess_finished_at = time.time()
        initial_program.metadata = with_pipeline_timing(
            initial_program.metadata,
            pipeline_started_at=pipeline_started_at,
            sampling_started_at=pipeline_started_at,
            sampling_finished_at=pipeline_started_at,
            evaluation_started_at=evaluation_started_at,
            evaluation_finished_at=evaluation_finished_at,
            postprocess_started_at=postprocess_started_at,
            postprocess_finished_at=postprocess_finished_at,
        )
        await self._persist_program_metadata_async(initial_program)
        self._log_program_to_wandb(initial_program)

        if self.verbose:
            logger.info(f"Setup initial repo: {initial_program.id}")
            logger.info("Generation 0 completed during setup")


    async def _verify_database_ready(self):
        """Verify that the database is ready for sampling with programs."""
        if self.verbose:
            logger.info("Verifying database is ready for sampling...")

        try:
            # Use a simple count check instead of sample_async() to avoid
            # printing the sampling summary table during verification
            program_count = await self.async_db.get_total_program_count_async()

            if program_count > 0:
                if self.verbose:
                    logger.info(
                        f"Database ready - {program_count} program(s) available for sampling"
                    )
            else:
                raise RuntimeError("Database sampling failed - no programs found")

        except Exception as e:
            logger.error(f"Database not ready for sampling: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")

        if self.verbose:
            logger.info(
                "Database verification completed - ready for proposal generation"
            )


    async def _job_monitor_task(self):
        """Monitor running jobs and process completed ones."""
        logger.info("🔄 Job monitor task started")

        while not self.should_stop.is_set():
            try:
                completed_jobs = []
                if not self.running_jobs:
                    logger.debug(
                        "🔍 Job monitor idle: completed_gens=%s, target=%s, "
                        "running_jobs=0, active_proposals=%s",
                        self.completed_generations,
                        self.evo_config.num_generations,
                        len(self.active_proposal_tasks),
                    )
                else:
                    monitored_jobs = list(self.running_jobs)
                    # Check job statuses concurrently
                    status_results = await self.scheduler.batch_check_status_async(
                        monitored_jobs
                    )
                    if self.verbose:
                        # Create safe status display to avoid race conditions
                        try:
                            status_display = []
                            for i, job in enumerate(monitored_jobs):
                                if i < len(status_results):
                                    status_display.append(
                                        f"{job.generation} - {status_results[i]}"
                                    )
                                else:
                                    status_display.append(
                                        f"{job.generation} - unknown"
                                    )

                            logger.debug(
                                f"Job statuses ({len(monitored_jobs)}): gen [{', '.join(status_display)}]"
                            )
                            logger.debug(
                                f"Active proposal jobs ({len(self.active_proposal_tasks)}): gen [{', '.join([task.get_name().split('_')[1] if task.get_name().startswith('proposal_') else 'unknown' for task in self.active_proposal_tasks.values()])}]"
                            )
                        except Exception as e:
                            logger.warning(f"Error in status logging: {e}")
                            logger.debug(
                                f"Running jobs: {len(self.running_jobs)}, Active proposals: {len(self.active_proposal_tasks)}"
                            )
                    still_running = []
                    current_running_jobs = list(self.running_jobs)
                    current_job_ids = {id(job) for job in current_running_jobs}
                    monitored_job_ids = {id(job) for job in monitored_jobs}
                    concurrently_added_jobs = [
                        job
                        for job in current_running_jobs
                        if id(job) not in monitored_job_ids
                    ]

                    for job, is_running in zip(monitored_jobs, status_results):
                        if id(job) not in current_job_ids:
                            continue
                        if isinstance(is_running, Exception):
                            logger.warning(
                                f"Error checking job {job.job_id}: {is_running}"
                            )
                            still_running.append(job)
                        elif not is_running:
                            completed_jobs.append(job)
                            runtime = time.time() - job.start_time
                            if self.verbose:
                                logger.info(
                                    f"✅ Job {job.job_id} completed (gen {job.generation}) after {runtime:.1f}s"
                                )
                        elif self._is_job_hung(job):
                            runtime = time.time() - (
                                job.evaluation_started_at
                                or job.evaluation_submitted_at
                                or job.start_time
                            )
                            logger.warning(
                                f"⏱️  Hung job detected for gen {job.generation}: "
                                f"runtime={runtime:.1f}s. Cancelling for recovery."
                            )
                            cancelled = await self.scheduler.cancel_job_async(
                                job.job_id
                            )
                            if not cancelled:
                                logger.warning(
                                    f"Failed to cancel hung job {job.job_id} "
                                    f"(gen {job.generation}); keeping it running"
                                )
                                still_running.append(job)
                                continue
                            completed_jobs.append(job)
                        else:
                            still_running.append(job)

                    self.running_jobs = still_running + concurrently_added_jobs

                if completed_jobs:
                    if self.verbose:
                        job_gens = [job.generation for job in completed_jobs]
                        # Format API cost info
                        if self.evo_config.max_api_costs is not None:
                            cost_pct = (
                                self.total_api_cost / self.evo_config.max_api_costs
                            ) * 100
                            cost_info = (
                                f" (cost: ${self.total_api_cost:.4f}, {cost_pct:.1f}%)"
                            )
                        else:
                            cost_info = f" (cost: ${self.total_api_cost:.4f})"

                        logger.info(
                            f"🔄 Processing {len(completed_jobs)} completed jobs: "
                            f"gens {job_gens}{cost_info}"
                        )

                    async with self.processing_lock:
                        old_retry_count = len(self.failed_jobs_for_retry)
                        self._mark_surplus_completed_jobs_for_discard(completed_jobs)
                        await self._process_completed_jobs_safely(completed_jobs)
                        old_completed = self.completed_generations
                        await self._update_completed_generations()

                        if self.verbose:
                            if self.completed_generations != old_completed:
                                if self.evo_config.max_api_costs is not None:
                                    cost_str = (
                                        f"${self.total_api_cost:.4f}/"
                                        f"${self.evo_config.max_api_costs:.2f}"
                                    )
                                    cost_pct = (
                                        self.total_api_cost
                                        / self.evo_config.max_api_costs
                                    ) * 100
                                    cost_info = f" (cost: {cost_str}, {cost_pct:.1f}%)"
                                else:
                                    cost_info = f" (cost: ${self.total_api_cost:.4f})"

                                logger.info(
                                    f"✅ Completed generations updated: "
                                    f"{old_completed} -> {self.completed_generations}"
                                    f"{cost_info}"
                                )
                            else:
                                retry_count = len(self.failed_jobs_for_retry)
                                new_retries = retry_count - old_retry_count
                                running_count = len(self.running_jobs)
                                at_target = (
                                    self.completed_generations
                                    >= self.evo_config.num_generations
                                )

                                if at_target:
                                    logger.debug(
                                        f"📊 Completed generations at target: "
                                        f"{self.completed_generations}"
                                    )
                                elif new_retries > 0:
                                    if self.evo_config.max_api_costs is not None:
                                        cost_str = (
                                            f"${self.total_api_cost:.4f}/"
                                            f"${self.evo_config.max_api_costs:.2f}"
                                        )
                                        cost_pct = (
                                            self.total_api_cost
                                            / self.evo_config.max_api_costs
                                        ) * 100
                                        cost_info = (
                                            f", cost: {cost_str} ({cost_pct:.1f}%)"
                                        )
                                    else:
                                        cost_info = (
                                            f", cost: ${self.total_api_cost:.4f}"
                                        )

                                    logger.info(
                                        f"📊 Completed generations: "
                                        f"{self.completed_generations} "
                                        f"({new_retries} new jobs in retry queue, "
                                        f"{retry_count} total pending retry"
                                        f"{cost_info})"
                                    )
                                elif retry_count > 0 or running_count > 0:
                                    logger.debug(
                                        f"📊 Completed generations: "
                                        f"{self.completed_generations} "
                                        f"(running={running_count}, "
                                        f"retry={retry_count})"
                                    )
                                else:
                                    logger.warning(
                                        f"⚠️  Completed generations unchanged "
                                        f"after processing jobs: "
                                        f"{self.completed_generations}"
                                    )

                    self._record_progress()
                    self.slot_available.set()

                # Retry any failed DB jobs
                if self.completed_generations >= self.evo_config.num_generations:
                    await self._cancel_surplus_inflight_work()

                if self.failed_jobs_for_retry:
                    try:
                        await self._retry_failed_db_jobs()
                    except Exception as e:
                        logger.error(f"Error retrying failed DB jobs: {e}")

                # Check if we've exceeded the API cost limit
                # Use committed cost for early detection, actual cost for final check
                if self.evo_config.max_api_costs is not None:
                    committed_cost = self._get_committed_cost()
                    if committed_cost >= self.evo_config.max_api_costs:
                        # Only log once when we first detect the limit
                        if not self.cost_limit_reached:
                            self.cost_limit_reached = True
                            in_flight_cost = committed_cost - self.total_api_cost
                            logger.info(
                                f"API cost budget reached: "
                                f"actual=${self.total_api_cost:.4f} + "
                                f"in-flight=${in_flight_cost:.4f} = "
                                f"${committed_cost:.4f} >= "
                                f"${self.evo_config.max_api_costs:.2f}. "
                                "Stopping evolution..."
                            )
                            pending_jobs = len(self.running_jobs)
                            pending_proposals = len(self.active_proposal_tasks)
                            if pending_jobs > 0 or pending_proposals > 0:
                                logger.info(
                                    f"⏳ Waiting for {pending_jobs} "
                                    f"running jobs and {pending_proposals} "
                                    "active proposals to complete..."
                                )

                        # Wait for ALL running jobs and proposals to
                        # complete and be processed
                        if (
                            len(self.running_jobs) == 0
                            and len(self.active_proposal_tasks) == 0
                        ):
                            # Final retry attempt for any remaining failed jobs
                            # before cost-limit shutdown
                            if self.failed_jobs_for_retry:
                                logger.info(
                                    f"🔄 FINAL RETRY: Attempting final retry of "
                                    f"{len(self.failed_jobs_for_retry)} failed "
                                    f"DB jobs before cost-limit shutdown"
                                )
                                try:
                                    await self._retry_failed_db_jobs()
                                except Exception as e:
                                    logger.error(f"Error in final retry attempt: {e}")

                                # Log any permanently failed jobs
                                if self.failed_jobs_for_retry:
                                    failed_gens = [
                                        job.generation
                                        for job in self.failed_jobs_for_retry.values()  # noqa: E501
                                    ]
                                    logger.error(
                                        f"❌ PERMANENT FAILURES: "
                                        f"{len(self.failed_jobs_for_retry)} jobs "
                                        f"could not be saved to database: "
                                        f"gens {failed_gens}"
                                    )

                            # Double-check that all jobs have been processed
                            logger.info(
                                f"✅ All running jobs completed. "
                                f"Total programs in database: "
                                f"{len(self.submitted_jobs)} submitted, "
                                f"{len(self.running_jobs)} still running."
                            )
                            # Stop evolution due to cost limit
                            logger.info(
                                "🛑 Job monitor setting should_stop signal "
                                "(cost limit reached, all jobs processed)"
                            )
                            self.should_stop.set()
                            self.slot_available.set()
                            logger.info(
                                "🏁 Job monitor setting finalization_complete signal"
                            )
                            self.finalization_complete.set()
                            break
                        else:
                            # Continue looping to process remaining jobs
                            # and proposals
                            pending_jobs = len(self.running_jobs)
                            pending_proposals = len(self.active_proposal_tasks)
                            if self.verbose and (
                                pending_jobs > 0 or pending_proposals > 0
                            ):
                                logger.debug(
                                    f"⏳ Still waiting for {pending_jobs} "
                                    f"jobs and {pending_proposals} proposals "
                                    "to complete..."
                                )
                            # Don't check other stop conditions when
                            # waiting for cost-limited jobs
                            await asyncio.sleep(0.1)
                            continue

                # Check if we should stop
                if (
                    self.completed_generations >= self.evo_config.num_generations
                    and len(self.running_jobs) == 0
                    and len(self.active_proposal_tasks) == 0
                ):
                    # Final retry attempt for any remaining failed jobs
                    # before shutdown
                    if self.failed_jobs_for_retry:
                        logger.info(
                            f"🔄 FINAL RETRY: Attempting final retry of "
                            f"{len(self.failed_jobs_for_retry)} failed "
                            f"DB jobs before shutdown"
                        )
                        try:
                            await self._retry_failed_db_jobs()
                        except Exception as e:
                            logger.error(f"Error in final retry attempt: {e}")

                        # Log any permanently failed jobs
                        if self.failed_jobs_for_retry:
                            failed_gens = [
                                job.generation
                                for job in self.failed_jobs_for_retry.values()
                            ]
                            logger.error(
                                f"❌ PERMANENT FAILURES: "
                                f"{len(self.failed_jobs_for_retry)} jobs "
                                f"could not be saved to database: "
                                f"gens {failed_gens}"
                            )

                    if self.verbose:
                        logger.info(
                            f"Evolution stopping: "
                            f"completed_generations="
                            f"{self.completed_generations}, "
                            f"target={self.evo_config.num_generations}, "
                            f"running_jobs={len(self.running_jobs)}, "
                            f"active_proposals="
                            f"{len(self.active_proposal_tasks)}"
                        )
                    # This is the final exit point.
                    logger.info("🛑 Job monitor setting should_stop signal")
                    self.should_stop.set()
                    # Wake up coordinator so it can see the stop signal.
                    self.slot_available.set()
                    # Signal that the entire run is complete.
                    logger.info("🏁 Job monitor setting finalization_complete signal")
                    self.finalization_complete.set()
                    break
                elif self.completed_generations >= self.evo_config.num_generations:
                    # We've reached target. Any extra work is being cancelled
                    # or discarded; only wait for the cleanup to settle.
                    if self.verbose:
                        pending_jobs = len(self.running_jobs)
                        pending_proposals = len(self.active_proposal_tasks)
                        logger.debug(
                            f"⏳ Target generations reached, draining surplus "
                            f"work: {pending_jobs} running jobs and "
                            f"{pending_proposals} proposal tasks still active..."
                        )

            except Exception as e:
                logger.error(f"Error in job monitor task: {e}")
                self.should_stop.set()  # Stop on error
                self.finalization_complete.set()
                break

            await asyncio.sleep(
                0.1
            )  # Check every 0.1 seconds for maximum responsiveness

        logger.info("Job monitor task exited")

    async def _proposal_coordinator_task(self):
        """Coordinate proposal generation to keep evaluation queue full."""
        while not self.should_stop.is_set():
            try:
                # Check for stuck system before normal processing
                if self._is_system_stuck():
                    recovery_success = await self._handle_stuck_system()
                    if not recovery_success:
                        # System determined to be permanently stuck, exit
                        break

                proposals_remaining = self._get_remaining_completed_work()
                proposal_slots_remaining = self._get_remaining_generation_slots()

                if (
                    proposal_slots_remaining == 0
                    and self._get_in_flight_work_count() == 0
                    and self.completed_generations < self.evo_config.num_generations
                ):
                    missing_generations = await self._get_missing_persisted_generations()
                    await self._record_generation_event(
                        generation=self.next_generation_to_submit,
                        status="stopped_generation_budget_exhausted",
                        details={
                            "completed_generations": self.completed_generations,
                            "target_generations": self.evo_config.num_generations,
                            "next_generation_to_submit": self.next_generation_to_submit,
                            "missing_generations": missing_generations,
                        },
                    )
                    logger.warning(
                        "Generation budget exhausted before reaching target "
                        f"completed generations: completed={self.completed_generations}, "
                        f"target={self.evo_config.num_generations}, "
                        f"next_generation={self.next_generation_to_submit}, "
                        f"missing_generations={missing_generations}. "
                        "Stopping without oversampling additional generations."
                    )
                    self.should_stop.set()
                    self.slot_available.set()
                    self.finalization_complete.set()
                    break

                # Check if cost limit has been reached using committed cost
                # Committed cost = actual cost + estimated cost of in-flight proposals
                # This prevents overshoot by stopping new proposals proactively
                should_generate_proposals = not self.cost_limit_reached
                if (
                    not self.cost_limit_reached
                    and self.evo_config.max_api_costs is not None
                ):
                    committed_cost = self._get_committed_cost()
                    if committed_cost >= self.evo_config.max_api_costs:
                        should_generate_proposals = False
                        self.cost_limit_reached = True
                        if self.verbose:
                            in_flight_cost = committed_cost - self.total_api_cost
                            logger.info(
                                f"Cost budget reached (using committed cost estimation): "
                                f"actual=${self.total_api_cost:.4f} + "
                                f"in-flight=${in_flight_cost:.4f} = "
                                f"${committed_cost:.4f} >= ${self.evo_config.max_api_costs:.2f} "
                                f"(avg proposal cost: ${self.avg_proposal_cost:.4f})"
                            )

                # Determine how many proposals to generate
                # Keep the pipeline full: aim for (running_jobs + active_proposals) = max_evaluation_jobs
                # This ensures proposals are ready when evaluation slots open up
                pipeline_capacity = len(self.running_jobs) + len(
                    self.active_proposal_tasks
                )
                pipeline_target = self._compute_proposal_pipeline_target()
                self._log_proposal_target_decision(pipeline_target)
                proposals_needed = min(
                    max(0, pipeline_target - pipeline_capacity),  # Fill the pipeline
                    proposals_remaining,
                    proposal_slots_remaining,
                    self.max_proposal_jobs - len(self.active_proposal_tasks),
                )

                if proposals_needed > 0 and should_generate_proposals:
                    # Start the needed proposals
                    if self.verbose:
                        logger.info(
                            f"Starting {proposals_needed} new proposals. "
                            f"Pipeline: {pipeline_capacity}/{pipeline_target} "
                            f"(running_jobs={len(self.running_jobs)}, active_proposals={len(self.active_proposal_tasks)}/{self.max_proposal_jobs}), "
                            f"Remaining completed work: {proposals_remaining} "
                            f"(completed={self.completed_generations}/{self.evo_config.num_generations}, "
                            f"next_generation={self.next_generation_to_submit})"
                        )
                    await self._start_proposals(proposals_needed)
                    # Record progress when we start new proposals
                    self._record_progress()

                # Clean up completed proposal tasks
                await self._cleanup_completed_proposal_tasks()

                # Wait for slot availability or the stop signal
                await self._wait_for_slot_or_stop(timeout=5.0)

            except Exception as e:
                logger.error(f"Error in proposal coordinator: {e}")
                await asyncio.sleep(1)

    async def _wait_for_slot_or_stop(self, timeout: float):
        """Wait for either the slot_available event or the should_stop event."""
        stop_task = asyncio.create_task(self.should_stop.wait())
        slot_task = asyncio.create_task(self.slot_available.wait())

        done, pending = await asyncio.wait(
            [stop_task, slot_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any pending tasks to avoid resource leaks
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # If the slot event was what completed, clear it
        if slot_task in done:
            self.slot_available.clear()

    async def _start_proposals(self, num_proposals: int):
        """Start the specified number of concurrent proposal generation tasks.

        Proposal assignment is hard-capped by num_generations so async runs do not
        oversample more proposal generations than the configured budget.
        """
        for _ in range(num_proposals):
            # Only stop if we've reached max proposal concurrency
            if len(self.active_proposal_tasks) >= self.max_proposal_jobs:
                break

            # Assign generation atomically to prevent duplicates
            generation = self.next_generation_to_submit

            if (
                self._generation_target_mode() == "proposal_ids"
                and generation >= self.evo_config.num_generations
            ):
                break

            # Double-check this generation hasn't been assigned already
            if generation in self.assigned_generations:
                logger.warning(f"Generation {generation} already assigned, skipping")
                continue

            # Mark generation as assigned and increment counter
            self.assigned_generations.add(generation)
            self.next_generation_to_submit += 1

            # Create proposal task
            task_id = str(uuid.uuid4())
            task = asyncio.create_task(
                self._generate_proposal_async(generation, task_id),
                name=f"proposal_{generation}",
            )
            self.active_proposal_tasks[task_id] = task

            if self.verbose:
                # Format API cost info
                if self.evo_config.max_api_costs is not None:
                    cost_pct = (
                        self.total_api_cost / self.evo_config.max_api_costs
                    ) * 100
                    cost_info = f" (cost: ${self.total_api_cost:.4f}, {cost_pct:.1f}%)"
                else:
                    cost_info = f" (cost: ${self.total_api_cost:.4f})"

                logger.info(
                    f"Started proposal task for generation {generation}{cost_info}"
                )

    async def _generate_proposal_async(
        self, generation: int, task_id: str
    ) -> Optional[AsyncRunningJob]:
        """Generate a single proposal asynchronously."""
        # Count all proposal attempts (including failures)
        self.total_proposals_generated += 1
        proposal_started_at = time.time()
        sampling_worker_id = None
        active_proposals_at_start = len(self.active_proposal_tasks)
        try:
            if self.verbose:
                logger.info(f"Generating proposal for generation {generation}")

            # Setup directories - create them synchronously to avoid race conditions
            gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{generation}"
            exec_fname = f"{gen_dir}/main.{self.lang_ext}"
            results_dir = f"{gen_dir}/results"
            lock_file = f"{gen_dir}/.generation_lock"
            # TODO: I dont think that these paths are used

            # Check if another task is already working on this generation
            # Run blocking file IO in executor
            loop = asyncio.get_event_loop()

            def sync_check_and_create_lock():
                if Path(lock_file).exists():
                    return False
                Path(gen_dir).mkdir(parents=True, exist_ok=False)  # Fail if exists
                Path(results_dir).mkdir(parents=True, exist_ok=True)
                Path(lock_file).touch()
                return True

            try:
                can_proceed = await loop.run_in_executor(
                    None, sync_check_and_create_lock
                )
                if not can_proceed:
                    logger.warning(
                        f"Generation {generation} already being processed or directory exists, aborting"
                    )
                    return None
            except FileExistsError:
                logger.warning(
                    f"Generation {generation} directory already exists, aborting duplicate task"
                )
                return None

            # Get current meta recommendations (no updates here - only during evaluation completion)
            meta_recs, meta_summary, meta_scratch = None, None, None
            if self.meta_summarizer:
                logger.info(
                    f"Getting meta recs for gen {generation}, "
                    f"sample_single_meta_rec={self.evo_config.sample_single_meta_rec}"
                )
                if self.evo_config.sample_single_meta_rec:
                    meta_recs = self.meta_summarizer.get_sampled_recommendation()
                    _, meta_summary, meta_scratch = self.meta_summarizer.get_current()
                else:
                    meta_recs, meta_summary, meta_scratch = (
                        self.meta_summarizer.get_current()
                    )
                logger.info(f"meta_recs result: {bool(meta_recs)}")

            # Handle initial generation - it's already evaluated and in database
            if generation == 0:
                if self.verbose:
                    logger.info(
                        "Generation 0 already processed during setup, skipping proposal generation"
                    )
                return None

            # Generate proposal for non-initial generation
            return await self._generate_evolved_proposal(
                generation,
                task_id,
                exec_fname,
                results_dir,
                meta_recs,
                meta_summary,
                meta_scratch,
                proposal_started_at,
                sampling_worker_id,
                active_proposals_at_start,
            )

        except Exception as e:
            logger.error(f"Error generating proposal for generation {generation}: {e}")
            return None
        finally:
            # Cleanup: remove lock file and task tracking
            try:
                gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{generation}"
                lock_file = f"{gen_dir}/.generation_lock"

                async def unlink_async():
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, Path(lock_file).unlink)

                if Path(lock_file).exists():
                    await unlink_async()

            except Exception as e:
                logger.warning(
                    f"Failed to cleanup lock file for generation {generation}: {e}"
                )

            await self._cleanup_proposal_task_state(
                generation=generation,
                task_id=task_id,
                sampling_worker_id=sampling_worker_id,
            )

    async def _generate_evolved_proposal(
        self,
        generation: int,
        task_id: str,
        exec_fname: str,
        results_dir: str,
        meta_recs: Optional[str],
        meta_summary: Optional[str],
        meta_scratch: Optional[str],
        proposal_started_at: float,
        sampling_worker_id: Optional[int],
        active_proposals_at_start: int,
    ) -> Optional[AsyncRunningJob]:
        """Generate an evolved proposal through the full pipeline."""
        api_costs = 0.0
        embed_cost = 0.0

        # Initialize novelty tracking variables (same as sync runner)
        novelty_checks_performed = 0
        novelty_total_cost = 0.0
        novelty_explanation = ""
        last_failure_stage = "proposal"
        last_failure_reason = "Agent failed to generate a valid proposal after all attempts"
        proposal_accepted = False
        parent_program: Optional[Program] = None
        archive_programs: List[Program] = []
        top_k_programs: List[Program] = []
        code_diff: Optional[str] = None
        meta_patch_data: Dict[str, Any] = {}
        text_embedding: Optional[List[float]] = None
        worktree: Optional[RepoWorktree] = None
        parent_commit: Optional[str] = None
        child_commit: Optional[str] = None
        summary_text = ""
        summary_version: Optional[str] = None
        changed_files: List[str] = []

        # Select LLM once per program generation (before all loops)
        model_sample_probs = None
        model_posterior = None
        if getattr(self, "llm_selection", None) is not None:
            healthy_routes = self.route_health.selectable_routes(
                self.evo_config.llm_models
            )
            model_sample_probs, model_posterior = self.llm_selection.select_llm(
                subset=healthy_routes
            )

        for attempt in range(self.evo_config.max_novelty_attempts):
            try:
                # Sample parent and inspirations with fix mode detection
                (
                    parent_program,
                    archive_programs,
                    top_k_programs,
                    needs_fix,
                ) = await self.async_db.sample_with_fix_mode_async(
                    target_generation=generation,
                    novelty_attempt=attempt + 1,
                    max_novelty_attempts=self.evo_config.max_novelty_attempts,
                    resample_attempt=1,
                    max_resample_attempts=1,
                )

                # Sync beam_search parent to main database if using beam_search strategy
                # (async sampling uses read-only thread-local DBs that can't persist state)
                if (
                    getattr(self.db_config, "parent_selection_strategy", "")
                    == "beam_search"
                    and parent_program
                ):
                    await self.async_db.update_beam_search_parent_async(
                        parent_program.id
                    )

                parent_commit = (
                    parent_program.repo_commit
                    or (parent_program.metadata or {}).get("repo_commit")
                    or self.seed_repo_commit
                )
                if not parent_commit:
                    raise ValueError("Parent program does not have a repo commit")

                individual_id = str(uuid.uuid4())
                worktree = self.repo_worktree_manager.create_child_worktree(
                    parent_commit=parent_commit,
                    generation=generation,
                    individual_id=individual_id,
                ) # create new worktree per attempt

                # Choose between fix mode and normal patch mode
                if needs_fix:
                    # FIX MODE: No correct programs exist, try to fix
                    # archive_programs contains ancestors in fix mode
                    if self.verbose:
                        logger.info(
                            f"FIX MODE: Attempting to fix incorrect repo "
                            f"{parent_program.id} (Gen: {parent_program.generation})"
                        )
                agent_result = await self._run_patch_async(
                    parent_program,
                    archive_programs,  # ancestors in fix mode
                    top_k_programs,
                    generation=generation,
                    novelty_attempt=attempt + 1,
                    resample_attempt=1,
                    model_sample_probs=model_sample_probs,
                    model_posterior=model_posterior,
                    fix_mode=needs_fix,
                    worktree=worktree,
                )

                if not agent_result:
                    last_failure_stage = "proposal"
                    last_failure_reason = "Program patch generation returned no result"
                    continue

                code_diff, meta_patch_data, success = agent_result
                api_costs += meta_patch_data.get("api_costs", 0.0)

                snapshot = self.repo_worktree_manager.diff_parent(worktree.path, parent_commit)
                self.repo_worktree_manager.validate_snapshot(worktree, snapshot)

                stashed_parent_id = WorktreeManager.read_parent_id(worktree.path)
                if stashed_parent_id != parent_program.id:
                    last_failure_stage = "proposal"
                    last_failure_reason = (
                        f"Parent ID mismatch: expected {parent_program.id!r}, "
                        f"got {stashed_parent_id!r}"
                    )
                    continue

                if not success:
                    last_failure_stage = "proposal"
                    last_failure_reason = (
                        meta_patch_data.get("last_error_msg")
                        or meta_patch_data.get("error_attempt")
                        or "Patch generation failed before evaluation"
                    )
                    if meta_patch_data.get("terminal_model_failure"):
                        break
                    continue

                # We have a successful patch, continue novelty check
                meta_patch_data["api_costs"] = api_costs
            except Exception as e:
                logger.warning(
                    f"Error in repo patch generation attempt: {e}"
                )
                last_failure_stage = "repo patch generation"
                last_failure_reason = str(e)
                continue

            # Get code embedding (only once per successful patch)
            if self.verbose:
                logger.info(f"Getting code embedding for generation {generation}...")
                
            snapshot = self.repo_worktree_manager.commit_child(worktree)
            child_commit = snapshot.commit_sha or parent_commit
            summary_path = worktree.path / self.evo_config.summary_filename
            summary_text = summary_path.read_text(encoding="utf-8")
            summary_check = validate_summary(
                summary_text,
                max_chars=self.evo_config.summary_max_chars,
            )
            if not summary_check.valid:
                raise ValueError("; ".join(summary_check.errors))
            summary_version = summary_check.schema_version
            changed_files = snapshot.changed_files
            meta_patch_data.update(
                {
                    "repo_path": str(worktree.path),
                    "repo_commit": child_commit,
                    "repo_parent_commit": parent_commit,
                    "repo_diff": snapshot.diff,
                    "repo_diff_stat": snapshot.diff_stat,
                    "repo_summary": summary_text,
                    "summary_version": summary_version,
                    "changed_files": changed_files,
                    "mutable_paths": self.evo_config.mutable_paths,
                    "immutable_paths": self.evo_config.immutable_paths,
                }
            )
            text_embedding, e_cost = await self._get_text_embedding_async(summary_text)
            
            embed_cost += e_cost
            if self.verbose:
                logger.info(
                    f"Text embedding completed for generation {generation} (cost: ${e_cost:.4f})"
                )

            if not text_embedding:
                if self.novelty_judge:
                    self.novelty_judge.log_novelty_skip_message("no embedding")
                proposal_accepted = True  # Accept program even without embedding
                break

            # Novelty check (same logic as sync runner)
            if self.novelty_judge:
                should_check = await self.novelty_judge.should_check_novelty_async(
                    text_embedding, generation, parent_program, self.db
                )

                if should_check:
                    (
                        should_accept,
                        novelty_metadata,
                    ) = await self.novelty_judge.assess_novelty_with_rejection_sampling_async(
                        summary_text, text_embedding, parent_program, self.db
                    )

                    # Update costs and metadata from novelty assessment (same as sync runner)
                    novelty_cost_from_check = novelty_metadata.get(
                        "novelty_total_cost", 0.0
                    )
                    novelty_total_cost += (
                        novelty_cost_from_check  # Accumulate novelty cost separately
                    )
                    novelty_checks_performed = novelty_metadata.get(
                        "novelty_checks_performed", 0
                    )
                    novelty_explanation = novelty_metadata.get(
                        "novelty_explanation", ""
                    )
                    meta_patch_data["novelty_checks_performed"] = (
                        novelty_checks_performed
                    )
                    meta_patch_data["novelty_cost"] = novelty_total_cost
                    meta_patch_data["novelty_explanation"] = novelty_explanation
                    meta_patch_data["max_similarity"] = novelty_metadata.get(
                        "max_similarity"
                    )
                    meta_patch_data["similarity_scores"] = novelty_metadata.get(
                        "similarity_scores", []
                    )

                    if should_accept:
                        proposal_accepted = True
                        break
                    # If not accepted, continue to next attempt (rejection sampling)
                    last_failure_stage = "novelty"
                    last_failure_reason = (
                        novelty_explanation
                        or "Proposal rejected by novelty check before downstream evaluation"
                    )
                else:
                    proposal_accepted = True
                    if not self.db.island_manager or not hasattr(
                        self.db.island_manager, "are_all_islands_initialized"
                    ):
                        self.novelty_judge.log_novelty_skip_message("no island manager")
                    elif not self.db.island_manager.are_all_islands_initialized():
                        self.novelty_judge.log_novelty_skip_message(
                            "not all islands initialized yet"
                        )
                    break
            else:
                # No novelty judge configured, accept the program
                proposal_accepted = True
                break

            # If proposal was accepted, break from the outer novelty loop
            if proposal_accepted:
                break

        # Add meta-recommendations/summary/scratchpad to meta_patch_data (same as sync runner)
        if meta_recs is not None:
            meta_patch_data["meta_recommendations"] = meta_recs
            meta_patch_data["meta_summary"] = meta_summary
            meta_patch_data["meta_scratch_pad"] = meta_scratch

        # If we have an accepted proposal, submit it
        if proposal_accepted:
            # Add novelty check information to meta_patch_data if any checks were performed (same as sync runner)
            if generation > 0 and novelty_checks_performed > 0:
                meta_patch_data["novelty_checks_performed"] = novelty_checks_performed
                meta_patch_data["novelty_cost"] = (
                    novelty_total_cost  # Use "novelty_cost" key like sync
                )
                meta_patch_data["novelty_explanation"] = novelty_explanation

            try:
                (
                    job_id,
                    evaluation_worker_id,
                    evaluation_submitted_at,
                    evaluation_started_at,
                    running_eval_jobs_at_submit,
                ) = await self._submit_evaluation_job_with_slot(
                    exec_fname=exec_fname,
                    results_dir=results_dir,
                    sampling_worker_id=sampling_worker_id,
                    repo_path=str(worktree.path) if worktree else None,
                )

                # Create running job
                running_job = AsyncRunningJob(
                    job_id=job_id,
                    exec_fname=exec_fname,
                    results_dir=results_dir,
                    start_time=proposal_started_at,
                    proposal_started_at=proposal_started_at,
                    evaluation_submitted_at=evaluation_submitted_at,
                    generation=generation,
                    individual_id=worktree.individual_id if worktree else None,
                    evaluation_started_at=evaluation_started_at,
                    sampling_worker_id=sampling_worker_id,
                    evaluation_worker_id=evaluation_worker_id,
                    active_proposals_at_start=active_proposals_at_start,
                    running_eval_jobs_at_submit=running_eval_jobs_at_submit,
                    parent_id=parent_program.id,
                    archive_insp_ids=[p.id for p in archive_programs],
                    top_k_insp_ids=[p.id for p in top_k_programs],
                    code_diff=code_diff,
                    meta_patch_data=meta_patch_data,
                    code_embedding=text_embedding,
                    embed_cost=embed_cost,
                    novelty_cost=novelty_total_cost,  # Store novelty cost in running job
                    proposal_task_id=task_id,
                    repo_path=str(worktree.path) if worktree else None,
                    repo_commit=child_commit,
                    repo_parent_commit=parent_commit,
                    repo_summary=summary_text,
                    summary_version=summary_version,
                    changed_files=changed_files,
                    agent_session_id=meta_patch_data.get("headless_session_id"),
                    agent_session_name=meta_patch_data.get("headless_session_name"),
                    agent_provider="headless",
                    agent_model=meta_patch_data.get("model_name"),
                )

                # Update costs
                meta_patch_data["api_costs"] = api_costs
                proposal_total_cost = api_costs + embed_cost + novelty_total_cost
                self.total_api_cost += proposal_total_cost

                # Update average proposal cost for in-flight estimation
                self._update_avg_proposal_cost(proposal_total_cost)

                # Track job in both running list and submitted registry
                self.running_jobs.append(running_job)
                self.submitted_jobs[str(job_id)] = running_job

                # Trigger immediate job status check to catch fast-completing jobs
                self.slot_available.set()
                if self.verbose:
                    total_cost = api_costs + embed_cost + novelty_total_cost

                    # Format total API cost info
                    if self.evo_config.max_api_costs is not None:
                        cost_pct = (
                            self.total_api_cost / self.evo_config.max_api_costs
                        ) * 100
                        total_cost_info = (
                            f", total: ${self.total_api_cost:.4f} ({cost_pct:.1f}%)"
                        )
                    else:
                        total_cost_info = f", total: ${self.total_api_cost:.4f}"

                    logger.info(
                        f"Proposal → Eval: gen {generation} submitted for eval "
                        f"(cost: ${total_cost:.4f}{total_cost_info}). "
                        f"Running jobs: {len(self.running_jobs)}/{self.max_evaluation_jobs}, "
                        f"Proposals: {len(self.active_proposal_tasks)}/{self.max_proposal_jobs}"
                    )

                return running_job

            except Exception as e:
                logger.error(f"Error submitting job: {e}")
                await self._record_terminal_failed_proposal(
                    generation=generation,
                    exec_fname=exec_fname,
                    proposal_started_at=proposal_started_at,
                    sampling_worker_id=sampling_worker_id,
                    active_proposals_at_start=active_proposals_at_start,
                    parent_program=parent_program,
                    archive_programs=archive_programs,
                    top_k_programs=top_k_programs,
                    code_diff=code_diff,
                    meta_patch_data=meta_patch_data,
                    code_embedding=text_embedding,
                    embed_cost=embed_cost,
                    novelty_cost=novelty_total_cost,
                    api_costs=api_costs,
                    failure_stage="evaluation_submit",
                    failure_reason=str(e),
                )
                return None

        logger.warning(
            f"Failed to generate proposal for generation {generation} after all attempts"
        )
        await self._record_terminal_failed_proposal(
            generation=generation,
            exec_fname=exec_fname,
            proposal_started_at=proposal_started_at,
            sampling_worker_id=sampling_worker_id,
            active_proposals_at_start=active_proposals_at_start,
            parent_program=parent_program,
            archive_programs=archive_programs,
            top_k_programs=top_k_programs,
            code_diff=code_diff,
            meta_patch_data=meta_patch_data,
            code_embedding=text_embedding,
            embed_cost=embed_cost,
            novelty_cost=novelty_total_cost,
            api_costs=api_costs,
            failure_stage=last_failure_stage,
            failure_reason=last_failure_reason,
        )
        return None

    def _program_summary_text(self, program: Optional[Program]) -> str:
        if program is None:
            return "No parent summary available."
        return (
            program.repo_summary
            or (program.metadata or {}).get("repo_summary")
            or "No summary recorded."
        )

    async def _save_patch_attempt_async(
        self,
        generation: int,
        novelty_attempt: int,
        resample_attempt: int,
        patch_attempt: int,
        response: Any,
        error_msg: Optional[str],
        patch_text: Optional[str],
        num_applied: int,
        patch_name: Optional[str],
        patch_description: Optional[str],
        success: bool,
        headless_artifacts: Optional[Dict[str, str]] = None,
    ):
        """Save patch attempt data to disk asynchronously for debugging and analysis."""
        # Create attempt directory structure
        attempt_dir = (
            Path(self.results_dir)
            / f"{FOLDER_PREFIX}_{generation}"
            / "attempts"
            / f"novelty_{novelty_attempt}"
            / f"resample_{resample_attempt}"
            / f"patch_{patch_attempt}"
        )

        # Run directory creation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: attempt_dir.mkdir(parents=True, exist_ok=True)
        )

        # Save LLM response
        if response and response.content:
            response_file = attempt_dir / "llm_response.txt"
            await write_file_async(str(response_file), response.content)

        response_kwargs = getattr(response, "kwargs", {}) if response else {}
        response_kwargs = {**(headless_artifacts or {}), **response_kwargs}
        headless_prompt_path = response_kwargs.get("headless_prompt_path")
        if headless_prompt_path:
            prompt_source = Path(headless_prompt_path)
            if prompt_source.exists():
                prompt_file = attempt_dir / "headless_prompt.md"
                await write_file_async(
                    str(prompt_file),
                    prompt_source.read_text(encoding="utf-8"),
                )
        for source_key, destination_name in (
            ("headless_stdout_path", "headless_stdout.log"),
            ("headless_stderr_path", "headless_stderr.log"),
        ):
            source_value = response_kwargs.get(source_key)
            if not source_value:
                continue
            source_path = Path(source_value)
            if source_path.is_file():
                await write_file_async(
                    str(attempt_dir / destination_name),
                    source_path.read_text(encoding="utf-8", errors="replace"),
                )

        # Save patch text if available
        if patch_text:
            patch_file = attempt_dir / "patch.txt"
            await write_file_async(str(patch_file), patch_text)

        # Save metadata as JSON
        metadata = {
            "generation": generation,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "patch_attempt": patch_attempt,
            "success": success,
            "num_applied": num_applied,
            "patch_name": patch_name,
            "patch_description": patch_description,
            "error_msg": error_msg,
            "timestamp": datetime.now().isoformat(),
        }

        if response:
            metadata["llm_cost"] = response.cost
            metadata["llm_model"] = getattr(response, "model_name", None)
            if headless_prompt_path:
                metadata["headless_prompt_path"] = str(
                    attempt_dir / "headless_prompt.md"
                )

        metadata_file = attempt_dir / "metadata.json"
        await write_file_async(str(metadata_file), json.dumps(metadata, indent=2))

    def _classify_failed_proposal(
        self,
        *,
        failure_stage: str,
        failure_reason: str,
        meta_patch_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Normalize pre-evaluation failures into stable classes for UI/analytics."""
        if failure_stage == "novelty":
            return "novelty_rejected"
        if failure_stage == "evaluation_submit":
            return "evaluation_submit_failed"

        reason = (failure_reason or "").lower()
        error_attempt = str((meta_patch_data or {}).get("error_attempt") or "").lower()
        last_error_msg = str((meta_patch_data or {}).get("last_error_msg") or "").lower()
        combined = " ".join(
            part for part in [reason, error_attempt, last_error_msg] if part
        )

        if (
            (meta_patch_data or {}).get("route_failure_class")
        ):
            return str((meta_patch_data or {})["route_failure_class"])
        if (
            "could not extract code" in combined
            or "llm response content was none" in combined
            or "no evolve-block regions found" in combined
        ):
            return "llm_output_invalid"
        if "did not apply any changes" in combined or "no repository changes" in combined:
            return "no_diff"
        if "summary" in combined and any(
            marker in combined
            for marker in ("missing", "empty", "placeholder", "schema", "heading")
        ):
            return "invalid_summary"
        if any(
            marker in combined
            for marker in (
                "immutable path",
                "outside mutable paths",
                "protected path",
                "policy file",
                "binary files are not allowed",
                "deletions are not allowed",
                "lockfile changes are not allowed",
                "symlink target escapes",
            )
        ):
            return "policy_violation"
        if (
            "search text not found" in combined
            or "no changes applied" in combined
            or "editable regions" in combined
        ):
            return "patch_apply_failed"

        return "proposal_generation_failed"

    def _collect_failure_artifacts(
        self,
        *,
        exec_fname: str,
        meta_patch_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect the well-known artifact paths for a failed generation."""
        exec_path = Path(exec_fname)
        gen_dir = exec_path.parent
        artifacts: Dict[str, Any] = {
            "generation_dir": str(gen_dir),
            "generated_code_path": str(exec_path),
            "results_dir": str(gen_dir / "results"),
        }

        def add_if_exists(key: str, path: Path) -> None:
            if path.exists():
                artifacts[key] = str(path)

        add_if_exists("generation_diff_path", gen_dir / "edit.diff")
        add_if_exists("generation_search_replace_path", gen_dir / "search_replace.txt")
        add_if_exists("generation_rewrite_path", gen_dir / "rewrite.txt")
        add_if_exists("generation_original_path", gen_dir / f"original.{self.lang_ext}")

        if meta_patch_data:
            novelty_attempt = meta_patch_data.get("novelty_attempt")
            resample_attempt = meta_patch_data.get("resample_attempt")
            patch_attempt = meta_patch_data.get("patch_attempt")
            if all(
                value is not None
                for value in [novelty_attempt, resample_attempt, patch_attempt]
            ):
                attempt_dir = (
                    gen_dir
                    / "attempts"
                    / f"novelty_{novelty_attempt}"
                    / f"resample_{resample_attempt}"
                    / f"patch_{patch_attempt}"
                )
                artifacts["attempt_dir"] = str(attempt_dir)
                add_if_exists("attempt_metadata_path", attempt_dir / "metadata.json")
                add_if_exists("llm_response_path", attempt_dir / "llm_response.txt")
                add_if_exists("attempt_patch_path", attempt_dir / "patch.txt")
                add_if_exists(
                    "headless_stdout_path", attempt_dir / "headless_stdout.log"
                )
                add_if_exists(
                    "headless_stderr_path", attempt_dir / "headless_stderr.log"
                )
                add_if_exists(
                    "headless_prompt_path", attempt_dir / "headless_prompt.md"
                )

        return artifacts

    def _get_failure_language(self, exec_path: Path) -> str:
        """Infer the failed proposal language from config or the generated filename."""
        configured_language = getattr(self.evo_config, "language", None)
        if configured_language:
            return configured_language

        ext = exec_path.suffix.lstrip(".").lower()
        return {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "cpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "cu": "cuda",
            "go": "go",
            "f90": "fortran",
            "f95": "fortran",
            "f03": "fortran",
            "f08": "fortran",
        }.get(ext, ext or "python")

    async def _write_failure_artifact_async(
        self,
        *,
        generation: int,
        exec_fname: str,
        parent_program: Optional[Program],
        archive_programs: List[Program],
        top_k_programs: List[Program],
        meta_patch_data: Optional[Dict[str, Any]],
        embed_cost: float,
        novelty_cost: float,
        api_costs: float,
        failure_stage: str,
        failure_class: str,
        failure_reason: str,
        code_embedding: Optional[List[float]],
        proposal_started_at: float,
        sampling_worker_id: Optional[int],
        active_proposals_at_start: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Write the durable `failure.json` payload for a failed generation."""
        exec_path = Path(exec_fname)
        failure_path = exec_path.parent / "failure.json"
        artifacts = self._collect_failure_artifacts(
            exec_fname=exec_fname,
            meta_patch_data=meta_patch_data,
        )

        payload = {
            "generation": generation,
            "node_kind": "failed_proposal",
            "language": self._get_failure_language(exec_path),
            "failure_stage": failure_stage,
            "failure_class": failure_class,
            "failure_reason": failure_reason,
            "timestamp": datetime.now().isoformat(),
            "proposal_started_at": proposal_started_at,
            "sampling_worker_id": sampling_worker_id,
            "active_proposals_at_start": active_proposals_at_start,
            "downstream_eval_submitted": False,
            "generated_code_available": exec_path.exists(),
            "code_embedding_available": bool(code_embedding),
            "parent_id": parent_program.id if parent_program else None,
            "archive_inspiration_ids": [p.id for p in archive_programs],
            "top_k_inspiration_ids": [p.id for p in top_k_programs],
            "patch_type": (meta_patch_data or {}).get("patch_type"),
            "patch_name": (meta_patch_data or {}).get("patch_name"),
            "patch_description": (meta_patch_data or {}).get("patch_description"),
            "system_prompt_id": (meta_patch_data or {}).get("system_prompt_id"),
            "api_costs": api_costs,
            "embed_cost": embed_cost,
            "novelty_cost": novelty_cost,
            "novelty_checks_performed": (meta_patch_data or {}).get(
                "novelty_checks_performed", 0
            ),
            "novelty_explanation": (meta_patch_data or {}).get(
                "novelty_explanation", ""
            ),
            "max_similarity": (meta_patch_data or {}).get("max_similarity"),
            "attempts": {
                "novelty_attempt": (meta_patch_data or {}).get("novelty_attempt"),
                "resample_attempt": (meta_patch_data or {}).get("resample_attempt"),
                "patch_attempt": (meta_patch_data or {}).get("patch_attempt"),
            },
            "error_attempt": (meta_patch_data or {}).get("error_attempt"),
            "last_error_msg": (meta_patch_data or {}).get("last_error_msg"),
            "artifacts": artifacts,
            "failure_json_path": str(failure_path),
        }

        await write_file_async(
            str(failure_path),
            json.dumps(payload, indent=2, sort_keys=True),
        )
        return str(failure_path), payload

    async def _record_terminal_failed_proposal(
        self,
        *,
        generation: int,
        exec_fname: str,
        proposal_started_at: float,
        sampling_worker_id: Optional[int],
        active_proposals_at_start: int,
        parent_program: Optional[Program],
        archive_programs: List[Program],
        top_k_programs: List[Program],
        code_diff: Optional[str],
        meta_patch_data: Optional[Dict[str, Any]],
        code_embedding: Optional[List[float]],
        embed_cost: float,
        novelty_cost: float,
        api_costs: float,
        failure_stage: str,
        failure_reason: str,
    ) -> None:
        """Record one terminal pre-eval failure via attempt_log plus failure.json."""
        terminal_failure_cost = float(api_costs) + float(embed_cost) + float(
            novelty_cost
        )
        self.total_api_cost += terminal_failure_cost
        if terminal_failure_cost > 0.0:
            self._update_avg_proposal_cost(terminal_failure_cost)
        failure_class = self._classify_failed_proposal(
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            meta_patch_data=meta_patch_data,
        )
        failure_json_path, failure_payload = await self._write_failure_artifact_async(
            generation=generation,
            exec_fname=exec_fname,
            parent_program=parent_program,
            archive_programs=archive_programs,
            top_k_programs=top_k_programs,
            meta_patch_data=meta_patch_data,
            embed_cost=embed_cost,
            novelty_cost=novelty_cost,
            api_costs=api_costs,
            failure_stage=failure_stage,
            failure_class=failure_class,
            failure_reason=failure_reason,
            code_embedding=code_embedding,
            proposal_started_at=proposal_started_at,
            sampling_worker_id=sampling_worker_id,
            active_proposals_at_start=active_proposals_at_start,
        )
        terminal_failure_at = time.time()
        await self._record_attempt_event(
            generation=generation,
            stage=failure_stage,
            status="failed",
            details={
                "node_kind": "failed_proposal",
                "language": failure_payload.get("language"),
                "failure_stage": failure_stage,
                "failure_class": failure_class,
                "failure_reason": failure_reason,
                "parent_id": parent_program.id if parent_program else None,
                "archive_inspiration_ids": [p.id for p in archive_programs],
                "top_k_inspiration_ids": [p.id for p in top_k_programs],
                "code_diff_available": bool(code_diff),
                "patch_type": (meta_patch_data or {}).get("patch_type"),
                "patch_name": (meta_patch_data or {}).get("patch_name"),
                "patch_description": (meta_patch_data or {}).get(
                    "patch_description"
                ),
                "system_prompt_id": (meta_patch_data or {}).get("system_prompt_id"),
                "model_name": (meta_patch_data or {}).get("model_name"),
                "api_costs": api_costs,
                "embed_cost": embed_cost,
                "novelty_cost": novelty_cost,
                "novelty_attempt": (meta_patch_data or {}).get("novelty_attempt"),
                "resample_attempt": (meta_patch_data or {}).get("resample_attempt"),
                "patch_attempt": (meta_patch_data or {}).get("patch_attempt"),
                "max_similarity": (meta_patch_data or {}).get("max_similarity"),
                "failure_json_path": failure_json_path,
                "pipeline_started_at": proposal_started_at,
                "sampling_started_at": proposal_started_at,
                "sampling_finished_at": terminal_failure_at,
                "evaluation_started_at": terminal_failure_at,
                "evaluation_finished_at": terminal_failure_at,
                "postprocess_started_at": terminal_failure_at,
                "postprocess_finished_at": terminal_failure_at,
                "timeline_lane_mode": "pool_slots",
                "sampling_worker_id": sampling_worker_id,
                "evaluation_worker_id": None,
                "postprocess_worker_id": None,
                "sampling_worker_capacity": self.max_proposal_jobs,
                "evaluation_worker_capacity": self.max_evaluation_jobs,
                "postprocess_worker_capacity": self.max_db_workers,
                "generated_code_available": failure_payload.get(
                    "generated_code_available", False
                ),
                "downstream_eval_submitted": False,
            },
        )
        model_name = (meta_patch_data or {}).get("model_name")
        route_failure_class = (meta_patch_data or {}).get("route_failure_class")
        if model_name and route_failure_class:
            self.route_health.record_failure(model_name, route_failure_class)
            logger.warning(
                "Route health failure for %s: %s; state=%s",
                model_name,
                route_failure_class,
                self.route_health.snapshot().get(model_name),
            )

    async def _run_patch_async(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: Optional[List[Program]] = None,
        generation: int = 0,
        meta_recs: Optional[str] = None,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
        model_sample_probs: Optional[List[float]] = None,
        model_posterior: Optional[List[float]] = None,
        fix_mode: bool = False,
        worktree: Optional[RepoWorktree] = None,
    ) -> Optional[Tuple[Optional[str], Dict[str, Any], bool]]:
        """Run one repo-backed mutation attempt with a headless coding agent.

        Args:
            parent_program: The parent repo to patch. In fix mode this is the
                incorrect repo to fix.
            archive_programs: Archive inspirations. In fix mode this is the list of
                ancestor inspirations (from ``sample_with_fix_mode``).
            top_k_programs: Top-k inspirations (unused in fix mode).
            generation: Current generation number.
            meta_recs: Meta recommendations (unused in fix mode).
            novelty_attempt: Current novelty attempt number.
            resample_attempt: Current resample attempt number.
            model_sample_probs: Model sampling probabilities.
            model_posterior: Model posterior probabilities.
            fix_mode: When True, generate a fix patch for an incorrect repo
                instead of a normal evolution patch. Fix patches always use a
                full rewrite and do not track an evolved system prompt.
            worktree: Mutable child worktree for the current proposal attempt.
        """
        current_prompt_id: Optional[str] = None
        current_sys_prompt: Optional[str] = None
        original_task_sys_msg = self.prompt_sampler.task_sys_msg
        required_summary_name = self.evo_config.summary_filename
        patch_name = "agent_repo_fix" if fix_mode else "agent_repo_mutation"
        patch_description = (
            "Repo fix produced by headless coding agent"
            if fix_mode
            else "Repo mutation produced by headless coding agent"
        )
        response_kwargs: Dict[str, Any] = {}
        summary_path: Optional[Path] = None
        policy_path: Optional[Path] = None
        agent_worktree: Optional[RepoWorktree] = None
        last_diff: Optional[str] = None
        last_diff_summary: Dict[str, Any] = {}
        last_summary_version: Optional[str] = None
        last_error_msg: Optional[str] = None
        terminal_model_failure = False

        try:
            if not fix_mode:
                current_sys_prompt, current_prompt_id = (
                    self._get_current_system_prompt()
                )

                if current_sys_prompt:
                    self.prompt_sampler.task_sys_msg = current_sys_prompt

            patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
                parent=parent_program,
                archive_inspirations=archive_programs,
                top_k_inspirations=top_k_programs or [],
                meta_recommendations=meta_recs,
            )
            if worktree is not None:
                agent_worktree = self.repo_worktree_manager.create_agent_worktree_view(
                    worktree,
                    hidden_paths=self._agent_hidden_paths_for_worktree(worktree.path),
                )
                self.repo_worktree_manager.write_parent_id(
                    worktree,
                    parent_program.id,
                )

            agent_target_worktree = agent_worktree or worktree
            required_summary_path = (
                agent_target_worktree.path / required_summary_name
                if agent_target_worktree is not None
                else None
            )
            required_summary_display = (
                str(required_summary_path.resolve())
                if required_summary_path is not None
                else required_summary_name
            )
            repo_contract = f"""
# Repository Mode Contract

The proposal repository root is `{agent_target_worktree.path.resolve() if agent_target_worktree else '<headless work directory>'}`.
Begin every filesystem and terminal operation in that exact directory. Never put
candidate files in a provider scratch directory. Shinka imports only
validated mutable-path changes from that view into the candidate evaluation
worktree.

Required constraints:
- If the mutable-path list is empty, the whole repository is mutable. Otherwise,
  modify only configured mutable paths, plus the summary file at `{required_summary_display}`.
- Immutable paths are read-only in the generation view; do not chmod, chown,
  edit, delete, or touch them.
- Evaluation code, private tests, and hidden paths are deliberately unavailable during generation.
- Do not modify `.git/` or any other Shinka control files under `.shinka/`.
- Complete the existing summary template at `{required_summary_display}` after generating changes.
- Preserve the summary schema/headings and replace every `TODO_AGENT_SUMMARY` placeholder.
- Keep the final assistant response brief; the repository files are the source of truth.
""".strip()
            if fix_mode:
                patch_msg = (
                    "# Incorrect Parent Program\n\n"
                    "The selected parent failed evaluation. Make the repository correct "
                    "first, then improve performance if possible.\n\n"
                    f"{patch_msg}"
                )
            patch_sys = f"{patch_sys}\n\n{repo_contract}"

            self.prompt_sampler.task_sys_msg = original_task_sys_msg

            if self.verbose:
                if not fix_mode and current_prompt_id:
                    logger.info(f"Using evolved prompt: {current_prompt_id[:8]}...")

            if agent_target_worktree is not None:
                if required_summary_path is None:
                    raise RuntimeError("Summary path was not prepared for worktree")
                summary_template = build_summary_template(
                    individual_id=agent_target_worktree.individual_id,
                    generation=generation,
                    parent_id=parent_program.id,
                    parent_commit=agent_target_worktree.parent_commit,
                )
                summary_written = await write_file_async(
                    str(required_summary_path),
                    summary_template,
                )
                if not summary_written:
                    raise RuntimeError(
                        f"Failed to write summary template: {required_summary_path}"
                    )
                policy_path = self.repo_worktree_manager.write_policy_files(
                    agent_target_worktree,
                    prompt_text=(
                        "# Headless System Prompt\n\n"
                        f"{patch_sys}\n\n"
                        "# Headless Patch Message\n\n"
                        f"{patch_msg}\n"
                    ),
                )
                self.repo_worktree_manager.write_parent_id(
                    agent_target_worktree,
                    parent_program.id,
                )

            total_costs = 0.0

            # Use provided model_sample_probs (selected once before all loops)
            llm_kwargs = self.llm.get_kwargs(model_sample_probs=model_sample_probs)
            headless_session_name = None
            if agent_target_worktree is not None:
                session_short_id = agent_target_worktree.individual_id.replace("-", "")[:12]
                headless_session_name = f"shinka-gen-{generation}-{session_short_id}"
                llm_kwargs = {
                    **llm_kwargs,
                    "headless_work_dir": str(agent_target_worktree.path),
                    "headless_session_name": headless_session_name,
                }

            # Update LLM selection with submission
            model_name = llm_kwargs.get("model_name", "unknown")
            if self.llm_selection is not None:
                self.llm_selection.update_submitted(model_name)

            for patch_attempt in range(self.evo_config.max_patch_attempts):
                summary_version = None
                diff_summary: Dict[str, Any] = {}
                summary_path = None
                last_diff = None
                response_kwargs = {}
                response = None

                try:
                    response = await self.llm.query(
                        msg=patch_msg,
                        system_msg=patch_sys,
                        llm_kwargs=llm_kwargs,
                        model_sample_probs=model_sample_probs,
                        model_posterior=model_posterior,
                    )
                except LLMRouteError as exc:
                    last_error_msg = str(exc)
                    terminal_model_failure = True
                    route_failure_class = exc.failure_class
                    response_kwargs = dict(exc.artifacts)
                    if agent_target_worktree is not None:
                        failed_snapshot = self.repo_worktree_manager.diff_parent(
                            agent_target_worktree.path,
                            agent_target_worktree.parent_commit,
                        )
                        last_diff = failed_snapshot.diff or None
                        last_diff_summary = {
                            "changed_files": failed_snapshot.changed_files,
                            "diff_stat": failed_snapshot.diff_stat.strip(),
                            "status": failed_snapshot.status,
                        }
                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=None,
                        error_msg=str(exc),
                        patch_text=last_diff,
                        num_applied=len(last_diff_summary.get("changed_files", [])),
                        patch_name=patch_name,
                        patch_description=patch_description,
                        success=False,
                        headless_artifacts=exc.artifacts,
                    )
                    break

                if not response:
                    error_str = "Agent response was None."
                    last_error_msg = error_str
                    last_diff_summary = {}
                    last_summary_version = None
                    terminal_model_failure = True

                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=error_str,
                        patch_text=last_diff,
                        num_applied=0,
                        patch_name=patch_name,
                        patch_description=patch_description,
                        success=False,
                    )

                    if (
                        patch_attempt < self.evo_config.max_patch_attempts - 1
                        and not terminal_model_failure
                    ):
                        continue
                    break

                total_costs += response.cost if response.cost else 0.0
                response_kwargs = getattr(response, "kwargs", {}) or {}
                agent_dir_raw = response_kwargs.get("headless_work_dir") or llm_kwargs.get(
                    "headless_work_dir"
                )
                error_str = None
                summary_text = ""
                num_applied = 0

                if agent_dir_raw is None:
                    error_str = "Headless agent did not report a working directory."
                else:
                    agent_dir = Path(agent_dir_raw).resolve()
                    summary_path = agent_dir / required_summary_name

                    expected_agent_dir = (
                        agent_target_worktree.path.resolve()
                        if agent_target_worktree is not None
                        else None
                    )
                    if expected_agent_dir is not None and agent_dir != expected_agent_dir:
                        error_str = (
                            f"Headless agent ran in unexpected worktree: {agent_dir}"
                        )
                    elif not summary_path.is_file():
                        error_str = (
                            f"Agent did not create required {required_summary_name} "
                            f"in {agent_dir}"
                        )
                    else:
                        summary_text = summary_path.read_text(encoding="utf-8")
                        if not summary_text.strip():
                            error_str = (
                                f"Agent created an empty {required_summary_name} "
                                f"in {agent_dir}"
                            )
                        else:
                            summary_check = validate_summary(
                                summary_text,
                                max_chars=self.evo_config.summary_max_chars,
                            )
                            summary_version = summary_check.schema_version
                            if not summary_check.valid:
                                error_str = "; ".join(summary_check.errors)

                    if agent_target_worktree is not None:
                        source_snapshot = self.repo_worktree_manager.diff_parent(
                            agent_target_worktree.path,
                            agent_target_worktree.parent_commit,
                        )
                        last_diff = source_snapshot.diff
                        num_applied = len(source_snapshot.changed_files)
                        diff_summary = {
                            "changed_files": source_snapshot.changed_files,
                            "diff_stat": source_snapshot.diff_stat.strip(),
                        }
                        try:
                            self.repo_worktree_manager.validate_snapshot(
                                agent_target_worktree,
                                source_snapshot,
                            )
                        except Exception as exc:
                            error_str = str(exc)

                        if error_str is None and num_applied == 0:
                            error_str = "Agent did not apply any changes to the worktree."

                        if error_str is None and worktree is not None:
                            if agent_worktree is not None:
                                self.repo_worktree_manager.apply_agent_view_changes(
                                    agent_worktree,
                                    worktree,
                                )
                            configured_summary_path = (
                                worktree.path / self.evo_config.summary_filename
                            )
                            configured_summary_path.parent.mkdir(
                                parents=True,
                                exist_ok=True,
                            )
                            configured_summary_path.write_text(
                                summary_text,
                                encoding="utf-8",
                            )
                            canonical_snapshot = self.repo_worktree_manager.diff_parent(
                                worktree.path,
                                worktree.parent_commit,
                            )
                            self.repo_worktree_manager.validate_snapshot(
                                worktree,
                                canonical_snapshot,
                            )
                            last_diff = canonical_snapshot.diff
                            num_applied = len(canonical_snapshot.changed_files)
                            diff_summary = {
                                "changed_files": canonical_snapshot.changed_files,
                                "diff_stat": canonical_snapshot.diff_stat.strip(),
                            }
                    else:
                        last_diff = response.content or None
                        num_applied = 1 if summary_path and summary_path.is_file() else 0
                        diff_summary = {
                            "summary_path": str(summary_path) if summary_path else None
                        }

                last_diff_summary = diff_summary
                last_summary_version = summary_version
                last_error_msg = error_str

                if error_str is None:
                    self.route_health.record_success(model_name)
                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=None,
                        patch_text=last_diff,
                        num_applied=num_applied,
                        patch_name=patch_name,
                        patch_description=patch_description,
                        success=True,
                    )

                    # Update LLM selection costs
                    if self.llm_selection is not None:
                        self.llm_selection.update_cost(arm=model_name, cost=total_costs)

                    meta_patch_data = {
                        "api_costs": total_costs,
                        "patch_type": patch_type,
                        "patch_name": patch_name,
                        "patch_description": patch_description,
                        "num_applied": num_applied,
                        "error_attempt": None,
                        "diff_summary": diff_summary,
                        "summary_path": str(summary_path) if summary_path else None,
                        "summary_version": summary_version,
                        "novelty_attempt": novelty_attempt,
                        "resample_attempt": resample_attempt,
                        "patch_attempt": patch_attempt + 1,
                        "headless_session_name": response_kwargs.get(
                            "headless_session_name", headless_session_name
                        ),
                        "headless_session_id": response_kwargs.get(
                            "headless_session_id", headless_session_name
                        ),
                        "headless_usage_unknown": response_kwargs.get(
                            "headless_usage_unknown", True
                        ),
                        "repo_policy_path": str(policy_path) if policy_path else None,
                        "headless_prompt_path": response_kwargs.get("headless_prompt_path"),
                        "headless_stdout_path": response_kwargs.get("headless_stdout_path"),
                        "headless_stderr_path": response_kwargs.get("headless_stderr_path"),
                        **llm_kwargs,
                        "llm_result": response.to_dict() if response else None,
                    }
                    if not fix_mode:
                        # Track evolved prompt
                        meta_patch_data["system_prompt_id"] = current_prompt_id

                    # Print metadata table for successful patches
                    if self.verbose:
                        if fix_mode:
                            logger.info(
                                f"  FIX ATTEMPT {patch_attempt + 1}/"
                                f"{self.evo_config.max_patch_attempts} SUCCESS"
                            )
                        self._print_metadata_table(meta_patch_data, generation)

                    return last_diff, meta_patch_data, True
                else:
                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=error_str,
                        patch_text=last_diff,
                        num_applied=num_applied,
                        patch_name=patch_name,
                        patch_description=patch_description,
                        success=False,
                    )

                    if self.verbose and fix_mode:
                        logger.info(
                            f"  FIX ATTEMPT {patch_attempt + 1}/"
                            f"{self.evo_config.max_patch_attempts} FAILURE: {error_str}"
                        )

                    if patch_attempt < self.evo_config.max_patch_attempts - 1:
                        patch_msg = (
                            f"{patch_msg}\n\n## Retry Instructions\n\n"
                            f"The previous attempt failed because: {error_str}\n"
                            "Retry in the same worktree, keep edits within the "
                            "allowed paths, and make sure "
                            f"`{required_summary_name}` is valid when you finish.\n"
                        )

            # Update LLM selection costs
            if self.llm_selection is not None:
                self.llm_selection.update_cost(arm=model_name, cost=total_costs)

            # All attempts failed
            meta_patch_data = {
                "api_costs": total_costs,
                "patch_type": patch_type,
                "patch_name": patch_name if "patch_name" in locals() else None,
                "patch_description": patch_description
                if "patch_description" in locals()
                else None,
                "error_attempt": (
                    "Agent returned no response after model retries."
                    if terminal_model_failure
                    else (
                        "Max fix attempts reached without success."
                        if fix_mode
                        else "Max attempts reached without successful patch"
                    )
                ),
                "last_error_msg": last_error_msg,
                "diff_summary": last_diff_summary,
                "summary_path": str(summary_path) if summary_path else None,
                "summary_version": last_summary_version,
                "novelty_attempt": novelty_attempt,
                "resample_attempt": resample_attempt,
                "patch_attempt": patch_attempt + 1,
                "terminal_model_failure": terminal_model_failure,
                "route_failure_class": (
                    route_failure_class
                    if "route_failure_class" in locals()
                    else None
                ),
                "headless_session_name": response_kwargs.get(
                    "headless_session_name",
                    llm_kwargs.get("headless_session_name") if "llm_kwargs" in locals() else None,
                ),
                "headless_session_id": response_kwargs.get(
                    "headless_session_id",
                    llm_kwargs.get("headless_session_name") if "llm_kwargs" in locals() else None,
                ),
                "headless_usage_unknown": response_kwargs.get(
                    "headless_usage_unknown", True
                ),
                "repo_policy_path": str(policy_path) if policy_path else None,
                "headless_prompt_path": response_kwargs.get("headless_prompt_path"),
                "headless_stdout_path": response_kwargs.get("headless_stdout_path"),
                "headless_stderr_path": response_kwargs.get("headless_stderr_path"),
                **llm_kwargs,
                "llm_result": response.to_dict()
                if "response" in locals() and response
                else None,
            }
            if fix_mode:
                meta_patch_data["num_applied"] = 0
            else:
                # Track evolved prompt
                meta_patch_data["system_prompt_id"] = current_prompt_id

            return None, meta_patch_data, False

        except Exception as e:
            if fix_mode:
                logger.error(f"Error in fix patch async: {e}")
                return None, {"api_costs": 0.0, "error_attempt": str(e)}, False

            logger.error(f"Error in async patch generation: {e}")
            # Restore original task_sys_msg in case of exception
            self.prompt_sampler.task_sys_msg = original_task_sys_msg
            return (
                None,
                {
                    "api_costs": 0.0,
                    "error_attempt": str(e),
                    "system_prompt_id": current_prompt_id,
                },
                False,
            )

    async def _persist_program_metadata_async(self, program: Program) -> None:
        """Persist updated program metadata using a thread-local DB handle."""
        if not program.metadata:
            return

        if hasattr(self.async_db, "update_program_metadata_async"):
            await self.async_db.update_program_metadata_async(
                program.id, dict(program.metadata)
            )
            return

        def update_metadata():
            from shinka.database import ProgramDatabase

            thread_db = ProgramDatabase(self.db.config)
            try:
                metadata_json = json.dumps(program.metadata)
                thread_db.cursor.execute(
                    "UPDATE programs SET metadata = ? WHERE id = ?",
                    (metadata_json, program.id),
                )
                thread_db.conn.commit()
            finally:
                thread_db.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_metadata)

    def _queue_failed_db_job(
        self, job: AsyncRunningJob, *, log_prefix: str, error_message: str
    ) -> None:
        """Queue a completed job for DB retry bookkeeping."""
        job.db_retry_count += 1
        logger.error(
            "%s %s (retry %s/%s)",
            log_prefix,
            error_message,
            job.db_retry_count,
            self.MAX_DB_RETRY_ATTEMPTS,
        )

        if job.db_retry_count < self.MAX_DB_RETRY_ATTEMPTS:
            self.failed_jobs_for_retry[str(job.job_id)] = job
            logger.info(
                "🔄 RETRY QUEUED: Job %s (gen %s) added to retry queue",
                job.job_id,
                job.generation,
            )
            return

        self.failed_jobs_for_retry.pop(str(job.job_id), None)
        logger.error(
            "❌ RETRY EXHAUSTED: Job %s (gen %s) exceeded max retry attempts (%s). Job permanently lost.",
            job.job_id,
            job.generation,
            self.MAX_DB_RETRY_ATTEMPTS,
        )

    async def _persist_failed_generation(
        self,
        *,
        generation: int,
        exec_fname: str,
        proposal_started_at: float,
        sampling_worker_id: Optional[int],
        active_proposals_at_start: int,
        parent_program: Optional[Program],
        archive_programs: List[Program],
        top_k_programs: List[Program],
        code_diff: Optional[str],
        meta_patch_data: Optional[Dict[str, Any]],
        code_embedding: Optional[List[float]],
        embed_cost: float,
        novelty_cost: float,
        api_costs: float,
        failure_stage: str,
        failure_reason: str,
    ) -> Optional[Program]:
        """Persist a pre-evaluation failure as an incorrect program row."""
        sampling_finished_at = time.time()
        postprocess_started_at = sampling_finished_at
        postprocess_finished_at = postprocess_started_at
        source_job_id = f"failed:{failure_stage}:{generation}"
        try:
            exec_path = Path(exec_fname)
            code = ""
            if exec_path.exists():
                code = await self._read_file_async(exec_fname) or ""
            failure_class = self._classify_failed_proposal(
                failure_stage=failure_stage,
                failure_reason=failure_reason,
                meta_patch_data=meta_patch_data,
            )
            failure_json_path, failure_payload = await self._write_failure_artifact_async(
                generation=generation,
                exec_fname=exec_fname,
                parent_program=parent_program,
                archive_programs=archive_programs,
                top_k_programs=top_k_programs,
                meta_patch_data=meta_patch_data,
                embed_cost=embed_cost,
                novelty_cost=novelty_cost,
                api_costs=api_costs,
                failure_stage=failure_stage,
                failure_class=failure_class,
                failure_reason=failure_reason,
                code_embedding=code_embedding,
                proposal_started_at=proposal_started_at,
                sampling_worker_id=sampling_worker_id,
                active_proposals_at_start=active_proposals_at_start,
            )
            metadata = with_pipeline_timing(
                {
                    **(meta_patch_data or {}),
                    "node_kind": "failed_proposal",
                    "api_costs": api_costs,
                    "embed_cost": embed_cost,
                    "novelty_cost": novelty_cost,
                    "failure_stage": failure_stage,
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                    "failure_persisted": True,
                    "results_missing": True,
                    "safe_processing": False,
                    "downstream_eval_submitted": False,
                    "failure_json_path": failure_json_path,
                    "generated_code_available": bool(code.strip()),
                    "failure_artifacts": failure_payload.get("artifacts", {}),
                    "source_job_id": source_job_id,
                    "source_generation": generation,
                    "stdout_log": "",
                    "stderr_log": failure_reason,
                    "sampling_worker_id": sampling_worker_id,
                    "evaluation_worker_id": None,
                    "postprocess_worker_id": None,
                    "active_proposals_at_start": active_proposals_at_start,
                    "running_eval_jobs_at_submit": 0,
                    "timeline_lane_mode": "pool_slots",
                    "sampling_worker_capacity": self.max_proposal_jobs,
                    "evaluation_worker_capacity": self.max_evaluation_jobs,
                    "postprocess_worker_capacity": self.max_db_workers,
                },
                pipeline_started_at=proposal_started_at,
                sampling_started_at=proposal_started_at,
                sampling_finished_at=sampling_finished_at,
                evaluation_started_at=sampling_finished_at,
                evaluation_finished_at=sampling_finished_at,
                postprocess_started_at=postprocess_started_at,
                postprocess_finished_at=postprocess_finished_at,
            )
            program = Program(
                id=str(uuid.uuid4()),
                code=code,
                generation=generation,
                correct=False,
                combined_score=0.0,
                public_metrics={},
                private_metrics={},
                text_feedback=failure_reason,
                timestamp=datetime.now().timestamp(),
                parent_id=parent_program.id if parent_program else None,
                archive_inspiration_ids=[p.id for p in archive_programs],
                top_k_inspiration_ids=[p.id for p in top_k_programs],
                code_diff=code_diff,
                embedding=code_embedding or [],
                system_prompt_id=(meta_patch_data or {}).get("system_prompt_id"),
                metadata=metadata,
            )

            await self.async_db.add_program_async(
                program,
                parent_id=program.parent_id,
                archive_insp_ids=program.archive_inspiration_ids,
                top_k_insp_ids=program.top_k_inspiration_ids,
                code_diff=program.code_diff,
                meta_patch_data=meta_patch_data,
                code_embedding=program.embedding,
                embed_cost=embed_cost,
                verbose=self.verbose,
                defer_maintenance=True,
            )

            await self._update_completed_generations()
            self._record_progress()
            self.slot_available.set()
            self._log_program_to_wandb(program)
            logger.info(
                "Persisted failed generation %s as incorrect program %s (%s)",
                generation,
                program.id,
                failure_stage,
            )
            return program
        except Exception as e:
            logger.error(
                "Failed to persist failed generation %s (%s): %r",
                generation,
                failure_stage,
                e,
            )
            return None

    async def _persist_completed_job(
        self, job: AsyncRunningJob
    ) -> CompletedJobPersistResult:
        """Persist a completed evaluation job without blocking on slower side effects."""
        postprocess_worker_id = None
        source_job_id = str(job.job_id)
        try:
            logger.info(
                f"🔄 SAFE PROCESSING: Starting job {job.job_id} (gen {job.generation})"
            )

            if job.discard_if_completed:
                logger.info(
                    f"⏭️  DISCARD SURPLUS: Skipping persistence for {job.job_id} "
                    f"(gen {job.generation}) after target was already reached"
                )
                return CompletedJobPersistResult(job=job, success=True)

            # Get job results with timeout to prevent hanging
            try:
                results = await asyncio.wait_for(
                    self.scheduler.get_job_results_async(job.job_id, job.results_dir),
                    timeout=30.0,  # 30 second timeout
                )
                results_retrieved_at = time.time()
                if job.results_retrieved_at is None:
                    job.results_retrieved_at = results_retrieved_at
                evaluation_finished_at = (
                    job.completion_detected_at or job.results_retrieved_at
                )
                logger.info(
                    f"📂 RESULTS: Got results for {job.job_id}: {results is not None}"
                )
            except asyncio.TimeoutError:
                logger.error(f"❌ TIMEOUT: Getting results for {job.job_id} timed out")
                await self._record_generation_event(
                    generation=job.generation,
                    status="results_timeout",
                    source_job_id=job.job_id,
                )
                return CompletedJobPersistResult(job=job, success=False)
            await self._release_evaluation_slot_once(job)
            postprocess_worker_id = await self.postprocess_slot_pool.acquire()
            db_workers_in_use_at_postprocess_start = self.postprocess_slot_pool.in_use
            postprocess_started_at = time.time()

            # Always create a program entry, even if results are missing
            if results:
                # Extract metrics properly like the sync version
                correct_val = results.get("correct", {}).get("correct", False)
                metrics_val = results.get("metrics", {})
                combined_score = metrics_val.get("combined_score", 0.0)
                public_metrics = metrics_val.get("public", {})
                private_metrics = metrics_val.get("private", {})
                text_feedback = metrics_val.get("text_feedback", "")
                stdout_log = truncate_log_tail(
                    results.get("stdout_log", ""),
                    self.db_config.max_stdout_log_chars,
                )
                stderr_log = results.get("stderr_log", "")

                logger.info(
                    f"✅ VALID RESULTS: {job.job_id} has valid results - correct={correct_val}, score={combined_score}"
                )
            else:
                # Handle missing results - don't lose the job!
                logger.warning(
                    f"⚠️  NO RESULTS: {job.job_id} (gen {job.generation}) has no results. "
                    f"Creating program entry with default values to avoid job loss."
                )
                correct_val = False
                combined_score = 0.0
                public_metrics = {}
                private_metrics = {}
                text_feedback = "Job completed but results could not be retrieved"
                stdout_log = ""
                stderr_log = "Results retrieval failed"

            # Extract system_prompt_id from meta_patch_data
            system_prompt_id = None
            if job.meta_patch_data:
                system_prompt_id = job.meta_patch_data.get("system_prompt_id")

            # Create repo from results (or defaults if results missing)
            evaluation_started_at = (
                job.evaluation_started_at or job.evaluation_submitted_at
            )
            persisted_summary = job.repo_summary or ""
            program = Program(
                id=job.individual_id or str(uuid.uuid4()),
                code=persisted_summary,
                language="repo",
                individual_type="repo",
                generation=job.generation,
                correct=correct_val,
                combined_score=combined_score,
                public_metrics=public_metrics,
                private_metrics=private_metrics,
                text_feedback=text_feedback,
                timestamp=datetime.now().timestamp(),
                parent_id=job.parent_id,
                archive_inspiration_ids=job.archive_insp_ids,
                top_k_inspiration_ids=job.top_k_insp_ids,
                code_diff=job.code_diff,
                repo_commit=job.repo_commit,
                repo_parent_commit=job.repo_parent_commit,
                repo_diff=job.code_diff,
                repo_summary=job.repo_summary,
                summary_version=job.summary_version,
                changed_files=job.changed_files,
                artifact_uri=job.repo_path,
                mutable_paths=getattr(self.evo_config, "mutable_paths", []),
                immutable_paths=getattr(self.evo_config, "immutable_paths", []),
                agent_session_id=job.agent_session_id,
                agent_session_name=job.agent_session_name,
                agent_provider=job.agent_provider,
                agent_model=job.agent_model,
                embedding=job.code_embedding or [],
                system_prompt_id=system_prompt_id,  # Track evolved prompt
                metadata=with_pipeline_timing(
                    {
                        **(job.meta_patch_data or {}),
                        "embed_cost": job.embed_cost,
                        "novelty_cost": job.novelty_cost,
                        "stdout_log": stdout_log,
                        "stderr_log": stderr_log,
                        "results_missing": results is None,
                        "safe_processing": True,
                        "source_job_id": source_job_id,
                        "source_generation": job.generation,
                        "timeline_lane_mode": "pool_slots",
                        "sampling_worker_id": job.sampling_worker_id,
                        "evaluation_worker_id": job.evaluation_worker_id,
                        "postprocess_worker_id": postprocess_worker_id,
                        "active_proposals_at_start": job.active_proposals_at_start,
                        "running_eval_jobs_at_submit": job.running_eval_jobs_at_submit,
                        "db_workers_in_use_at_postprocess_start": db_workers_in_use_at_postprocess_start,
                        "sampling_worker_capacity": self.max_proposal_jobs,
                        "evaluation_worker_capacity": self.max_evaluation_jobs,
                        "postprocess_worker_capacity": self.max_db_workers,
                    },
                    pipeline_started_at=job.proposal_started_at,
                    sampling_started_at=job.proposal_started_at,
                    sampling_finished_at=evaluation_started_at,
                    evaluation_started_at=evaluation_started_at,
                    evaluation_finished_at=evaluation_finished_at,
                    postprocess_started_at=postprocess_started_at,
                    postprocess_finished_at=postprocess_started_at,
                ),
            )

            # Add to database with timeout protection
            logger.info(
                f"💾 DB ADD: Adding program to database for {job.job_id} (gen {job.generation})..."
            )

            try:
                added = await asyncio.wait_for(
                    self.async_db.add_program_async(
                        program,
                        parent_id=job.parent_id,
                        archive_insp_ids=job.archive_insp_ids,
                        top_k_insp_ids=job.top_k_insp_ids,
                        code_diff=job.code_diff,
                        meta_patch_data=job.meta_patch_data,
                        code_embedding=job.code_embedding,
                        embed_cost=job.embed_cost,
                        verbose=self.verbose,
                        defer_maintenance=True,
                    ),
                    timeout=90.0,  # 90 second timeout for DB operations
                )
                if added:
                    logger.info(
                        f"✅ DB SUCCESS: Program {program.id} successfully added to database for {job.job_id} (gen {job.generation})"
                    )
                else:
                    existing_program = None
                    if hasattr(self.async_db, "get_program_by_source_job_id_async"):
                        existing_program = (
                            await self.async_db.get_program_by_source_job_id_async(
                                source_job_id
                            )
                        )

                    if existing_program is None:
                        self._queue_failed_db_job(
                            job,
                            log_prefix="⏳ DB STILL IN FLIGHT:",
                            error_message=(
                                "Program insert is still in flight for "
                                f"{job.job_id}; will retry side effects"
                            ),
                        )
                        return CompletedJobPersistResult(job=job, success=False)

                    postprocess_finished_at = time.time()
                    existing_metadata = existing_program.metadata or {}
                    existing_program.metadata = with_pipeline_timing(
                        existing_metadata,
                        pipeline_started_at=float(
                            existing_metadata.get(
                                "pipeline_started_at", job.proposal_started_at
                            )
                        ),
                        sampling_started_at=float(
                            existing_metadata.get(
                                "sampling_started_at", job.proposal_started_at
                            )
                        ),
                        sampling_finished_at=float(
                            existing_metadata.get(
                                "sampling_finished_at", evaluation_started_at
                            )
                        ),
                        evaluation_started_at=float(
                            existing_metadata.get(
                                "evaluation_started_at", evaluation_started_at
                            )
                        ),
                        evaluation_finished_at=float(
                            existing_metadata.get(
                                "evaluation_finished_at", evaluation_finished_at
                            )
                        ),
                        postprocess_started_at=float(
                            existing_metadata.get(
                                "postprocess_started_at", postprocess_started_at
                            )
                        ),
                        postprocess_finished_at=max(
                            float(
                                existing_metadata.get(
                                    "postprocess_finished_at", postprocess_started_at
                                )
                            ),
                            postprocess_finished_at,
                        ),
                    )
                    self._record_oversubscription_timing_sample(
                        existing_program.metadata or {}
                    )

                    if (existing_program.metadata or {}).get(
                        "postprocess_side_effects_applied"
                    ):
                        logger.info(
                            "⏭️  SKIP DUPLICATE SIDE EFFECTS: Job %s already fully processed",
                            job.job_id,
                        )
                        return CompletedJobPersistResult(job=job, success=True)

                    logger.info(
                        "♻️  REUSE DUPLICATE ROW: Job %s already persisted as program %s",
                        job.job_id,
                        existing_program.id,
                    )
                    return CompletedJobPersistResult(
                        job=job,
                        success=True,
                        persisted_event=self._make_persisted_event(
                            job=job,
                            program=existing_program,
                            evaluation_finished_at=float(
                                (existing_program.metadata or {}).get(
                                    "evaluation_finished_at", evaluation_finished_at
                                )
                            ),
                            postprocess_started_at=float(
                                (existing_program.metadata or {}).get(
                                    "postprocess_started_at", postprocess_started_at
                                )
                            ),
                            postprocess_finished_at=float(
                                (existing_program.metadata or {}).get(
                                    "postprocess_finished_at",
                                    postprocess_finished_at,
                                )
                            ),
                        ),
                    )

            except asyncio.TimeoutError:
                self._queue_failed_db_job(
                    job,
                    log_prefix="❌ DB TIMEOUT:",
                    error_message=(
                        f"Adding program to database for {job.job_id} timed out"
                    ),
                )
                return CompletedJobPersistResult(job=job, success=False)
            except Exception as e:
                self._queue_failed_db_job(
                    job,
                    log_prefix="❌ DB ERROR:",
                    error_message=(
                        f"Failed to add program to database for {job.job_id}: {e}"
                    ),
                )
                return CompletedJobPersistResult(job=job, success=False)

            postprocess_finished_at = time.time()
            program.metadata = with_pipeline_timing(
                program.metadata,
                pipeline_started_at=job.proposal_started_at,
                sampling_started_at=job.proposal_started_at,
                sampling_finished_at=evaluation_started_at,
                evaluation_started_at=evaluation_started_at,
                evaluation_finished_at=evaluation_finished_at,
                postprocess_started_at=postprocess_started_at,
                postprocess_finished_at=postprocess_finished_at,
            )
            self._record_oversubscription_timing_sample(program.metadata or {})

            return CompletedJobPersistResult(
                job=job,
                success=True,
                persisted_event=self._make_persisted_event(
                    job=job,
                    program=program,
                    evaluation_finished_at=evaluation_finished_at,
                    postprocess_started_at=postprocess_started_at,
                    postprocess_finished_at=postprocess_finished_at,
                ),
            )

        except Exception as e:
            logger.error(
                f"❌ CRITICAL: Exception in safe processing for job {job.job_id} (gen {job.generation}): {e}"
            )
            logger.error(
                f"   Job details: exec_fname={job.exec_fname}, results_dir={job.results_dir}"
            )
            return CompletedJobPersistResult(job=job, success=False)
        finally:
            await self._release_evaluation_slot_once(job)
            await self.postprocess_slot_pool.release(postprocess_worker_id)

    async def _apply_persisted_program_side_effects(
        self, persisted_event: PersistedProgramEvent
    ) -> None:
        """Apply slower post-persistence side effects for one completed program."""
        job = persisted_event.job
        program = persisted_event.program
        apply_started_at = time.time()
        metadata_persist_needed = False
        side_effects_completed = False

        try:
            if hasattr(self.async_db, "run_program_maintenance_async"):
                await self.async_db.run_program_maintenance_async(
                    program,
                    verbose=self.verbose,
                )

            system_prompt_id = None
            if job.meta_patch_data:
                system_prompt_id = job.meta_patch_data.get("system_prompt_id")

            # Update prompt fitness if prompt evolution is enabled
            if system_prompt_id and self.evo_config.evolve_prompts:
                prompt_lock = getattr(self, "_prompt_side_effect_lock", None)
                if prompt_lock is None:
                    prompt_lock = asyncio.Lock()
                    self._prompt_side_effect_lock = prompt_lock
                async with prompt_lock:
                    parent_score = 0.0
                    if job.parent_id:
                        parent_program = await self.async_db.get_async(job.parent_id)
                        if parent_program:
                            parent_score = parent_program.combined_score or 0.0

                    program_score = program.combined_score or 0.0
                    improvement = program_score - parent_score
                    await self._update_prompt_fitness(
                        system_prompt_id,
                        program.id,
                        program_score=program_score,
                        improvement=improvement,
                        correct=program.correct,
                    )
                    await self._maybe_evolve_prompt()

            if self.meta_summarizer:
                try:
                    meta_lock = getattr(self, "_meta_side_effect_lock", None)
                    if meta_lock is None:
                        meta_lock = asyncio.Lock()
                        self._meta_side_effect_lock = meta_lock
                    async with meta_lock:
                        self.meta_summarizer.add_evaluated_program(program)

                        if self.meta_summarizer.should_update_meta(
                            self.evo_config.meta_rec_interval
                        ):
                            logger.info("Updating meta memory...")
                            best_program = await self.async_db.get_best_program_async()
                            # Use async meta summarizer for non-blocking meta analysis
                            (
                                updated_recs,
                                meta_cost,
                            ) = await self.meta_summarizer.update_meta_memory_async(
                                best_program
                            )
                            if updated_recs:
                                # Write meta output file asynchronously
                                await self.meta_summarizer.write_meta_output_async(
                                    str(self.results_dir)
                                )
                                if meta_cost > 0:
                                    logger.info(
                                        f"Meta recommendation cost: ${meta_cost:.4f}"
                                    )
                                    # Add meta cost to in-memory total for accurate budget tracking
                                    self.total_api_cost += meta_cost

                                    # Add meta cost to this program's metadata
                                    if program.metadata is None:
                                        program.metadata = {}
                                    program.metadata["meta_cost"] = meta_cost
                                    metadata_persist_needed = True
                except Exception as e:
                    logger.warning(f"Meta summarizer error for {job.job_id}: {e}")
                    # Don't fail the whole job for meta summarizer issues

            # Update LLM selection
            if self.llm_selection is not None and "model_name" in (
                program.metadata or {}
            ):
                try:
                    parent = None
                    if program.parent_id:
                        parent = await self.async_db.get_async(program.parent_id)
                    baseline = parent.combined_score if parent else None
                    reward = program.combined_score if program.correct else None
                    model_name = program.metadata["model_name"]
                    self.llm_selection.update(
                        arm=model_name, reward=reward, baseline=baseline
                    )
                    if self.verbose:
                        self.llm_selection.print_summary(console=self.console)
                except Exception as e:
                    logger.warning(f"LLM selection update error for {job.job_id}: {e}")
                    # Don't fail the whole job for LLM selection issues

            # Update best solution
            try:
                await self._update_best_solution_async()
            except Exception as e:
                logger.warning(f"Best solution update error for {job.job_id}: {e}")
                # Don't fail the whole job for best solution update issues

            side_effects_completed = True

        finally:
            if side_effects_completed:
                base_metadata = program.metadata or {}
                program.metadata = with_pipeline_timing(
                    base_metadata,
                    pipeline_started_at=float(
                        base_metadata.get("pipeline_started_at", job.proposal_started_at)
                    ),
                    sampling_started_at=float(
                        base_metadata.get("sampling_started_at", job.proposal_started_at)
                    ),
                    sampling_finished_at=float(
                        base_metadata.get(
                            "sampling_finished_at",
                            job.evaluation_started_at
                            or job.evaluation_submitted_at
                            or job.proposal_started_at,
                        )
                    ),
                    evaluation_started_at=float(
                        base_metadata.get(
                            "evaluation_started_at",
                            job.evaluation_started_at
                            or job.evaluation_submitted_at
                            or job.proposal_started_at,
                        )
                    ),
                    evaluation_finished_at=float(
                        base_metadata.get(
                            "evaluation_finished_at",
                            persisted_event.evaluation_finished_at,
                        )
                    ),
                    postprocess_started_at=float(
                        base_metadata.get(
                            "postprocess_started_at",
                            persisted_event.postprocess_started_at,
                        )
                    ),
                    postprocess_finished_at=max(
                        float(
                            base_metadata.get(
                                "postprocess_finished_at",
                                persisted_event.postprocess_finished_at,
                            )
                        ),
                        float(persisted_event.postprocess_finished_at),
                    ),
                )
                program.metadata = with_side_effect_timing(
                    program.metadata,
                    apply_started_at=apply_started_at,
                    apply_finished_at=time.time(),
                )
                program.metadata["postprocess_side_effects_applied"] = True
                metadata_persist_needed = True
            if metadata_persist_needed:
                try:
                    await self._persist_program_metadata_async(program)
                except Exception as e:
                    logger.warning(
                        f"Apply-stage metadata persistence error for {job.job_id}: {e}"
                    )

        self._log_program_to_wandb(program)
        logger.info(
            "✅ JOB COMPLETE: Finished processing %s - program %s added (gen %s)",
            job.job_id,
            program.id,
            job.generation,
        )

    def _make_persisted_event(
        self,
        job: AsyncRunningJob,
        program: Program,
        evaluation_finished_at: float,
        postprocess_started_at: float,
        postprocess_finished_at: float,
    ) -> PersistedProgramEvent:
        """Construct a persisted-program event for follow-up side effects."""
        return PersistedProgramEvent(
            job=job,
            program=program,
            evaluation_finished_at=evaluation_finished_at,
            postprocess_started_at=postprocess_started_at,
            postprocess_finished_at=postprocess_finished_at,
        )

    async def _mark_completed_jobs_detected(
        self, completed_jobs: List[AsyncRunningJob]
    ) -> None:
        """Stamp completion time and free eval slots before slower persistence."""
        detected_at = time.time()
        for job in completed_jobs:
            if job.completion_detected_at is None:
                job.completion_detected_at = detected_at
            await self._release_evaluation_slot_once(job)
        self.slot_available.set()

    def _get_completed_job_work_count(self) -> int:
        """Return queued completed-job persistence work plus the active batch."""
        pending = int(getattr(self, "_completed_jobs_pending", 0) or 0)
        if pending > 0:
            return pending
        return int(hasattr(self, "processing_lock") and self.processing_lock.locked())

    def _schedule_completed_jobs_for_processing(
        self, completed_jobs: List[AsyncRunningJob]
    ) -> None:
        """Persist completed jobs in the background without blocking monitoring."""
        if not completed_jobs:
            return

        self._completed_jobs_pending = int(
            getattr(self, "_completed_jobs_pending", 0)
        ) + len(completed_jobs)
        if getattr(self, "_completed_job_batch_tasks", None) is None:
            self._completed_job_batch_tasks = set()
        task = asyncio.create_task(
            self._process_completed_job_batch(completed_jobs),
            name=(
                "completed_job_batch_"
                + "_".join(str(job.generation) for job in completed_jobs[:3])
            ),
        )
        self._completed_job_batch_tasks.add(task)

        def _cleanup(finished_task: asyncio.Task) -> None:
            self._completed_job_batch_tasks.discard(finished_task)
            try:
                exc = finished_task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.error("Completed-job batch task failed: %s", exc)

        task.add_done_callback(_cleanup)

    async def _process_completed_job_batch(
        self, completed_jobs: List[AsyncRunningJob]
    ) -> None:
        """Serialize batch persistence while allowing the monitor to keep polling."""
        old_retry_count = len(self.failed_jobs_for_retry)
        old_completed = self.completed_generations
        try:
            async with self.processing_lock:
                self._mark_surplus_completed_jobs_for_discard(completed_jobs)
                await self._process_completed_jobs_safely(completed_jobs)
                await self._update_completed_generations()
                self.slot_available.set()
        finally:
            self._completed_jobs_pending = max(
                0,
                int(getattr(self, "_completed_jobs_pending", 0))
                - len(completed_jobs),
            )

        self._record_progress()

        if not self.verbose:
            return

        if self.completed_generations != old_completed:
            if self.evo_config.max_api_costs is not None:
                cost_str = (
                    f"${self.total_api_cost:.4f}/"
                    f"${self.evo_config.max_api_costs:.2f}"
                )
                cost_pct = (
                    self.total_api_cost / self.evo_config.max_api_costs
                ) * 100
                cost_info = f" (cost: {cost_str}, {cost_pct:.1f}%)"
            else:
                cost_info = f" (cost: ${self.total_api_cost:.4f})"

            logger.info(
                f"✅ Completed generations updated: "
                f"{old_completed} -> {self.completed_generations}{cost_info}"
            )
            return

        retry_count = len(self.failed_jobs_for_retry)
        new_retries = retry_count - old_retry_count
        running_count = len(self.running_jobs)
        at_target = self.completed_generations >= self.evo_config.num_generations

        if at_target:
            logger.debug(
                f"📊 Completed generations at target: {self.completed_generations}"
            )
        elif new_retries > 0:
            if self.evo_config.max_api_costs is not None:
                cost_str = (
                    f"${self.total_api_cost:.4f}/"
                    f"${self.evo_config.max_api_costs:.2f}"
                )
                cost_pct = (
                    self.total_api_cost / self.evo_config.max_api_costs
                ) * 100
                cost_info = f", cost: {cost_str} ({cost_pct:.1f}%)"
            else:
                cost_info = f", cost: ${self.total_api_cost:.4f}"

            logger.info(
                f"📊 Completed generations: "
                f"{self.completed_generations} "
                f"({new_retries} new jobs in retry queue, "
                f"{retry_count} total pending retry{cost_info})"
            )
        elif retry_count > 0 or running_count > 0:
            logger.debug(
                f"📊 Completed generations: "
                f"{self.completed_generations} "
                f"(running={running_count}, retry={retry_count})"
            )
        else:
            logger.warning(
                f"⚠️  Completed generations unchanged after processing jobs: "
                f"{self.completed_generations}"
            )

    async def _wait_for_completed_job_batches(self) -> None:
        """Wait until all queued completed-job persistence batches have drained."""
        while getattr(self, "_completed_job_batch_tasks", set()):
            tasks = list(self._completed_job_batch_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cancel_completed_job_batches(self) -> None:
        """Cancel background completed-job persistence during exceptional shutdown."""
        tasks = list(getattr(self, "_completed_job_batch_tasks", set()))
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._completed_job_batch_tasks.clear()
        self._completed_jobs_pending = 0

    async def _ensure_background_side_effect_worker(self) -> None:
        """Start the background side-effect worker pool if needed."""
        tasks = {
            task
            for task in getattr(self, "_background_side_effect_tasks", set())
            if not task.done()
        }
        self._background_side_effect_tasks = tasks

        if getattr(self, "side_effect_event_queue", None) is None:
            self.side_effect_event_queue = asyncio.Queue()

        target_workers = max(1, int(getattr(self, "max_db_workers", 1) or 1))
        while len(tasks) < target_workers:
            task = asyncio.create_task(
                self._background_side_effect_worker_loop(),
                name=f"background_side_effects_{len(tasks) + 1}",
            )
            tasks.add(task)

        self._background_side_effect_task = next(iter(tasks), None)

    def _get_background_side_effect_work_count(self) -> int:
        """Return queued side effects plus the currently running one."""
        pending = int(getattr(self, "_background_side_effects_pending", 0) or 0)
        busy = int(getattr(self, "_background_side_effects_busy_count", 0) or 0)
        if busy == 0:
            busy = int(bool(getattr(self, "_background_side_effects_busy", False)))
        return max(pending, busy)

    async def _enqueue_background_side_effects(
        self, persisted_events: List[PersistedProgramEvent]
    ) -> None:
        """Queue persisted events so slow side effects do not block persistence."""
        if not persisted_events:
            return

        await self._ensure_background_side_effect_worker()
        self._background_side_effects_pending = int(
            getattr(self, "_background_side_effects_pending", 0)
        )

        for persisted_event in persisted_events:
            self._background_side_effects_pending += 1
            await self.side_effect_event_queue.put(persisted_event)

        logger.debug(
            "Queued %s persisted side-effect event(s); backlog=%s",
            len(persisted_events),
            self._background_side_effects_pending,
        )

    async def _background_side_effect_worker_loop(self) -> None:
        """Drain side effects in generation order without blocking completions."""
        while True:
            persisted_event = await self.side_effect_event_queue.get()
            if persisted_event is None:
                self.side_effect_event_queue.task_done()
                break

            self._background_side_effects_busy_count = (
                int(getattr(self, "_background_side_effects_busy_count", 0)) + 1
            )
            self._background_side_effects_busy = True
            try:
                await self._apply_persisted_program_side_effects(persisted_event)
                self._record_progress()
            except Exception as e:
                logger.error(
                    "❌ APPLY ERROR: Side effects failed for job %s (gen %s): %s",
                    persisted_event.job.job_id,
                    persisted_event.job.generation,
                    e,
                )
            finally:
                self._background_side_effects_busy_count = max(
                    0,
                    int(getattr(self, "_background_side_effects_busy_count", 0)) - 1,
                )
                self._background_side_effects_busy = (
                    self._background_side_effects_busy_count > 0
                )
                self._background_side_effects_pending = max(
                    0,
                    int(getattr(self, "_background_side_effects_pending", 0)) - 1,
                )
                self.side_effect_event_queue.task_done()

    async def _wait_for_background_side_effects(self) -> None:
        """Wait until the background side-effect backlog has drained."""
        queue = getattr(self, "side_effect_event_queue", None)
        if queue is None:
            return
        await queue.join()

    async def _shutdown_background_side_effect_worker(self) -> None:
        """Gracefully stop the background side-effect worker pool after draining."""
        tasks = {
            task
            for task in getattr(self, "_background_side_effect_tasks", set())
            if not task.done()
        }
        if not tasks:
            return

        await self._wait_for_background_side_effects()
        for _ in tasks:
            await self.side_effect_event_queue.put(None)
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_side_effect_tasks.clear()
        self._background_side_effect_task = None

    async def _cancel_background_side_effect_worker(self) -> None:
        """Cancel the background side-effect worker pool during exceptional shutdown."""
        tasks = list(getattr(self, "_background_side_effect_tasks", set()))
        if not tasks:
            return
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_side_effect_tasks.clear()
        self._background_side_effect_task = None

    async def _process_completed_jobs_safely(
        self, completed_jobs: List[AsyncRunningJob]
    ):
        """Process completed jobs inline, closer to the amd_shinka execution model."""
        successfully_processed: List[AsyncRunningJob] = []

        for job in completed_jobs:
            try:
                success = await self._process_single_job_safely(job)
                if success:
                    successfully_processed.append(job)
                    if str(job.job_id) in self.submitted_jobs:
                        del self.submitted_jobs[str(job.job_id)]
                else:
                    logger.error(
                        f"❌ CRITICAL: Failed to process job {job.job_id} (gen {job.generation})"
                    )
            except Exception as e:
                logger.error(
                    f"❌ CRITICAL: Exception processing job {job.job_id} (gen {job.generation}): {e}"
                )

        logger.info(
            f"✅ Successfully processed {len(successfully_processed)}/{len(completed_jobs)} jobs"
        )

        if len(successfully_processed) < len(completed_jobs):
            failed_jobs = [
                job for job in completed_jobs if job not in successfully_processed
            ]
            failed_gens = [job.generation for job in failed_jobs]
            logger.error(
                f"❌ FAILED JOBS: {len(failed_jobs)} jobs failed processing: gens {failed_gens}"
            )
            logger.error(
                "   These jobs remain in submitted_jobs registry for potential recovery"
            )

    async def _process_single_job_safely(self, job: AsyncRunningJob) -> bool:
        """Process a single job with comprehensive error handling. Returns True on success."""
        persist_result = await self._persist_completed_job(job)
        if not persist_result.success:
            return False

        if persist_result.persisted_event is None:
            return True

        await self._apply_persisted_program_side_effects(persist_result.persisted_event)
        return True

    async def _process_completed_jobs(self, completed_jobs: List[AsyncRunningJob]):
        """Legacy method - now redirects to safe processing."""
        await self._process_completed_jobs_safely(completed_jobs)

    async def _retry_failed_db_jobs(self):
        """Retry jobs that failed to write to the database.

        This method attempts to re-process jobs that completed evaluation
        successfully but failed to write to the database due to timeouts
        or errors.
        """
        if not self.failed_jobs_for_retry:
            return

        # Take a snapshot of jobs to retry (avoid modification during iter)
        jobs_to_retry = list(self.failed_jobs_for_retry.values())

        logger.info(
            f"🔄 RETRY: Attempting to retry {len(jobs_to_retry)} failed DB jobs"
        )

        successfully_retried = []

        for job in jobs_to_retry:
            try:
                logger.info(
                    f"🔄 RETRY ATTEMPT: Retrying job {job.job_id} "
                    f"(gen {job.generation}) - "
                    f"attempt {job.db_retry_count + 1}/"
                    f"{self.MAX_DB_RETRY_ATTEMPTS}"
                )

                success = await self._process_single_job_safely(job)

                if success:
                    successfully_retried.append(job)
                    # Remove from retry queue
                    if str(job.job_id) in self.failed_jobs_for_retry:
                        del self.failed_jobs_for_retry[str(job.job_id)]
                    # Also remove from submitted_jobs
                    if str(job.job_id) in self.submitted_jobs:
                        del self.submitted_jobs[str(job.job_id)]
                    await self._update_completed_generations()
                    self._record_progress()
                    self.slot_available.set()
                    logger.info(
                        f"✅ RETRY SUCCESS: Job {job.job_id} "
                        f"(gen {job.generation}) "
                        f"successfully retried and added to database"
                    )
                else:
                    # Job will either be re-queued (if retries remain)
                    # or marked as lost
                    # This is handled in _process_single_job_safely
                    pass

            except Exception as e:
                logger.error(
                    f"❌ RETRY ERROR: Exception retrying job "
                    f"{job.job_id} (gen {job.generation}): {e}"
                )
                # Keep job in retry queue for next attempt

        if successfully_retried:
            logger.info(
                f"✅ RETRY COMPLETE: Successfully retried "
                f"{len(successfully_retried)}/{len(jobs_to_retry)} jobs"
            )

        # Log remaining failed jobs
        if self.failed_jobs_for_retry:
            logger.warning(
                f"⚠️  RETRY PENDING: {len(self.failed_jobs_for_retry)} "
                f"jobs still in retry queue"
            )

    async def _count_completed_generations_from_db(self) -> int:
        """Count persisted completed generations, excluding island copies."""
        total_programs = await self.async_db.get_total_program_count_async()
        island_copies = max(0, getattr(self.db_config, "num_islands", 1) - 1)
        completed_generations = max(0, total_programs - island_copies)
        return min(completed_generations, self.evo_config.num_generations)

    def _generation_target_mode(self) -> str:
        return getattr(self.evo_config, "generation_target_mode", "proposal_ids")

    async def _get_missing_persisted_generations(self) -> List[int]:
        """Return budgeted generations that do not yet have persisted rows."""
        if self._generation_target_mode() != "proposal_ids":
            return []
        persisted_generations = await self.async_db.get_persisted_generation_ids_async()
        budgeted_generations = set(range(self.evo_config.num_generations))
        return sorted(budgeted_generations - set(persisted_generations))

    async def _restore_resume_progress(self) -> None:
        """Restore progress counters from persisted database state."""
        self.completed_generations = await self._count_completed_generations_from_db()
        max_attempt_generation = -1
        try:
            self.db.cursor.execute("SELECT MAX(generation) FROM attempt_log")
            row = self.db.cursor.fetchone()
            if row and row[0] is not None:
                max_attempt_generation = int(row[0])
        except Exception as exc:
            logger.debug("Could not restore max attempted generation: %s", exc)
        self.next_generation_to_submit = max(
            self.db.last_iteration + 1,
            max_attempt_generation + 1,
            1,
        )

    def _get_in_flight_work_count(self) -> int:
        """Return work that is expected to complete without new proposals."""
        return (
            len(self.running_jobs)
            + len(self.active_proposal_tasks)
            + len(self.failed_jobs_for_retry)
            + self._get_completed_job_work_count()
        )

    def _has_background_side_effect_work(self) -> bool:
        """Return whether background side effects are still queued or running."""
        return self._get_background_side_effect_work_count() > 0

    def _has_persistence_work_in_progress(self) -> bool:
        """Return whether persistence or retry bookkeeping is still active."""
        return self._get_in_flight_work_count() > 0

    def _get_remaining_completed_work(self) -> int:
        """Return how many completed generations are still needed."""
        return max(
            0,
            self.evo_config.num_generations
            - self.completed_generations
            - self._get_in_flight_work_count(),
        )

    def _get_remaining_generation_slots(self) -> int:
        """Return how many proposal generations can still be assigned."""
        if self._generation_target_mode() == "evaluated_candidates":
            return max(0, self.evo_config.num_generations - self.completed_generations)
        return max(0, self.evo_config.num_generations - self.next_generation_to_submit)

    def _mark_surplus_completed_jobs_for_discard(
        self, completed_jobs: List[AsyncRunningJob]
    ) -> None:
        """Mark completed jobs beyond the target budget so they are not persisted."""
        remaining_slots = max(
            0, self.evo_config.num_generations - self.completed_generations
        )
        if remaining_slots >= len(completed_jobs):
            return

        keep_job_ids = {
            id(job)
            for job in sorted(completed_jobs, key=lambda job: job.generation)[
                :remaining_slots
            ]
        }
        discarded_generations = []
        for job in completed_jobs:
            should_discard = id(job) not in keep_job_ids
            job.discard_if_completed = should_discard
            if should_discard:
                discarded_generations.append(job.generation)

        if discarded_generations:
            logger.info(
                f"🧹 Discarding {len(discarded_generations)} completed surplus "
                f"job(s) from the current batch after reaching target: "
                f"gens {sorted(discarded_generations)}"
            )

    async def _cleanup_proposal_task_state(
        self,
        generation: int,
        task_id: str,
        sampling_worker_id: Optional[int],
    ) -> None:
        """Release proposal bookkeeping after a proposal attempt finishes."""
        self.assigned_generations.discard(generation)
        if task_id in self.active_proposal_tasks:
            del self.active_proposal_tasks[task_id]

    async def _release_evaluation_slot_once(self, job: AsyncRunningJob) -> None:
        """Release an evaluation slot at most once per job."""
        if job.evaluation_slot_released:
            return
        await self.evaluation_slot_pool.release(job.evaluation_worker_id)
        job.evaluation_slot_released = True

    async def _submit_evaluation_job_with_slot(
        self,
        exec_fname: str,
        results_dir: str,
        sampling_worker_id: Optional[int],
        repo_path: Optional[str] = None,
    ) -> tuple[Union[str, Any], int, float, float, int]:
        """Reserve an evaluation slot before submitting the evaluation job."""
        if repo_path is None:
            raise ValueError("repo_path is required for evaluation")
        if sampling_worker_id is not None:
            await self.sampling_slot_pool.release(sampling_worker_id)

        evaluation_worker_id = await self.evaluation_slot_pool.acquire()
        try:
            job_id = await self.scheduler.submit_async_nonblocking(
                exec_fname, results_dir, repo_path
            )
        except Exception:
            await self.evaluation_slot_pool.release(evaluation_worker_id)
            raise

        evaluation_submitted_at = time.time()
        evaluation_started_at = evaluation_submitted_at
        running_eval_jobs_at_submit = self.evaluation_slot_pool.in_use
        return (
            job_id,
            evaluation_worker_id,
            evaluation_submitted_at,
            evaluation_started_at,
            running_eval_jobs_at_submit,
        )

    def _agent_hidden_paths_for_worktree(self, worktree_path: Path) -> List[str]:
        hidden_paths = list(getattr(self.evo_config, "agent_hidden_paths", []) or [])
        eval_program_path = getattr(self.job_config, "eval_program_path", None)
        if eval_program_path:
            eval_path = Path(str(eval_program_path))
            rel_eval_path: Optional[str] = None
            if eval_path.is_absolute():
                try:
                    rel_eval_path = eval_path.resolve().relative_to(
                        worktree_path.resolve()
                    ).as_posix()
                except ValueError:
                    rel_eval_path = None
            else:
                rel_eval_path = eval_path.as_posix()

            if rel_eval_path:
                normalized = rel_eval_path.replace("\\", "/").strip("/")
                if (
                    normalized
                    and normalized != "."
                    and not normalized.startswith("../")
                    and "/../" not in normalized
                ):
                    hidden_paths.append(normalized)

        return list(dict.fromkeys(hidden_paths))

    def _get_evaluation_runtime_limit_seconds(self) -> Optional[float]:
        """Return the wall-clock runtime limit for a single evaluation job."""
        job_timeout = getattr(self.job_config, "time", None)
        if job_timeout:
            return float(parse_time_to_seconds(job_timeout))

        if getattr(self.scheduler, "job_type", None) == "local":
            if self._evaluation_seconds_ewma is not None:
                return max(60.0, self._evaluation_seconds_ewma * 5.0 + 60.0)

        return None

    def _is_job_hung(self, job: AsyncRunningJob, now: Optional[float] = None) -> bool:
        """Detect a single evaluation job that has exceeded its runtime budget."""
        runtime_limit = self._get_evaluation_runtime_limit_seconds()
        if runtime_limit is None:
            return False

        started_at = (
            job.evaluation_started_at or job.evaluation_submitted_at or job.start_time
        )
        if started_at is None:
            return False

        current_time = time.time() if now is None else now
        return (current_time - started_at) > runtime_limit

    async def _update_completed_generations(self):
        """Update completed generations count for async evolution.

        In async evolution, generations can complete out of order. For termination
        and progress tracking, what matters is the total count of completed work,
        not whether it's contiguous. This counts all generations that have:
        1. No running jobs AND
        2. Programs in the database (successful or persisted-failed generations)
        """
        # Get all generations that have running jobs
        running_generations = {job.generation for job in self.running_jobs}

        # More efficient approach: get total program count and subtract running jobs
        # This avoids expensive per-generation database queries
        try:
            calculated_completed = await self._count_completed_generations_from_db()

            # Debug logging when count doesn't change
            if (
                self.verbose
                and hasattr(self, "completed_generations")
                and calculated_completed == self.completed_generations
            ):
                logger.debug(
                    f"📊 Completion calc: persisted={calculated_completed}, "
                    f"inflight={self._get_in_flight_work_count()}"
                )

            self.completed_generations = calculated_completed

            # Periodically save bandit state (every 5 generations)
            if self.completed_generations % 5 == 0 and self.completed_generations > 0:
                self._save_bandit_state()

        except Exception as e:
            logger.warning(f"Error in optimized completion counting: {e}")
            # Fallback to old method but with timeout protection
            await self._update_completed_generations_fallback(running_generations)

    async def _update_completed_generations_fallback(self, running_generations):
        """Fallback method for completion counting with timeout protection."""
        total_completed = 0

        # Limit the check to reasonable range to avoid infinite database queries
        target_gens = self.evo_config.num_generations
        next_gen = self.next_generation_to_submit
        max_check_gen = (
            min(target_gens, next_gen + 10)
            if self._generation_target_mode() == "proposal_ids"
            else next_gen + 10
        )

        for gen in range(max_check_gen):
            if gen not in running_generations:
                try:
                    # Add timeout to prevent hanging on individual queries
                    get_programs_coro = self.async_db.get_programs_by_generation_async(
                        gen
                    )
                    programs_in_gen = await asyncio.wait_for(
                        get_programs_coro,
                        timeout=5.0,  # 5 second timeout per generation
                    )
                    if programs_in_gen:
                        total_completed += 1
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout checking programs for generation {gen}")
                    break  # Stop checking if we hit timeouts
                except Exception as e:
                    logger.warning(f"Error checking generation {gen}: {e}")
                    continue

        self.completed_generations = total_completed

    async def _cleanup_completed_proposal_tasks(self):
        """Clean up completed proposal tasks."""
        completed_tasks = []
        for task_id, task in self.active_proposal_tasks.items():
            if task.done():
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            del self.active_proposal_tasks[task_id]

    async def _cancel_surplus_inflight_work(self) -> None:
        """Cancel or discard work that is no longer needed after hitting target."""
        if self.failed_jobs_for_retry:
            dropped_retry_gens = [
                job.generation for job in self.failed_jobs_for_retry.values()
            ]
            logger.info(
                f"🧹 Dropping {len(dropped_retry_gens)} retry-queued jobs after "
                f"reaching target: gens {dropped_retry_gens}"
            )
            for job in self.failed_jobs_for_retry.values():
                job.discard_if_completed = True
                self.submitted_jobs.pop(str(job.job_id), None)
            self.failed_jobs_for_retry.clear()

        if self.active_proposal_tasks:
            logger.info(
                f"🛑 Cancelling {len(self.active_proposal_tasks)} surplus proposal "
                "task(s) after reaching target generations"
            )
            proposal_tasks = list(self.active_proposal_tasks.values())
            for task in proposal_tasks:
                task.cancel()
            await asyncio.gather(*proposal_tasks, return_exceptions=True)
            await self._cleanup_completed_proposal_tasks()

        if self.running_jobs:
            cancelled_jobs: List[AsyncRunningJob] = []
            surviving_jobs: List[AsyncRunningJob] = []

            for job in list(self.running_jobs):
                job.discard_if_completed = True
                try:
                    cancelled = await self.scheduler.cancel_job_async(job.job_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to cancel surplus job {job.job_id} "
                        f"(gen {job.generation}): {e}"
                    )
                    cancelled = False

                if cancelled:
                    cancelled_jobs.append(job)
                    self.submitted_jobs.pop(str(job.job_id), None)
                    await self._release_evaluation_slot_once(job)
                else:
                    surviving_jobs.append(job)

            self.running_jobs = surviving_jobs

            if cancelled_jobs:
                cancelled_gens = [job.generation for job in cancelled_jobs]
                logger.info(
                    f"🧹 Cancelled {len(cancelled_jobs)} surplus evaluation job(s): "
                    f"gens {cancelled_gens}"
                )

        remaining_task_generations = {
            int(task.get_name().split("_", 1)[1])
            for task in self.active_proposal_tasks.values()
            if task.get_name().startswith("proposal_")
            and task.get_name().split("_", 1)[1].isdigit()
        }
        self.assigned_generations = {
            job.generation for job in self.running_jobs
        } | remaining_task_generations
        self.slot_available.set()

    def _is_system_stuck(self) -> bool:
        """
        Detect if the system is stuck with no progress.

        Returns True if:
        - No running evaluation jobs AND
        - No running proposal jobs AND
        - Not signaled to stop AND
        - Not waiting for cost-limited jobs to complete AND
        - Still have uncompleted work (completed < target)
        """
        running_eval_jobs = len(self.running_jobs)
        running_proposal_jobs = len(self.active_proposal_tasks)
        should_stop = self.should_stop.is_set()
        # Check based on actual completed work, not submitted generations
        # This properly handles failures/rejections in async evolution
        completed_target = self.completed_generations >= self.evo_config.num_generations

        # Don't consider system stuck if we're waiting for cost-limited jobs
        if self.cost_limit_reached and running_eval_jobs > 0:
            return False

        if self._has_persistence_work_in_progress():
            return False

        if self._has_background_side_effect_work():
            return False

        # System is stuck if all conditions are met
        is_stuck = (
            running_eval_jobs == 0
            and running_proposal_jobs == 0
            and not should_stop
            and not completed_target
        )

        return is_stuck

    async def _handle_stuck_system(self) -> bool:
        """
        Handle a stuck system by attempting recovery.

        Returns True if recovery was attempted, False if system should stop.
        """
        current_time = time.time()

        # Initialize progress tracking
        if self.last_progress_time is None:
            self.last_progress_time = current_time
            return True

        # Check if we've been stuck for too long
        time_since_progress = current_time - self.last_progress_time

        if time_since_progress > self.stuck_detection_timeout:
            self.stuck_detection_count += 1

            pending_work = self._get_remaining_completed_work()
            logger.warning(
                f"🚨 STUCK SYSTEM DETECTED (#{self.stuck_detection_count}/{self.max_stuck_detections}): "
                f"No progress for {time_since_progress:.1f}s. "
                f"running_eval_jobs=0, running_proposal_jobs=0, should_stop=False, "
                f"pending_work={pending_work} (target={self.evo_config.num_generations}, completed={self.completed_generations})"
            )

            # If we've exceeded max stuck detections, stop the system
            if self.stuck_detection_count >= self.max_stuck_detections:
                logger.error(
                    f"❌ SYSTEM PERMANENTLY STUCK: Exceeded max stuck detections "
                    f"({self.max_stuck_detections}). Stopping evolution to prevent infinite loop."
                )
                logger.error(
                    f"   Final state: completed_gens={self.completed_generations}, "
                    f"target_gens={self.evo_config.num_generations}, "
                    f"next_to_submit={self.next_generation_to_submit}"
                )
                self.should_stop.set()
                self.finalization_complete.set()
                return False

            # Attempt recovery by forcing proposal generation
            logger.info("🔧 ATTEMPTING RECOVERY: Force-starting proposal generation...")

            try:
                # Force start at least one proposal if we have uncompleted work
                # Use completed_generations to determine pending work, not next_generation_to_submit
                pending_work = self._get_remaining_completed_work()
                if pending_work > 0:
                    proposals_to_start = min(1, pending_work, self.max_proposal_jobs)
                    await self._start_proposals(proposals_to_start)
                    logger.info(
                        f"✅ Recovery attempt: Started {proposals_to_start} proposal(s)"
                    )
                else:
                    logger.warning(
                        "⚠️  No pending work to complete - system may be complete"
                    )
                    # Double-check completion status
                    await self._update_completed_generations()
                    if self.completed_generations >= self.evo_config.num_generations:
                        logger.info("✅ System is actually complete, stopping")
                        self.should_stop.set()
                        self.finalization_complete.set()
                        return False

            except Exception as e:
                logger.error(f"❌ Recovery attempt failed: {e}")

            # Reset progress timer after recovery attempt
            self.last_progress_time = current_time

        return True

    def _record_progress(self):
        """Record that progress has been made (jobs completed, proposals started, etc.)."""
        self.last_progress_time = time.time()
        # Reset stuck detection count on successful progress
        if self.stuck_detection_count > 0:
            logger.info(
                f"✅ Progress detected, resetting stuck detection count (was {self.stuck_detection_count})"
            )
            self.stuck_detection_count = 0

    async def _meta_summarizer_task(self):
        """Background task for meta summarization."""
        if not self.meta_summarizer:
            return

        logger.info("🔄 Meta summarizer task started")

        while not self.should_stop.is_set():
            try:
                # Debug: Check evolution state including stuck detection info
                running_eval_jobs = len(self.running_jobs)
                running_proposal_jobs = len(self.active_proposal_tasks)
                is_stuck = self._is_system_stuck()
                time_since_progress = None
                if self.last_progress_time:
                    time_since_progress = time.time() - self.last_progress_time

                time_since_str = (
                    f"{time_since_progress:.1f}s" if time_since_progress else "None"
                )
                pending_work = (
                    self.evo_config.num_generations - self.completed_generations
                )

                # Format API cost info
                if self.evo_config.max_api_costs is not None:
                    cost_str = f"${self.total_api_cost:.4f}/${self.evo_config.max_api_costs:.2f}"
                    cost_pct = (
                        self.total_api_cost / self.evo_config.max_api_costs
                    ) * 100
                    cost_info = f"api_costs={cost_str} ({cost_pct:.1f}%), "
                else:
                    cost_info = f"api_costs=${self.total_api_cost:.4f}, "

                # Determine if we should log at INFO level (meaningful change) or DEBUG
                current_state = {
                    "completed_generations": self.completed_generations,
                    "is_stuck": is_stuck,
                    "stuck_count": self.stuck_detection_count,
                }
                current_time = time.time()

                # Log at INFO if: state changed, or it's been 5+ minutes since last INFO log
                should_log_info = False
                if self._last_meta_log_state is None:
                    should_log_info = True  # First log
                elif current_state != self._last_meta_log_state:
                    should_log_info = True  # State changed
                elif (
                    self._last_meta_log_info_time is None
                    or current_time - self._last_meta_log_info_time >= 300
                ):
                    should_log_info = True  # 5 minutes since last INFO log

                log_msg = (
                    f"🔍 Meta task check: completed_gens={self.completed_generations}, target={self.evo_config.num_generations}, pending_work={pending_work}, "
                    f"running_eval_jobs={running_eval_jobs}, running_proposal_jobs={running_proposal_jobs}, "
                    f"{cost_info}"
                    f"should_stop={self.should_stop.is_set()}, is_stuck={is_stuck}, "
                    f"stuck_count={self.stuck_detection_count}, time_since_progress={time_since_str}"
                )

                if should_log_info:
                    logger.info(log_msg)
                    self._last_meta_log_state = current_state
                    self._last_meta_log_info_time = current_time
                else:
                    logger.debug(log_msg)

                # Check if we should exit (same logic as job monitor)
                if (
                    self.completed_generations >= self.evo_config.num_generations
                    and len(self.running_jobs) == 0
                ):
                    logger.info("Meta summarizer task detected completion, exiting")
                    break

                # Update meta summarizer periodically
                if self.completed_generations > 0:
                    best_program = await self.async_db.get_best_program_async()
                    if best_program:
                        # This would need to be made async in MetaSummarizer
                        pass  # Placeholder for now

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error in meta summarizer task: {e}")
                await asyncio.sleep(5)

        logger.info("Meta summarizer task exited")

    async def _cleanup_async(self):
        """Cleanup async resources."""
        try:
            # Cancel remaining proposal tasks
            for task in self.active_proposal_tasks.values():
                if not task.done():
                    task.cancel()

            # Wait for tasks to finish
            if self.active_proposal_tasks:
                await asyncio.gather(
                    *self.active_proposal_tasks.values(), return_exceptions=True
                )

            # Final recomputation of prompt percentiles to ensure fitness is accurate
            if self.prompt_db is not None and self.db is not None:
                try:
                    # Get all correct program scores from main database
                    all_programs = self.db.get_all_programs()
                    all_correct_scores = [
                        p.combined_score
                        for p in all_programs
                        if p.correct and p.combined_score is not None
                    ]
                    # Build mapping from program_id to current score
                    program_id_to_score = {
                        p.id: p.combined_score
                        for p in all_programs
                        if p.correct and p.combined_score is not None
                    }
                    self.prompt_db.recompute_all_percentiles(
                        all_correct_scores, program_id_to_score
                    )
                    logger.info(
                        f"Final recomputation of prompt fitness percentiles complete "
                        f"(using {len(all_correct_scores)} correct program scores)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to recompute prompt percentiles: {e}")

            # Cleanup database
            self._finish_wandb_logging()
            await self.async_db.close_async()

            # Cleanup scheduler
            self.scheduler.shutdown()

        except Exception as e:
            logger.error(f"Error in async cleanup: {e}")

    def _log_program_to_wandb(self, program: Program) -> None:
        self.wandb_logger.log_program(
            program=program,
            db=self.db,
            prompt_db=self.prompt_db,
        )

    def _finish_wandb_logging(self) -> None:
        self.wandb_logger.log_final(
            db=self.db,
            prompt_db=self.prompt_db,
            results_dir=Path(self.results_dir),
            total_proposals_generated=self.total_proposals_generated,
            total_api_cost=self.total_api_cost,
        )
        self.wandb_logger.finish()

    async def _print_final_summary(self):
        """Print final evolution summary."""
        if not self.verbose:
            return

        end_time = time.time()
        total_time = end_time - (self.start_time or end_time)
        missing_generations = (
            self._get_generations_without_program_due_to_proposal_failure()
        )

        logger.info("=" * 80)
        logger.info("ASYNC EVOLUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Target generations: {self.evo_config.num_generations}")
        logger.info(f"Stored programs: {self.completed_generations}")
        logger.info(
            "Generations without program (proposal generation exhausted retries): %s",
            len(missing_generations),
        )
        logger.info(
            "Generation IDs without program after proposal retries: %s",
            missing_generations,
        )
        logger.info(f"Total proposals generated: {self.total_proposals_generated}")
        logger.info(f"Total API cost: ${self.total_api_cost:.4f}")

        # Log cost budget usage if max_api_costs was set
        if self.evo_config.max_api_costs is not None:
            percentage = (self.total_api_cost / self.evo_config.max_api_costs) * 100
            logger.info(
                f"API cost budget usage: {percentage:.1f}% "
                f"(${self.total_api_cost:.4f} / "
                f"${self.evo_config.max_api_costs:.2f})"
            )

        logger.info(f"Total runtime: {total_time:.2f} seconds")

        if self.total_proposals_generated > 0:
            avg_time_per_proposal = total_time / self.total_proposals_generated
            logger.info(
                f"Average time per proposal: {avg_time_per_proposal:.2f} seconds"
            )

        # Report final operations status
        logger.info("-" * 40)
        logger.info("FINAL OPERATIONS STATUS:")
        if self.embedding_client:
            logger.info("PCA/Embedding recomputation: COMPLETED")
        else:
            logger.info("PCA/Embedding recomputation: SKIPPED (no embedding client)")

        if self.meta_summarizer:
            logger.info("Meta summary generation: COMPLETED")
        else:
            logger.info("Meta summary generation: SKIPPED (no meta summarizer)")

        # Print database summary
        if self.db:
            logger.info("-" * 40)
            self._log_timing_bottleneck_summary()
            self.db.print_summary(
                console=self.console,
                total_program_target=self.evo_config.num_generations,
            )

    def _log_timing_bottleneck_summary(self) -> None:
        """Print aggregate timing stats to identify queueing bottlenecks."""
        if not self.db:
            return

        try:
            programs = [
                program
                for program in self.db.get_all_programs()
                if program.generation > 0 and isinstance(program.metadata, dict)
            ]
            if not programs:
                return

            metadata_rows = [program.metadata or {} for program in programs]
            metrics = [
                "sampling_seconds",
                "evaluation_seconds",
                "post_eval_queue_wait_seconds",
                "postprocess_seconds",
                "postprocess_apply_wait_seconds",
                "postprocess_apply_seconds",
                "pipeline_unaccounted_seconds",
                "end_to_end_with_side_effects_seconds",
            ]
            summary = summarize_timing_metadata(metadata_rows, metrics)
            if not summary:
                return

            label_map = {
                "sampling_seconds": "Sampling",
                "evaluation_seconds": "Evaluation",
                "post_eval_queue_wait_seconds": "Eval->Postprocess Wait",
                "postprocess_seconds": "Postprocess Hot Path",
                "postprocess_apply_wait_seconds": "Postprocess->Apply Wait",
                "postprocess_apply_seconds": "Side-Effect Apply",
                "pipeline_unaccounted_seconds": "Pipeline Unaccounted",
                "end_to_end_with_side_effects_seconds": "End-to-End w/ Side Effects",
            }

            logger.info("-" * 40)
            logger.info("TIMING BOTTLENECK SUMMARY:")
            logger.info(
                "Programs analyzed: %s (generation > 0 with persisted metadata)",
                len(programs),
            )
            for metric in metrics:
                stats = summary.get(metric)
                if not stats:
                    continue
                logger.info(
                    "%s: mean=%.2fs median=%.2fs p90=%.2fs max=%.2fs (n=%d)",
                    label_map[metric],
                    stats["mean"],
                    stats["median"],
                    stats["p90"],
                    stats["max"],
                    int(stats["count"]),
                )

            for metric, label in [
                ("post_eval_queue_wait_seconds", "Top Eval->Postprocess Waits"),
                ("postprocess_apply_wait_seconds", "Top Postprocess->Apply Waits"),
                ("postprocess_apply_seconds", "Top Side-Effect Apply Durations"),
            ]:
                ranked = []
                for program in programs:
                    value = (program.metadata or {}).get(metric)
                    if isinstance(value, (int, float)) and value > 0:
                        ranked.append((float(value), program.generation))
                if not ranked:
                    continue
                top_rows = ", ".join(
                    f"gen {generation}={value:.1f}s"
                    for value, generation in sorted(ranked, reverse=True)[:5]
                )
                logger.info("%s: %s", label, top_rows)
        except Exception as e:
            logger.warning(f"Failed to compute timing bottleneck summary: {e}")

    def _get_generations_without_program_due_to_proposal_failure(self) -> List[int]:
        """Return target-range generation IDs that exhausted proposal retries and produced no program row."""
        if not self.db or not getattr(self.db, "cursor", None):
            return []

        try:
            self.db.cursor.execute(
                """
                SELECT DISTINCT attempt_log.generation
                FROM attempt_log
                WHERE attempt_log.status = 'failed'
                  AND json_valid(attempt_log.details)
                  AND json_extract(attempt_log.details, '$.node_kind') = 'failed_proposal'
                  AND attempt_log.generation < ?
                  AND attempt_log.generation NOT IN (
                      SELECT DISTINCT generation FROM programs
                  )
                ORDER BY attempt_log.generation
                """,
                (self.evo_config.num_generations,),
            )
            return [int(row[0]) for row in self.db.cursor.fetchall()]
        except Exception as e:
            logger.warning(
                "Failed to load proposal-failure generations without program rows: %s",
                e,
            )
            return []

    def _print_metadata_table(self, meta_data: dict, generation: int = None):
        """Display metadata in a formatted rich table."""
        # Create title with generation and attempt information
        title_parts = ["[bold magenta]Patch Metadata"]

        # Add generation if present
        if generation is not None:
            # Check if we have attempt information in meta_data
            if all(
                key in meta_data
                for key in ["novelty_attempt", "resample_attempt", "patch_attempt"]
            ):
                title_parts.append(
                    f" - Gen {generation}/{self.evo_config.num_generations} - "
                    f"Novelty: {meta_data['novelty_attempt']}/{self.evo_config.max_novelty_attempts} - "
                    f"Resample: {meta_data['resample_attempt']}/{self.evo_config.max_patch_resamples} - "
                    f"Patch: {meta_data['patch_attempt']}/{self.evo_config.max_patch_attempts}"
                )
            else:
                title_parts.append(
                    f" - Gen {generation}/{self.evo_config.num_generations}"
                )

        title_parts.append("[/bold magenta]")
        table = Table(
            title="".join(title_parts),
            show_header=True,
            header_style="bold cyan",
            border_style="magenta",
            box=rich.box.ROUNDED,
            width=120,  # Match display.py table width
        )
        table.add_column("Field", style="cyan bold", no_wrap=True, width=25)
        table.add_column("Value", style="green", overflow="fold", width=90)

        # Define display order and formatting for specific fields
        display_order = [
            "patch_type",
            "patch_name",
            "patch_description",
            "num_applied",
            "api_costs",
            "error_attempt",
        ]

        # Add ordered fields first
        for field_name in display_order:
            if field_name in meta_data:
                value = meta_data[field_name]
                if value is None:
                    formatted_value = "[dim]None[/dim]"
                elif field_name == "api_costs":
                    formatted_value = f"${value:.4f}"
                elif field_name == "error_attempt" and value is None:
                    formatted_value = "[green]Success[/green]"
                elif field_name == "error_attempt":
                    formatted_value = (
                        f"[red]{str(value)[:100]}...[/red]"
                        if len(str(value)) > 100
                        else f"[red]{value}[/red]"
                    )
                else:
                    formatted_value = str(value)

                table.add_row(field_name, formatted_value)

        # Add remaining fields (excluding llm_result, diff_summary, and attempt info for brevity)
        skip_fields = set(
            display_order
            + [
                "llm_result",
                "diff_summary",
                "novelty_attempt",
                "resample_attempt",
                "patch_attempt",
            ]
        )
        for field_key, field_value in meta_data.items():
            if field_key not in skip_fields:
                if field_value is None:
                    formatted_value = "[dim]None[/dim]"
                else:
                    formatted_value = (
                        str(field_value)[:100] + "..."
                        if len(str(field_value)) > 100
                        else str(field_value)
                    )
                table.add_row(field_key, formatted_value)

        # Add diff summary if available
        if "diff_summary" in meta_data and meta_data["diff_summary"]:
            diff_summary = meta_data["diff_summary"]
            if isinstance(diff_summary, dict):
                summary_text = ""
                for k, v in diff_summary.items():
                    summary_text += f"{k}: {v}; "
                table.add_row("diff_summary", summary_text.strip())
            else:
                table.add_row("diff_summary", str(diff_summary)[:200])

        self.console.print(table)

    async def _update_best_solution_async(self):
        """Checks and updates the best program asynchronously."""
        if not self.async_db:
            return
        best_lock = getattr(self, "_best_solution_lock", None)
        if best_lock is None:
            best_lock = asyncio.Lock()
            self._best_solution_lock = best_lock

        async with best_lock:
            best_programs = await self.async_db.get_top_programs_async(
                n=1, correct_only=True
            )
            if not best_programs:
                if self.verbose:
                    logger.info(
                        "No correct programs found yet, cannot determine best solution."
                    )
                return

            best_program = best_programs[0]

            if best_program.id == self.best_program_id:
                return  # No change

            self.best_program_id = best_program.id

            source_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{best_program.generation}"
            best_dir = Path(self.results_dir) / "best"

            loop = asyncio.get_event_loop()

            def sync_file_operations():
                """Synchronous file operations to run in executor."""
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(source_dir, best_dir)

            await loop.run_in_executor(None, sync_file_operations)

            if self.verbose:
                logger.info(
                    f"New best program found: gen {best_program.generation}, "
                    f"id {best_program.id[:6]}... "
                    f"Copied to {best_dir}"
                )

    def _extract_code_from_response(self, response_content: str) -> Optional[str]:
        """Extract code from LLM response."""
        # Look for code blocks
        import re

        # Try to find code between triple backticks
        code_match = re.search(
            r"```(?:python|py)?\s*\n(.*?)\n```", response_content, re.DOTALL
        )
        if code_match:
            return code_match.group(1).strip()

        # If no code block found, return the whole response
        return response_content.strip()

    async def _read_file_async(self, file_path: str) -> Optional[str]:
        """Read file asynchronously."""
        try:

            def read_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, read_file)
            return content
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None
