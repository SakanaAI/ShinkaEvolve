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
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Union
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
)
from shinka.embed import AsyncEmbeddingClient
from shinka.launch import JobScheduler, JobConfig
from shinka.edit.async_apply import (
    apply_patch_async,
    get_code_embedding_async,
    write_file_async,
)
from shinka.edit import summarize_diff
from shinka.core.sampler import PromptSampler
from shinka.core.summarizer import MetaSummarizer
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.core.novelty_judge import NoveltyJudge
from shinka.core.runner import EvolutionConfig, FOLDER_PREFIX
from shinka.core.prompt_evolver import (
    SystemPromptSampler,
    AsyncSystemPromptEvolver,
)
from shinka.logo import print_gradient_logo
from shinka.utils import get_language_extension

logger = logging.getLogger(__name__)


@dataclass
class AsyncRunningJob:
    """Async version of RunningJob with additional async metadata."""

    job_id: Union[str, Any]
    exec_fname: str
    results_dir: str
    start_time: float
    generation: int
    parent_id: Optional[str] = None
    archive_insp_ids: List[str] = field(default_factory=list)
    top_k_insp_ids: List[str] = field(default_factory=list)
    code_diff: Optional[str] = None
    meta_patch_data: Dict[str, Any] = field(default_factory=dict)
    code_embedding: Optional[List[float]] = None
    embed_cost: float = 0.0
    novelty_cost: float = 0.0  # Track novelty checking cost
    proposal_task_id: Optional[str] = None  # Track which proposal task created this job
    db_retry_count: int = 0  # Track number of DB write retry attempts


class AsyncEvolutionRunner:
    """Fully async evolution runner with concurrent proposal generation."""

    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        verbose: bool = True,
        max_evaluation_jobs: int = None,
        max_proposal_jobs: int = 10,
        max_db_workers: int = 4,
        debug: bool = False,
        init_program_str: Optional[str] = None,
        evaluate_str: Optional[str] = None,
    ):
        """Initialize async evolution runner.

        Args:
            evo_config: Evolution configuration
            job_config: Job configuration
            db_config: Database configuration
            verbose: Enable verbose logging
            max_evaluation_jobs: Maximum concurrent evaluation jobs
                (defaults to evo_config.max_parallel_jobs)
            max_proposal_jobs: Maximum concurrent proposal generation tasks
            init_program_str: Optional string content for initial program
                (will be saved to results dir and path updated in evo_config)
            evaluate_str: Optional string content for evaluate script
                (will be saved to results dir and path updated in job_config)
        """
        print_gradient_logo((255, 0, 0), (255, 255, 255))
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
        self.enable_deadlock_debugging = debug

        if self.verbose:
            # Set up logging like the sync version
            log_filename = f"{self.results_dir}/evolution_run.log"
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

        # Handle init_program_str: write to file and update config path
        if init_program_str is not None:
            lang_ext = get_language_extension(evo_config.language)
            init_program_path = Path(self.results_dir) / f"init_program.{lang_ext}"
            init_program_path.write_text(init_program_str, encoding="utf-8")
            self.evo_config.init_program_path = str(init_program_path)
            if self.verbose:
                logger.info(f"Saved init_program_str to {init_program_path}")

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
                max_evaluation_jobs
                if max_evaluation_jobs is not None
                else evo_config.max_parallel_jobs,
                max_proposal_jobs,
                max_db_workers,
                cpu_count,
            )
        )

        self.max_evaluation_jobs = max_evaluation_jobs
        self.max_proposal_jobs = max_proposal_jobs
        self.max_db_workers = max_db_workers

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
        logger.info(f"Language: {self.evo_config.language}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Log file: {log_filename}")
        if self.evo_config.max_api_costs is not None:
            logger.info(f"Max API costs: ${self.evo_config.max_api_costs:.2f}")
        logger.info("=" * 80)

        # Initialize rich console for formatted output
        self.console = Console()

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

        # Store db_config for later initialization (after results_dir is set)
        # Database will be initialized in _setup_async()
        self.db = None
        self.async_db = None

        # LLM clients
        self.llm = AsyncLLMClient(
            model_names=evo_config.llm_models,
            **evo_config.llm_kwargs,
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
                **evo_config.meta_llm_kwargs,
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
                **evo_config.novelty_llm_kwargs,
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
                **evo_config.prompt_llm_kwargs,
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

        # Robust job tracking - ensure no jobs are lost
        self.submitted_jobs: Dict[str, AsyncRunningJob] = {}  # All jobs ever submitted
        self.processing_lock = asyncio.Lock()  # Prevent concurrent processing issues

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
                    self.llm_selection.print_summary()
            else:
                logger.debug(
                    f"No bandit state file found at {bandit_path}, "
                    "starting with fresh bandit state"
                )
        except Exception as e:
            logger.warning(f"Failed to load bandit state: {e}")

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
        except:
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
                    f"⚠️  High concurrency settings may cause CPU oversubscription"
                )
            if max_evaluation_jobs + max_proposal_jobs > memory_based_limit:
                logger.warning(
                    f"⚠️  High concurrency settings may cause memory pressure (limit: {memory_based_limit})"
                )

        return max_evaluation_jobs, max_proposal_jobs, max_db_workers

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

    async def run(self):
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
        self.async_db = AsyncProgramDatabase(
            self.db,
            max_workers=self.max_db_workers,
            enable_deadlock_debugging=self.enable_deadlock_debugging,
        )

        # Initialize prompt evolution database if enabled
        if self.evo_config.evolve_prompts:
            await self._setup_prompt_evolution()

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
            self.completed_generations = self.db.last_iteration + 1
            self.next_generation_to_submit = self.completed_generations
        else:
            # Generate or copy initial program only if NOT resuming
            if (
                self.evo_config.init_program_path
                and Path(self.evo_config.init_program_path).exists()
            ):
                # Copy existing initial program
                if self.verbose:
                    logger.info(
                        f"Copying initial program from {self.evo_config.init_program_path}"
                    )
                initial_code = await self._read_file_async(
                    self.evo_config.init_program_path
                )
                if initial_code:
                    await self._setup_initial_program(initial_code)
            else:
                # Generate initial program with LLM
                if self.verbose:
                    logger.info(
                        "`init_program_path` not provided, "
                        "generating initial program with LLM..."
                    )
                await self._generate_initial_program()

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
            self.prompt_percentile_recompute_counter += 1
            recompute_interval = self.evo_config.prompt_percentile_recompute_interval
            if (
                recompute_interval > 0
                and self.prompt_percentile_recompute_counter >= recompute_interval
            ):
                self.prompt_percentile_recompute_counter = 0
                try:
                    # Get all correct program scores from main database
                    # This matches what the webUI uses for beat percentage calculation
                    all_programs = self.db.get_all_programs()
                    all_correct_scores = [
                        p.combined_score
                        for p in all_programs
                        if p.correct and p.combined_score is not None
                    ]
                    # Build mapping from program_id to current score
                    # This ensures we use actual current scores, not stale stored ones
                    program_id_to_score = {
                        p.id: p.combined_score
                        for p in all_programs
                        if p.correct and p.combined_score is not None
                    }
                    self.prompt_db.recompute_all_percentiles(
                        all_correct_scores, program_id_to_score
                    )
                    logger.info(
                        f"Recomputed prompt fitness percentiles "
                        f"(every {recompute_interval} programs, "
                        f"using {len(all_correct_scores)} correct program scores)"
                    )
                except Exception as recompute_err:
                    logger.warning(
                        f"Failed to recompute prompt percentiles: {recompute_err}"
                    )

        except Exception as e:
            logger.error(f"Failed to update prompt fitness: {e}")

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

    async def _setup_initial_program(self, code: str):
        """Setup initial program in database."""
        await self._setup_initial_program_with_metadata(
            code, "initial_program", "Initial program setup", 0.0
        )

    async def _setup_initial_program_with_metadata(
        self,
        code: str,
        patch_name: Optional[str],
        patch_description: Optional[str],
        api_cost: float,
        llm_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Setup initial program in database with metadata."""
        # Create generation 0 directory structure first
        gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"
        results_dir = f"{gen_dir}/results"

        # Create directories synchronously to avoid race conditions
        Path(gen_dir).mkdir(parents=True, exist_ok=True)
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Write the initial program file
        exec_fname = f"{gen_dir}/main.{self.lang_ext}"
        await write_file_async(exec_fname, code)

        # Run initial evaluation to get proper metrics
        try:
            if self.verbose:
                logger.info(f"Starting initial program evaluation: {exec_fname}")

            # Run the evaluation synchronously for generation 0
            loop = asyncio.get_event_loop()
            results, rtime = await loop.run_in_executor(
                None, self.scheduler.run, exec_fname, results_dir
            )

            if self.verbose:
                logger.info(f"Initial program evaluation completed in {rtime:.2f}s")

            # Get code embedding for initial program
            code_embedding, e_cost = await self._get_code_embedding_async(exec_fname)
            if self.verbose and code_embedding:
                logger.info(f"Initial program embedding computed (cost: ${e_cost:.4f})")

            # Extract metrics properly like the sync version
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            combined_score = metrics_val.get("combined_score", 0.0)
            public_metrics = metrics_val.get("public", {})
            private_metrics = metrics_val.get("private", {})
            text_feedback = metrics_val.get("text_feedback", "")
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

            # Build base metadata
            base_metadata = {
                "compute_time": rtime,
                "embed_cost": e_cost,
                "novelty_cost": 0.0,  # No novelty cost for generation 0
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            }

            # For file-based initial programs, add default metadata
            if not llm_metadata:
                base_metadata.update(
                    {
                        "api_costs": api_cost,
                        "patch_type": "init",
                        "patch_name": patch_name or "initial_program",
                        "patch_description": patch_description
                        or "Initial program setup",
                    }
                )
            else:
                # LLM-generated: llm_metadata already contains structured data
                base_metadata.update(llm_metadata)

            # Create program with actual evaluation results
            initial_program = Program(
                id=str(uuid.uuid4()),
                code=code,
                generation=0,
                correct=correct_val,
                combined_score=combined_score,
                public_metrics=public_metrics,
                private_metrics=private_metrics,
                text_feedback=text_feedback,
                timestamp=datetime.now().timestamp(),
                embedding=code_embedding,
                metadata=base_metadata,
            )

            if self.verbose:
                logger.info(
                    f"Initial program evaluated - correct: {initial_program.correct}, "
                    f"combined_score: {initial_program.combined_score}"
                )

        except Exception as e:
            logger.warning(f"Initial program evaluation failed: {e}")

            # Still try to compute embedding even if evaluation failed
            try:
                code_embedding, e_cost = await self._get_code_embedding_async(
                    exec_fname
                )
            except Exception:
                code_embedding, e_cost = None, 0.0

            # Build base metadata for fallback
            base_metadata = {
                "compute_time": 0.0,
                "embed_cost": e_cost,
                "novelty_cost": 0.0,  # No novelty cost for generation 0 fallback
                "evaluation_failed": True,
                "stdout_log": "",
                "stderr_log": "",
            }

            # For file-based initial programs, add default metadata
            if not llm_metadata:
                base_metadata.update(
                    {
                        "api_costs": api_cost,
                        "patch_type": "init",
                        "patch_name": patch_name or "initial_program",
                        "patch_description": patch_description
                        or "Initial program setup (fallback)",
                    }
                )
            else:
                # LLM-generated: llm_metadata already contains structured data
                base_metadata.update(llm_metadata)

            # Fall back to assuming it's correct
            initial_program = Program(
                id=str(uuid.uuid4()),
                code=code,
                generation=0,
                public_metrics={"score": 0.0},
                correct=True,
                timestamp=datetime.now().timestamp(),
                embedding=code_embedding,
                metadata=base_metadata,
            )

        # Add to database
        await self.async_db.add_program_async(initial_program)

        # Add initial program costs to in-memory total for accurate budget tracking
        initial_api_cost = (initial_program.metadata or {}).get("api_costs", 0.0)
        initial_embed_cost = (initial_program.metadata or {}).get("embed_cost", 0.0)
        initial_novelty_cost = (initial_program.metadata or {}).get("novelty_cost", 0.0)
        self.total_api_cost += (
            initial_api_cost + initial_embed_cost + initial_novelty_cost
        )

        # Add the initial program to meta memory tracking
        if self.meta_summarizer:
            self.meta_summarizer.add_evaluated_program(initial_program)

            # Check if we should update meta memory after adding this program
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

                        # Add meta cost to this program's metadata (the one that triggered the update)
                        if initial_program.metadata is None:
                            initial_program.metadata = {}
                        initial_program.metadata["meta_cost"] = meta_cost
                        # Update the program in the database with the new metadata (thread-safe)

                        def update_metadata():
                            # Create a new database connection in this thread to avoid conflicts
                            from shinka.database import ProgramDatabase

                            thread_db = ProgramDatabase(self.db.config)
                            try:
                                metadata_json = json.dumps(initial_program.metadata)
                                thread_db.cursor.execute(
                                    "UPDATE programs SET metadata = ? WHERE id = ?",
                                    (metadata_json, initial_program.id),
                                )
                                thread_db.conn.commit()
                            finally:
                                thread_db.close()

                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, update_metadata)

        # Set baseline score for LLM selection
        if self.llm_selection is not None:
            self.llm_selection.set_baseline_score(
                initial_program.combined_score if initial_program.correct else 0.0,
            )

        # Mark generation 0 as completed
        self.completed_generations = 1

        # Record progress after initial setup
        self._record_progress()

        if self.verbose:
            logger.info(f"Setup initial program: {initial_program.id}")
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

    async def _generate_initial_program(self):
        """Generate initial program using LLM, with retries."""
        sys_msg, user_msg = self.prompt_sampler.initial_program_prompt()

        # Select LLM once per program generation (before all attempts)
        model_sample_probs = None
        model_posterior = None
        if self.llm_selection is not None:
            model_sample_probs, model_posterior = self.llm_selection.select_llm()

        # Get LLM kwargs for metadata storage
        llm_kwargs = self.llm.get_kwargs(model_sample_probs=model_sample_probs)

        total_costs = 0.0

        for attempt in range(self.evo_config.max_patch_attempts):
            response = await self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                model_sample_probs=model_sample_probs,
                model_posterior=model_posterior,
            )

            if response is None or response.content is None:
                error_msg = "LLM response content was None."
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        f"FAILURE. Error: {error_msg}"
                    )
                # Save failed attempt
                await self._save_patch_attempt_async(
                    generation=0,
                    novelty_attempt=1,
                    resample_attempt=1,
                    patch_attempt=attempt + 1,
                    response=response,
                    error_msg=error_msg,
                    patch_text=None,
                    num_applied=0,
                    patch_name=None,
                    patch_description=None,
                    success=False,
                )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full code."
                    )
                    continue
                else:
                    break

            total_costs += response.cost or 0.0

            # Extract code using language-specific markers
            initial_code = extract_between(
                response.content,
                f"```{self.evo_config.language}",
                "```",
                False,
            )

            if initial_code:
                # Extract patch name and description
                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                )

                # Add EVOLVE-BLOCK markers
                if self.evo_config.language == "python":
                    comment_char = "#"
                else:
                    comment_char = "//"

                initial_code = (
                    f"{comment_char} EVOLVE-BLOCK-START\n"
                    f"{initial_code}\n"
                    f"{comment_char} EVOLVE-BLOCK-END\n"
                )

                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "SUCCESS."
                    )

                # Save successful attempt
                await self._save_patch_attempt_async(
                    generation=0,
                    novelty_attempt=1,
                    resample_attempt=1,
                    patch_attempt=attempt + 1,
                    response=response,
                    error_msg=None,
                    patch_text=initial_code,
                    num_applied=1,
                    patch_name=patch_name,
                    patch_description=patch_description,
                    success=True,
                )

                # Include LLM metadata for storage (structured like meta_edit_data)
                llm_metadata = {
                    "patch_type": "init",
                    "api_costs": total_costs,
                    "num_applied": 1,  # Initial program counts as 1 application
                    "patch_name": patch_name,
                    "patch_description": patch_description,
                    "error_attempt": None,  # No error on success
                    "novelty_attempt": 1,
                    "resample_attempt": 1,
                    "patch_attempt": attempt + 1,
                    **llm_kwargs,
                    "llm_result": response.to_dict() if response else None,
                    "diff_summary": {},  # No diff for initial program
                }

                # Pass the metadata to setup method
                await self._setup_initial_program_with_metadata(
                    initial_code,
                    patch_name,
                    patch_description,
                    total_costs,
                    llm_metadata,
                )
                return
            else:  # code extraction failed
                error_msg = "Could not extract code from response."
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        f"FAILURE. Error: {error_msg}"
                    )
                # Save failed attempt
                await self._save_patch_attempt_async(
                    generation=0,
                    novelty_attempt=1,
                    resample_attempt=1,
                    patch_attempt=attempt + 1,
                    response=response,
                    error_msg=error_msg,
                    patch_text=None,
                    num_applied=0,
                    patch_name=None,
                    patch_description=None,
                    success=False,
                )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "Could not extract code from your last response. "
                        "Please make sure to enclose the code in "
                        f"```{self.evo_config.language}...``` tags."
                    )
                else:  # last attempt
                    break

        raise RuntimeError(
            "LLM failed to generate a valid initial program after "
            f"{self.evo_config.max_patch_attempts} attempts."
        )

    async def _job_monitor_task(self):
        """Monitor running jobs and process completed ones."""
        logger.info("🔄 Job monitor task started")

        while not self.should_stop.is_set():
            if not self.running_jobs:
                # Debug: Log when waiting with no jobs
                logger.debug(
                    f"🔍 Job monitor waiting: completed_gens={self.completed_generations}, target={self.evo_config.num_generations}, no running jobs"
                )
                await asyncio.sleep(1)
                continue

            try:
                # Check job statuses concurrently
                status_results = await self.scheduler.batch_check_status_async(
                    self.running_jobs
                )
                if self.verbose:
                    # Create safe status display to avoid race conditions
                    try:
                        status_display = []
                        for i, job in enumerate(self.running_jobs):
                            if i < len(status_results):
                                status_display.append(
                                    f"{job.generation} - {status_results[i]}"
                                )
                            else:
                                status_display.append(f"{job.generation} - unknown")

                        logger.debug(
                            f"Job statuses ({len(self.running_jobs)}): gen [{', '.join(status_display)}]"
                        )
                        logger.debug(
                            f"Active proposal jobs ({len(self.active_proposal_tasks)}): gen [{', '.join([task.get_name().split('_')[1] if task.get_name().startswith('proposal_') else 'unknown' for task in self.active_proposal_tasks.values()])}]"
                        )
                    except Exception as e:
                        logger.warning(f"Error in status logging: {e}")
                        logger.debug(
                            f"Running jobs: {len(self.running_jobs)}, Active proposals: {len(self.active_proposal_tasks)}"
                        )
                completed_jobs = []
                still_running = []

                for job, is_running in zip(self.running_jobs, status_results):
                    if isinstance(is_running, Exception):
                        logger.warning(f"Error checking job {job.job_id}: {is_running}")
                        still_running.append(job)
                    elif not is_running:
                        completed_jobs.append(job)
                        runtime = time.time() - job.start_time
                        if self.verbose:
                            logger.info(
                                f"✅ Job {job.job_id} completed (gen {job.generation}) after {runtime:.1f}s"
                            )
                    else:
                        still_running.append(job)

                self.running_jobs = still_running

                # Process completed jobs atomically with lock to prevent race conditions
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
                        await self._process_completed_jobs_safely(completed_jobs)
                        old_completed = self.completed_generations
                        await self._update_completed_generations()

                        if self.verbose:
                            if self.completed_generations != old_completed:
                                # Format API cost info
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
                                # Log why count didn't change
                                retry_count = len(self.failed_jobs_for_retry)
                                new_retries = retry_count - old_retry_count
                                running_count = len(self.running_jobs)

                                # Check if we're at target (capped)
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
                                    # Format API cost info
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
                                    # Only warn if truly unexpected
                                    logger.warning(
                                        f"⚠️  Completed generations unchanged "
                                        f"after processing jobs: "
                                        f"{self.completed_generations}"
                                    )

                    # Record progress when jobs complete
                    self._record_progress()

                    # Signal that slots are available
                    self.slot_available.set()

                # Retry any failed DB jobs
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
                    # We've reached target but still have running jobs
                    # or proposals
                    if self.verbose:
                        pending_jobs = len(self.running_jobs)
                        pending_proposals = len(self.active_proposal_tasks)
                        logger.debug(
                            f"⏳ Target generations reached, waiting for "
                            f"{pending_jobs} jobs and "
                            f"{pending_proposals} proposals to complete..."
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

                # Calculate available slots
                available_slots = self.max_evaluation_jobs - len(self.running_jobs)

                # Simple approach: use next_generation_to_submit as hard cap
                # This tracks total submitted proposals and prevents any overshoot
                proposals_remaining = max(
                    0, self.evo_config.num_generations - self.next_generation_to_submit
                )

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
                pipeline_target = self.max_evaluation_jobs
                proposals_needed = min(
                    max(0, pipeline_target - pipeline_capacity),  # Fill the pipeline
                    proposals_remaining,
                    self.max_proposal_jobs - len(self.active_proposal_tasks),
                )

                if proposals_needed > 0 and should_generate_proposals:
                    # Start the needed proposals
                    if self.verbose:
                        logger.info(
                            f"Starting {proposals_needed} new proposals. "
                            f"Pipeline: {pipeline_capacity}/{pipeline_target} "
                            f"(running_jobs={len(self.running_jobs)}, active_proposals={len(self.active_proposal_tasks)}/{self.max_proposal_jobs}), "
                            f"Proposals remaining: {proposals_remaining} (submitted={self.next_generation_to_submit}/{self.evo_config.num_generations})"
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

        In async evolution, we continue generating proposals beyond num_generations
        if needed to compensate for failed/rejected proposals, until we reach the
        target number of completed generations.
        """
        for _ in range(num_proposals):
            # Only stop if we've reached max proposal concurrency
            # Don't check num_generations here - the coordinator handles that
            if len(self.active_proposal_tasks) >= self.max_proposal_jobs:
                break

            # Assign generation atomically to prevent duplicates
            generation = self.next_generation_to_submit

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
        try:
            if self.verbose:
                logger.info(f"Generating proposal for generation {generation}")

            # Setup directories - create them synchronously to avoid race conditions
            gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{generation}"
            exec_fname = f"{gen_dir}/main.{self.lang_ext}"
            results_dir = f"{gen_dir}/results"
            lock_file = f"{gen_dir}/.generation_lock"

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

            # Remove from active tasks
            if task_id in self.active_proposal_tasks:
                del self.active_proposal_tasks[task_id]

    async def _generate_evolved_proposal(
        self,
        generation: int,
        task_id: str,
        exec_fname: str,
        results_dir: str,
        meta_recs: Optional[str],
        meta_summary: Optional[str],
        meta_scratch: Optional[str],
    ) -> Optional[AsyncRunningJob]:
        """Generate an evolved proposal through the full pipeline."""
        api_costs = 0.0
        embed_cost = 0.0

        # Initialize novelty tracking variables (same as sync runner)
        novelty_checks_performed = 0
        novelty_total_cost = 0.0
        novelty_explanation = ""
        proposal_accepted = False

        # Select LLM once per program generation (before all loops)
        model_sample_probs = None
        model_posterior = None
        if self.llm_selection is not None:
            model_sample_probs, model_posterior = self.llm_selection.select_llm()

        for attempt in range(self.evo_config.max_novelty_attempts):
            for resample in range(self.evo_config.max_patch_resamples):
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
                        resample_attempt=resample + 1,
                        max_resample_attempts=self.evo_config.max_patch_resamples,
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

                    # Choose between fix mode and normal patch mode
                    if needs_fix:
                        # FIX MODE: No correct programs exist, try to fix
                        # archive_programs contains ancestors in fix mode
                        if self.verbose:
                            logger.info(
                                f"FIX MODE: Attempting to fix incorrect program "
                                f"{parent_program.id} (Gen: {parent_program.generation})"
                            )
                        patch_result = await self._run_fix_patch_async(
                            parent_program,
                            archive_programs,  # ancestors in fix mode
                            generation,
                            novelty_attempt=attempt + 1,
                            resample_attempt=resample + 1,
                            model_sample_probs=model_sample_probs,
                            model_posterior=model_posterior,
                        )
                    else:
                        # NORMAL MODE: Generate patch
                        patch_result = await self._run_patch_async(
                            parent_program,
                            archive_programs,
                            top_k_programs,
                            generation,
                            meta_recs,
                            novelty_attempt=attempt + 1,
                            resample_attempt=resample + 1,
                            model_sample_probs=model_sample_probs,
                            model_posterior=model_posterior,
                        )

                    if not patch_result:
                        continue

                    code_diff, meta_patch_data, success = patch_result
                    api_costs += meta_patch_data.get("api_costs", 0.0)

                    if not success:
                        continue

                    # We have a successful patch, break from resample loop
                    meta_patch_data["api_costs"] = api_costs
                    break
                except Exception as e:
                    logger.warning(
                        f"Error in patch generation attempt {resample + 1}: {e}"
                    )
                    continue
            else:
                # No successful patch in all resamples, continue to next novelty attempt
                continue

            # Get code embedding (only once per successful patch)
            if self.verbose:
                logger.info(f"Getting code embedding for generation {generation}...")
            code_embedding, e_cost = await self._get_code_embedding_async(exec_fname)
            embed_cost += e_cost
            if self.verbose:
                logger.info(
                    f"Code embedding completed for generation {generation} (cost: ${e_cost:.4f})"
                )

            if not code_embedding:
                if self.novelty_judge:
                    self.novelty_judge.log_novelty_skip_message("no embedding")
                proposal_accepted = True  # Accept program even without embedding
                break

            # Novelty check (same logic as sync runner)
            if self.novelty_judge:
                should_check = await self.novelty_judge.should_check_novelty_async(
                    code_embedding, generation, parent_program, self.db
                )

                if should_check:
                    (
                        should_accept,
                        novelty_metadata,
                    ) = await self.novelty_judge.assess_novelty_with_rejection_sampling_async(
                        exec_fname, code_embedding, parent_program, self.db
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

                    if should_accept:
                        proposal_accepted = True
                        break
                    # If not accepted, continue to next attempt (rejection sampling)
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
                # Submit job
                job_id = await self.scheduler.submit_async_nonblocking(
                    exec_fname, results_dir
                )

                # Create running job
                running_job = AsyncRunningJob(
                    job_id=job_id,
                    exec_fname=exec_fname,
                    results_dir=results_dir,
                    start_time=time.time(),
                    generation=generation,
                    parent_id=parent_program.id,
                    archive_insp_ids=[p.id for p in archive_programs],
                    top_k_insp_ids=[p.id for p in top_k_programs],
                    code_diff=code_diff,
                    meta_patch_data=meta_patch_data,
                    code_embedding=code_embedding,
                    embed_cost=embed_cost,
                    novelty_cost=novelty_total_cost,  # Store novelty cost in running job
                    proposal_task_id=task_id,
                )

                # Update costs
                meta_patch_data["api_costs"] = api_costs
                proposal_total_cost = api_costs + embed_cost + novelty_total_cost
                self.total_api_cost += proposal_total_cost

                # Update average proposal cost for in-flight estimation
                self._update_avg_proposal_cost(proposal_total_cost)

                # Check if we're exceeding max evaluation jobs (race condition protection)
                while len(self.running_jobs) >= self.max_evaluation_jobs:
                    if self.verbose:
                        logger.info(
                            f"⏳ Waiting for evaluation slot: {len(self.running_jobs)}/{self.max_evaluation_jobs} "
                            f"jobs running, gen {generation} proposal ready"
                        )
                    # Wait for a slot to become available
                    await asyncio.sleep(0.5)  # Short wait to avoid busy loop

                    # Check if we should stop (in case system is shutting down)
                    if self.should_stop.is_set():
                        logger.warning(
                            f"System shutting down, cancelling job for gen {generation}"
                        )
                        try:
                            await self.scheduler.cancel_job_async(job_id)
                        except Exception as e:
                            logger.warning(f"Failed to cancel job during shutdown: {e}")
                        return None

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
                return None

        logger.warning(
            f"Failed to generate proposal for generation {generation} after all attempts"
        )
        return None

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
            metadata["llm_model"] = getattr(response, "model", None)

        metadata_file = attempt_dir / "metadata.json"
        await write_file_async(str(metadata_file), json.dumps(metadata, indent=2))

    async def _run_fix_patch_async(
        self,
        incorrect_program: Program,
        ancestor_inspirations: List[Program],
        generation: int,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
        model_sample_probs: Optional[List[float]] = None,
        model_posterior: Optional[List[float]] = None,
    ) -> Optional[Tuple[Optional[str], Dict[str, Any], bool]]:
        """
        Run async fix patch generation for an incorrect program.

        This is used when no correct programs exist in the database.

        Args:
            incorrect_program: The incorrect program to fix
            ancestor_inspirations: Ancestors of the program (from sample_with_fix_mode)
            generation: Current generation number
            meta_recs: Meta recommendations
            novelty_attempt: Current novelty attempt number
            resample_attempt: Current resample attempt number
            model_sample_probs: Model sampling probabilities
            model_posterior: Model posterior probabilities
        """
        try:
            # Generate fix prompt with ancestor inspirations
            patch_sys, patch_msg, patch_type = self.prompt_sampler.sample_fix(
                incorrect_program=incorrect_program,
                ancestor_inspirations=ancestor_inspirations,
            )

            patch_type = str(patch_type)

            if self.verbose:
                logger.info(f"Generated FIX patch type: {patch_type}")

            total_costs = 0.0
            msg_history = []

            llm_kwargs = self.llm.get_kwargs(model_sample_probs=model_sample_probs)

            if self.llm_selection is not None:
                model_name = llm_kwargs.get("model_name", "unknown")
                self.llm_selection.update_submitted(model_name)

            for patch_attempt in range(self.evo_config.max_patch_attempts):
                response = await self.llm.query(
                    msg=patch_msg,
                    system_msg=patch_sys,
                    msg_history=msg_history,
                    model_sample_probs=model_sample_probs,
                    model_posterior=model_posterior,
                )

                if not response or not response.content:
                    error_str = "LLM response content was None."

                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=error_str,
                        patch_text=None,
                        num_applied=0,
                        patch_name=None,
                        patch_description=None,
                        success=False,
                    )

                    if patch_attempt < self.evo_config.max_patch_attempts - 1:
                        patch_msg = "The previous fix attempt failed. Try again."
                        continue
                    else:
                        break

                total_costs += response.cost if response.cost else 0.0

                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                )

                patch_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{generation}"
                language_str = str(self.evo_config.language)

                # Fix patches always use full rewrite
                result = await apply_patch_async(
                    original_str=incorrect_program.code,
                    patch_str=response.content,
                    patch_dir=patch_dir,
                    language=language_str,
                    patch_type="full",  # Fix always uses full rewrite
                    verbose=False,
                )

                (
                    modified_code,
                    num_applied,
                    output_path,
                    error_msg,
                    patch_txt,
                    patch_path,
                ) = result

                if error_msg is None and num_applied > 0:
                    diff_summary = {}
                    if patch_path:
                        diff_summary = summarize_diff(str(patch_path))
                        if f"original.{self.lang_ext}" in diff_summary:
                            diff_summary = diff_summary[f"original.{self.lang_ext}"]

                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=None,
                        patch_text=patch_txt,
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
                        "novelty_attempt": novelty_attempt,
                        "resample_attempt": resample_attempt,
                        "patch_attempt": patch_attempt + 1,
                        **llm_kwargs,  # Spread llm_kwargs like _run_patch_async
                        "llm_result": response.to_dict() if response else None,
                    }

                    if self.verbose:
                        logger.info(
                            f"  FIX ATTEMPT {patch_attempt + 1}/"
                            f"{self.evo_config.max_patch_attempts} SUCCESS"
                        )
                        self._print_metadata_table(meta_patch_data, generation)

                    return patch_txt, meta_patch_data, True

                # Patch application failed
                error_str = str(error_msg) if error_msg else "No changes applied."

                await self._save_patch_attempt_async(
                    generation=generation,
                    novelty_attempt=novelty_attempt,
                    resample_attempt=resample_attempt,
                    patch_attempt=patch_attempt + 1,
                    response=response,
                    error_msg=error_str,
                    patch_text=patch_txt,
                    num_applied=num_applied,
                    patch_name=patch_name,
                    patch_description=patch_description,
                    success=False,
                )

                if self.verbose:
                    logger.info(
                        f"  FIX ATTEMPT {patch_attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} FAILURE: {error_str}"
                    )

                if patch_attempt < self.evo_config.max_patch_attempts - 1:
                    patch_msg = (
                        f"The previous fix attempt failed. Error: {error_str}. "
                        f"Try again with a different approach."
                    )
                    msg_history = response.new_msg_history if response else []

            # Update LLM selection costs
            if self.llm_selection is not None:
                self.llm_selection.update_cost(arm=model_name, cost=total_costs)

            # All attempts failed
            meta_patch_data = {
                "api_costs": total_costs,
                "patch_type": patch_type,
                "patch_name": patch_name,
                "patch_description": patch_description,
                "num_applied": 0,
                "error_attempt": "Max fix attempts reached without success.",
                "novelty_attempt": novelty_attempt,
                "resample_attempt": resample_attempt,
                "patch_attempt": self.evo_config.max_patch_attempts,
                **llm_kwargs,  # Spread llm_kwargs like _run_patch_async
                "llm_result": response.to_dict() if response else None,
            }

            return None, meta_patch_data, False

        except Exception as e:
            logger.error(f"Error in fix patch async: {e}")
            return None, {"api_costs": 0.0, "error_attempt": str(e)}, False

    async def _run_patch_async(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        generation: int,
        meta_recs: Optional[str] = None,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
        model_sample_probs: Optional[List[float]] = None,
        model_posterior: Optional[List[float]] = None,
    ) -> Optional[Tuple[Optional[str], Dict[str, Any], bool]]:
        """Run async patch generation."""
        # Initialize prompt-related variables outside try block for exception handling
        current_prompt_id: Optional[str] = None
        original_task_sys_msg = self.prompt_sampler.task_sys_msg

        try:
            # Get system prompt (potentially evolved)
            current_sys_prompt, current_prompt_id = self._get_current_system_prompt()

            # Temporarily update prompt_sampler with evolved prompt
            if current_sys_prompt:
                self.prompt_sampler.task_sys_msg = current_sys_prompt

            # Generate patch prompt
            patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
                parent=parent_program,
                archive_inspirations=archive_programs,
                top_k_inspirations=top_k_programs,
                meta_recommendations=meta_recs,
            )

            # Restore original task_sys_msg
            self.prompt_sampler.task_sys_msg = original_task_sys_msg

            # Convert numpy string to regular Python string
            patch_type = str(patch_type)

            if self.verbose:
                logger.info(f"Generated patch type: {patch_type}")
                if current_prompt_id:
                    logger.info(f"Using evolved prompt: {current_prompt_id[:8]}...")

            total_costs = 0.0
            msg_history = []

            # Use provided model_sample_probs (selected once before all loops)
            llm_kwargs = self.llm.get_kwargs(model_sample_probs=model_sample_probs)

            # Update LLM selection with submission
            if self.llm_selection is not None:
                model_name = llm_kwargs.get("model_name", "unknown")
                self.llm_selection.update_submitted(model_name)

            for patch_attempt in range(self.evo_config.max_patch_attempts):
                # Query LLM for patch
                response = await self.llm.query(
                    msg=patch_msg,
                    system_msg=patch_sys,
                    msg_history=msg_history,
                    model_sample_probs=model_sample_probs,
                    model_posterior=model_posterior,
                )

                if not response or not response.content:
                    error_str = "LLM response content was None."

                    # Save failed attempt data
                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=error_str,
                        patch_text=None,
                        num_applied=0,
                        patch_name=None,
                        patch_description=None,
                        success=False,
                    )

                    if patch_attempt < self.evo_config.max_patch_attempts - 1:
                        patch_msg = "The previous attempt failed. Try again."
                        continue
                    else:
                        break

                total_costs += response.cost if response.cost else 0.0

                # Extract patch name and description from LLM response
                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                )

                # Apply patch asynchronously
                patch_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{generation}"

                if self.verbose:
                    logger.info(
                        f"Applying patch with language: {self.evo_config.language}"
                    )
                    logger.info(f"Patch type for application: {patch_type}")

                # Ensure language is a string
                language_str = str(self.evo_config.language)

                result = await apply_patch_async(
                    original_str=parent_program.code,
                    patch_str=response.content,
                    patch_dir=patch_dir,
                    language=language_str,
                    patch_type=patch_type,
                    verbose=False,
                )

                (
                    modified_code,
                    num_applied,
                    output_path,
                    error_msg,
                    patch_txt,
                    patch_path,
                ) = result

                if error_msg is None and num_applied > 0:
                    # Success - generate diff summary
                    diff_summary = {}
                    if patch_path:
                        diff_summary = summarize_diff(str(patch_path))
                        # Only consider the diff summary for the original.py file
                        if "original.py" in diff_summary:
                            diff_summary = diff_summary["original.py"]

                    # Save successful attempt data
                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=None,
                        patch_text=patch_txt,
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
                        "novelty_attempt": novelty_attempt,
                        "resample_attempt": resample_attempt,
                        "patch_attempt": patch_attempt + 1,
                        "system_prompt_id": current_prompt_id,  # Track evolved prompt
                        **llm_kwargs,
                        "llm_result": response.to_dict() if response else None,
                    }

                    # Print metadata table for successful patches
                    if self.verbose:
                        self._print_metadata_table(meta_patch_data, generation)

                    return patch_txt, meta_patch_data, True
                else:
                    # Failure, try again
                    error_str = error_msg or "No changes applied"

                    # Save failed attempt data
                    await self._save_patch_attempt_async(
                        generation=generation,
                        novelty_attempt=novelty_attempt,
                        resample_attempt=resample_attempt,
                        patch_attempt=patch_attempt + 1,
                        response=response,
                        error_msg=error_str,
                        patch_text=patch_txt,
                        num_applied=num_applied,
                        patch_name=patch_name,
                        patch_description=patch_description,
                        success=False,
                    )

                    patch_msg = f"The previous edit was not successful. Error: {error_str}\n\nTry again."
                    msg_history = (
                        response.new_msg_history
                        if hasattr(response, "new_msg_history")
                        else []
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
                "error_attempt": "Max attempts reached without successful patch",
                "novelty_attempt": novelty_attempt,
                "resample_attempt": resample_attempt,
                "patch_attempt": self.evo_config.max_patch_attempts,
                "system_prompt_id": current_prompt_id,  # Track evolved prompt
            }

            return None, meta_patch_data, False

        except Exception as e:
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

    async def _get_code_embedding_async(
        self, exec_fname: str
    ) -> Tuple[Optional[List[float]], float]:
        """Get code embedding asynchronously."""
        if not self.embedding_client:
            return None, 0.0

        return await get_code_embedding_async(exec_fname, self.embedding_client)

    async def _process_completed_jobs_safely(
        self, completed_jobs: List[AsyncRunningJob]
    ):
        """Process completed jobs with robust error handling to ensure no jobs are lost."""
        successfully_processed = []

        for job in completed_jobs:
            try:
                success = await self._process_single_job_safely(job)
                if success:
                    successfully_processed.append(job)
                    # Mark job as processed in registry
                    if str(job.job_id) in self.submitted_jobs:
                        del self.submitted_jobs[str(job.job_id)]
                else:
                    logger.error(
                        f"❌ CRITICAL: Failed to process job {job.job_id} (gen {job.generation})"
                    )
                    # Keep job in registry for potential retry

            except Exception as e:
                logger.error(
                    f"❌ CRITICAL: Exception processing job {job.job_id} (gen {job.generation}): {e}"
                )
                # Keep job in registry for potential retry

        logger.info(
            f"✅ Successfully processed {len(successfully_processed)}/{len(completed_jobs)} jobs"
        )

        # If some jobs failed, log the issue but don't lose them
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
        try:
            logger.info(
                f"🔄 SAFE PROCESSING: Starting job {job.job_id} (gen {job.generation})"
            )

            # Get job results with timeout to prevent hanging
            try:
                results = await asyncio.wait_for(
                    self.scheduler.get_job_results_async(job.job_id, job.results_dir),
                    timeout=30.0,  # 30 second timeout
                )
                logger.info(
                    f"📂 RESULTS: Got results for {job.job_id}: {results is not None}"
                )
            except asyncio.TimeoutError:
                logger.error(f"❌ TIMEOUT: Getting results for {job.job_id} timed out")
                return False

            # Always create a program entry, even if results are missing
            if results:
                # Extract metrics properly like the sync version
                correct_val = results.get("correct", {}).get("correct", False)
                metrics_val = results.get("metrics", {})
                combined_score = metrics_val.get("combined_score", 0.0)
                public_metrics = metrics_val.get("public", {})
                private_metrics = metrics_val.get("private", {})
                text_feedback = metrics_val.get("text_feedback", "")
                stdout_log = results.get("stdout_log", "")
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

            # Create program from results (or defaults if results missing)
            program = Program(
                id=str(uuid.uuid4()),
                code=await self._read_file_async(job.exec_fname) or "",
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
                embedding=job.code_embedding or [],
                system_prompt_id=system_prompt_id,  # Track evolved prompt
                metadata={
                    "compute_time": time.time() - job.start_time,
                    **(job.meta_patch_data or {}),
                    "embed_cost": job.embed_cost,
                    "novelty_cost": job.novelty_cost,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "results_missing": results is None,
                    "safe_processing": True,
                },
            )

            # Add to database with timeout protection
            logger.info(
                f"💾 DB ADD: Adding program to database for {job.job_id} (gen {job.generation})..."
            )

            try:
                await asyncio.wait_for(
                    self.async_db.add_program_async(
                        program,
                        parent_id=job.parent_id,
                        archive_insp_ids=job.archive_insp_ids,
                        top_k_insp_ids=job.top_k_insp_ids,
                        code_diff=job.code_diff,
                        meta_patch_data=job.meta_patch_data,
                        code_embedding=job.code_embedding,
                        embed_cost=job.embed_cost,
                    ),
                    timeout=30.0,  # 30 second timeout for DB operations
                )
                logger.info(
                    f"✅ DB SUCCESS: Program {program.id} successfully added to database for {job.job_id} (gen {job.generation})"
                )

                # Update prompt fitness if prompt evolution is enabled
                if system_prompt_id and self.evo_config.evolve_prompts:
                    # Calculate improvement (need parent score)
                    parent_score = 0.0
                    if job.parent_id:
                        parent_program = await self.async_db.get_async(job.parent_id)
                        if parent_program:
                            parent_score = parent_program.combined_score or 0.0

                    program_score = combined_score or 0.0
                    improvement = program_score - parent_score
                    await self._update_prompt_fitness(
                        system_prompt_id,
                        program.id,
                        program_score=program_score,
                        improvement=improvement,
                        correct=correct_val,
                    )

                    # Check if we should evolve a new prompt
                    await self._maybe_evolve_prompt()

            except asyncio.TimeoutError:
                job.db_retry_count += 1
                logger.error(
                    f"❌ DB TIMEOUT: Adding program to database for "
                    f"{job.job_id} timed out "
                    f"(retry {job.db_retry_count}/"
                    f"{self.MAX_DB_RETRY_ATTEMPTS})"
                )

                if job.db_retry_count < self.MAX_DB_RETRY_ATTEMPTS:
                    # Add to retry queue for later retry
                    self.failed_jobs_for_retry[str(job.job_id)] = job
                    logger.info(
                        f"🔄 RETRY QUEUED: Job {job.job_id} "
                        f"(gen {job.generation}) added to retry queue"
                    )
                else:
                    logger.error(
                        f"❌ RETRY EXHAUSTED: Job {job.job_id} "
                        f"(gen {job.generation}) exceeded "
                        f"max retry attempts "
                        f"({self.MAX_DB_RETRY_ATTEMPTS}). "
                        f"Job permanently lost."
                    )
                return False
            except Exception as e:
                job.db_retry_count += 1
                logger.error(
                    f"❌ DB ERROR: Failed to add program to database "
                    f"for {job.job_id}: {e} "
                    f"(retry {job.db_retry_count}/"
                    f"{self.MAX_DB_RETRY_ATTEMPTS})"
                )

                if job.db_retry_count < self.MAX_DB_RETRY_ATTEMPTS:
                    # Add to retry queue for later retry
                    self.failed_jobs_for_retry[str(job.job_id)] = job
                    logger.info(
                        f"🔄 RETRY QUEUED: Job {job.job_id} "
                        f"(gen {job.generation}) added to retry queue"
                    )
                else:
                    logger.error(
                        f"❌ RETRY EXHAUSTED: Job {job.job_id} "
                        f"(gen {job.generation}) exceeded "
                        f"max retry attempts "
                        f"({self.MAX_DB_RETRY_ATTEMPTS}). "
                        f"Job permanently lost."
                    )
                return False

            # Handle meta summarizer and other post-processing
            if self.meta_summarizer:
                try:
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

                                # Update the program in the database
                                def update_metadata():
                                    from shinka.database import (
                                        ProgramDatabase,
                                    )

                                    thread_db = ProgramDatabase(self.db.config)
                                    try:
                                        metadata_json = json.dumps(program.metadata)
                                        thread_db.cursor.execute(
                                            (
                                                "UPDATE programs SET "
                                                "metadata = ? WHERE id = ?"
                                            ),
                                            (metadata_json, program.id),
                                        )
                                        thread_db.conn.commit()
                                    finally:
                                        thread_db.close()

                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, update_metadata)
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
                        self.llm_selection.print_summary()
                except Exception as e:
                    logger.warning(f"LLM selection update error for {job.job_id}: {e}")
                    # Don't fail the whole job for LLM selection issues

            # Update best solution
            try:
                await self._update_best_solution_async()
            except Exception as e:
                logger.warning(f"Best solution update error for {job.job_id}: {e}")
                # Don't fail the whole job for best solution update issues

            logger.info(
                f"✅ JOB COMPLETE: Finished processing {job.job_id} - program {program.id} added (gen {job.generation})"
            )
            return True

        except Exception as e:
            logger.error(
                f"❌ CRITICAL: Exception in safe processing for job {job.job_id} (gen {job.generation}): {e}"
            )
            logger.error(
                f"   Job details: exec_fname={job.exec_fname}, results_dir={job.results_dir}"
            )
            return False

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

    async def _update_completed_generations(self):
        """Update completed generations count for async evolution.

        In async evolution, generations can complete out of order. For termination
        and progress tracking, what matters is the total count of completed work,
        not whether it's contiguous. This counts all generations that have:
        1. No running jobs AND
        2. Programs in the database (successful evaluation)
        """
        # Get all generations that have running jobs
        running_generations = {job.generation for job in self.running_jobs}

        # More efficient approach: get total program count and subtract running jobs
        # This avoids expensive per-generation database queries
        try:
            # Get total number of programs in database (much faster single query)
            total_programs = await self.async_db.get_total_program_count_async()

            # Account for island copies: the initial program gets duplicated
            # (num_islands - 1) times, so we need to subtract these extra copies
            num_islands = getattr(self.db_config, "num_islands", 1)
            if num_islands > 1:
                # Subtract the extra island copies of generation 0
                island_copies = num_islands - 1
                total_programs -= island_copies

            # Each generation should have exactly 1 program when completed
            # So completed generations = total programs - programs from running jobs
            programs_from_running = len(self.running_jobs)

            # Account for jobs in retry queue (completed eval but failed DB write)
            # These jobs are not in running_jobs but also not in the database yet
            programs_in_retry = len(self.failed_jobs_for_retry)

            # The completed count is total programs minus running jobs
            # and minus jobs waiting for DB retry
            # (since each successful evaluation adds exactly 1 program)
            calculated_completed = (
                total_programs - programs_from_running - programs_in_retry
            )

            # Debug logging when count doesn't change
            if (
                self.verbose
                and hasattr(self, "completed_generations")
                and calculated_completed == self.completed_generations
            ):
                logger.debug(
                    f"📊 Completion calc: total_programs={total_programs}, "
                    f"running={programs_from_running}, "
                    f"retry={programs_in_retry}, "
                    f"result={calculated_completed}"
                )

            self.completed_generations = calculated_completed

            # Ensure we don't exceed target generations
            max_gens = self.evo_config.num_generations
            self.completed_generations = min(self.completed_generations, max_gens)

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
        max_check_gen = min(target_gens, next_gen + 10)

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

            pending_work = self.evo_config.num_generations - self.completed_generations
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
            logger.info(
                f"🔧 ATTEMPTING RECOVERY: Force-starting proposal generation..."
            )

            try:
                # Force start at least one proposal if we have uncompleted work
                # Use completed_generations to determine pending work, not next_generation_to_submit
                pending_work = (
                    self.evo_config.num_generations - self.completed_generations
                )
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
            await self.async_db.close_async()

            # Cleanup scheduler
            self.scheduler.shutdown()

        except Exception as e:
            logger.error(f"Error in async cleanup: {e}")

    async def _print_final_summary(self):
        """Print final evolution summary."""
        if not self.verbose:
            return

        end_time = time.time()
        total_time = end_time - (self.start_time or end_time)

        logger.info("=" * 80)
        logger.info("ASYNC EVOLUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total generations: {self.completed_generations}")
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
            self.db.print_summary()

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
                with open(file_path, "r") as f:
                    return f.read()

            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, read_file)
            return content
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None
