import difflib
import hashlib
import json
import logging
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from subprocess import Popen
from typing import Any, Dict, List, Literal, Optional, Union, cast

import rich.box
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from shinka.core.embedding_corpus import extract_file_content
from shinka.core.novelty_judge import NoveltyJudge
from shinka.core.sampler import PromptSampler
from shinka.core.summarizer import MetaSummarizer
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.edit import (
    AgentContext,
    AgenticEditor,
    CommandResult,
    apply_diff_patch,
    apply_full_patch,
    redact_immutable,
    summarize_diff,
)
from shinka.edit.codex_cli import (
    CodexAuthError,
    CodexExecutionError,
    CodexUnavailableError,
    ensure_codex_available,
    run_codex_task,
    validate_codex_setup,
)
from shinka.edit.shinka_agent import (
    ShinkaExecutionError,
    ShinkaUnavailableError,
    ensure_shinka_available,
    run_shinka_task,
)
from shinka.launch import JobConfig, JobScheduler, ProcessWithLogging
from shinka.llm import (
    AsymmetricUCB,
    BanditBase,
    EmbeddingClient,
    LLMClient,
    extract_between,
)
from shinka.logo import print_gradient_logo
from shinka.eval.agentic import AgenticEvaluator, AgenticEvaluatorResult

FOLDER_PREFIX = "gen"

# Number of session events to include in agentic evaluator metadata
AGENTIC_EVAL_PREVIEW_LIMIT = 50

# Directories to exclude when copying workspace files for agentic edits
WORKSPACE_EXCLUDE_DIRS = {
    "results",
    "workspace_snapshot",
    "agent_sessions",
    ".hydra",
    "__pycache__",
}
WORKSPACE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
WORKSPACE_EXCLUDE_FILES = {
    "rewrite.txt",
    "edit.diff",
    "session_log.jsonl",
}


@dataclass
class AgenticConfig:
    """Configuration options for agentic editing sessions.

    This config supports Codex CLI and ShinkaAgent backends.
    The `backend` field selects which one to use.
    """

    backend: str = "shinka"  # "shinka" or "codex"
    cli_profile: Optional[str] = None
    sandbox: str = "workspace-write"
    approval_mode: str = "full-auto"
    max_turns: int = 50
    max_events: int = 240  # Event limit for Codex CLI streaming (3x default)
    max_seconds: int = 0
    cli_path: Optional[str] = None
    extra_cli_config: Dict[str, Any] = field(default_factory=dict)
    resume_parent_session: bool = False
    # Base directory for scratch workspaces. Using /tmp ensures scratch dirs are
    # outside any git repo, preventing CLI from discovering parent AGENTS.md files.
    scratch_dir_base: Optional[str] = "/tmp/shinka_scratch"


@dataclass
class AgenticEvaluatorConfig:
    """Configuration for agentic evaluation sessions.

    The evaluator can use a different backend than the editor.
    If backend is None, inherits from AgenticConfig.backend.
    """

    backend: Optional[str] = None  # If None, use agentic.backend
    cli_profile: Optional[str] = None
    sandbox: str = "workspace-write"
    approval_mode: str = "full-auto"
    max_events: int = 240  # Event limit for Codex CLI streaming (3x default)
    max_seconds: int = 0
    cli_path: Optional[str] = None
    extra_cli_config: Dict[str, Any] = field(default_factory=dict)
    eval_prompt: Optional[str] = None  # Custom evaluation criteria for LLM judge


@dataclass
class EvaluatorConfig:
    """Evaluator selection configuration."""

    mode: Literal["auto", "legacy", "agentic"] = "auto"
    agentic: AgenticEvaluatorConfig = field(default_factory=AgenticEvaluatorConfig)


@dataclass
class EvolutionConfig:
    task_sys_msg: Optional[str] = None
    patch_types: List[str] = field(default_factory=lambda: ["diff"])
    patch_type_probs: List[float] = field(default_factory=lambda: [1.0])
    num_generations: int = 10
    max_parallel_jobs: int = 2
    max_patch_resamples: int = 3
    max_patch_attempts: int = 5
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=lambda: ["azure-gpt-4.1-mini"])
    llm_dynamic_selection: Optional[Union[str, BanditBase]] = None
    llm_dynamic_selection_kwargs: dict = field(default_factory=lambda: {})
    llm_kwargs: dict = field(default_factory=lambda: {})
    meta_rec_interval: Optional[int] = None
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    embedding_model: Optional[str] = None
    init_program_path: Optional[str] = "initial.py"
    results_dir: Optional[str] = None
    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 1.0
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False
    # Agentic editing configuration
    agentic_mode: bool = False
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    # Maximum possible score for evaluation (used by agentic evaluator prompts)
    max_score: float = 100.0
    # Multi-file support: directory containing additional files to copy
    init_support_dir: Optional[str] = None


@dataclass
class RunningJob:
    """Represents a running job in the queue."""

    job_id: Union[str, Popen, ProcessWithLogging]
    exec_fname: str
    results_dir: str
    generation_dir: Path
    start_time: float
    generation: int
    parent_id: Optional[str]
    archive_insp_ids: List[str]
    top_k_insp_ids: List[str]
    code_diff: Optional[str]
    meta_patch_data: Optional[dict]
    code_embedding: List[float] = field(default_factory=list)
    embed_cost: float = 0.0
    novelty_cost: float = 0.0
    # For multi-file embedding corpus
    corpus_text: str = ""
    corpus_meta: dict = field(default_factory=dict)
    # For agentic evaluator results (pre-computed when agentic mode)
    agentic_result: Optional[tuple] = None
    # For async agentic evaluation (Future object)
    agentic_future: Optional[Future] = None


# Set up logging
logger = logging.getLogger(__name__)


class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        verbose: bool = True,
    ):
        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.verbose = verbose

        print_gradient_logo((255, 0, 0), (255, 255, 255))
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results_{timestamp}"
        else:
            self.results_dir = Path(evo_config.results_dir)

        if self.verbose:
            # Create log file path in results directory
            log_filename = f"{self.results_dir}/evolution_run.log"
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

            # Set up logging with both console and file handlers
            logging.basicConfig(
                level=logging.INFO,
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
            )

            # Also log the initial setup information
            logger.info("=" * 80)
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Evolution run started at {start_time}")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Log file: {log_filename}")
            logger.info("=" * 80)

        # Validate agentic backend setup early (fail fast, not mid-evolution)
        if evo_config.agentic_mode:
            if evo_config.agentic.backend == "codex":
                logger.info("Validating Codex backend setup...")
                validate_codex_setup(evo_config.agentic.cli_path)
                logger.info("Codex backend validated successfully")
            else:
                logger.info("Validating ShinkaAgent backend setup...")
                ensure_shinka_available()
                logger.info("ShinkaAgent backend validated successfully")

        # Check if we are resuming a run
        resuming_run = False
        db_path = Path(f"{self.results_dir}/{db_config.db_path}")
        if self.evo_config.results_dir is not None and db_path.exists():
            resuming_run = True

        # Initialize LLM selection strategy
        if evo_config.llm_dynamic_selection is None:
            self.llm_selection = None
        elif isinstance(evo_config.llm_dynamic_selection, BanditBase):
            self.llm_selection = evo_config.llm_dynamic_selection
        elif (evo_config.llm_dynamic_selection.lower() == "ucb") or (
            evo_config.llm_dynamic_selection.lower() == "ucb1"
        ):
            self.llm_selection = AsymmetricUCB(
                arm_names=evo_config.llm_models,
                **evo_config.llm_dynamic_selection_kwargs,
            )
        else:
            raise ValueError("Invalid llm_dynamic_selection")

        # Initialize database and scheduler
        db_config.db_path = str(db_path)
        embedding_model_to_use = evo_config.embedding_model or "text-embedding-3-small"
        self.db = ProgramDatabase(
            config=db_config, embedding_model=embedding_model_to_use
        )
        self.scheduler = JobScheduler(
            job_type=evo_config.job_type,
            config=job_config,  # type: ignore
            verbose=verbose,
        )

        # Initialize agentic evaluator if enabled
        self.evaluator_mode = self._resolve_evaluator_mode()
        if self.evaluator_mode == "agentic":
            # Use evaluator-specific backend if set, else fall back to agentic backend
            eval_backend = (
                self.evo_config.evaluator.agentic.backend
                or self.evo_config.agentic.backend
            )
            if eval_backend == "shinka":
                runner_fn = run_shinka_task
            else:
                runner_fn = run_codex_task
            self.agentic_evaluator: Optional[AgenticEvaluator] = AgenticEvaluator(
                self.evo_config.evaluator.agentic,
                agent_runner=runner_fn,
            )
            if self.verbose:
                logger.info(f"Agentic evaluator using backend: {eval_backend}")
        else:
            self.agentic_evaluator = None
        self.agentic_eval_sessions_dir = (
            Path(self.results_dir) / "agentic_eval_sessions"
        )
        # Thread pool for parallel job execution (uses max_parallel_jobs workers)
        # Enabled when agentic editing mode is on (works with both legacy and agentic eval)
        self._eval_executor: Optional[ThreadPoolExecutor] = None
        if evo_config.agentic_mode:
            max_workers = evo_config.max_parallel_jobs or 6
            self._eval_executor = ThreadPoolExecutor(max_workers=max_workers)
            if self.verbose:
                logger.info(f"Parallel agentic editing enabled with {max_workers} workers")

        self.llm = LLMClient(
            model_names=evo_config.llm_models,
            model_selection=self.llm_selection,
            **evo_config.llm_kwargs,
            verbose=verbose,
        )
        if evo_config.embedding_model is not None:
            self.embedding = EmbeddingClient(
                model_name=evo_config.embedding_model,
                verbose=verbose,
            )
        else:
            self.embedding = None

        if evo_config.meta_llm_models is not None:
            self.meta_llm = LLMClient(
                model_names=evo_config.meta_llm_models,
                **evo_config.meta_llm_kwargs,
                verbose=verbose,
            )
        else:
            self.meta_llm = None

        if evo_config.novelty_llm_models is not None:
            self.novelty_llm = LLMClient(
                model_names=evo_config.novelty_llm_models,
                **evo_config.novelty_llm_kwargs,
                verbose=verbose,
            )
        else:
            self.novelty_llm = None

        # Initialize PromptSampler for handling LLM code prompts
        self.prompt_sampler = PromptSampler(
            task_sys_msg=evo_config.task_sys_msg,
            language=evo_config.language,
            patch_types=evo_config.patch_types,
            patch_type_probs=evo_config.patch_type_probs,
            use_text_feedback=evo_config.use_text_feedback,
            agentic_mode=evo_config.agentic_mode,
        )

        # Initialize MetaSummarizer for meta-recommendations
        self.meta_summarizer = MetaSummarizer(
            meta_llm_client=self.meta_llm,
            language=evo_config.language,
            use_text_feedback=evo_config.use_text_feedback,
            max_recommendations=evo_config.meta_max_recommendations,
        )

        # Initialize NoveltyJudge for novelty assessment
        # Pass agentic config for potential future use, with graceful fallback
        self.novelty_judge = NoveltyJudge(
            novelty_llm_client=self.novelty_llm,
            language=evo_config.language,
            similarity_threshold=evo_config.code_embed_sim_threshold,
            max_novelty_attempts=evo_config.max_novelty_attempts,
            # Agentic novelty (falls back to legacy if agent_runner not set)
            agentic_mode=evo_config.agentic_mode,
            agent_runner=None,  # Not implemented in minimal PR
            agent_config=evo_config.agentic if evo_config.agentic_mode else None,
        )

        # Initialize rich console for formatted output
        self.console = Console()

        if self.evo_config.language == "cuda":
            self.lang_ext = "cu"
        elif self.evo_config.language == "cpp":
            self.lang_ext = "cpp"
        elif self.evo_config.language == "python":
            self.lang_ext = "py"
        elif self.evo_config.language == "rust":
            self.lang_ext = "rs"
        elif self.evo_config.language == "swift":
            self.lang_ext = "swift"
        elif self.evo_config.language in ["json", "json5"]:
            self.lang_ext = "json"
        else:
            msg = f"Language {self.evo_config.language} not supported"
            raise ValueError(msg)

        # Queue for managing parallel jobs
        self.running_jobs: List[RunningJob] = []
        self.best_program_id: Optional[str] = None
        self.next_generation_to_submit = 0

        if resuming_run:
            self.completed_generations = self.db.last_iteration + 1
            self.next_generation_to_submit = self.completed_generations
            logger.info("=" * 80)
            logger.info("RESUMING PREVIOUS EVOLUTION RUN")
            logger.info("=" * 80)
            logger.info(
                f"Resuming evolution from: {self.results_dir}\n"
                f"Found {self.completed_generations} "
                "previously completed generations."
            )
            logger.info("=" * 80)
            self._update_best_solution()
            # Restore meta memory state when resuming
            self._restore_meta_memory()
        else:
            self.completed_generations = 0

        # Save experiment configuration to a YAML file
        self._save_experiment_config(evo_config, job_config, db_config)

    def _save_experiment_config(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
    ) -> None:
        """Save experiment configuration to a YAML file."""
        config_data = {
            "evolution_config": asdict(evo_config),
            "job_config": asdict(job_config),
            "database_config": asdict(db_config),
            "timestamp": datetime.now().isoformat(),
            "results_directory": str(self.results_dir),
        }

        config_path = Path(self.results_dir) / "experiment_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Experiment configuration saved to {config_path}")

    def run(self):
        """Run evolution with parallel job queue."""
        max_jobs = self.evo_config.max_parallel_jobs
        target_gens = self.evo_config.num_generations
        logger.info(
            f"Starting evolution with {max_jobs} parallel jobs, "
            f"target: {target_gens} generations"
        )

        # First, run generation 0 sequentially to populate the database
        if self.completed_generations == 0 and target_gens > 0:
            logger.info("Running generation 0 sequentially to initialize database...")
            self._run_generation_0()
            self.completed_generations = 1
            self.next_generation_to_submit = 1
            logger.info(f"Completed generation 0, total: 1/{target_gens}")

        # Now start parallel execution for remaining generations
        if self.completed_generations < target_gens:
            logger.info("Starting parallel execution for remaining generations...")

            # Main loop: monitor jobs and submit new ones
            while (
                self.completed_generations < target_gens or len(self.running_jobs) > 0
            ):
                # Check for completed jobs
                completed_jobs = self._check_completed_jobs()

                # Process completed jobs
                if completed_jobs:
                    for job in completed_jobs:
                        self._process_completed_job(job)

                    # Update completed generations count
                    self._update_completed_generations()

                    if self.verbose:
                        logger.info(
                            f"Processed {len(completed_jobs)} jobs. "
                            f"Total completed generations: "
                            f"{self.completed_generations}/{target_gens}"
                        )

                # Check if we've completed all generations
                if self.completed_generations >= target_gens:
                    logger.info("All generations completed, exiting...")
                    break

                # Submit new jobs to fill the queue (only if we have capacity)
                while (
                    len(self.running_jobs) < max_jobs
                    and self.next_generation_to_submit < target_gens
                ):
                    if self.evo_config.agentic_mode:
                        # Full parallelism: parent sampling in main thread (thread-safe),
                        # edit + eval in worker threads (works with both legacy and agentic eval)
                        self._submit_agentic_job_async()
                    else:
                        self._submit_new_job()
                        break  # Legacy editing mode submits one job at a time

                # Wait a bit before checking again
                time.sleep(2)

            # All jobs are now handled by the main loop above

        # Perform final meta summary for any remaining unprocessed programs
        best_program = self.db.get_best_program()
        self.meta_summarizer.perform_final_summary(str(self.results_dir), best_program)

        # Save final meta memory state
        self._save_meta_memory()

        self.db.print_summary()
        logger.info(f"Evolution completed! {self.completed_generations} generations")
        logger.info("=" * 80)
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Evolution run ended at {end_time}")
        logger.info("=" * 80)

        # Cleanup thread pool executors
        if self._eval_executor is not None:
            self._eval_executor.shutdown(wait=False)
            self._eval_executor = None
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.shutdown()

    def generate_initial_program(self):
        """Generate initial program with LLM, with retries."""
        llm_kwargs = self.llm.get_kwargs()

        sys_msg, user_msg = self.prompt_sampler.initial_program_prompt()
        msg_history = []
        total_costs = 0.0

        for attempt in range(self.evo_config.max_patch_attempts):
            response = self.llm.query(
                msg=user_msg,
                system_msg=sys_msg,
                llm_kwargs=llm_kwargs,
                msg_history=msg_history,
            )
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: LLM response content was None."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "The previous response was empty. Please try again "
                        "and provide the full code."
                    )
                    if response and response.new_msg_history:
                        msg_history = response.new_msg_history
                    continue
                else:
                    break

            total_costs += response.cost or 0
            initial_code = extract_between(
                response.content,
                f"```{self.evo_config.language}",
                "```",
                False,
            )

            if initial_code:
                patch_name = extract_between(
                    response.content, "<NAME>", "</NAME>", False
                )
                patch_description = extract_between(
                    response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
                )
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
                return initial_code, patch_name, patch_description, total_costs
            else:  # code extraction failed
                if self.verbose:
                    logger.info(
                        f"  INITIAL PROGRAM ATTEMPT {attempt + 1}/"
                        f"{self.evo_config.max_patch_attempts} "
                        "FAILURE. Error: Could not extract code from response."
                    )
                if attempt < self.evo_config.max_patch_attempts - 1:
                    user_msg = (
                        "Could not extract code from your last response. "
                        "Please make sure to enclose the code in "
                        "`<CODE>`...`</CODE>` tags."
                    )
                    msg_history = response.new_msg_history
                else:  # last attempt
                    break

        raise ValueError(
            "LLM failed to generate a valid initial program after "
            f"{self.evo_config.max_patch_attempts} attempts."
        )

    def _run_generation_0(self):
        """Setup and run generation 0 to initialize the database."""
        initial_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0"
        Path(initial_dir).mkdir(parents=True, exist_ok=True)
        exec_fname = f"{initial_dir}/main.{self.lang_ext}"
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_0/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        api_costs = 0.0
        patch_name = "initial_program"
        patch_description = "Initial program from file."
        patch_type = "init"

        # Multi-file support: copy additional support files into generation 0 directory
        if self.evo_config.init_support_dir:
            support_dir = Path(self.evo_config.init_support_dir)
            if support_dir.is_dir():
                for path in support_dir.rglob("*"):
                    rel = path.relative_to(support_dir)
                    # Skip excluded dirs/files
                    if any(part in WORKSPACE_EXCLUDE_DIRS for part in rel.parts):
                        continue
                    if path.is_dir():
                        continue
                    if path.suffix in WORKSPACE_EXCLUDE_SUFFIXES:
                        continue
                    if path.name in WORKSPACE_EXCLUDE_FILES:
                        continue
                    target = Path(initial_dir) / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            else:
                logger.warning(
                    f"init_support_dir provided but not a directory: {support_dir}"
                )

        if self.evo_config.init_program_path:
            if self.verbose:
                logger.info(
                    f"Copying initial program from {self.evo_config.init_program_path}"
                )
            shutil.copy(self.evo_config.init_program_path, exec_fname)
        else:
            if self.verbose:
                logger.info(
                    "`init_program_path` not provided, "
                    "generating initial program with LLM..."
                )
            initial_code, patch_name, patch_description, api_costs = (
                self.generate_initial_program()
            )
            with open(exec_fname, "w", encoding="utf-8") as f:
                f.write(initial_code)

            if self.verbose:
                logger.info(f"Initial program generated and saved to {exec_fname}")

        # Run the evaluation synchronously
        if self.evaluator_mode == "agentic":
            results, rtime = self._run_agentic_evaluation(
                exec_fname=exec_fname,
                results_dir=results_dir,
                generation_dir=Path(initial_dir),
                generation=0,
                parent_id=None,
            )
        else:
            results, rtime = self.scheduler.run(exec_fname, results_dir)

        code_embedding, e_cost = self.get_code_embedding(exec_fname)

        # Read the evaluated code for database insertion
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=None,
            generation=0,
            archive_inspiration_ids=[],
            top_k_inspiration_ids=[],
            code_diff=None,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                "api_costs": api_costs,
                "embed_cost": e_cost,
                "novelty_cost": 0.0,  # No novelty cost for generation 0
                "patch_type": patch_type,
                "patch_name": patch_name,
                "patch_description": patch_description,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            },
        )

        self.db.add(db_program, verbose=True)
        if self.llm_selection is not None:
            self.llm_selection.set_baseline_score(
                db_program.combined_score if correct_val else 0.0,
            )
        self.db.save()
        self._update_best_solution()

        # Add the evaluated program to meta memory tracking
        self.meta_summarizer.add_evaluated_program(db_program)

        # Check if we should update meta memory after adding this program
        if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
            logger.info(
                f"Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(
                best_program
            )
            if updated_recs:
                # Write meta output file for generation 0
                self.meta_summarizer.write_meta_output(str(self.results_dir))
                # Store meta cost for tracking
                if meta_cost > 0:
                    logger.info(
                        f"Meta recommendation generation cost: ${meta_cost:.4f}"
                    )
                    # Add meta cost to this program's metadata (the one that triggered the update)
                    if db_program.metadata is None:
                        db_program.metadata = {}
                    db_program.metadata["meta_cost"] = meta_cost
                    # Update the program in the database with the new metadata
                    import json

                    metadata_json = json.dumps(db_program.metadata)
                    self.db.cursor.execute(
                        "UPDATE programs SET metadata = ? WHERE id = ?",
                        (metadata_json, db_program.id),
                    )
                    self.db.conn.commit()

        # Save meta memory state after each job completion
        self._save_meta_memory()

    def _update_completed_generations(self):
        """
        Update the count of completed generations from the database.
        A generation `g` is considered complete if all generations from 0..g
        have at least one program in the database. This ensures the count
        advances sequentially without gaps.
        """
        last_gen = self.db.last_iteration
        if last_gen == -1:
            self.completed_generations = 0
            return

        # Check for contiguous generations from 0 up to last_gen
        completed_up_to = 0
        for i in range(last_gen + 1):
            if self.db.get_programs_by_generation(i):
                completed_up_to = i + 1
            else:
                # Found a gap, so contiguous sequence is broken
                self.completed_generations = completed_up_to
                return

        self.completed_generations = completed_up_to

    def _submit_new_job(self):
        """Submit a new job to the queue."""
        current_gen = self.next_generation_to_submit

        if current_gen >= self.evo_config.num_generations:
            return

        self.next_generation_to_submit += 1

        generation_dir = Path(self.results_dir) / f"{FOLDER_PREFIX}_{current_gen}"
        exec_fname = str(generation_dir / f"main.{self.lang_ext}")
        results_dir = str(generation_dir / "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get current meta-recommendations for this job
        meta_recs, meta_summary, meta_scratch = self.meta_summarizer.get_current()

        # Sample parent and inspiration programs
        if current_gen == 0:
            parent_id = None
            archive_insp_ids = []
            top_k_insp_ids = []
            code_diff = None
            meta_patch_data = {}
            # Initial program already copied in setup_initial_program
        else:
            api_costs = 0
            embed_cost = 0
            novelty_cost = 0.0
            novelty_checks_performed = 0
            # Loop over novelty attempts
            for nov_attempt in range(self.evo_config.max_novelty_attempts):
                # Loop over patch resamples - including parents
                for resample in range(self.evo_config.max_patch_resamples):
                    (
                        parent_program,
                        archive_programs,
                        top_k_programs,
                    ) = self.db.sample(
                        target_generation=current_gen,
                        novelty_attempt=nov_attempt + 1,
                        max_novelty_attempts=self.evo_config.max_novelty_attempts,
                        resample_attempt=resample + 1,
                        max_resample_attempts=self.evo_config.max_patch_resamples,
                    )
                    archive_insp_ids = [p.id for p in archive_programs]
                    top_k_insp_ids = [p.id for p in top_k_programs]
                    parent_id = parent_program.id
                    # Run patch (until success with max attempts)
                    code_diff, meta_patch_data, num_applied_attempt = self.run_patch(
                        parent_program,
                        archive_programs,
                        top_k_programs,
                        current_gen,
                        novelty_attempt=nov_attempt + 1,
                        resample_attempt=resample + 1,
                    )
                    api_costs += meta_patch_data["api_costs"]
                    if (
                        meta_patch_data["error_attempt"] is None
                        and num_applied_attempt > 0
                    ):
                        meta_patch_data["api_costs"] = api_costs
                        break

                # Get the code embedding for the evaluated code
                code_embedding, e_cost = self.get_code_embedding(exec_fname)
                embed_cost += e_cost

                if not code_embedding:
                    self.novelty_judge.log_novelty_skip_message("no embedding")
                    break

                # Use NoveltyJudge for novelty assessment with rejection sampling
                if self.novelty_judge.should_check_novelty(
                    code_embedding, current_gen, parent_program, self.db
                ):
                    should_accept, novelty_metadata = (
                        self.novelty_judge.assess_novelty_with_rejection_sampling(
                            exec_fname, code_embedding, parent_program, self.db
                        )
                    )

                    # Update costs and metadata from novelty assessment
                    novelty_cost += novelty_metadata.get("novelty_total_cost", 0.0)
                    novelty_checks_performed = novelty_metadata.get(
                        "novelty_checks_performed", 0
                    )
                    novelty_explanation = novelty_metadata.get(
                        "novelty_explanation", ""
                    )

                    if should_accept:
                        break
                    # If not accepted, continue to next attempt (rejection sampling)
                else:
                    if not self.db.island_manager or not hasattr(
                        self.db.island_manager, "are_all_islands_initialized"
                    ):
                        self.novelty_judge.log_novelty_skip_message("no island manager")
                    elif not self.db.island_manager.are_all_islands_initialized():
                        self.novelty_judge.log_novelty_skip_message(
                            "not all islands initialized yet"
                        )
                    break

        # Add meta-recommendations/summary/scratchpad to meta_patch_data
        if meta_recs is not None:
            meta_patch_data["meta_recommendations"] = meta_recs
            meta_patch_data["meta_summary"] = meta_summary
            meta_patch_data["meta_scratch_pad"] = meta_scratch

        # Add novelty check information to meta_patch_data if any checks were performed
        if current_gen > 0 and novelty_checks_performed > 0:
            meta_patch_data["novelty_checks_performed"] = novelty_checks_performed
            meta_patch_data["novelty_cost"] = novelty_cost
            meta_patch_data["novelty_explanation"] = novelty_explanation

        # Submit the job (agentic uses async thread pool, legacy uses async scheduler)
        if self.evaluator_mode == "agentic":
            # Submit agentic evaluation to thread pool for parallel execution
            future = self._eval_executor.submit(
                self._run_agentic_evaluation,
                exec_fname=exec_fname,
                results_dir=results_dir,
                generation_dir=generation_dir,
                generation=current_gen,
                parent_id=parent_id,
            )
            # Create job with future for async completion checking
            running_job = RunningJob(
                job_id=f"agentic_gen_{current_gen}",
                exec_fname=exec_fname,
                results_dir=results_dir,
                generation_dir=generation_dir,
                start_time=time.time(),
                generation=current_gen,
                parent_id=parent_id,
                archive_insp_ids=archive_insp_ids,
                top_k_insp_ids=top_k_insp_ids,
                code_diff=code_diff,
                meta_patch_data=meta_patch_data,
                code_embedding=code_embedding,
                embed_cost=embed_cost,
                novelty_cost=novelty_cost,
                agentic_future=future,  # Store future for completion checking
            )
            self.running_jobs.append(running_job)
        else:
            job_id = self.scheduler.submit_async(exec_fname, results_dir)
            # Add to running jobs queue
            running_job = RunningJob(
                job_id=job_id,
                exec_fname=exec_fname,
                results_dir=results_dir,
                generation_dir=generation_dir,
                start_time=time.time(),
                generation=current_gen,
                parent_id=parent_id,
                archive_insp_ids=archive_insp_ids,
                top_k_insp_ids=top_k_insp_ids,
                code_diff=code_diff,
                meta_patch_data=meta_patch_data,
                code_embedding=code_embedding,
                embed_cost=embed_cost,
                novelty_cost=novelty_cost,
            )
            self.running_jobs.append(running_job)

        if self.verbose:
            logger.info(
                f"Submitted job for generation {current_gen}, "
                f"queue size: {len(self.running_jobs)}"
            )

    def _submit_agentic_job_async(self):
        """Submit an agentic job asynchronously (non-blocking).

        This method samples the parent in the main thread (thread-safe DB access),
        then submits the edit + eval to the thread pool for parallel execution.
        """
        current_gen = self.next_generation_to_submit

        if current_gen >= self.evo_config.num_generations:
            return

        self.next_generation_to_submit += 1

        generation_dir = Path(self.results_dir) / f"{FOLDER_PREFIX}_{current_gen}"
        exec_fname = str(generation_dir / f"main.{self.lang_ext}")
        results_dir = str(generation_dir / "results")

        # Sample parent in main thread (DB access is NOT thread-safe)
        parent_program, archive_programs, top_k_programs = self.db.sample(
            target_generation=current_gen,
            novelty_attempt=1,
            max_novelty_attempts=self.evo_config.max_novelty_attempts,
            resample_attempt=1,
            max_resample_attempts=self.evo_config.max_patch_resamples,
        )
        parent_id = parent_program.id
        archive_insp_ids = [p.id for p in archive_programs]
        top_k_insp_ids = [p.id for p in top_k_programs]

        # Get meta-recommendations in main thread
        meta_recs, meta_summary, meta_scratch = self.meta_summarizer.get_current()

        # Submit the edit + eval to thread pool (no DB access in worker)
        future = self._eval_executor.submit(
            self._run_full_agentic_job,
            current_gen=current_gen,
            generation_dir=generation_dir,
            exec_fname=exec_fname,
            results_dir=results_dir,
            parent_program=parent_program,
            archive_programs=archive_programs,
            top_k_programs=top_k_programs,
            meta_recs=meta_recs,
            meta_summary=meta_summary,
            meta_scratch=meta_scratch,
        )

        # Create job with known parent info
        running_job = RunningJob(
            job_id=f"agentic_async_gen_{current_gen}",
            exec_fname=exec_fname,
            results_dir=results_dir,
            generation_dir=generation_dir,
            start_time=time.time(),
            generation=current_gen,
            parent_id=parent_id,
            archive_insp_ids=archive_insp_ids,
            top_k_insp_ids=top_k_insp_ids,
            code_diff=None,
            meta_patch_data={},
            agentic_future=future,
        )
        self.running_jobs.append(running_job)

        if self.verbose:
            logger.info(
                f"Submitted async agentic job for gen {current_gen}, "
                f"queue size: {len(self.running_jobs)}"
            )

    def _run_full_agentic_job(
        self,
        current_gen: int,
        generation_dir: Path,
        exec_fname: str,
        results_dir: str,
        parent_program: "Program",
        archive_programs: List["Program"],
        top_k_programs: List["Program"],
        meta_recs: Optional[str],
        meta_summary: Optional[str],
        meta_scratch: Optional[str],
    ) -> tuple:
        """Run the full agentic job (edit + eval) in a thread.

        NOTE: This runs in a worker thread. It must NOT access self.db directly
        because SQLite connections are not thread-safe. All parent/inspiration
        data is passed in from the main thread.

        Returns tuple of (results, rtime, job_metadata).
        """
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        parent_id = parent_program.id
        archive_insp_ids = [p.id for p in archive_programs]
        top_k_insp_ids = [p.id for p in top_k_programs]

        # Run the edit (patch generation)
        code_diff, meta_patch_data, num_applied = self.run_patch(
            parent_program,
            archive_programs,
            top_k_programs,
            current_gen,
            novelty_attempt=1,
            resample_attempt=1,
        )

        # Get code embedding (thread-safe - uses HTTP calls)
        code_embedding, embed_cost = self.get_code_embedding(exec_fname)

        # Add meta info
        if meta_recs is not None:
            meta_patch_data["meta_recommendations"] = meta_recs
            meta_patch_data["meta_summary"] = meta_summary
            meta_patch_data["meta_scratch_pad"] = meta_scratch

        # Run evaluation (legacy or agentic based on evaluator_mode)
        if self.evaluator_mode == "legacy":
            results, rtime = self._run_legacy_evaluation_sync(
                exec_fname=exec_fname,
                results_dir=results_dir,
            )
        else:
            results, rtime = self._run_agentic_evaluation(
                exec_fname=exec_fname,
                results_dir=results_dir,
                generation_dir=generation_dir,
                generation=current_gen,
                parent_id=parent_id,
            )

        # Return all data needed to process the job
        # Note: novelty_cost is 0 because we skip novelty checks in parallel mode
        # (novelty checks require DB access which is not thread-safe)
        job_metadata = {
            "parent_id": parent_id,
            "archive_insp_ids": archive_insp_ids,
            "top_k_insp_ids": top_k_insp_ids,
            "code_diff": code_diff,
            "meta_patch_data": meta_patch_data,
            "code_embedding": code_embedding,
            "embed_cost": embed_cost,
            "novelty_cost": 0.0,
        }

        return (results, rtime, job_metadata)

    def _check_completed_jobs(self) -> List[RunningJob]:
        """Check for completed jobs and return them."""
        completed = []
        still_running = []

        for job in self.running_jobs:
            # Agentic jobs with pre-computed results are already complete
            if job.agentic_result is not None:
                if self.verbose:
                    logger.info(f"Agentic job for gen {job.generation} completed!")
                completed.append(job)
            # Agentic jobs with futures - check if future is done
            elif job.agentic_future is not None:
                if job.agentic_future.done():
                    # Future completed - get results and store them
                    try:
                        future_result = job.agentic_future.result()
                        # Handle both 2-tuple (results, rtime) and 3-tuple (results, rtime, metadata)
                        if len(future_result) == 3:
                            results, rtime, job_metadata = future_result
                            # Update job with metadata from async execution
                            job.parent_id = job_metadata.get("parent_id")
                            job.archive_insp_ids = job_metadata.get("archive_insp_ids", [])
                            job.top_k_insp_ids = job_metadata.get("top_k_insp_ids", [])
                            job.code_diff = job_metadata.get("code_diff")
                            job.meta_patch_data = job_metadata.get("meta_patch_data", {})
                            job.code_embedding = job_metadata.get("code_embedding", [])
                            job.embed_cost = job_metadata.get("embed_cost", 0.0)
                            job.novelty_cost = job_metadata.get("novelty_cost", 0.0)
                        else:
                            results, rtime = future_result
                        job.agentic_result = (results, rtime)
                        if self.verbose:
                            logger.info(f"Agentic job for gen {job.generation} completed (async)!")
                        completed.append(job)
                    except Exception as e:
                        # Evaluation failed - create error result
                        logger.error(f"Agentic evaluation for gen {job.generation} failed: {e}")
                        job.agentic_result = (
                            {"correct": {"correct": False}, "metrics": {"error": str(e)}},
                            time.time() - job.start_time,
                        )
                        completed.append(job)
                else:
                    # Future still running
                    still_running.append(job)
            else:
                is_running = self.scheduler.check_job_status(job)
                if not is_running:
                    # Job completed
                    if self.verbose:
                        logger.info(f"Job {job.job_id} completed!")
                    completed.append(job)
                else:
                    # Job still running
                    still_running.append(job)

        self.running_jobs = still_running
        return completed

    def _process_completed_job(self, job: RunningJob):
        """Process a completed job and add results to database."""
        end_time = time.time()

        # Get job results (agentic has pre-computed results, legacy uses scheduler)
        if job.agentic_result is not None:
            results, rtime = job.agentic_result
        else:
            rtime = end_time - job.start_time
            results = self.scheduler.get_job_results(job.job_id, job.results_dir)

        # Read the evaluated code
        try:
            evaluated_code = Path(job.exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {job.job_id}. Error: {e}")
            evaluated_code = ""

        # Use pre-computed embedding and novelty costs
        code_embedding = job.code_embedding
        e_cost = job.embed_cost
        n_cost = job.novelty_cost
        if self.verbose:
            logger.debug(
                f"=> Using pre-computed embedding for job {job.job_id}, "
                f"embed cost: {e_cost:.4f}, novelty cost: {n_cost:.4f}"
            )

        correct_val = False
        metrics_val = {}
        stdout_log = ""
        stderr_log = ""
        if results:
            correct_val = results.get("correct", {}).get("correct", False)
            metrics_val = results.get("metrics", {})
            stdout_log = results.get("stdout_log", "")
            stderr_log = results.get("stderr_log", "")

        combined_score = metrics_val.get("combined_score", 0.0)
        public_metrics = metrics_val.get("public", {})
        private_metrics = metrics_val.get("private", {})
        text_feedback = metrics_val.get("text_feedback", "")

        # Add the program to the database
        db_program = Program(
            id=str(uuid.uuid4()),
            code=evaluated_code,
            language=self.evo_config.language,
            parent_id=job.parent_id,
            generation=job.generation,
            archive_inspiration_ids=job.archive_insp_ids,
            top_k_inspiration_ids=job.top_k_insp_ids,
            code_diff=job.code_diff,
            embedding=code_embedding,
            correct=correct_val,
            combined_score=combined_score,
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            text_feedback=text_feedback,
            metadata={
                "compute_time": rtime,
                **(job.meta_patch_data or {}),
                "embed_cost": e_cost,
                "novelty_cost": n_cost,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            },
        )
        self.db.add(db_program, verbose=True)

        # Add the evaluated program to meta memory tracking
        self.meta_summarizer.add_evaluated_program(db_program)

        # Check if we should update meta memory after adding this program
        if self.meta_summarizer.should_update_meta(self.evo_config.meta_rec_interval):
            logger.info(
                f"Updating meta memory after processing "
                f"{len(self.meta_summarizer.evaluated_since_last_meta)} programs..."
            )
            best_program = self.db.get_best_program()
            updated_recs, meta_cost = self.meta_summarizer.update_meta_memory(
                best_program
            )
            if updated_recs:
                # Write meta output file using accumulated program count
                self.meta_summarizer.write_meta_output(str(self.results_dir))
                # Store meta cost for tracking
                if meta_cost > 0:
                    logger.info(
                        f"Meta recommendation generation cost: ${meta_cost:.4f}"
                    )
                    # Add meta cost to this program's metadata (the one that triggered the update)
                    if db_program.metadata is None:
                        db_program.metadata = {}
                    db_program.metadata["meta_cost"] = meta_cost
                    # Update the program in the database with the new metadata
                    import json

                    metadata_json = json.dumps(db_program.metadata)
                    self.db.cursor.execute(
                        "UPDATE programs SET metadata = ? WHERE id = ?",
                        (metadata_json, db_program.id),
                    )
                    self.db.conn.commit()

        if self.llm_selection is not None:
            if "model_name" not in db_program.metadata:
                logger.warning(
                    "No model_name found in program metadata, "
                    "unable to update model selection algorithm."
                )
            else:
                parent = (
                    self.db.get(db_program.parent_id) if db_program.parent_id else None
                )
                baseline = parent.combined_score if parent else None
                reward = db_program.combined_score if correct_val else None
                model_name = db_program.metadata["model_name"]
                result = self.llm_selection.update(
                    arm=model_name,
                    reward=reward,
                    baseline=baseline,
                )
                if result and self.verbose:
                    normalized_score, baseline = result

                    def fmt(x):
                        return f"{x:.4f}" if isinstance(x, (float, int)) else "None"

                    logger.debug(
                        f"==> UPDATED LLM SELECTION: model: "
                        f"{model_name.split('/')[-1][-25:]}..., "
                        f"score: {fmt(normalized_score)}, "
                        f"raw score: {fmt(reward)}, baseline: {fmt(baseline)}"
                    )
                    self.llm_selection.print_summary()

        self.db.save()
        self._update_best_solution()

        # Note: Meta summarization check is now done after completed generations
        # are updated in the main loop to ensure correct timing

        # Save meta memory state after each job completion
        self._save_meta_memory()

    def _update_best_solution(self):
        """Checks and updates the best program."""
        best_programs = self.db.get_top_programs(n=1, correct_only=True)
        if not best_programs:
            if self.verbose:
                logger.debug(
                    "No correct programs found yet, cannot determine best solution."
                )
            return

        best_program = best_programs[0]

        if best_program.id == self.best_program_id:
            return  # No change

        self.best_program_id = best_program.id

        source_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{best_program.generation}"
        best_dir = Path(self.results_dir) / "best"

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(source_dir, best_dir)

        if self.verbose:
            logger.info(
                f"New best program found: gen {best_program.generation}, "
                f"id {best_program.id[:6]}... "
                f"Copied to {best_dir}"
            )

    def run_patch(
        self,
        parent_program: Program,
        archive_programs: List[Program],
        top_k_programs: List[Program],
        generation: int,
        novelty_attempt: int = 1,
        resample_attempt: int = 1,
    ) -> tuple[Optional[str], dict, int]:
        """Run patch generation for a specific generation."""
        max_patch_attempts = self.evo_config.max_patch_attempts
        if self.verbose:
            logger.info(
                f"Edit Cycle {generation} -> {generation + 1}, "
                f"Max Patch Attempts: {max_patch_attempts}"
            )
        # Get current meta recommendations
        meta_recs, _, _ = self.meta_summarizer.get_current()
        # Construct edit / code change message
        patch_sys, patch_msg, patch_type = self.prompt_sampler.sample(
            parent=parent_program,
            archive_inspirations=archive_programs,
            top_k_inspirations=top_k_programs,
            meta_recommendations=meta_recs,
        )

        # Route to agentic patch if enabled
        if self.evo_config.agentic_mode:
            return self._run_agentic_patch(
                parent_program=parent_program,
                generation=generation,
                patch_sys=patch_sys,
                patch_msg=patch_msg,
                patch_type=patch_type,
                novelty_attempt=novelty_attempt,
                resample_attempt=resample_attempt,
            )

        if patch_type in ["full", "cross"]:
            apply_patch = apply_full_patch
        elif patch_type == "diff":
            apply_patch = apply_diff_patch
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
            # apply_patch = apply_paper_patch
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        # Multi-file support (legacy patch path): ensure helper files are present.
        # Agentic mode hydrates the workspace explicitly; for legacy patches we
        # hydrate from the parent generation directory so multi-file tasks can run.
        generation_dir = Path(self.results_dir) / f"{FOLDER_PREFIX}_{generation}"
        if generation_dir.is_dir():
            # Clear any stale workspace files from earlier patch attempts/resamples.
            # Keep evaluation artifacts directories (e.g., results/) intact.
            for child in generation_dir.iterdir():
                if child.name in WORKSPACE_EXCLUDE_DIRS:
                    continue
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except OSError:
                    continue
            self._hydrate_generation_directory(parent_program, generation_dir)

        total_costs = 0
        msg_history = []
        llm_kwargs = self.llm.get_kwargs()
        if self.llm_selection is not None:
            model_name = llm_kwargs["model_name"]
            self.llm_selection.update_submitted(model_name)
        code_diff = None  # Initialize code_diff
        num_applied_attempt = 0  # Initialize num_applied_attempt
        error_attempt = (
            "Max attempts reached without successful patch."  # Default error
        )
        patch_name = None
        patch_description = None
        output_path_attempt = None
        patch_txt_attempt = None
        patch_path = None
        diff_summary = {}

        for patch_attempt in range(max_patch_attempts):
            response = self.llm.query(
                msg=patch_msg,
                system_msg=patch_sys,
                msg_history=msg_history,
                llm_kwargs=llm_kwargs,
            )
            # print(response.content)
            if response is None or response.content is None:
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: LLM response content was None."
                    )
                # Prepare for next attempt or exit
                error_attempt = "LLM response content was None."
                num_applied_attempt = 0
                patch_txt_attempt = None
                if patch_attempt < max_patch_attempts - 1:
                    patch_msg = (
                        "The previous attempt to get an edit was not "
                        "successful because the LLM response was empty. "
                        "Try again."
                    )
                    if response:
                        msg_history = response.new_msg_history
                    continue
                else:  # Last attempt
                    break

            total_costs += response.cost  # Acc. cost
            patch_name = extract_between(
                response.content,
                "<NAME>",
                "</NAME>",
                False,
            )
            patch_description = extract_between(
                response.content,
                "<DESCRIPTION>",
                "</DESCRIPTION>",
                False,
            )

            # Apply the code patch (diff/full rewrite)
            (
                _,
                num_applied_attempt,
                output_path_attempt,
                error_attempt,
                patch_txt_attempt,
                patch_path,
            ) = apply_patch(
                original_str=parent_program.code,
                patch_str=response.content,
                patch_dir=f"{self.results_dir}/{FOLDER_PREFIX}_{generation}",
                language=self.evo_config.language,
                verbose=False,
            )

            if error_attempt is None and num_applied_attempt > 0:
                if patch_path:  # Ensure patch_path is not None
                    diff_summary = summarize_diff(
                        str(patch_path)
                    )  # Convert Path to str
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} SUCCESS. "
                        f"Output: {output_path_attempt}, "
                        f"Patches Applied: {num_applied_attempt}."
                    )

                code_diff = patch_txt_attempt
                break  # Break from patch attempts
            else:
                error_str = (
                    str(error_attempt) if error_attempt else "No changes applied."
                )
                patch_msg = (
                    "The previous edit was not successful."
                    + " This was the error message: \n\n"
                    + error_str
                    + "\n\n Try again."
                )
                if self.verbose:
                    logger.info(
                        f"  PATCH ATTEMPT {patch_attempt + 1}/{max_patch_attempts} FAILURE. "
                        f"Error: '{error_str}', "
                        f"Patches Applied: {num_applied_attempt}."
                    )
                msg_history = response.new_msg_history
                code_diff = None
                if patch_attempt == max_patch_attempts - 1:  # Last attempt failed
                    # error_attempt is already set from apply_patch or default
                    pass

        # Only consider the diff summary for the original source file
        original_filename = f"original.{self.lang_ext}"
        if original_filename in diff_summary:
            diff_summary = diff_summary[original_filename]

        meta_edit_data = {
            "patch_type": patch_type,
            "api_costs": total_costs,
            "num_applied": num_applied_attempt,
            "patch_name": patch_name,
            "patch_description": patch_description,
            "error_attempt": error_attempt,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "patch_attempt": patch_attempt + 1,
            **llm_kwargs,
            "llm_result": response.to_dict() if response else None,
            "diff_summary": diff_summary,
        }
        if self.verbose and num_applied_attempt > 0:
            self._print_metadata_table(meta_edit_data, generation)
        # Delete generation from meta_edit_data
        return code_diff, meta_edit_data, num_applied_attempt

    def get_code_embedding(self, exec_fname: str) -> tuple[List[float], float]:
        """Get the embedding of the code."""
        # Read the evaluated code
        try:
            evaluated_code = Path(exec_fname).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read code for job {exec_fname}. Error: {e}")
            evaluated_code = ""
        if evaluated_code != "":
            # Get the embedding of the initial program
            try:
                if self.embedding is not None:
                    redacted_code = redact_immutable(evaluated_code, no_state=True)
                    if self.verbose:
                        logger.debug(
                            "=> EMBED: Code length - "
                            f"Original: {len(evaluated_code)} - "
                            f"Redacted: {len(redacted_code)}"
                        )

                    embedding_result, e_cost = self.embedding.get_embedding(
                        redacted_code
                    )
                else:
                    if self.verbose:
                        logger.debug("=> EMBED: No embedding model configured.")
                    embedding_result = []
                    e_cost = 0.0
                code_embedding = cast(List[float], embedding_result)
            except Exception as e:
                logger.warning(f"Could not embed code for job {exec_fname}. Error: {e}")
                code_embedding = []
                e_cost = 0.0
        else:
            code_embedding = []
            e_cost = 0.0
        return code_embedding, e_cost

    def _print_metadata_table(self, meta_data: dict, generation: int):
        """Display metadata in a formatted rich table."""
        # Create title with generation and attempt information
        title_parts = ["[bold magenta]Patch Metadata"]

        # Add generation if present
        if generation is not None:
            title_parts.append(
                f" - Gen {generation}/{self.evo_config.num_generations} - Novelty: {meta_data['novelty_attempt']}/{self.evo_config.max_novelty_attempts} - Resample: {meta_data['resample_attempt']}/{self.evo_config.max_patch_resamples} - Patch: {meta_data['patch_attempt']}/{self.evo_config.max_patch_attempts}"
            )

        # Add attempt information if present
        if all(
            key in meta_data
            for key in [
                "novelty_attempt",
                "resample_attempt",
                "patch_attempt",
                "generation",
            ]
        ):
            title_parts.append(
                f" (Novelty: {meta_data['novelty_attempt']}, "
                f"Resample: {meta_data['resample_attempt']}, "
                f"Patch: {meta_data['patch_attempt']})"
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

        # Add remaining fields (excluding llm_result, diff_summary, and header info)
        skip_fields = set(
            display_order
            + [
                "llm_result",
                "diff_summary",
                "generation",
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

    def _save_meta_memory(self) -> None:
        """Save the meta memory state to disk."""
        meta_memory_path = Path(self.results_dir) / "meta_memory.json"
        self.meta_summarizer.save_meta_state(str(meta_memory_path))

    def _restore_meta_memory(self) -> None:
        """Restore the meta memory state from disk."""
        meta_memory_path = Path(self.results_dir) / "meta_memory.json"

        if self.verbose:
            logger.info(f"Attempting to restore meta memory from: {meta_memory_path}")

        success = self.meta_summarizer.load_meta_state(str(meta_memory_path))
        if success:
            logger.info("Successfully restored meta memory state")
        else:
            if meta_memory_path.exists():
                logger.warning(
                    f"Meta memory file exists but failed to load: {meta_memory_path}"
                )
            else:
                logger.info("No previous meta memory state found - starting fresh")

    def _collect_parent_workspace_files(
        self, parent_program: Program
    ) -> Dict[Path, str]:
        """Collect workspace files from parent program's generation directory."""
        workspace_files: Dict[Path, str] = {}
        parent_generation_dir = (
            Path(self.results_dir) / f"{FOLDER_PREFIX}_{parent_program.generation}"
        )
        if parent_generation_dir.is_dir():
            for file_path in parent_generation_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                rel_path = file_path.relative_to(parent_generation_dir)
                if any(part in WORKSPACE_EXCLUDE_DIRS for part in rel_path.parts):
                    continue
                if file_path.suffix in WORKSPACE_EXCLUDE_SUFFIXES:
                    continue
                if file_path.name in WORKSPACE_EXCLUDE_FILES:
                    continue
                try:
                    workspace_files[rel_path] = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
            return workspace_files

        parent_metadata = parent_program.metadata or {}

        # Fallback: Check if parent has stored changed files from agentic edit
        agent_changed = parent_metadata.get("agent_changed_files")
        if agent_changed and isinstance(agent_changed, dict):
            for rel_path_str, content in agent_changed.items():
                workspace_files[Path(rel_path_str)] = content

        return workspace_files

    def _hydrate_generation_directory(
        self, parent_program: Program, generation_dir: Path
    ) -> None:
        """Copy workspace files from parent to new generation directory."""
        parent_generation_dir = (
            Path(self.results_dir) / f"{FOLDER_PREFIX}_{parent_program.generation}"
        )
        if parent_generation_dir.is_dir():
            for src_path in parent_generation_dir.rglob("*"):
                rel_path = src_path.relative_to(parent_generation_dir)
                if any(part in WORKSPACE_EXCLUDE_DIRS for part in rel_path.parts):
                    continue
                if src_path.is_dir():
                    continue
                if src_path.suffix in WORKSPACE_EXCLUDE_SUFFIXES:
                    continue
                if src_path.name in WORKSPACE_EXCLUDE_FILES:
                    continue
                dst_path = generation_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            return

        # Fallback to metadata-stored files
        workspace_files = self._collect_parent_workspace_files(parent_program)
        for rel_path, content in workspace_files.items():
            target_path = generation_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")

    def _run_agentic_patch(
        self,
        *,
        parent_program: Program,
        generation: int,
        patch_sys: str,
        patch_msg: str,
        patch_type: str,
        novelty_attempt: int,
        resample_attempt: int,
    ) -> tuple[Optional[str], dict, int]:
        """Execute an agentic editing session via CLI backend (Codex or ShinkaAgent)."""
        logger.info(f"_run_agentic_patch: START gen={generation} nov={novelty_attempt} resamp={resample_attempt}")

        primary_filename = Path(f"main.{self.lang_ext}")

        # Extract content from corpus; fallback to raw code if not a corpus
        primary_content = extract_file_content(
            parent_program.code, str(primary_filename)
        )
        if primary_content is None:
            if "=== FILE:" not in parent_program.code:
                primary_content = parent_program.code
            else:
                primary_content = extract_file_content(parent_program.code, "main.py")
                if primary_content is None:
                    primary_content = parent_program.code

        base_files: Dict[Path, str] = {primary_filename: primary_content}
        base_files.update(self._collect_parent_workspace_files(parent_program))

        session_root: Optional[Path] = None
        parent_metadata = parent_program.metadata or {}
        resume_session_id: Optional[str] = None
        resumed_from_parent = False

        if self.evo_config.agentic.resume_parent_session:
            candidate = parent_metadata.get("agent_session_id")
            if isinstance(candidate, str) and candidate.strip():
                resume_session_id = candidate.strip()
                resumed_from_parent = True

        def _serialize_changed_files(
            changed_files: Optional[Dict[Path, str]],
        ) -> Dict[str, str]:
            if not changed_files:
                return {}
            serialized: Dict[str, str] = {}
            for rel_path, content in changed_files.items():
                if rel_path == primary_filename:
                    continue
                serialized[str(rel_path)] = content
            return serialized

        def _build_code_diffs(
            changed_files: Optional[Dict[Path, str]],
        ) -> List[Dict[str, str]]:
            """Build multi-file diffs for frontend display."""
            if not changed_files:
                return []
            diffs: List[Dict[str, str]] = []
            for rel_path, new_content in changed_files.items():
                before = base_files.get(rel_path, "")
                before_lines = before.splitlines(keepends=True)
                after_lines = new_content.splitlines(keepends=True)
                diff_text = "".join(
                    difflib.unified_diff(
                        before_lines,
                        after_lines,
                        fromfile=f"a/{rel_path}",
                        tofile=f"b/{rel_path}",
                    )
                )
                diffs.append({"path": str(rel_path), "diff": diff_text})
            return diffs

        def _agent_model_name(backend: str, actual_model: Optional[str] = None) -> str:
            """Determine model name with priority: actual > config > profile > fallback."""
            if actual_model:
                return actual_model
            extra_cli = self.evo_config.agentic.extra_cli_config
            if extra_cli:
                model_override = (
                    extra_cli.get("model") if isinstance(extra_cli, dict) else None
                )
                if model_override:
                    return str(model_override)
            if self.evo_config.agentic.cli_profile:
                return self.evo_config.agentic.cli_profile
            return f"{backend}-default"

        selected_backend = self.evo_config.agentic.backend

        # Bandit model selection (same as legacy path at lines 1150-1153)
        bandit_model: Optional[str] = None
        if self.llm_selection is not None:
            llm_kwargs = self.llm.get_kwargs()
            bandit_model = llm_kwargs.get("model_name")
            if bandit_model:
                self.llm_selection.update_submitted(bandit_model)

        def failure_meta(
            message: str,
            *,
            session_log: Optional[List[str]] = None,
            commands: Optional[List[CommandResult]] = None,
            metrics: Optional[Dict[str, float]] = None,
            session_id: Optional[str] = None,
            changed_files: Optional[Dict[Path, str]] = None,
        ) -> tuple[Optional[str], dict, int]:
            api_cost = 0.0
            if metrics:
                api_cost = (
                    metrics.get("total_cost")
                    or metrics.get("estimated_total_cost")
                    or 0.0
                )
            serialized_changed = _serialize_changed_files(changed_files)
            meta_edit_data = {
                "patch_type": "agentic",
                "api_costs": api_cost,
                "num_applied": 0,
                "patch_name": None,
                "patch_description": None,
                "error_attempt": message,
                "novelty_attempt": novelty_attempt,
                "resample_attempt": resample_attempt,
                "patch_attempt": 1,
                "agent_session_path": str(session_root) if session_root else None,
                "agent_session_log": session_log or [],
                "agent_commands": [asdict(cmd) for cmd in commands or []],
                "agent_metrics": metrics or {},
                "agent_changed_files": serialized_changed,
                "agent_code_diffs": _build_code_diffs(changed_files),
                "agent_primary_file": str(primary_filename),
                # Use bandit-selected model for bandit learning, fall back to backend default
                "model_name": bandit_model or _agent_model_name(selected_backend),
                "agent_backend": selected_backend,
                "agent_session_id": session_id,
                "agent_resumed_from_parent": resumed_from_parent,
            }
            return None, meta_edit_data, 0

        # Ensure backend is available
        try:
            if selected_backend == "shinka":
                ensure_shinka_available()
            else:
                ensure_codex_available(self.evo_config.agentic.cli_path)
        except (CodexUnavailableError, ShinkaUnavailableError) as exc:
            return failure_meta(str(exc))

        # Create scratch directory
        session_uuid = str(uuid.uuid4())
        if self.evo_config.agentic.scratch_dir_base:
            scratch_base = Path(self.evo_config.agentic.scratch_dir_base)
            scratch_base.mkdir(parents=True, exist_ok=True)
            session_root = scratch_base / session_uuid
        else:
            session_root = Path(self.results_dir) / "agent_sessions" / session_uuid

        session_root.mkdir(parents=True, exist_ok=True)

        # Write session metadata
        session_meta = {
            "parent_id": parent_program.id,
            "generation": generation,
            "patch_type": patch_type,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "start_time": time.time(),
            "results_dir": str(self.results_dir),
        }
        try:
            with open(session_root / "session_meta.json", "w") as f:
                json.dump(session_meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write session_meta.json: {e}")

        # Build context for agent
        helper_files = [p for p in base_files.keys() if p != primary_filename]
        system_prompt = patch_sys.strip()
        if helper_files:
            helper_listing = "\n".join(
                f"- {path.as_posix()}" for path in sorted(helper_files)
            )
            system_prompt += (
                "\n\n# Workspace Files\n"
                "The following helper files were copied from the parent program:\n"
                f"{helper_listing}"
            )

        context = AgentContext(
            user_prompt=patch_msg.strip(),
            system_prompt=system_prompt,
            language=self.evo_config.language,
            base_files=base_files,
            primary_file=primary_filename,
            metadata={
                "generation": generation,
                "novelty_attempt": novelty_attempt,
                "resample_attempt": resample_attempt,
                "patch_type": patch_type,
                "results_dir": str(self.results_dir),
            },
            resume_session_id=resume_session_id,
        )

        # Create config with bandit-selected model if available
        agentic_config = self.evo_config.agentic
        if bandit_model:
            # Create modified extra_cli_config with bandit model
            modified_extra_cli = dict(agentic_config.extra_cli_config)
            modified_extra_cli["model"] = bandit_model
            # Create new config with modified extra_cli_config
            # Handle both dataclass instances and DictConfig from Hydra CLI overrides
            if is_dataclass(agentic_config) and not isinstance(agentic_config, type):
                agentic_config = replace(
                    agentic_config, extra_cli_config=modified_extra_cli
                )
            else:
                # DictConfig from Hydra - create a mutable copy preserving attribute access
                from omegaconf import OmegaConf
                agentic_config = OmegaConf.create(OmegaConf.to_container(agentic_config, resolve=True))
                agentic_config.extra_cli_config = modified_extra_cli

        editor = AgenticEditor(
            scratch_dir=session_root,
            config=agentic_config,
            runner=run_shinka_task if selected_backend == "shinka" else run_codex_task,
        )

        try:
            agent_result = editor.run_session(context)
            logger.info(f"_run_agentic_patch: session completed, changed_files={list(agent_result.changed_files.keys())}")
        except (CodexExecutionError, ShinkaExecutionError) as exc:
            logger.info(f"_run_agentic_patch: session FAILED with {type(exc).__name__}: {exc}")
            return failure_meta(str(exc))

        # Create generation directory
        generation_dir = Path(self.results_dir) / f"{FOLDER_PREFIX}_{generation}"
        if generation_dir.exists():
            shutil.rmtree(generation_dir)
        generation_dir.mkdir(parents=True, exist_ok=True)
        self._hydrate_generation_directory(parent_program, generation_dir)

        # Get primary file content from agent result
        primary_content = agent_result.changed_files.get(
            context.primary_file, base_files[context.primary_file]
        )
        original_for_patch = base_files[context.primary_file]

        # Write ALL changed files directly to generation directory
        # (Agentic mode: no EVOLVE-BLOCK markers needed)
        logger.info(
            f"Agentic edit: writing {len(agent_result.changed_files)} changed files "
            f"to {generation_dir}"
        )
        for rel_path, content in agent_result.changed_files.items():
            target = generation_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            logger.info(f"  Wrote: {rel_path} ({len(content)} bytes)")

        # If agent didn't change the primary file, ensure it exists
        primary_target = generation_dir / context.primary_file
        if not primary_target.exists():
            primary_target.write_text(primary_content, encoding="utf-8")
            logger.info(f"  Wrote primary (unchanged): {context.primary_file}")

        # In agentic mode, we consider the patch applied if any files were written
        # (either changed files or the primary file was created)
        num_applied = 1 if agent_result.changed_files or primary_target.exists() else 0
        logger.info(f"Agentic edit: num_applied={num_applied}")

        # Build code diff for display
        original_lines = original_for_patch.splitlines(keepends=True)
        new_lines = primary_content.splitlines(keepends=True)
        code_diff = "".join(
            difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile="a/main." + self.lang_ext,
                tofile="b/main." + self.lang_ext,
            )
        )

        api_cost = 0.0
        if agent_result.metrics:
            api_cost = (
                agent_result.metrics.get("total_cost")
                or agent_result.metrics.get("estimated_total_cost")
                or 0.0
            )

        serialized_changed = _serialize_changed_files(agent_result.changed_files)
        actual_model = agent_result.model

        meta_edit_data = {
            "patch_type": "agentic",
            "api_costs": api_cost,
            "num_applied": num_applied,
            "patch_name": None,
            "patch_description": None,
            "error_attempt": None,
            "novelty_attempt": novelty_attempt,
            "resample_attempt": resample_attempt,
            "patch_attempt": 1,
            "agent_session_path": str(session_root),
            "agent_session_log": agent_result.session_log,
            "agent_commands": [asdict(cmd) for cmd in agent_result.commands_run],
            "agent_metrics": agent_result.metrics,
            "agent_changed_files": serialized_changed,
            "agent_code_diffs": _build_code_diffs(agent_result.changed_files),
            "agent_primary_file": str(primary_filename),
            # Use bandit-selected model for bandit learning, fall back to actual model
            "model_name": bandit_model
            or _agent_model_name(selected_backend, actual_model),
            "agent_backend": selected_backend,
            "agent_session_id": agent_result.session_id,
            "agent_resumed_from_parent": resumed_from_parent,
            "bandit_selected_model": bandit_model,
        }

        # Note: Bandit update happens in _process_completed_job() after evaluation,
        # using the model_name stored in metadata (same pattern as legacy path)

        return code_diff, meta_edit_data, num_applied

    def _resolve_evaluator_mode(self) -> str:
        """Resolve evaluator mode after considering agentic defaults."""
        mode = (self.evo_config.evaluator.mode or "auto").lower()
        if mode == "legacy":
            return "legacy"
        if mode == "agentic":
            return "agentic"
        if mode == "auto":
            return "agentic" if self.evo_config.agentic_mode else "legacy"
        raise ValueError(f"Unknown evaluator mode: {self.evo_config.evaluator.mode}")

    def _run_legacy_evaluation_sync(
        self, exec_fname: str, results_dir: str
    ) -> tuple[dict, float]:
        """Run legacy evaluation synchronously via subprocess.

        This is thread-safe and can be called from worker threads.
        Returns (results_dict, runtime_seconds) in the expected format:
        {"correct": {"correct": bool}, "metrics": {...}}
        """
        import subprocess

        eval_command = self._build_eval_command(exec_fname, results_dir)
        if not eval_command:
            logger.warning("No eval command configured for legacy evaluation")
            return {"correct": {"correct": False}, "metrics": {"combined_score": 0.0}}, 0.0

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = Path(results_dir) / "metrics.json"
        correct_path = Path(results_dir) / "correct.json"

        start_time = time.time()
        try:
            result = subprocess.run(
                eval_command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode != 0:
                logger.warning(
                    f"Legacy eval failed (exit {result.returncode}): {result.stderr[:500]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning("Legacy eval timed out after 5 minutes")
        except Exception as e:
            logger.warning(f"Legacy eval error: {e}")

        rtime = time.time() - start_time

        # Parse correct.json
        correct_val = False
        if correct_path.exists():
            try:
                content = correct_path.read_text(encoding="utf-8").strip()
                if content:
                    correct_data = json.loads(content)
                    correct_val = correct_data.get("correct", False)
            except Exception as e:
                logger.warning(f"Failed to parse correct.json: {e}")

        # Parse metrics.json
        metrics_val = {"combined_score": 0.0}
        if metrics_path.exists():
            try:
                content = metrics_path.read_text(encoding="utf-8").strip()
                if content:
                    metrics_val = json.loads(content)
            except Exception as e:
                logger.warning(f"Failed to parse metrics.json: {e}")

        # Return in expected format
        return {
            "correct": {"correct": correct_val},
            "metrics": metrics_val,
        }, rtime

    def _build_eval_command(self, exec_fname: str, results_dir: str) -> List[str]:
        """Build the evaluation command from job config."""
        eval_program = self.job_config.eval_program_path
        if not eval_program:
            return []
        # Build command: python3 <eval_program> --program_path <exec_fname> --results_dir <results_dir>
        # Or use the raw eval_command if set in job_config
        if hasattr(self.job_config, "eval_command") and self.job_config.eval_command:
            return self.job_config.eval_command.split()
        # Resolve to absolute path if relative (important for agentic eval which changes workdir)
        eval_program_path = Path(eval_program)
        if not eval_program_path.is_absolute():
            eval_program_path = (Path.cwd() / eval_program_path).resolve()
        # Resolve exec_fname and results_dir to absolute paths too
        exec_fname_path = Path(exec_fname)
        if not exec_fname_path.is_absolute():
            exec_fname_path = (Path.cwd() / exec_fname_path).resolve()
        results_dir_path = Path(results_dir)
        if not results_dir_path.is_absolute():
            results_dir_path = (Path.cwd() / results_dir_path).resolve()
        return [
            "python3", str(eval_program_path),
            "--program_path", str(exec_fname_path),
            "--results_dir", str(results_dir_path),
        ]

    def _run_agentic_evaluation(
        self,
        *,
        exec_fname: str,
        results_dir: str,
        generation_dir: Path,
        generation: int,
        parent_id: Optional[str] = None,
        patch_type: Optional[str] = None,
    ) -> tuple[Dict[str, Any], float]:
        """Run evaluation using the agentic evaluator (LLM-powered)."""
        if self.agentic_evaluator is None:
            raise RuntimeError("Agentic evaluator not initialized")

        repo_root = generation_dir.resolve()
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = Path(results_dir) / "metrics.json"
        eval_sessions_root = self.agentic_eval_sessions_dir
        eval_sessions_root.mkdir(parents=True, exist_ok=True)
        eval_command = self._build_eval_command(exec_fname, results_dir)
        run_root = Path(self.results_dir).resolve()

        def _rel_to_run_path(raw: Union[str, Path]) -> str:
            try:
                resolved = Path(raw).resolve()
                return str(resolved.relative_to(run_root))
            except Exception:
                return str(raw)

        # --- Evaluation integrity snapshot ---
        # Policy: evaluator may create new artifacts but must not modify pre-existing files
        results_path = Path(results_dir).resolve()
        try:
            results_rel = results_path.relative_to(repo_root)
        except Exception:
            results_rel = None

        ignored_dir_parts = {"__pycache__", ".pytest_cache", ".hydra", ".git", ".venv"}
        ignored_suffixes = {".pyc", ".pyo"}

        def _should_ignore_integrity_path(rel_path: Path) -> bool:
            if not rel_path.parts:
                return True
            if (
                results_rel is not None
                and rel_path.parts[: len(results_rel.parts)] == results_rel.parts
            ):
                return True
            if rel_path.suffix in ignored_suffixes:
                return True
            if any(part in ignored_dir_parts for part in rel_path.parts):
                return True
            return False

        def _snapshot_integrity(root: Path) -> Dict[str, str]:
            snapshot: Dict[str, str] = {}
            for abs_path in root.rglob("*"):
                if not abs_path.is_file():
                    continue
                rel = abs_path.relative_to(root)
                if _should_ignore_integrity_path(rel):
                    continue
                try:
                    digest = hashlib.sha256(abs_path.read_bytes()).hexdigest()
                except Exception:
                    continue
                snapshot[rel.as_posix()] = digest
            return snapshot

        integrity_pre = _snapshot_integrity(repo_root)

        # Convert paths to be relative to repo_root for the evaluator
        # The agent runs with workdir=repo_root, so paths need to be relative
        try:
            rel_program_path = Path(exec_fname).resolve().relative_to(repo_root)
        except ValueError:
            rel_program_path = Path(exec_fname).name  # Fallback to just filename

        try:
            rel_results_path = Path(results_dir).resolve().relative_to(repo_root)
        except ValueError:
            rel_results_path = Path("results")  # Fallback

        try:
            rel_metrics_path = metrics_path.resolve().relative_to(repo_root)
        except ValueError:
            rel_metrics_path = Path("results/metrics.json")  # Fallback

        start = time.time()
        result = None
        try:
            result = self.agentic_evaluator.evaluate(
                repo_root=repo_root,
                eval_command=eval_command,
                program_path=rel_program_path,
                results_path=rel_results_path,
                metrics_path=rel_metrics_path,
                eval_sessions_root=eval_sessions_root,
                task_name=self.job_config.eval_program_path or "agentic_evaluator",
                results_dir=str(self.results_dir),
                eval_prompt=getattr(
                    self.evo_config.evaluator.agentic, "eval_prompt", None
                ),
                max_score=self.evo_config.max_score,
            )
        except (CodexExecutionError, ShinkaExecutionError) as exc:
            # If metrics missing or empty, emit fallback so run can proceed
            metrics_content = ""
            if metrics_path.exists():
                metrics_content = metrics_path.read_text(encoding="utf-8").strip()
            if not metrics_content:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                fallback = {
                    "combined_score": 0.0,
                    "correct": False,
                    "details": f"Agentic evaluator failed: {exc}",
                }
                metrics_path.write_text(json.dumps(fallback), encoding="utf-8")
                metrics_content = json.dumps(fallback)
            try:
                metrics = json.loads(metrics_content)
            except json.JSONDecodeError:
                metrics = {"combined_score": 0.0, "error": "Invalid metrics JSON"}
            # If metrics exist and have a correct flag, use it; otherwise default to False
            correct_from_metrics = bool(metrics.get("correct", False))
            result = AgenticEvaluatorResult(
                metrics=metrics,
                correct=correct_from_metrics,
                error_message=str(exc),
                stdout_log="",
                stderr_log="",
                session_log=[],
                commands_run=[],
                session_log_path=metrics_path.parent / "session_log.missing",
                session_events=[],
                session_id=None,
                session_dir=metrics_path.parent,
                elapsed_seconds=time.time() - start,
            )
        rtime = time.time() - start

        integrity_post = _snapshot_integrity(repo_root)
        modified_existing = sorted(
            p
            for p in integrity_pre.keys()
            if p in integrity_post and integrity_pre[p] != integrity_post[p]
        )
        deleted_existing = sorted(
            p for p in integrity_pre.keys() if p not in integrity_post
        )
        new_files_created = sorted(
            p for p in integrity_post.keys() if p not in integrity_pre
        )

        integrity_status = "clean"
        if modified_existing or deleted_existing:
            integrity_status = "violation"
        elif new_files_created:
            integrity_status = "artifacts_only"

        integrity_meta = {
            "policy": "no_modify_preexisting_files",
            "status": integrity_status,
            "modified_existing_count": len(modified_existing),
            "deleted_existing_count": len(deleted_existing),
            "new_files_created_count": len(new_files_created),
        }

        # If integrity violated, force incorrect
        effective_correct = result.correct
        effective_error = result.error_message
        effective_metrics = dict(result.metrics or {})

        if integrity_status == "violation":
            effective_correct = False
            sample_paths = (modified_existing + deleted_existing)[:10]
            integrity_msg = f"Evaluation integrity violation: evaluator modified files ({', '.join(sample_paths)})"
            effective_error = (
                f"{effective_error} | {integrity_msg}"
                if effective_error
                else integrity_msg
            )

        events_preview = result.session_events[-AGENTIC_EVAL_PREVIEW_LIMIT:]
        agentic_meta = {
            "session_dir": _rel_to_run_path(result.session_dir),
            "session_log_path": _rel_to_run_path(result.session_log_path),
            "session_id": result.session_id,
            "commands_run": [asdict(cmd) for cmd in result.commands_run],
            "generation": generation,
            "elapsed_seconds": result.elapsed_seconds,
            "status": "error" if effective_error else "success",
            "correct": effective_correct,
            "metrics_path": _rel_to_run_path(metrics_path),
            "metrics": effective_metrics,
            "error_message": effective_error,
            "stdout_log": result.stdout_log,
            "stderr_log": result.stderr_log,
            "events_preview": events_preview,
            "system_prompt": result.system_prompt,
            "user_prompt": result.user_prompt,
            "integrity": integrity_meta,
        }

        results_payload = {
            "metrics": effective_metrics,
            "correct": {
                "correct": effective_correct,
                "error": effective_error,
            },
            "stdout_log": result.stdout_log,
            "stderr_log": result.stderr_log,
            "agentic_eval": agentic_meta,
        }

        return results_payload, rtime
