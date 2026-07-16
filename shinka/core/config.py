from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from shinka.llm import BanditBase
from shinka.defaults import (
    DEFAULT_TASK_SYS_MSG,
    default_llm_dynamic_selection_kwargs,
    default_llm_kwargs,
    default_llm_models,
    default_patch_type_probs,
    default_patch_types,
    default_prompt_patch_type_probs,
    default_prompt_patch_types,
)

FOLDER_PREFIX = "gen"


@dataclass
class EvolutionConfig:
    # Repo-backed evolution settings.
    seed_repo_path: Optional[str] = None
    worktree_root: Optional[str] = None
    base_ref: str = "HEAD"
    mutable_paths: List[str] = field(default_factory=list)
    immutable_paths: List[str] = field(default_factory=list)
    agent_hidden_paths: List[str] = field(default_factory=list)
    ignore_paths: List[str] = field(default_factory=lambda: [".git", ".shinka"])
    allow_deletions: bool = True
    allow_lockfile_changes: bool = True
    allow_binary_files: bool = True
    max_file_bytes: Optional[int] = None
    summary_filename: str = ".shinka/individual.md"
    summary_max_chars: int = 12000

    # Headless proposal execution. Coding agents may legitimately inspect,
    # test, and optimize for a long time before producing a candidate.
    headless_proposal_timeout_seconds: float = 7200.0
    headless_cleanup_grace_seconds: float = 60.0
    headless_output_mode: str = "json"
    headless_model_timeouts: Dict[str, float] = field(default_factory=dict)

    # Provider controls are independent from the model-quality bandit.
    route_failure_threshold: int = 3
    route_cooldown_seconds: float = 900.0
    llm_rate_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    llm_daily_quotas: Dict[str, int] = field(default_factory=dict)

    # Headless coding-agent model used for repo-backed evolution.
    agent_model: Optional[str] = None

    task_sys_msg: Optional[str] = DEFAULT_TASK_SYS_MSG
    patch_types: List[str] = field(default_factory=default_patch_types)
    patch_type_probs: List[float] = field(default_factory=default_patch_type_probs)
    num_generations: int = 50
    generation_target_mode: str = "evaluated_candidates"
    max_patch_resamples: int = 3
    max_patch_attempts: int = 1
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=default_llm_models)
    llm_dynamic_selection: Optional[Union[str, Any]] = "ucb"
    llm_dynamic_selection_kwargs: dict = field(
        default_factory=default_llm_dynamic_selection_kwargs
    )
    llm_kwargs: dict = field(default_factory=default_llm_kwargs)
    meta_rec_interval: Optional[int] = 10
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    sample_single_meta_rec: bool = True
    embedding_model: Optional[str] = "text-embedding-3-small"
    results_dir: Optional[str] = None

    # Optional W&B logging is additive to the existing database/WebUI logging.
    enable_wandb_logging: bool = False
    wandb_project: Optional[str] = "shinka-evolve"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_dir: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_resume: str = "allow"
    wandb_config: Dict[str, Any] = field(default_factory=dict)

    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 0.99
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False
    max_api_costs: Optional[float] = None
    inspiration_sort_order: str = "ascending"
    enable_controlled_oversubscription: bool = False
    proposal_target_mode: str = "adaptive"
    proposal_target_min_samples: int = 5
    proposal_target_ratio_cap: float = 2.0
    proposal_buffer_max: int = 2
    proposal_target_hard_cap: Optional[int] = None
    proposal_target_ewma_alpha: float = 0.3

    # Meta-prompt evolution settings.
    evolve_prompts: bool = False
    prompt_patch_types: List[str] = field(default_factory=default_prompt_patch_types)
    prompt_patch_type_probs: List[float] = field(
        default_factory=default_prompt_patch_type_probs
    )
    prompt_evolution_interval: Optional[int] = None
    prompt_archive_size: int = 10
    prompt_llm_models: Optional[List[str]] = None
    prompt_llm_kwargs: dict = field(default_factory=lambda: {})
    prompt_ucb_exploration_constant: float = 1.0
    prompt_epsilon: float = 0.1
    prompt_evo_top_k_programs: int = 3
    prompt_percentile_recompute_interval: int = 20
