from .load_df import (
    load_programs_to_df,
    get_path_to_best_node,
    store_best_path,
    load_prompts_to_df,
)
from .general import (
    parse_time_to_seconds,
    load_results,
    load_configs_from_yaml,
    truncate_log_tail,
)
from .utils_hydra import (
    build_cfgs_from_python,
    add_evolve_markers,
    chdir_to_function_dir,
    wrap_object,
    load_hydra_config,
)
from .languages import get_language_extension


def __getattr__(name):
    if name in {
        "load_programs_to_df",
        "get_path_to_best_node",
        "store_best_path",
        "load_prompts_to_df",
    }:
        from . import load_df

        return getattr(load_df, name)
    if name in {
        "build_cfgs_from_python",
        "add_evolve_markers",
        "chdir_to_function_dir",
        "wrap_object",
        "load_hydra_config",
    }:
        from . import utils_hydra

        return getattr(utils_hydra, name)
    raise AttributeError(name)

__all__ = [
    "load_programs_to_df",
    "get_path_to_best_node",
    "store_best_path",
    "parse_time_to_seconds",
    "load_results",
    "build_cfgs_from_python",
    "add_evolve_markers",
    "chdir_to_function_dir",
    "wrap_object",
    "load_hydra_config",
    "load_configs_from_yaml",
    "truncate_log_tail",
    "get_language_extension",
    "load_prompts_to_df",
]
