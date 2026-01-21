# from .load_df import load_programs_to_df, get_path_to_best_node, store_best_path
# from .general import parse_time_to_seconds, load_results
# from .utils_hydra import build_cfgs_from_python, add_evolve_markers, chdir_to_function_dir, wrap_object, load_hydra_config
# from .custom import truncate_log_blocks

# __all__ = [
#     "load_programs_to_df",
#     "get_path_to_best_node",
#     "store_best_path",
#     "parse_time_to_seconds",
#     "load_results",
#     "build_cfgs_from_python",
#     "add_evolve_markers",
#     "chdir_to_function_dir",
#     "wrap_object",
#     "load_hydra_config",
#     "truncate_log_blocks"
# ]


# Lazy access so importing shinka.utils doesn't eagerly import heavy submodules.
# Only import what is actually accessed (PEP 562).

# Lazy, per-symbol import (PEP 562) so importing shinka.utils doesn't eagerly import submodules.
# Each exported name maps to the specific module + attribute that defines it.

from importlib import import_module

_EXPORTS = {
    # load_df.py
    "load_programs_to_df": (".load_df", "load_programs_to_df"),
    "get_path_to_best_node": (".load_df", "get_path_to_best_node"),
    "store_best_path": (".load_df", "store_best_path"),

    # general.py
    "parse_time_to_seconds": (".general", "parse_time_to_seconds"),
    "load_results": (".general", "load_results"),

    # utils_hydra.py (Hydra is imported lazily inside functions where needed)
    "build_cfgs_from_python": (".utils_hydra", "build_cfgs_from_python"),
    "add_evolve_markers": (".utils_hydra", "add_evolve_markers"),
    "chdir_to_function_dir": (".utils_hydra", "chdir_to_function_dir"),
    "wrap_object": (".utils_hydra", "wrap_object"),
    "load_hydra_config": (".utils_hydra", "load_hydra_config"),

    # custom.py
    "truncate_log_blocks": (".custom", "truncate_log_blocks"),
}

__all__ = list(_EXPORTS.keys())

def __getattr__(name: str):
    try:
        mod_path, attr = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(f"module 'shinka.utils' has no attribute {name!r}") from e
    module = import_module(mod_path, package=__name__)
    return getattr(module, attr)

def __dir__():
    return sorted(__all__)
