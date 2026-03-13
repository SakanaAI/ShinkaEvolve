import hydra
from hydra import compose, initialize_config_dir
import ast
import pathlib
from pathlib import Path
import inspect
from functools import wraps
from typing import Optional, Union
import os
import sys
from omegaconf import DictConfig, OmegaConf

from shinka.configs import config_root


def load_hydra_config(
    output_dir: str, max_parent_depth: int = 2
) -> Optional[DictConfig]:
    """Check for .hydra in this directory or its parents and get the configs."""
    hydra_dir = os.path.join(output_dir, ".hydra")
    if os.path.isdir(hydra_dir):
        config_file = os.path.join(hydra_dir, "config.yaml")
        if os.path.isfile(config_file):
            return OmegaConf.load(config_file)
        return None
    # stop if no remaining depth
    if max_parent_depth <= 0:
        return None
    parent = os.path.dirname(output_dir)
    if not parent or parent == output_dir:
        return None
    return load_hydra_config(parent, max_parent_depth - 1)


def build_cfgs_from_python(*launcher_args, **launcher_kwargs):
    with config_root() as cfgs_root:
        global_list = [p.name for p in cfgs_root.iterdir() if p.is_dir()]

        def tag_global(overrides, keys):
            out = []
            for s in overrides:
                if "@_global_=" in s:
                    out.append(s)
                    continue
                for k in keys:
                    p = f"{k}="
                    if s.startswith(p):
                        s = s.replace(p, f"{k}@_global_=", 1)
                        break
                out.append(s)
            return out

        hydra_overrides = list(launcher_args)
        hydra_overrides += [f"{k}={v}" for k, v in launcher_kwargs.items()]
        hydra_overrides = tag_global(hydra_overrides, global_list)

        with initialize_config_dir(
            version_base=None, config_dir=str(cfgs_root), job_name="shinka"
        ):
            cfg = compose(config_name="config", overrides=hydra_overrides)

    run_dir = pathlib.Path(cfg.output_dir)
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, hydra_dir / "config.yaml")
    hydra_node = cfg.get("hydra", {})
    OmegaConf.save(OmegaConf.create(hydra_node), hydra_dir / "hydra.yaml")
    OmegaConf.save(
        OmegaConf.create(list(hydra_overrides)), hydra_dir / "overrides.yaml"
    )

    job_cfg = hydra.utils.instantiate(cfg.job_config)
    db_cfg = hydra.utils.instantiate(cfg.db_config)
    evo_cfg = hydra.utils.instantiate(cfg.evo_config)
    return job_cfg, db_cfg, evo_cfg, cfg


def add_evolve_markers(
    initial_file_path: Union[str, os.PathLike],
    save_dir: Union[str, os.PathLike],
    insert_start: Union[int, str],
    insert_end: Union[int, str],
    *,
    zero_indexed: bool = True,
) -> str:
    """
    Copy *initial_file_path* to *save_dir*/initial.py, inserting EVOLVE markers.

    • *insert_start* / *insert_end* may be **int** (line index) or **str** (symbol
      name).  If a str is supplied, it is passed through get_line() with:
          start=True   for insert_start
          start=False  for insert_end
      -> indices are therefore always 0-based in that case.

    • If both arguments are ints you can still opt-in to 1-based indexing by
      setting `zero_indexed=False` (legacy behaviour).
    """
    if isinstance(insert_start, str):
        idx_start = get_line(
            fn_or_class_name=insert_start, file_path=initial_file_path, start=True
        )
        zero_indexed = True
    else:
        idx_start = insert_start

    if isinstance(insert_end, str):
        idx_end = get_line(
            fn_or_class_name=insert_end, file_path=initial_file_path, start=False
        )
        zero_indexed = True
    else:
        idx_end = insert_end

    with open(initial_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = len(lines)

    if zero_indexed:
        if not (0 <= idx_start <= n and 0 <= idx_end <= n):
            raise IndexError("insert_* out of range (0-based).")
        if idx_end < idx_start:
            raise ValueError("insert_end must be ≥ insert_start.")
    else:  # treat supplied ints as 1-based
        if not (1 <= idx_start <= n and 1 <= idx_end <= n):
            raise IndexError("insert_* out of range (1-based).")
        if idx_end < idx_start:
            raise ValueError("insert_end must be ≥ insert_start.")
        idx_start -= 1
        idx_end -= 0

    def _put_marker(index: int, marker: str) -> None:
        if index < len(lines) and lines[index].strip() == "":
            lines[index] = marker
        else:
            lines.insert(index, marker)

    _put_marker(idx_end, "# EVOLVE-BLOCK-END\n")
    _put_marker(idx_start, "# EVOLVE-BLOCK-START\n")

    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "initial.py"
    out_file.write_text("".join(lines), encoding="utf-8")

    return str(out_file.resolve())


def get_line(
    fn_or_class_name: Optional[str],
    file_path: Union[str, os.PathLike],
    start: bool,
) -> int:
    """
    Locate a line boundary in *file_path*.

    Parameters
    ----------
    fn_or_class_name : str | None
        • If **None** – work on the whole file.
        • Otherwise – the exact name of a function **or** class whose
          boundaries you want.
    file_path : str | PathLike
        Path to the Python source file.
    start : bool
        • When *fn_or_class_name* is **None**
            – ``True`` ⇒ return *0* (first line)
            – ``False`` ⇒ return *len(lines)* (one-past-last line)

        • When *fn_or_class_name* **is given**
            – ``True`` ⇒ 0-based line **right before** the definition
            – ``False`` ⇒ 0-based line **right after** the definition
              (i.e. one-past the last line of its body)

    Returns
    -------
    int
        A **0-based** line index suitable for later insertions.

    Raises
    ------
    ValueError
        If *fn_or_class_name* is supplied but no matching function/class
        definition is found.
    """
    path = Path(file_path).expanduser().resolve()
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    if fn_or_class_name is None:
        return 0 if start else len(lines)

    tree = ast.parse(source, filename=str(path))
    target_node: Optional[ast.AST] = None

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == fn_or_class_name
        ):
            target_node = node
            break

    if target_node is None:
        raise ValueError(
            f"No function or class named '{fn_or_class_name}' found in {path}"
        )

    # AST line numbers are 1-based; convert to 0-based indices.
    before_idx = max(target_node.lineno - 2, 0)
    after_idx = target_node.end_lineno

    return before_idx if start else after_idx


def chdir_to_function_dir(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_cwd = os.getcwd()
        src_file = inspect.getsourcefile(func)
        here = Path(src_file).resolve().parent

        sys.path.insert(0, str(here))
        os.chdir(here)
        try:
            return func(*args, **kwargs)
        finally:
            os.chdir(old_cwd)
            try:
                sys.path.remove(str(here))
            except ValueError:
                pass

    return wrapper


def wrap_object(object_config) -> Optional[DictConfig]:
    """Allows wrapping a callable function/class without automatically instantiating it."""

    def instantiator():
        return hydra.utils.instantiate(object_config)

    return instantiator
