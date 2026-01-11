#!/usr/bin/env python3
import os
import sys
import argparse
import importlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import omegaconf
import hydra
from shinka.utils import load_hydra_config


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


@contextmanager
def chdir_to_target(cfg):
    module_path, _ = cfg._target_.rsplit(".", 1)
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise ModuleNotFoundError(f"Cannot find module {module_path}")
    source_dir = Path(spec.origin).resolve().parent
    main_cwd = os.getcwd()
    sys.path.insert(0, str(main_cwd))
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    try:
        yield
    finally:
        os.chdir(main_cwd)
        for p in (str(source_dir), str(main_cwd), str(repo_root)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shinka hydra-based evaluation launcher"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program to evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results and logs",
    )

    parser.add_argument(
        "--prover_model",
        type=str,
        default="gpt-5-nano",  # or an actual prover like "deepseek-ai/DeepSeek-Prover-V2-7B" (requires local LLM support)
        help="LLM agent used to construct LEAN proofs based on the initial header and formalization.",
    )

    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    cfg = load_hydra_config(args.results_dir, max_parent_depth=2)
    cfg.evaluate_function.program_path = os.path.abspath(args.program_path)
    cfg.evaluate_function.results_dir = os.path.abspath(args.results_dir)
    cfg.evaluate_function.prover_model = args.prover_model
    print(os.getcwd())
    print("Launching evaluation of function:")
    print(omegaconf.OmegaConf.to_yaml(cfg.evaluate_function))

    # import & run under the target directory
    with chdir_to_target(cfg.evaluate_function):
        hydra.utils.instantiate(cfg.evaluate_function)
