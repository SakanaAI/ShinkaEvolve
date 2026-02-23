#!/usr/bin/env python3
"""Agent-friendly async CLI launcher for Shinka tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin

from shinka.core import AsyncEvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

DEFAULT_TASK_SYS_MSG = (
    "You are an expert optimization and algorithm design assistant. "
    "Improve the program while preserving correctness and immutable regions."
)

SUPPORTED_INITIAL_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".jl": "julia",
    ".rs": "rust",
    ".swift": "swift",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cu": "cuda",
    ".json": "json",
}

INITIAL_EXTENSION_PRIORITY: list[str] = [
    ".py",
    ".jl",
    ".rs",
    ".cpp",
    ".cc",
    ".cxx",
    ".cu",
    ".swift",
    ".json",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "Run async Shinka evolution from a task directory.\n\n"
        "Task directory contract:\n"
        "  - evaluate.py\n"
        "  - initial.<ext> (e.g. initial.py, initial.jl)"
    )
    epilog = (
        "Override grammar:\n"
        "  --set <namespace>.<field>=<value>\n"
        "  namespaces: evo, db, job\n"
        "  list/dict values must be valid JSON\n"
        "  bool values: true,false,1,0,yes,no (case-insensitive)\n\n"
        "Examples:\n"
        "  Minimal:\n"
        "    shinka_run --task-dir examples/circle_packing "
        "--results_dir results/circle_small --num_generations 20\n\n"
        "  With overrides:\n"
        "    shinka_run --task-dir examples/circle_packing "
        "--results_dir results/circle_custom --num_generations 50 "
        "--set evo.max_parallel_jobs=6 --set db.num_islands=3 "
        "--set job.time=00:10:00 "
        "--set evo.llm_models='[\"gpt-5-mini\",\"gpt-5-nano\"]'\n\n"
        "Failure behavior:\n"
        "  - unknown namespace/field: non-zero exit\n"
        "  - invalid value type: non-zero exit\n"
        "  - missing evaluate.py or initial.<ext>: non-zero exit\n\n"
        "Precedence:\n"
        "  - --results_dir always sets evo.results_dir\n"
        "  - --num_generations always sets evo.num_generations"
    )
    parser = argparse.ArgumentParser(
        prog="shinka_run",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required_group = parser.add_argument_group("required arguments")
    required_group.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Directory containing evaluate.py and initial.<ext>.",
    )
    required_group.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help=(
            "Output directory for run artifacts/logs/databases. "
            "Authoritative: always sets evo.results_dir."
        ),
    )
    required_group.add_argument(
        "--num_generations",
        type=_positive_int,
        required=True,
        help=(
            "Number of generations to run. "
            "Authoritative: always sets evo.num_generations."
        ),
    )

    override_group = parser.add_argument_group("overrides")
    override_group.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NS.FIELD=VALUE",
        help=(
            "Repeatable namespaced override.\n"
            "Examples: --set evo.max_parallel_jobs=4 "
            "--set db.num_islands=2 "
            "--set job.extra_cmd_args='{\"seed\":42}'"
        ),
    )

    concurrency_group = parser.add_argument_group("concurrency")
    concurrency_group.add_argument(
        "--max-evaluation-jobs",
        type=_positive_int,
        default=None,
        help="Override AsyncEvolutionRunner max_evaluation_jobs.",
    )
    concurrency_group.add_argument(
        "--max-proposal-jobs",
        type=_positive_int,
        default=None,
        help="Override AsyncEvolutionRunner max_proposal_jobs.",
    )
    concurrency_group.add_argument(
        "--max-db-workers",
        type=_positive_int,
        default=None,
        help="Override AsyncEvolutionRunner max_db_workers.",
    )

    output_group = parser.add_argument_group("output/verbosity")
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose runner logging.",
    )
    output_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra async runner diagnostics.",
    )
    return parser


def _field_types() -> Dict[str, Dict[str, Any]]:
    return {
        "evo": {field.name: field.type for field in fields(EvolutionConfig)},
        "db": {field.name: field.type for field in fields(DatabaseConfig)},
        "job": {field.name: field.type for field in fields(LocalJobConfig)},
    }


def _coerce_bool(raw_value: str, key: str) -> bool:
    lowered = raw_value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid bool for {key}: {raw_value}. "
        "Use true/false/1/0/yes/no."
    )


def _coerce_scalar(raw_value: str, target_type: type, key: str) -> Any:
    if target_type is str:
        return raw_value
    if target_type is bool:
        return _coerce_bool(raw_value, key)
    if target_type is int:
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid int for {key}: {raw_value}") from exc
    if target_type is float:
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {key}: {raw_value}") from exc
    return raw_value


def _coerce_json_container(
    raw_value: str, expected_container: type, key: str
) -> Any:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON for {key}: {raw_value}. "
            f"Expected {expected_container.__name__} JSON."
        ) from exc

    if not isinstance(parsed, expected_container):
        raise ValueError(
            f"Invalid type for {key}: expected {expected_container.__name__}, "
            f"got {type(parsed).__name__}."
        )
    return parsed


def _coerce_override_value(raw_value: str, target_type: Any, key: str) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    if target_type is Any:
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value

    if target_type in {dict, list, str, bool, int, float}:
        if target_type is dict:
            return _coerce_json_container(raw_value, dict, key)
        if target_type is list:
            return _coerce_json_container(raw_value, list, key)
        return _coerce_scalar(raw_value, target_type, key)

    if origin is Union:
        if type(None) in args and raw_value.strip().lower() in {"none", "null"}:
            return None
        candidates = [candidate for candidate in args if candidate is not type(None)]
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return _coerce_override_value(raw_value, candidate, key)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise ValueError(str(last_error))
        return raw_value

    if origin in {dict, Dict}:
        return _coerce_json_container(raw_value, dict, key)

    if origin in {list, tuple}:
        return _coerce_json_container(raw_value, list, key)

    return _coerce_scalar(raw_value, str, key)


def _parse_override_token(token: str) -> tuple[str, str, str]:
    if "=" not in token:
        raise ValueError(f"Invalid override '{token}'. Expected NS.FIELD=VALUE.")
    key, raw_value = token.split("=", 1)
    if "." not in key:
        raise ValueError(
            f"Invalid override key '{key}'. Expected namespaced key NS.FIELD."
        )
    namespace, field_name = key.split(".", 1)
    if not field_name:
        raise ValueError(f"Missing field name in override '{token}'.")
    return namespace, field_name, raw_value


def _parse_overrides(
    tokens: list[str], allowed_field_types: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {"evo": {}, "db": {}, "job": {}}
    for token in tokens:
        namespace, field_name, raw_value = _parse_override_token(token)
        if namespace not in allowed_field_types:
            valid_namespaces = ", ".join(sorted(allowed_field_types.keys()))
            raise ValueError(
                f"Invalid namespace '{namespace}' in '{token}'. "
                f"Use one of: {valid_namespaces}."
            )
        field_types_for_ns = allowed_field_types[namespace]
        if field_name not in field_types_for_ns:
            valid_fields = ", ".join(sorted(field_types_for_ns.keys()))
            raise ValueError(
                f"Unknown field '{namespace}.{field_name}'. "
                f"Valid {namespace} fields: {valid_fields}"
            )
        target_type = field_types_for_ns[field_name]
        parsed[namespace][field_name] = _coerce_override_value(
            raw_value, target_type, f"{namespace}.{field_name}"
        )
    return parsed


def _detect_initial_program(task_dir: Path) -> Path:
    candidates = sorted(task_dir.glob("initial.*"))
    if not candidates:
        raise FileNotFoundError(
            f"No initial.<ext> found in task dir: {task_dir}"
        )
    sorted_candidates = sorted(
        candidates,
        key=lambda path: (
            INITIAL_EXTENSION_PRIORITY.index(path.suffix)
            if path.suffix in INITIAL_EXTENSION_PRIORITY
            else len(INITIAL_EXTENSION_PRIORITY),
            path.name,
        ),
    )
    return sorted_candidates[0]


def _infer_language(initial_program_path: Path) -> str:
    suffix = initial_program_path.suffix.lower()
    if suffix not in SUPPORTED_INITIAL_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_INITIAL_EXTENSIONS.keys()))
        raise ValueError(
            f"Unsupported initial program extension '{suffix}' for "
            f"{initial_program_path}. Supported: {supported}"
        )
    return SUPPORTED_INITIAL_EXTENSIONS[suffix]


def _build_default_evo_values(
    *,
    language: str,
    init_program_path: Path,
    results_dir: Path,
    num_generations: int,
) -> Dict[str, Any]:
    return {
        "task_sys_msg": DEFAULT_TASK_SYS_MSG,
        "patch_types": ["diff", "full", "cross"],
        "patch_type_probs": [0.6, 0.3, 0.1],
        "num_generations": num_generations,
        "max_parallel_jobs": 2,
        "max_patch_resamples": 3,
        "max_patch_attempts": 3,
        "job_type": "local",
        "language": language,
        "llm_models": ["gpt-5-mini"],
        "llm_kwargs": {
            "temperatures": [0.2, 0.6, 1.0],
            "reasoning_efforts": ["medium"],
            "max_tokens": 16384,
        },
        "embedding_model": "text-embedding-3-small",
        "code_embed_sim_threshold": 0.995,
        "init_program_path": str(init_program_path),
        "results_dir": str(results_dir),
        "max_novelty_attempts": 3,
    }


def _build_default_db_values() -> Dict[str, Any]:
    return {}


def _build_default_job_values(evaluate_path: Path) -> Dict[str, Any]:
    return {"eval_program_path": str(evaluate_path)}


def _validate_task_dir(task_dir: Path) -> tuple[Path, Path]:
    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir does not exist: {task_dir}")
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task dir is not a directory: {task_dir}")
    evaluate_path = task_dir / "evaluate.py"
    if not evaluate_path.exists():
        raise FileNotFoundError(f"Missing evaluate.py in task dir: {task_dir}")
    initial_path = _detect_initial_program(task_dir)
    return evaluate_path, initial_path


def _build_runner(
    *,
    args: argparse.Namespace,
    evo_config: EvolutionConfig,
    db_config: DatabaseConfig,
    job_config: LocalJobConfig,
    init_program_str: str,
    evaluate_str: str,
) -> AsyncEvolutionRunner:
    runner_kwargs: Dict[str, Any] = {
        "evo_config": evo_config,
        "job_config": job_config,
        "db_config": db_config,
        "verbose": args.verbose,
        "debug": args.debug,
        "init_program_str": init_program_str,
        "evaluate_str": evaluate_str,
    }
    if args.max_evaluation_jobs is not None:
        runner_kwargs["max_evaluation_jobs"] = args.max_evaluation_jobs
    if args.max_proposal_jobs is not None:
        runner_kwargs["max_proposal_jobs"] = args.max_proposal_jobs
    if args.max_db_workers is not None:
        runner_kwargs["max_db_workers"] = args.max_db_workers
    return AsyncEvolutionRunner(**runner_kwargs)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    task_dir = args.task_dir.resolve()
    results_dir = args.results_dir.resolve()

    try:
        evaluate_path, initial_path = _validate_task_dir(task_dir)
        language = _infer_language(initial_path)
        allowed_types = _field_types()
        parsed_overrides = _parse_overrides(args.overrides, allowed_types)

        evo_values = _build_default_evo_values(
            language=language,
            init_program_path=initial_path,
            results_dir=results_dir,
            num_generations=args.num_generations,
        )
        evo_values.update(parsed_overrides["evo"])
        evo_values["results_dir"] = str(results_dir)
        evo_values["num_generations"] = args.num_generations

        db_values = _build_default_db_values()
        db_values.update(parsed_overrides["db"])

        job_values = _build_default_job_values(evaluate_path)
        job_values.update(parsed_overrides["job"])

        evo_config = EvolutionConfig(**evo_values)
        db_config = DatabaseConfig(**db_values)
        job_config = LocalJobConfig(**job_values)

        init_program_str = initial_path.read_text(encoding="utf-8")
        evaluate_str = evaluate_path.read_text(encoding="utf-8")

        runner = _build_runner(
            args=args,
            evo_config=evo_config,
            db_config=db_config,
            job_config=job_config,
            init_program_str=init_program_str,
            evaluate_str=evaluate_str,
        )
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))

    asyncio.run(runner.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
