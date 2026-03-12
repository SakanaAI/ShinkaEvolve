from __future__ import annotations

import sys

from shinka import launch_hydra


def preprocess_args(argv: list[str]) -> list[str]:
    processed_args: list[str] = []
    for arg in argv:
        if "=" not in arg:
            processed_args.append(arg)
            continue

        key, value = arg.split("=", 1)
        if key in {"database", "evolution", "task", "cluster", "variant"} and not value.startswith("@"):
            processed_args.append(f"{key}@_global_={value}")
            continue

        processed_args.append(arg)

    return processed_args


def main(argv: list[str] | None = None) -> None:
    args = preprocess_args(list(sys.argv[1:] if argv is None else argv))
    sys.argv = [sys.argv[0], *args]
    launch_hydra.main()
