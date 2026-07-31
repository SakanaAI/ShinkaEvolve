"""Restore the evaluator environment after the results lease is released."""

import json
import os
import sys


def main(environment_fd: int, command: list[str]) -> int:
    if not command:
        return 127
    try:
        with os.fdopen(environment_fd, "r", encoding="utf-8") as environment_file:
            environment = json.load(environment_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 127
    if not isinstance(environment, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in environment.items()
    ):
        return 127
    os.execve(command[0], command, environment)
    return 127


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(127)
    raise SystemExit(main(int(sys.argv[1]), sys.argv[2:]))
