"""Result directories owned by individual executions of a Slurm seed job."""

from pathlib import Path
import shlex


def _execution_prefix(results_dir: str, job_name: str) -> Path:
    destination = Path(results_dir)
    return destination.with_name(f"{destination.name}.{job_name}.execution-")


def render_seed_evaluator(
    cmd: list[str], results_dir: str | None, job_name: str
) -> tuple[str, str, str]:
    if results_dir is None:
        return "", shlex.join(cmd), "exit $?"
    results_option = cmd.index("--results_dir")
    if cmd[results_option + 1 : results_option + 2] != [results_dir]:
        raise ValueError("Slurm seed command does not match its results directory")

    prefix = shlex.quote(str(_execution_prefix(results_dir, job_name)))
    setup = f"""
shinka_restart=${{SLURM_RESTART_COUNT:-0}}
if ! [[ "$shinka_restart" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "Invalid Slurm restart count: $shinka_restart" >&2
    exit 1
fi
shinka_execution_dir={prefix}"$shinka_restart"
(set -o noclobber; : > "$shinka_execution_dir.receipt") || exit 1
mkdir -- "$shinka_execution_dir" || exit 1
"""
    before = shlex.join(cmd[: results_option + 1])
    after = shlex.join(cmd[results_option + 2 :])
    command = f'{before} "$shinka_execution_dir" {after}'.rstrip()
    finish = """
shinka_status=$?
if [[ "$shinka_status" -eq 0 ]]; then
    printf 'complete\\n' >> "$shinka_execution_dir.receipt" || exit 1
fi
exit "$shinka_status"
"""
    return setup, command, finish


def latest_execution(results_dir: str, job_name: str) -> tuple[int, Path] | None:
    prefix = _execution_prefix(results_dir, job_name)
    executions: dict[int, Path] = {}
    for path in prefix.parent.iterdir():
        if not path.name.startswith(prefix.name) or path.suffix != ".receipt":
            continue
        restart = path.stem.removeprefix(prefix.name)
        if restart.isascii() and restart.isdecimal() and restart == str(int(restart)):
            executions[int(restart)] = path.with_suffix("")
    if not executions:
        return None
    restart = max(executions)
    return restart, executions[restart]


def require_completed_execution(execution_dir: Path) -> None:
    # Slurm retains the job ID and name across requeues. An incomplete latest
    # execution cannot fall back to the completion receipt of an older one.
    receipt = execution_dir.with_name(f"{execution_dir.name}.receipt")
    if (
        receipt.is_symlink()
        or not receipt.is_file()
        or receipt.read_bytes() != b"complete\n"
    ):
        raise RuntimeError(
            f"Slurm seed execution has no completion receipt: {execution_dir}"
        )
    if execution_dir.is_symlink() or not execution_dir.is_dir():
        raise RuntimeError(
            f"Slurm seed execution directory disappeared: {execution_dir}"
        )
