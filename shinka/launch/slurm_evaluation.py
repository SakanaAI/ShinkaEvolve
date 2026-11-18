"""Strict completion checks for synchronous Slurm evaluations.

The asynchronous scheduler keeps its existing status/retrieval contract.
"""

from pathlib import Path
import subprocess
import time
from typing import Any

from shinka.utils import load_results
from . import slurm
from .slurm_results import latest_execution, require_completed_execution


TERMINAL_STATUS_ATTEMPTS = 6
TERMINAL_STATUS_RETRY_SECONDS = 10
CONTROL_COMMAND_TIMEOUT_SECONDS = 30
STATUS_ERROR_LIMIT = 3
FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
    "REVOKED",
}


def monitor_evaluation(
    job_id: str, results_dir: str, *, job_name: str, poll_interval: float = 10
) -> dict[str, Any]:
    """Load artifacts only after the evaluator has verifiably succeeded."""
    if job_id.startswith("local-"):
        _wait_for_local_success(job_id, poll_interval)
        return load_results(results_dir)

    queue_restart = _wait_for_queue_exit(job_id, job_name, poll_interval)
    execution_dir = _require_terminal_success(job_id, job_name, results_dir, queue_restart)
    results = load_results(str(execution_dir), log_dir=results_dir)
    results["_artifact_dir"] = execution_dir
    return results


def _wait_for_queue_exit(job_id: str, job_name: str, poll_interval: float) -> int | None:
    consecutive_errors = 0
    latest_restart = None
    while True:
        try:
            status, restart = _queue_status(job_id, job_name)
        except (OSError, subprocess.SubprocessError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= STATUS_ERROR_LIMIT:
                raise RuntimeError(
                    f"Slurm status lookup for {job_id} failed "
                    f"{consecutive_errors} consecutive times: {exc}"
                ) from exc
        else:
            if restart is not None:
                latest_restart = max(latest_restart or 0, restart)
            if not status or status == "COMPLETED" or status in FAILED_STATES:
                return latest_restart
            consecutive_errors = 0
        time.sleep(poll_interval)


def _queue_status(job_id: str, job_name: str) -> tuple[str, int | None]:
    try:
        result = subprocess.run(
            [
                "squeue",
                "-j",
                job_id,
                f"--name={job_name}",
                "--noheader",
                "--states=all",
                "--Format=State:40|,RestartCnt",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=CONTROL_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        # The controller may have purged a finished job before our next poll.
        # Absence is only a cue to verify terminal status, never proof of success.
        if (exc.stderr or "").strip().endswith("Invalid job id specified"):
            return "", None
        raise
    state, _, restart = result.stdout.strip().partition("|")
    return state.strip(), int(restart) if restart.strip().isdecimal() else None


def _wait_for_local_success(job_id: str, poll_interval: float) -> None:
    while True:
        job = slurm.LOCAL_JOBS.get(job_id)
        if job is None:
            raise RuntimeError(f"Unknown local fallback job {job_id}")
        process = job.get("popen")
        # A fallback may still be waiting for GPU capacity before it launches.
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                if returncode != 0:
                    raise RuntimeError(
                        f"Local fallback job {job_id} exited with status {returncode}"
                    )
                return
        time.sleep(poll_interval)


def _require_terminal_success(
    job_id: str, job_name: str, results_dir: str, queue_restart: int | None
) -> Path:
    details = {}
    execution = None
    minimum_restart = queue_restart or 0
    for attempt in range(TERMINAL_STATUS_ATTEMPTS):
        accounting_success = False
        verified_success = False
        for source, read_status in (
            ("accounting", _read_accounting),
            ("controller", _read_controller),
        ):
            try:
                record = read_status(job_id, job_name)
            except (OSError, subprocess.SubprocessError) as exc:
                details[source] = str(exc)
                continue
            details[source] = "no unambiguous matching job record"
            if record is None:
                continue
            visible = latest_execution(results_dir, job_name)
            if visible is not None and (execution is None or visible[0] > execution[0]):
                execution = visible
                accounting_success = False
            restart = max(minimum_restart, execution[0] if execution else 0)
            state, exit_code, observed_restart = record
            details[source] = f"state={state}, ExitCode={exit_code}"
            success = state == "COMPLETED" and exit_code == "0:0"
            if source == "accounting" and (restart or (success and queue_restart is None)):
                # Accounting may lack a restart count. Check the retained
                # controller record when a requeue is known or squeue had no count.
                accounting_success = success
                continue
            if source == "controller" and observed_restart is not None:
                accounting_success = False
                minimum_restart = max(minimum_restart, observed_restart)
                if observed_restart != restart:
                    details[source] += (
                        f", Restarts={observed_restart}, expected {restart}"
                    )
                    continue
            elif restart:
                details[source] += ", no restart count"
                continue
            if success:
                verified_success = True
                break
            if state == "COMPLETED" or state.split()[0] in FAILED_STATES:
                raise RuntimeError(
                    f"Slurm job {job_id} failed: {source}: {details[source]}"
                )
        if verified_success or accounting_success:
            if execution is not None and execution[0] >= minimum_restart:
                try:
                    require_completed_execution(execution[1])
                except (OSError, RuntimeError) as exc:
                    details["results"] = str(exc)
                else:
                    visible = latest_execution(results_dir, job_name)
                    if visible == execution:
                        return execution[1]
                    if visible is not None and visible[0] > execution[0]:
                        execution = visible
                    details["results"] = "execution receipt changed during verification"
            else:
                details["results"] = f"no visible receipt for execution {minimum_restart}"
        if attempt + 1 < TERMINAL_STATUS_ATTEMPTS:
            time.sleep(TERMINAL_STATUS_RETRY_SECONDS)
    diagnostic = "; ".join(f"{source}: {detail}" for source, detail in details.items())
    raise RuntimeError(
        f"Could not verify successful Slurm accounting/controller status for {job_id}: {diagnostic}"
    )


def _read_accounting(job_id: str, job_name: str) -> tuple[str, str, int | None] | None:
    # Allocation-only output prevents a successful batch/extern step from
    # masking a failed job. Widen State to avoid Slurm's truncated labels.
    result = subprocess.run(
        [
            "sacct",
            "--jobs",
            job_id,
            "--allocations",
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName%80,State%40,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=CONTROL_COMMAND_TIMEOUT_SECONDS,
    )
    records = [
        [field.strip() for field in line.split("|")]
        for line in result.stdout.splitlines()
    ]
    matches = [
        record
        for record in records
        if len(record) == 4 and record[:2] == [job_id, job_name]
    ]
    if len(matches) != 1 or not matches[0][2] or not matches[0][3]:
        return None
    return matches[0][2], matches[0][3], None


def _read_controller(job_id: str, job_name: str) -> tuple[str, str, int | None] | None:
    result = subprocess.run(
        ["scontrol", "--oneliner", "show", "job", job_id],
        capture_output=True,
        text=True,
        check=True,
        timeout=CONTROL_COMMAND_TIMEOUT_SECONDS,
    )
    records = [line for line in result.stdout.splitlines() if line.strip()]
    if len(records) != 1:
        return None
    fields: dict[str, list[str]] = {
        "JobId": [],
        "JobName": [],
        "JobState": [],
        "ExitCode": [],
        "Restarts": [],
    }
    for token in records[0].split():
        name, _, value = token.partition("=")
        if name in fields:
            fields[name].append(value)
    restart_counts = fields.pop("Restarts")
    if restart_counts and (
        len(restart_counts) != 1 or not restart_counts[0].isdecimal()
    ):
        return None
    if any(len(values) != 1 or not values[0] for values in fields.values()):
        return None
    if fields["JobId"][0] != job_id or fields["JobName"][0] != job_name:
        return None
    restart = int(restart_counts[0]) if restart_counts else None
    return fields["JobState"][0], fields["ExitCode"][0], restart
