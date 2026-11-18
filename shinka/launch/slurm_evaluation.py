"""Strict completion checks for synchronous Slurm evaluations.

The asynchronous scheduler keeps its existing status/retrieval contract.
"""

import subprocess
import time
from typing import Any

from shinka.utils import load_results
from . import slurm


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
    else:
        _wait_for_queue_exit(job_id, job_name, poll_interval)
        _require_terminal_success(job_id, job_name)
    return load_results(results_dir)


def _wait_for_queue_exit(job_id: str, job_name: str, poll_interval: float) -> None:
    consecutive_errors = 0
    while True:
        try:
            status = _queue_status(job_id, job_name)
        except (OSError, subprocess.SubprocessError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= STATUS_ERROR_LIMIT:
                raise RuntimeError(
                    f"Slurm status lookup for {job_id} failed "
                    f"{consecutive_errors} consecutive times: {exc}"
                ) from exc
        else:
            if not status or status == "COMPLETED" or status in FAILED_STATES:
                return
            consecutive_errors = 0
        time.sleep(poll_interval)


def _queue_status(job_id: str, job_name: str) -> str:
    try:
        result = subprocess.run(
            [
                "squeue",
                "-j",
                job_id,
                f"--name={job_name}",
                "--noheader",
                "--states=all",
                "--format=%T",
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
            return ""
        raise
    return result.stdout.strip()


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


def _require_terminal_success(job_id: str, job_name: str) -> None:
    details = {}
    for attempt in range(TERMINAL_STATUS_ATTEMPTS):
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
            state, exit_code = record
            details[source] = f"state={state}, ExitCode={exit_code}"
            if state == "COMPLETED" and exit_code == "0:0":
                return
            if state == "COMPLETED" or state.split()[0] in FAILED_STATES:
                raise RuntimeError(
                    f"Slurm job {job_id} failed: {source}: {details[source]}"
                )
        if attempt + 1 < TERMINAL_STATUS_ATTEMPTS:
            time.sleep(TERMINAL_STATUS_RETRY_SECONDS)
    diagnostic = "; ".join(f"{source}: {detail}" for source, detail in details.items())
    raise RuntimeError(
        f"Could not verify successful Slurm accounting/controller status for {job_id}: {diagnostic}"
    )


def _read_accounting(job_id: str, job_name: str) -> tuple[str, str] | None:
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
    return matches[0][2], matches[0][3]


def _read_controller(job_id: str, job_name: str) -> tuple[str, str] | None:
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
    }
    for token in records[0].split():
        name, _, value = token.partition("=")
        if name in fields:
            fields[name].append(value)
    if any(len(values) != 1 or not values[0] for values in fields.values()):
        return None
    if fields["JobId"][0] != job_id or fields["JobName"][0] != job_name:
        return None
    return fields["JobState"][0], fields["ExitCode"][0]
