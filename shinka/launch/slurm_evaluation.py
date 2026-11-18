"""Strict completion checks for synchronous Slurm evaluations.

The asynchronous scheduler keeps its existing status/retrieval contract.
"""

import subprocess
import time
from typing import Any

from shinka.utils import load_results
from . import slurm


ACCOUNTING_ATTEMPTS = 6
ACCOUNTING_RETRY_SECONDS = 10
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
    job_id: str, results_dir: str, poll_interval: float = 10
) -> dict[str, Any]:
    """Load artifacts only after the evaluator has verifiably succeeded."""
    if job_id.startswith("local-"):
        _wait_for_local_success(job_id, poll_interval)
    else:
        while slurm.get_job_status(job_id) != "":
            time.sleep(poll_interval)
        _require_accounting_success(job_id)
    return load_results(results_dir)


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


def _require_accounting_success(job_id: str) -> None:
    detail = "no matching allocation record"
    for attempt in range(ACCOUNTING_ATTEMPTS):
        try:
            record = _read_accounting(job_id)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = str(exc)
        else:
            if record is not None:
                state, exit_code = record
                detail = f"state={state}, ExitCode={exit_code}"
                if state == "COMPLETED" and exit_code == "0:0":
                    return
                if state == "COMPLETED" or state.split()[0] in FAILED_STATES:
                    raise RuntimeError(f"Slurm job {job_id} failed: {detail}")
        if attempt + 1 < ACCOUNTING_ATTEMPTS:
            time.sleep(ACCOUNTING_RETRY_SECONDS)
    raise RuntimeError(
        f"Could not verify successful Slurm accounting for {job_id}: {detail}"
    )


def _read_accounting(job_id: str) -> tuple[str, str] | None:
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
            "--format=JobIDRaw,State%40,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    records = [
        [field.strip() for field in line.split("|")]
        for line in result.stdout.splitlines()
    ]
    matches = [record for record in records if len(record) == 3 and record[0] == job_id]
    if len(matches) != 1 or not matches[0][1] or not matches[0][2]:
        return None
    return matches[0][1], matches[0][2]
