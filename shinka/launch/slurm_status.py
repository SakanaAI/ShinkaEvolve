"""Queue status for asynchronously tracked Slurm jobs."""

import logging
import subprocess
import threading
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

STATUS_COMMAND_TIMEOUT_SECONDS = 30
# The runner asks for statuses many times per second; the controller is asked
# at most this often, with one command for every tracked job.
STATUS_POLL_SECONDS = 5.0
# A controller outage stops no jobs, so a failing lookup means "still active"
# for as long as it lasts. The run stalls and resumes when the controller
# answers; until then the outage is reported this often.
STATUS_OUTAGE_LOG_SECONDS = 300.0


def is_unknown_job_error(stderr: Optional[str]) -> bool:
    """Whether Slurm rejected a job id it no longer keeps a record of."""
    return (stderr or "").strip().endswith("Invalid job id specified")


def query_active_jobs(job_ids: list[str]) -> set[str]:
    """Return the given jobs that are still pending, running or completing."""
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--format=%i", "--jobs", ",".join(job_ids)],
            capture_output=True,
            text=True,
            check=True,
            timeout=STATUS_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as error:
        if not is_unknown_job_error(error.stderr):
            raise
        # The controller purges a finished job after MinJobAge. It reports that
        # for a single id; ask one by one if it ever does so for a whole list.
        if len(job_ids) == 1:
            return set()
        return {job for job_id in job_ids for job in query_active_jobs([job_id])}
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


class SlurmQueuePoller:
    """Tells which tracked jobs are still active, from batched ``squeue`` calls."""

    def __init__(self, poll_interval: float = STATUS_POLL_SECONDS) -> None:
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._polled_at: Optional[float] = None
        self._polled: set[str] = set()
        self._active: set[str] = set()
        # Jobs asked about since the last poll that it did not cover.
        self._unpolled: set[str] = set()
        self._failures = 0
        self._failing_since: Optional[float] = None
        self._failure_reported_at: Optional[float] = None

    def active_jobs(self, job_ids: Iterable[str]) -> set[str]:
        """Return the subset of ``job_ids`` that has not left the queue."""
        wanted = set(job_ids)
        with self._lock:
            now = time.monotonic()
            if self._polled_at is None or now - self._polled_at >= self._poll_interval:
                self._poll(wanted | self._unpolled, now)
            # A job submitted after the last poll is active until one says otherwise.
            self._unpolled |= wanted - self._polled
            return {
                job_id
                for job_id in wanted
                if job_id in self._active or job_id not in self._polled
            }

    def _poll(self, job_ids: set[str], now: float) -> None:
        self._polled_at = now
        self._polled = job_ids
        self._unpolled = set()
        if not job_ids:
            self._active = set()
            return
        try:
            self._active = query_active_jobs(sorted(job_ids))
        except (OSError, subprocess.SubprocessError) as error:
            self._report_failure(len(job_ids), now, error)
            self._active = set(job_ids)
            return
        if self._failures:
            logger.info("Slurm status lookup recovered after %s failure(s)", self._failures)
        self._failures = 0
        self._failing_since = None
        self._failure_reported_at = None

    def _report_failure(self, job_count: int, now: float, error: Exception) -> None:
        self._failures += 1
        if self._failing_since is None:
            self._failing_since = now
        reported_at = self._failure_reported_at
        if reported_at is not None and now - reported_at < STATUS_OUTAGE_LOG_SECONDS:
            return
        self._failure_reported_at = now
        outage = now - self._failing_since
        # The first failure may be a blip; one that outlasts a report interval is not.
        log = logger.error if outage >= STATUS_OUTAGE_LOG_SECONDS else logger.warning
        log(
            "Slurm status lookup has failed for %.0fs (%s attempts); still tracking "
            "%s job(s), the run resumes when the controller answers: %s",
            outage,
            self._failures,
            job_count,
            getattr(error, "stderr", None) or error,
        )
