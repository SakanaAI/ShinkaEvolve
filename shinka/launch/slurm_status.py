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
# A controller restart or failover must not cost the jobs that are running
# through it, so a failing lookup means "still active" for this long.
STATUS_OUTAGE_LIMIT_SECONDS = 900.0
STATUS_OUTAGE_MIN_FAILURES = 3
_FAILURE_LOG_EVERY = 12


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

    def __init__(
        self,
        poll_interval: float = STATUS_POLL_SECONDS,
        outage_limit: float = STATUS_OUTAGE_LIMIT_SECONDS,
    ) -> None:
        self._poll_interval = poll_interval
        self._outage_limit = outage_limit
        self._lock = threading.Lock()
        self._polled_at: Optional[float] = None
        self._polled: set[str] = set()
        self._active: set[str] = set()
        # Jobs asked about since the last poll that it did not cover.
        self._unpolled: set[str] = set()
        self._failures = 0
        self._failing_since: Optional[float] = None

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
            self._active = self._jobs_assumed_active(job_ids, now, error)
            return
        if self._failures:
            logger.info("Slurm status lookup recovered after %s failure(s)", self._failures)
        self._failures = 0
        self._failing_since = None

    def _jobs_assumed_active(
        self, job_ids: set[str], now: float, error: Exception
    ) -> set[str]:
        self._failures += 1
        if self._failing_since is None:
            self._failing_since = now
        outage = now - self._failing_since
        detail = getattr(error, "stderr", None) or error
        if (
            self._failures >= STATUS_OUTAGE_MIN_FAILURES
            and outage >= self._outage_limit
        ):
            # Nothing can confirm these jobs any more. Hand them back so their
            # result directories decide, instead of tracking them forever.
            logger.error(
                "Slurm status lookup has failed for %.0fs (%s attempts); treating "
                "%s job(s) as finished: %s",
                outage,
                self._failures,
                len(job_ids),
                detail,
            )
            return set()
        if self._failures % _FAILURE_LOG_EVERY == 1:
            logger.warning(
                "Slurm status lookup failed (%s in a row); assuming %s job(s) are "
                "still active: %s",
                self._failures,
                len(job_ids),
                detail,
            )
        return set(job_ids)
