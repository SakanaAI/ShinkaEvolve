import logging
import subprocess

import pytest

from job_lifecycle_test_utils import (
    PURGED_JOB_ERROR,
    FakeClock,
    fake_squeue,
    squeue_failure,
)
from shinka.launch import slurm, slurm_status
from shinka.launch.slurm_status import SlurmQueuePoller


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    fake.install(monkeypatch, slurm, slurm_status)
    return fake


def test_purged_job_id_counts_as_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: squeue_failure(PURGED_JOB_ERROR))

    assert slurm.get_job_status("123") == ""
    assert SlurmQueuePoller().active_jobs(["123"]) == set()
    assert all(call["timeout"] is not None for call in calls)


@pytest.mark.parametrize(
    "error",
    [
        squeue_failure("slurm_load_jobs error: Unable to contact slurm controller"),
        subprocess.TimeoutExpired(["squeue"], 30),
        FileNotFoundError("squeue"),
    ],
    ids=["controller-down", "timeout", "missing-binary"],
)
def test_failed_lookup_is_reported_as_unknown(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake_squeue(monkeypatch, lambda job_ids: error)

    assert slurm.get_job_status("123") is None


def test_one_squeue_call_covers_every_tracked_job(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: ["101", "103"])

    assert SlurmQueuePoller().active_jobs(["101", "102", "103"]) == {"101", "103"}
    assert [call["job_ids"] for call in calls] == [["101", "102", "103"]]


def test_status_requests_between_polls_reuse_the_last_answer(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    active = {"101", "102"}
    calls = fake_squeue(monkeypatch, lambda job_ids: sorted(active & set(job_ids)))
    poller = SlurmQueuePoller(poll_interval=5.0)

    assert poller.active_jobs(["101", "102"]) == {"101", "102"}
    active.discard("102")
    clock.now += 4.9
    assert poller.active_jobs(["101", "102"]) == {"101", "102"}
    # A job submitted since the last poll has not been asked about yet.
    assert poller.active_jobs(["101", "102", "104"]) == {"101", "102", "104"}
    assert len(calls) == 1

    clock.now += 0.2
    assert poller.active_jobs(["101", "102", "104"]) == {"101"}
    assert [call["job_ids"] for call in calls][1:] == [["101", "102", "104"]]


def test_single_job_checks_cannot_starve_each_other(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    fake_squeue(monkeypatch, lambda job_ids: [])
    poller = SlurmQueuePoller(poll_interval=5.0)

    assert poller.active_jobs(["101"]) == set()
    assert poller.active_jobs(["102"]) == {"102"}
    clock.now += 5.0
    assert poller.active_jobs(["101"]) == set()
    assert poller.active_jobs(["102"]) == set()


def test_purged_id_in_a_list_falls_back_to_single_lookups(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    def reply(job_ids: list[str]):
        if len(job_ids) > 1 or job_ids == ["102"]:
            return squeue_failure(PURGED_JOB_ERROR)
        return job_ids

    calls = fake_squeue(monkeypatch, reply)

    assert SlurmQueuePoller().active_jobs(["101", "102"]) == {"101"}
    assert [call["job_ids"] for call in calls] == [["101", "102"], ["101"], ["102"]]


@pytest.mark.parametrize(
    "error",
    [
        squeue_failure("slurm_load_jobs error: Unable to contact slurm controller"),
        subprocess.TimeoutExpired(["squeue"], 30),
        FileNotFoundError("squeue"),
    ],
    ids=["controller-down", "timeout", "missing-binary"],
)
def test_failed_lookup_keeps_jobs_only_for_a_bounded_outage(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: error)
    poller = SlurmQueuePoller(poll_interval=5.0, outage_limit=60.0)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        for _ in range(12):
            assert poller.active_jobs(["101", "102"]) == {"101", "102"}
            clock.now += 5.0
        assert poller.active_jobs(["101", "102"]) == set()

    assert len(calls) == 13
    assert "treating 2 job(s) as finished" in caplog.text


def test_lookup_recovery_restarts_the_outage_window(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    failing = True
    fake_squeue(
        monkeypatch,
        lambda job_ids: squeue_failure("Socket timed out") if failing else job_ids,
    )
    poller = SlurmQueuePoller(poll_interval=5.0, outage_limit=60.0)

    assert poller.active_jobs(["101"]) == {"101"}
    # The runner was busy for longer than the outage limit before asking again:
    # two failures are not yet an outage.
    clock.now += 600.0
    assert poller.active_jobs(["101"]) == {"101"}
    failing = False
    clock.now += 5.0
    assert poller.active_jobs(["101"]) == {"101"}
    failing = True
    for _ in range(12):
        clock.now += 5.0
        assert poller.active_jobs(["101"]) == {"101"}


def test_verbose_monitor_survives_a_job_that_is_still_listed(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    statuses = iter([["123"], ["123"], []])
    calls = fake_squeue(monkeypatch, lambda job_ids: next(statuses))

    with caplog.at_level(logging.INFO, logger=slurm.__name__):
        assert slurm.monitor("123", verbose=True) is None

    assert len(calls) == 3
    assert "Job 123 status: 123" in caplog.text


def test_unbounded_monitor_stops_after_repeated_lookup_failures(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: squeue_failure("Socket timed out"))

    with pytest.raises(RuntimeError, match="123 failed 3 consecutive times"):
        slurm.monitor("123")

    assert len(calls) == 3


def test_bounded_monitor_stops_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: job_ids)
    started = clock.now

    assert slurm.monitor("123", poll_interval=2, max_wait=20.0) is None

    assert clock.now - started == pytest.approx(20.0)
    assert all(0 < call["timeout"] <= 30 for call in calls)


def test_bounded_monitor_does_not_wait_on_a_failing_lookup(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    calls = fake_squeue(monkeypatch, lambda job_ids: squeue_failure("Socket timed out"))
    started = clock.now

    assert slurm.monitor("123", poll_interval=2, max_wait=20.0) is None

    assert clock.now == started
    assert len(calls) == 1
