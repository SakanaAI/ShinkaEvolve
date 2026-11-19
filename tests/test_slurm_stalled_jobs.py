import logging

import pytest

from job_lifecycle_test_utils import FakeClock, fake_squeue
from shinka.launch import slurm_status
from shinka.launch.slurm_status import SlurmQueuePoller

POLL_SECONDS = 5.0


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    fake.install(monkeypatch, slurm_status)
    return fake


def _poll_for(
    poller: SlurmQueuePoller, clock: FakeClock, job_ids: list[str], minutes: float
) -> None:
    """Ask for the jobs' status on every poll interval; they must stay tracked."""
    for _ in range(int(minutes * 60 / POLL_SECONDS)):
        assert poller.active_jobs(job_ids) == set(job_ids)
        clock.now += POLL_SECONDS


def _stalled_reports(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "not running" in record.getMessage()
    ]


def test_job_pending_for_long_is_reported_with_state_and_reason(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = fake_squeue(
        monkeypatch,
        lambda job_ids: ["101|PENDING|PartitionTimeLimit", "102|RUNNING|None"],
    )
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101", "102"], minutes=29)
        assert _stalled_reports(caplog) == []
        _poll_for(poller, clock, ["101", "102"], minutes=2)

    (report,) = _stalled_reports(caplog)
    assert "101 PENDING (PartitionTimeLimit) for 30 min" in report
    assert "102" not in report
    assert "--format=%i|%T|%r" in calls[0]["cmd"]


def test_stalled_job_is_reported_again_while_it_stays_stalled(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_squeue(monkeypatch, lambda job_ids: ["101|PENDING|Resources"])
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101"], minutes=125)

    assert len(_stalled_reports(caplog)) == 4


@pytest.mark.parametrize(
    "reason", ["JobHeldAdmin", "JobHeldUser", "launch failed requeued held"]
)
def test_held_job_is_reported_at_once(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
    reason: str,
) -> None:
    fake_squeue(monkeypatch, lambda job_ids: [f"101|PENDING|{reason}"])
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101"], minutes=1)

    (report,) = _stalled_reports(caplog)
    assert f"101 PENDING ({reason}) for 0 min" in report


def test_job_that_starts_in_time_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = "PENDING|Priority"
    fake_squeue(monkeypatch, lambda job_ids: [f"101|{state}"])
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101"], minutes=20)
        state = "RUNNING|None"
        _poll_for(poller, clock, ["101"], minutes=600)

    assert _stalled_reports(caplog) == []


def test_job_stuck_after_running_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = "RUNNING|None"
    fake_squeue(monkeypatch, lambda job_ids: [f"101|{state}"])
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101"], minutes=120)
        state = "COMPLETING|None"
        _poll_for(poller, clock, ["101"], minutes=31)

    (report,) = _stalled_reports(caplog)
    assert "101 COMPLETING (None) for 30 min" in report


def test_checking_one_job_does_not_restart_the_wait_of_another(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_squeue(
        monkeypatch, lambda job_ids: [f"{job_id}|PENDING|Priority" for job_id in job_ids]
    )
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101", "102"], minutes=20)
        _poll_for(poller, clock, ["102"], minutes=5)
        _poll_for(poller, clock, ["101", "102"], minutes=6)

    assert any("101 PENDING" in report for report in _stalled_reports(caplog))


def test_queue_line_without_state_still_tracks_the_job(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A site wrapper may ignore the requested format and print default columns.
    fake_squeue(monkeypatch, lambda job_ids: ["   101  gpu  shinka  rob  PD  0:00  1"])
    poller = SlurmQueuePoller(poll_interval=POLL_SECONDS)

    with caplog.at_level(logging.WARNING, logger=slurm_status.__name__):
        _poll_for(poller, clock, ["101"], minutes=45)

    assert _stalled_reports(caplog) == []
