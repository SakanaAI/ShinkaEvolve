"""Helpers shared by the tests of what fatal signals do to a run."""

import asyncio
import os
import signal
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable, Iterator

import pytest

from shinka.core.async_runner import AsyncRunningJob, ShinkaEvolveRunner
from shinka.launch import fatal_signals, local

FATAL_SIGNALS = [signal.SIGTERM, signal.SIGHUP]


@contextmanager
def recorded_deaths(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Fatal signals as a fresh process has them; yields what would have ended it."""
    died_of: list[int] = []
    previous = {signum: signal.getsignal(signum) for signum in FATAL_SIGNALS}
    monkeypatch.setattr(fatal_signals, "_stops", [])
    monkeypatch.setattr(fatal_signals, "_asked_to_stop", None)
    monkeypatch.setattr(fatal_signals, "_unhandled_since", {})
    monkeypatch.setattr(fatal_signals, "_warned_that_nothing_is_hooked", False)
    monkeypatch.setattr(local, "_LIVE_PROCESSES", set())
    # Without the relay, only the signals a test raises reach the handler.
    monkeypatch.setattr(
        fatal_signals, "_relay_fatal_signals_to_the_main_thread", lambda: None
    )
    reported_to = signal.set_wakeup_fd(-1)
    monkeypatch.setattr(os, "kill", lambda pid, signum: died_of.append(signum))
    for signum in FATAL_SIGNALS:
        signal.signal(signum, signal.SIG_DFL)
    try:
        yield died_of
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        signal.set_wakeup_fd(reported_to)


def raise_hooked(signum: int) -> None:
    """Send this process a signal whose handler is the hook."""
    # Unhooked, the signal would end the test run.
    assert fatal_signals._is_hooked(signum)
    signal.raise_signal(signum)


CLEANUP = [
    "flush_charges",
    "finish_wandb",
    "close_db",
    "cancel:101",
    "scheduler.shutdown",
]


def waiting_runner(record: Callable[[str], None]) -> ShinkaEvolveRunner:
    """A runner with one evaluation running, which records "waiting" and waits.

    Its cleanup records the `CLEANUP` events, in that order when it is complete.
    """
    runner = object.__new__(ShinkaEvolveRunner)
    runner.verbose = False
    runner.pricing_snapshot = None
    runner.meta_summarizer = None
    runner.finalization_complete = asyncio.Event()
    runner.should_stop = asyncio.Event()
    runner.active_proposal_tasks = {}
    runner._completed_job_batch_tasks = set()
    runner.prompt_db = None
    runner.db = None
    now = time.time()
    runner.running_jobs = [
        AsyncRunningJob(
            job_id="101",
            exec_fname="gen_1/main.py",
            results_dir="gen_1/results",
            start_time=now,
            proposal_started_at=now,
            evaluation_submitted_at=now,
            generation=1,
        )
    ]

    async def note(event: str) -> bool:
        record(event)
        return True

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    runner._setup_async = lambda: note("setup")
    runner._verify_database_ready = lambda: note("waiting")
    runner._job_monitor_task = wait_forever
    runner._proposal_coordinator_task = wait_forever
    runner._flush_unwritten_charges = lambda: note("flush_charges")
    runner._finish_wandb_logging = lambda: note("finish_wandb")
    runner.async_db = SimpleNamespace(close_async=lambda: note("close_db"))
    runner.scheduler = SimpleNamespace(
        cancel_job_async=lambda job_id: note(f"cancel:{job_id}"),
        shutdown=lambda: record("scheduler.shutdown"),
    )
    return runner
