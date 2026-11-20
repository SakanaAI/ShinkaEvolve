"""A proposal starts only when the cost cap covers a reservation for it.

The numbers are those of a real run (cap $0.45, gpt-5.4-mini, two proposal
workers). With $0.2386 charged and one proposal in flight the runner started a
second one, because it reserved the average proposal cost ($0.0597) for the one
in flight and nothing for the new one. Both replies spent their whole output
budget on reasoning, $0.15 each, and the run ended at $0.5387.
"""

import asyncio
import logging
import math
from types import SimpleNamespace
from typing import Any

import pytest

from shinka.api_budget import cost_scope
from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.billing import program_db_costs, stored_proposal_costs
from shinka.database import DatabaseConfig, Program, ProgramDatabase

CAP = 0.45
FIRST_PROPOSALS = [0.0162, 0.0371, 0.0717, 0.1137]
CHARGED = sum(FIRST_PROPOSALS)  # $0.2387


def _runner(
    *, charged: float, in_flight: int, history: list[float], cap: float | None = CAP
) -> Any:
    runner: Any = object.__new__(ShinkaEvolveRunner)
    runner.evo_config = SimpleNamespace(max_api_costs=cap, num_generations=50)
    runner.total_api_cost = charged
    runner.active_proposal_tasks = {f"task-{n}": object() for n in range(in_flight)}
    runner.running_jobs = []
    runner.completed_proposal_costs = []
    runner.avg_proposal_cost = 0.0
    for cost in history:
        runner._update_avg_proposal_cost(cost)
    runner.cost_limit_reached = False
    runner.verbose = True
    return runner


def _coordinator_tick(runner: Any, *, wanted: int) -> list[int]:
    """Run the proposal coordinator once and return how many proposals it started."""
    started: list[int] = []

    async def start_proposals(count: int) -> None:
        started.append(count)

    async def stop_after_one_tick(timeout: float) -> None:
        runner.should_stop.set()

    async def run() -> None:
        runner.should_stop = asyncio.Event()
        runner.slot_available = asyncio.Event()
        runner.finalization_complete = asyncio.Event()
        runner.completed_generations = 5
        runner.next_generation_to_submit = 6
        runner.max_proposal_jobs = wanted + len(runner.active_proposal_tasks)
        runner._is_system_stuck = lambda: False
        runner._get_remaining_completed_work = lambda: 40
        runner._get_remaining_generation_slots = lambda: 40
        runner._get_in_flight_work_count = lambda: len(runner.active_proposal_tasks)
        runner._compute_proposal_pipeline_target = lambda: runner.max_proposal_jobs
        runner._log_proposal_target_decision = lambda target: None
        runner._record_progress = lambda: None
        runner._cleanup_completed_proposal_tasks = lambda: asyncio.sleep(0)
        runner._start_proposals = start_proposals
        runner._wait_for_slot_or_stop = stop_after_one_tick
        await asyncio.wait_for(runner._proposal_coordinator_task(), timeout=5)

    asyncio.run(run())
    return started


def test_second_proposal_waits_when_the_cap_cannot_cover_two_reservations():
    runner = _runner(charged=CHARGED, in_flight=1, history=FIRST_PROPOSALS)

    started = _coordinator_tick(runner, wanted=1)

    # $0.2387 + 2 x $0.1137 > $0.45: wait for the proposal in flight.
    assert started == []
    assert runner.cost_limit_reached is False


def test_run_winds_down_when_not_even_one_more_proposal_fits(caplog):
    history = [*FIRST_PROPOSALS, 0.1497]
    runner = _runner(charged=CHARGED + 0.1497, in_flight=0, history=history)

    with caplog.at_level(logging.INFO, logger="shinka.core.async_runner"):
        started = _coordinator_tick(runner, wanted=2)

    # $0.3884 is below the cap, and another $0.1497 would pass it.
    assert started == []
    assert runner.cost_limit_reached is True
    assert "cannot cover another proposal" in caplog.text
    assert "$0.3884" in caplog.text and "$0.1497" in caplog.text


def test_only_as_many_proposals_start_as_the_cap_covers():
    runner = _runner(charged=0.10, in_flight=0, history=[0.03, 0.10], cap=0.35)

    started = _coordinator_tick(runner, wanted=4)

    # $0.25 of room holds two reservations of $0.10.
    assert started == [2]


def test_first_proposals_start_without_a_cost_history():
    runner = _runner(charged=0.0, in_flight=0, history=[])

    assert _coordinator_tick(runner, wanted=2) == [2]


def test_a_cap_that_exactly_covers_the_next_proposal_admits_it():
    # In floating point (0.3 - 0.2) / 0.1 is a hair below one.
    runner = _runner(charged=0.2, in_flight=0, history=[0.1], cap=0.3)

    assert _coordinator_tick(runner, wanted=3) == [1]


def test_proposals_are_not_limited_without_a_cap():
    runner = _runner(charged=5.0, in_flight=1, history=[1.0], cap=None)

    assert _coordinator_tick(runner, wanted=3) == [3]


def test_committed_cost_reserves_the_most_expensive_proposal_for_each_in_flight():
    runner = _runner(charged=CHARGED, in_flight=2, history=FIRST_PROPOSALS)

    assert runner._get_committed_cost() == pytest.approx(CHARGED + 2 * 0.1137)


def test_a_charge_already_received_does_not_use_up_a_reservation():
    """A proposal that has paid for one reply may still pay for another."""
    runner = _runner(charged=0.20, in_flight=1, history=[0.10])
    scope, token = runner._begin_api_cost_scope()
    cost_scope.reset(token)
    scope.record(0.08)

    # $0.28 known + $0.10 in flight + $0.10 for the new one > $0.45.
    assert runner._get_committed_cost() == pytest.approx(0.38)
    assert _coordinator_tick(runner, wanted=1) == []
    assert runner.cost_limit_reached is False


@pytest.mark.parametrize("charged", [math.nan, math.inf])
def test_a_total_that_is_not_a_number_admits_nothing(charged):
    """It cannot be compared with the cap, so it must not count as room."""
    runner = _runner(charged=charged, in_flight=0, history=[0.05])

    assert _coordinator_tick(runner, wanted=2) == []
    assert runner.cost_limit_reached is True


def test_a_negligible_reservation_does_not_limit_proposals():
    runner = _runner(charged=0.1, in_flight=0, history=[1e-320])

    assert _coordinator_tick(runner, wanted=3) == [3]


def test_monitor_stops_a_run_whose_cap_cannot_cover_another_proposal():
    """Nothing is in flight and nothing may start: the run must end, not wait."""
    history = [*FIRST_PROPOSALS, 0.1497]
    runner = _runner(charged=CHARGED + 0.1497, in_flight=0, history=history)

    async def run() -> None:
        runner.should_stop = asyncio.Event()
        runner.slot_available = asyncio.Event()
        runner.finalization_complete = asyncio.Event()
        runner.processing_lock = asyncio.Lock()
        runner.failed_jobs_for_retry = {}
        runner.submitted_jobs = {}
        runner.completed_generations = 6
        runner._restore_hang_limit_from_history = lambda: asyncio.sleep(0)
        await asyncio.wait_for(runner._job_monitor_task(), timeout=2)

    asyncio.run(run())

    assert runner.should_stop.is_set()
    assert runner.finalization_complete.is_set()
    assert runner.total_api_cost < CAP


def test_resume_restores_the_cost_history_of_the_stored_proposals(tmp_path):
    """Without it a resumed run starts its first proposals with no reservation."""
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite"), num_islands=2),
        embedding_model="",
        read_only=False,
    )
    try:
        # The seed and its island copy are not proposals.
        db.add(Program(id="seed", code="x = 0\n", generation=0, correct=True,
                       combined_score=1.0, metadata={"api_costs": 0.9}))
        db.add(Program(id="gen1", code="x = 1\n", generation=1, parent_id="seed",
                       correct=True, combined_score=2.0,
                       metadata={"api_costs": 0.07, "embed_cost": 0.001,
                                 "novelty_cost": 0.002, "meta_cost": 0.5}))
        db.record_attempt_event(
            2, "proposal", "failed",
            {"node_kind": "failed_proposal", "api_costs": 0.1497, "embed_cost": 0.0},
        )
        db.record_attempt_event(
            3, "meta", "failed", {"node_kind": "unattached_cost", "api_costs": 0.8}
        )

        # A database of an early release can hold a NaN literal as a cost.
        db.conn.execute(
            "INSERT INTO attempt_log (generation, stage, status, details, created_at) "
            "VALUES (4, 'proposal', 'failed', "
            """'{"node_kind": "failed_proposal", "api_costs": NaN}', 0)"""
        )
        db.conn.commit()

        assert sorted(stored_proposal_costs(db.conn)) == pytest.approx([0.073, 0.1497])
        assert math.isfinite(program_db_costs(db.conn).total)

        runner = _runner(charged=0.0, in_flight=0, history=[])
        runner.db = db
        asyncio.run(runner._restore_proposal_costs())
        assert runner._proposal_reservation() == pytest.approx(0.1497)
    finally:
        db.close()
