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
from collections import deque
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
    runner.completed_generations = 5
    runner.next_generation_to_submit = 6
    runner._settled_resume_generations = deque()
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


def test_proposals_are_held_back_released_and_then_stopped_as_charges_arrive():
    runner = _runner(charged=0.20, in_flight=2, history=[0.10])

    # $0.20 + 2 x $0.10 in flight leaves no room for a third reservation.
    assert _coordinator_tick(runner, wanted=1) == []
    assert runner.cost_limit_reached is False

    # One of the two comes back cheap: $0.22 + $0.10 in flight + $0.10 fits.
    runner.active_proposal_tasks.popitem()
    runner.total_api_cost += 0.02
    runner._update_avg_proposal_cost(0.02)
    assert _coordinator_tick(runner, wanted=1) == [1]

    # Both in flight then cost $0.09: $0.40 + $0.10 cannot fit under $0.45.
    runner.active_proposal_tasks.clear()
    runner.total_api_cost += 0.18
    assert _coordinator_tick(runner, wanted=1) == []
    assert runner.cost_limit_reached is True


def test_one_proposal_goes_first_when_a_capped_run_has_no_cost_history():
    """Nothing can be reserved yet, so only one request is exposed to the cap."""
    runner = _runner(charged=0.0, in_flight=0, history=[])

    assert _coordinator_tick(runner, wanted=4) == [1]


def test_further_proposals_wait_until_the_cost_of_the_first_is_known(caplog):
    runner = _runner(charged=0.0, in_flight=1, history=[])

    with caplog.at_level(logging.INFO, logger="shinka.core.async_runner"):
        assert _coordinator_tick(runner, wanted=3) == []
    assert runner.cost_limit_reached is False
    assert "until the cost of the first proposal is known" in caplog.text

    runner.active_proposal_tasks.clear()
    runner.total_api_cost = 0.05
    runner._update_avg_proposal_cost(0.05)

    assert _coordinator_tick(runner, wanted=3) == [3]


def test_proposals_run_one_at_a_time_until_one_has_joined_the_cost_history():
    runner = _runner(charged=0.0, in_flight=0, history=[])
    assert _coordinator_tick(runner, wanted=4) == [1]

    runner.active_proposal_tasks["first"] = object()
    assert _coordinator_tick(runner, wanted=3) == []

    # It failed before any reply was charged, so nothing joined the history.
    runner.active_proposal_tasks.clear()
    assert _coordinator_tick(runner, wanted=4) == [1]
    assert runner.cost_limit_reached is False


def test_a_first_proposal_that_cost_nothing_opens_up_the_others():
    runner = _runner(charged=0.0, in_flight=0, history=[0.0])

    assert _coordinator_tick(runner, wanted=3) == [3]


def test_first_proposals_all_start_at_once_without_a_cap():
    runner = _runner(charged=0.0, in_flight=0, history=[], cap=None)

    assert _coordinator_tick(runner, wanted=4) == [4]


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


def test_an_expensive_proposal_is_reserved_only_while_it_is_among_the_last_fifty():
    """One outlier must not hold back proposals for the rest of a long run."""
    runner = _runner(charged=1.0, in_flight=0, history=[0.15, *[0.01] * 49])
    assert runner._proposal_reservation() == pytest.approx(0.15)

    runner._update_avg_proposal_cost(0.01)

    assert runner._proposal_reservation() == pytest.approx(0.01)


def test_free_proposals_do_not_push_a_charged_one_out_of_the_last_fifty():
    """With a free and a paid model in one run, the free replies say nothing
    about what the next paid one costs. A resumed run restores charged
    proposals only, so the running one must look at the same fifty."""
    runner = _runner(charged=1.0, in_flight=0, history=[0.15, *[0.0] * 60])

    assert runner._proposal_reservation() == pytest.approx(0.15)


def test_run_is_not_stopped_while_a_proposal_in_flight_can_lower_the_reservation():
    """The expensive proposal leaves the last fifty when the one in flight ends."""
    runner = _runner(
        charged=CAP - 0.10, in_flight=1, history=[0.15, *[0.01] * 49], cap=CAP
    )

    # $0.10 of room does not hold the $0.15 reservation: nothing starts.
    assert _coordinator_tick(runner, wanted=1) == []
    assert runner.cost_limit_reached is False

    runner.active_proposal_tasks.clear()
    runner.total_api_cost += 0.01
    runner._update_avg_proposal_cost(0.01)

    # $0.09 of room holds nine reservations of $0.01.
    assert _coordinator_tick(runner, wanted=3) == [3]
    assert runner.cost_limit_reached is False


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


def _prepare_monitor(runner: Any) -> None:
    runner.should_stop = asyncio.Event()
    runner.slot_available = asyncio.Event()
    runner.finalization_complete = asyncio.Event()
    runner.processing_lock = asyncio.Lock()
    runner.failed_jobs_for_retry = {}
    runner.submitted_jobs = {}
    runner._restore_hang_limit_from_history = lambda: asyncio.sleep(0)


def test_monitor_stops_a_run_whose_cap_cannot_cover_another_proposal():
    """Nothing is in flight and nothing may start: the run must end, not wait."""
    history = [*FIRST_PROPOSALS, 0.1497]
    runner = _runner(charged=CHARGED + 0.1497, in_flight=0, history=history)

    async def run() -> None:
        _prepare_monitor(runner)
        await asyncio.wait_for(runner._job_monitor_task(), timeout=2)

    asyncio.run(run())

    assert runner.cost_limit_reached is True
    assert runner.should_stop.is_set()
    assert runner.finalization_complete.is_set()
    assert runner.total_api_cost < CAP


def test_reservation_does_not_stop_a_run_with_no_proposal_left_to_start(caplog):
    """Every generation has been assigned: the run ends as any finished run."""
    history = [*FIRST_PROPOSALS, 0.1497]
    runner = _runner(charged=CHARGED + 0.1497, in_flight=0, history=history)
    runner.next_generation_to_submit = runner.evo_config.num_generations

    with caplog.at_level(logging.INFO, logger="shinka.core.async_runner"):
        assert runner._close_proposal_budget() is False

    assert runner.cost_limit_reached is False
    assert "cannot cover another proposal" not in caplog.text


def test_monitor_leaves_a_run_out_of_generations_to_the_coordinator():
    """Two generations failed for good: the coordinator records that and stops."""
    history = [*FIRST_PROPOSALS, 0.1497]
    runner = _runner(charged=CHARGED + 0.1497, in_flight=0, history=history)
    runner.next_generation_to_submit = runner.evo_config.num_generations
    runner.completed_generations = runner.evo_config.num_generations - 2

    async def run() -> None:
        _prepare_monitor(runner)
        monitor = asyncio.create_task(runner._job_monitor_task())
        await asyncio.sleep(0.35)
        stopped_by_monitor = runner.should_stop.is_set()
        runner.should_stop.set()
        await asyncio.wait_for(monitor, timeout=2)
        assert not stopped_by_monitor

    asyncio.run(run())

    assert runner.cost_limit_reached is False
    assert not runner.finalization_complete.is_set()


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


def test_resume_restores_a_proposal_that_was_in_evaluation_at_the_stop(tmp_path):
    """It was paid and journaled when handed to evaluation; its row never landed."""
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")),
        embedding_model="",
        read_only=False,
    )

    def hand_to_evaluation(generation: int, cost: float) -> str:
        charge_id = f"task-{generation}"
        db.record_attempt_event(
            generation, "evaluation", "charged",
            {"node_kind": "evaluation_cost", "api_charge_id": charge_id,
             "api_costs": cost},
        )
        return charge_id

    try:
        db.add(Program(id="seed", code="x = 0\n", generation=0, correct=True,
                       combined_score=1.0))
        for generation in (1, 2):
            charge_id = hand_to_evaluation(generation, 0.02)
            db.add(Program(id=f"gen{generation}", code=f"x = {generation}\n",
                           generation=generation, parent_id="seed", correct=True,
                           combined_score=2.0,
                           metadata={"api_costs": 0.02, "api_charge_id": charge_id}))
        # The run stopped while generation 3 was being evaluated.
        hand_to_evaluation(3, 0.15)
        # The paid seed of a run is not a proposal.
        db.record_attempt_event(
            0, "evaluation", "charged",
            {"node_kind": "evaluation_cost", "api_charge_id": "seed-charge",
             "api_costs": 0.9},
        )

        # A stored proposal counts once, not again for its journaled charge.
        assert stored_proposal_costs(db.conn) == pytest.approx([0.02, 0.02, 0.15])

        runner = _runner(charged=0.0, in_flight=0, history=[])
        runner.db = db
        asyncio.run(runner._restore_proposal_costs())
        assert runner._proposal_reservation() == pytest.approx(0.15)
    finally:
        db.close()


def test_a_proposal_that_failed_after_its_charge_was_journaled_counts_once(tmp_path):
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")),
        embedding_model="",
        read_only=False,
    )
    try:
        db.record_attempt_event(
            1, "evaluation", "charged",
            {"node_kind": "evaluation_cost", "api_charge_id": "task-1",
             "api_costs": 0.05},
        )
        db.record_attempt_event(
            1, "proposal", "failed",
            {"node_kind": "failed_proposal", "api_charge_id": "task-1",
             "api_costs": 0.04, "embed_cost": 0.01},
        )

        assert stored_proposal_costs(db.conn) == pytest.approx([0.05])
    finally:
        db.close()


def test_resume_reserves_from_the_last_fifty_stored_proposals(tmp_path):
    """An expensive proposal that 50 newer ones followed is not reserved again."""
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")),
        embedding_model="",
        read_only=False,
    )
    try:
        # Stored out of order, as evaluations finish: the generation orders them.
        for generation in (*range(2, 52), 1):
            db.record_attempt_event(
                generation, "proposal", "failed",
                {"node_kind": "failed_proposal",
                 "api_costs": 0.15 if generation == 1 else 0.01},
            )
        runner = _runner(charged=0.0, in_flight=0, history=[])
        runner.db = db

        asyncio.run(runner._restore_proposal_costs())

        assert len(runner.completed_proposal_costs) == 51
        assert runner._proposal_reservation() == pytest.approx(0.01)
    finally:
        db.close()
