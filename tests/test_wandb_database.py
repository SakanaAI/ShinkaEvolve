import threading
from types import SimpleNamespace

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.wandb_logging import ShinkaWandbLogger
from shinka.wandb_metrics import (
    build_population_progress_payload,
    build_population_progress_payload_from_telemetry,
)
from wandb_test_utils import make_db


def test_population_progress_uses_aggregate_query_without_heavy_fields(
    tmp_path, monkeypatch
):
    db, _, _ = make_db(tmp_path)
    statements = []
    db.conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        db,
        "get_all_programs",
        lambda: (_ for _ in ()).throw(AssertionError("full rows were materialized")),
    )
    logged = []
    logger = ShinkaWandbLogger(enabled=True)
    logger._run = SimpleNamespace(log=logged.append)

    logger.log_population_progress(db=db)

    select = next(
        statement for statement in statements if "WITH metadata_source AS" in statement
    )
    assert " code" not in select.lower()
    assert "embedding" not in select.lower()
    assert "private_metrics" not in select.lower()
    assert "public_metrics" not in select.lower()
    assert logged[0]["population/count"] == 2
    assert logged[0]["population/evaluated_count"] == 2


def test_population_aggregate_matches_full_program_metric_semantics(tmp_path):
    db, _, _ = make_db(tmp_path)
    db.add(
        Program(
            id="copy",
            code="pass",
            combined_score=1.0,
            correct=True,
            island_idx=2,
            metadata={"_is_island_copy": True, "api_costs": 99.0},
        ),
        defer_maintenance=True,
    )
    db.add(
        Program(
            id="non-finite",
            code="pass",
            combined_score=float("inf"),
            metadata={
                "api_costs": 2.0,
                "meta_cost": float("inf"),
                "pipeline_seconds": float("inf"),
            },
        ),
        defer_maintenance=True,
    )
    db.add(
        Program(
            id="uppercase-headless",
            code="pass",
            metadata={
                "model_name": "HEADLESS/codex",
                "headless_pricing_unknown": True,
                "api_costs": 12.0,
            },
        ),
        defer_maintenance=True,
    )

    aggregate_payload = build_population_progress_payload_from_telemetry(
        db.get_population_telemetry()
    )
    full_payload = build_population_progress_payload(db.get_all_programs())

    assert aggregate_payload == full_payload


def _make_runner(db, logger):
    runner = object.__new__(ShinkaEvolveRunner)
    runner.db = db
    runner.wandb_logger = logger
    runner.total_api_cost = 0.5
    runner.total_proposals_generated = 3
    runner._wandb_population_task = None
    runner._wandb_population_delay_task = None
    runner._wandb_population_pending = False
    runner._wandb_last_population_snapshot_at = None
    runner._wandb_population_interval_seconds = 5.0
    runner._wandb_executor = None
    runner._wandb_candidate_queue = None
    runner._wandb_candidate_worker_task = None
    runner._wandb_dropped_candidate_events = 0
    runner._wandb_last_drop_warning_at = None
    return runner


def test_in_memory_snapshots_read_sqlite_on_owner_thread():
    async def run():
        owner_thread = threading.get_ident()
        db = ProgramDatabase(DatabaseConfig())
        db.add(Program(id="program", code="pass"), defer_maintenance=True)
        observed = []

        class Logger:
            active = True

            def log_population_snapshot(self, snapshot):
                observed.append(("population", snapshot, threading.get_ident()))

            def log_final_programs(self, programs, **kwargs):
                observed.append(
                    (
                        "final",
                        [program.id for program in programs],
                        threading.get_ident(),
                    )
                )

            def finish(self):
                observed.append(("finish", None, threading.get_ident()))

        runner = _make_runner(db, Logger())
        runner._log_wandb_population_progress()
        await runner._finish_wandb_logging()
        db.close()

        assert observed[0][0] == "population"
        assert observed[0][1][0]["count"] == 1
        assert observed[1][0:2] == ("final", ["program"])
        assert all(thread_id != owner_thread for _, _, thread_id in observed)

    import asyncio

    asyncio.run(run())


def test_wandb_sdk_operations_share_one_nonblocking_worker():
    async def run():
        import asyncio

        loop = asyncio.get_running_loop()
        owner_thread = threading.get_ident()
        first_started = asyncio.Event()
        release_first = threading.Event()
        operation_lock = threading.Lock()
        calls = []

        class Database:
            config = SimpleNamespace(db_path=None)

            def get_population_telemetry(self):
                assert threading.get_ident() == owner_thread
                return []

            def get_all_programs(self):
                assert threading.get_ident() == owner_thread
                return []

        class Logger:
            active = True

            def _record(self, name, *, wait=False):
                assert operation_lock.acquire(blocking=False), "overlapping W&B call"
                try:
                    calls.append((name, threading.get_ident()))
                    if wait:
                        loop.call_soon_threadsafe(first_started.set)
                        release_first.wait(timeout=1)
                finally:
                    operation_lock.release()

            def log_program_payload(self, program_id, payload):
                self._record("candidate", wait=True)

            def log_population_snapshot(self, snapshot):
                self._record("population")

            def log_final_programs(self, programs, **kwargs):
                self._record("final")

            def finish(self):
                self._record("finish")

        runner = _make_runner(Database(), Logger())
        runner._log_program_to_wandb(Program(id="candidate", code="pass"))
        await asyncio.wait_for(first_started.wait(), timeout=0.2)
        runner._log_wandb_population_progress()
        finish_task = asyncio.create_task(runner._finish_wandb_logging())
        await asyncio.sleep(0)
        release_first.set()
        await finish_task

        assert [name for name, _ in calls] == [
            "candidate",
            "population",
            "final",
            "finish",
        ]
        worker_threads = {thread_id for _, thread_id in calls}
        assert len(worker_threads) == 1
        assert owner_thread not in worker_threads

    import asyncio

    asyncio.run(run())


def test_wandb_candidate_queue_is_bounded_compact_and_rate_limited(caplog):
    async def run():
        import asyncio

        loop = asyncio.get_running_loop()
        first_started = asyncio.Event()
        release_first = threading.Event()
        logged_ids = []

        class Logger:
            active = True

            def log_program_payload(self, program_id, payload):
                logged_ids.append(program_id)
                if len(logged_ids) == 1:
                    loop.call_soon_threadsafe(first_started.set)
                    release_first.wait(timeout=1)

        runner = _make_runner(SimpleNamespace(), Logger())
        runner._wandb_candidate_queue_size = 2
        runner._log_program_to_wandb(
            Program(id="first", code="unique-private-code-first")
        )
        await asyncio.wait_for(first_started.wait(), timeout=0.2)

        for program_id in ("second", "third", "fourth", "fifth"):
            runner._log_program_to_wandb(
                Program(id=program_id, code=f"unique-private-code-{program_id}")
            )

        queue = runner._wandb_candidate_queue
        assert queue.maxsize == 2
        assert queue.qsize() == 2
        assert "unique-private-code" not in repr(list(queue._queue))
        assert "Program(" not in repr(list(queue._queue))
        assert runner._wandb_dropped_candidate_events == 2
        assert caplog.text.count("W&B candidate queue full") == 1

        release_first.set()
        await runner._drain_wandb_candidate_queue()
        await runner._stop_wandb_candidate_worker()
        await runner._shutdown_wandb_executor()

        assert logged_ids == ["first", "second", "third"]

    import asyncio

    asyncio.run(run())


def test_wandb_population_queries_are_time_throttled_and_final_skips_delay():
    async def run():
        import asyncio

        query_count = 0

        class Database:
            config = SimpleNamespace(db_path=None)

            def get_population_telemetry(self):
                nonlocal query_count
                query_count += 1
                return []

            def get_all_programs(self):
                return []

        class Logger:
            active = True

            def log_population_snapshot(self, snapshot):
                pass

            def log_final_programs(self, programs, **kwargs):
                pass

            def finish(self):
                pass

        runner = _make_runner(Database(), Logger())
        runner._wandb_population_interval_seconds = 60.0
        runner._log_wandb_population_progress()
        await runner._wait_for_wandb_population_progress()

        runner._log_wandb_population_progress()
        runner._log_wandb_population_progress()
        runner._log_wandb_population_progress()
        await asyncio.sleep(0)

        assert query_count == 1
        assert runner._wandb_population_delay_task is not None

        await runner._finish_wandb_logging()

        assert query_count == 1
        assert runner._wandb_population_delay_task is None

    import asyncio

    asyncio.run(run())


def test_in_memory_population_query_failure_is_non_fatal(caplog):
    class Database:
        config = SimpleNamespace(db_path=None)

        def get_population_telemetry(self):
            raise RuntimeError("telemetry query failed")

    logger = SimpleNamespace(active=True, log_population_snapshot=lambda snapshot: None)
    runner = _make_runner(Database(), logger)

    async def run():
        runner._log_wandb_population_progress()

    import asyncio

    asyncio.run(run())

    assert runner._wandb_population_task is None
    assert "Failed to query in-memory W&B population snapshot" in caplog.text
