import asyncio
import threading
from types import SimpleNamespace

from shinka.api_budget import cost_scope
from shinka.core.async_runner import ShinkaEvolveRunner


def test_budget_reader_cannot_see_partial_charge_transfer(monkeypatch):
    async def run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.evo_config = SimpleNamespace(max_api_costs=1.0)
        runner.total_api_cost = 0.0
        scope, token = runner._begin_api_cost_scope()
        scope.pending = 1.0
        transferring = threading.Event()
        release = threading.Event()
        reading = threading.Event()
        acknowledge = scope.acknowledge

        def pause_after_acknowledging(cost):
            acknowledge(cost)
            transferring.set()
            assert release.wait(2)

        def check_budget():
            reading.set()
            return runner._api_budget_exhausted()

        monkeypatch.setattr(scope, "acknowledge", pause_after_acknowledging)
        transfer = asyncio.create_task(asyncio.to_thread(runner._post_api_cost, 1.0))
        try:
            assert await asyncio.to_thread(transferring.wait, 2)
            reader = asyncio.create_task(asyncio.to_thread(check_budget))
            assert await asyncio.to_thread(reading.wait, 2)
            await asyncio.wait({reader}, timeout=0.05)
        finally:
            release.set()
            await transfer
            cost_scope.reset(token)
        assert await reader

    asyncio.run(run())
