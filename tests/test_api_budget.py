import asyncio
import threading
from types import SimpleNamespace

import httpx
import openai
import pytest

import shinka.embed.embedding as embedding_module
from shinka.api_budget import (
    APIBudgetExceeded,
    check_provider_attempt,
    cost_scope,
    run_request,
)
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


@pytest.mark.parametrize("exhausted_on_retry", [False, True])
def test_budgeted_openai_embedding_rechecks_transient_retries(
    monkeypatch, exhausted_on_retry
):
    async def run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.evo_config = SimpleNamespace(max_api_costs=1.0)
        runner.total_api_cost = 0.0
        scope, token = runner._begin_api_cost_scope()
        requests = []

        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(503, json={"error": {"message": "retry"}})
            return httpx.Response(
                200, json={"data": [{"embedding": [0.1]}], "usage": {"total_tokens": 20}}
            )

        async def backoff(delay):
            if exhausted_on_retry:
                runner.total_api_cost = 1.0

        monkeypatch.setattr(embedding_module.asyncio, "sleep", backoff)
        try:
            async with openai.AsyncOpenAI(
                api_key="mock",
                max_retries=1,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as sdk:
                monkeypatch.setattr(
                    embedding_module, "get_async_client_embed", lambda name: (sdk, name)
                )
                client = embedding_module.AsyncEmbeddingClient()
                embedding, charge = await client.embed_async("code")
        finally:
            cost_scope.reset(token)

        assert len(requests) == (1 if exhausted_on_retry else 2)
        if exhausted_on_retry:
            assert embedding == []
            assert charge == 0.0
        else:
            assert embedding == [0.1]
            assert charge > 0.0
        assert scope.pending == charge

    asyncio.run(run())


def test_uncapped_gemini_embedding_records_charge_when_cancelled(monkeypatch):
    async def run():
        started = threading.Event()
        release = threading.Event()
        runner = object.__new__(ShinkaEvolveRunner)
        runner.evo_config = SimpleNamespace(max_api_costs=None)
        runner.total_api_cost = 0.0
        scope, token = runner._begin_api_cost_scope()
        model = "gemini-embedding-001"

        def embed(**kwargs):
            started.set()
            assert release.wait(2)
            return [[0.1]], 0.6

        monkeypatch.setattr(embedding_module, "_get_google_embeddings_and_cost", embed)
        monkeypatch.setattr(
            embedding_module, "get_async_client_embed", lambda name: (object(), name)
        )
        client = embedding_module.AsyncEmbeddingClient(model)
        task = asyncio.create_task(client.embed_async("code"))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        try:
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            cost_scope.reset(token)
        assert scope.pending == pytest.approx(0.6)

    asyncio.run(run())


def test_uncapped_transport_rechecks_admission_after_a_journal_error():
    async def run():
        runner = object.__new__(ShinkaEvolveRunner)
        runner.evo_config = SimpleNamespace(max_api_costs=None)
        runner.total_api_cost = 0.0
        scope, token = runner._begin_api_cost_scope()

        async def provider():
            check_provider_attempt()
            runner._billing_writes_blocked = True
            check_provider_attempt()

        try:
            with pytest.raises(APIBudgetExceeded):
                await run_request(provider, lambda _: 0.0, "gpt-5-mini")
        finally:
            await runner._finish_api_cost_scope(scope, token, "proposal", 1)

    asyncio.run(run())
