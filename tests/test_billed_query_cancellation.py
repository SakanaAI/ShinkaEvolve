"""Cancellation propagates with usage from replies already billed."""

import asyncio

import pytest

import shinka.llm.llm as llm_module
from shinka.llm.llm import AsyncLLMClient
from shinka.llm.providers.errors import BilledCancellationError, EmptyResponseError


@pytest.mark.parametrize("boundary", ["retry_delay", "next_request"])
@pytest.mark.parametrize("billed_replies", [1, 2])
def test_cancelled_query_carries_received_usage(monkeypatch, boundary, billed_replies):
    async def run():
        reached_boundary = asyncio.Event()
        calls = []

        async def provider(**kwargs):
            calls.append(kwargs)
            if len(calls) <= billed_replies:
                if len(calls) == billed_replies and boundary == "retry_delay":
                    reached_boundary.set()
                raise EmptyResponseError(
                    "no text",
                    None,
                    {
                        "cost": 0.5,
                        "input_cost": 0.1,
                        "output_cost": 0.4,
                        "input_tokens": 1000,
                        "output_tokens": 20,
                        "thinking_tokens": 20,
                    },
                )
            reached_boundary.set()
            await asyncio.Event().wait()

        real_sleep = asyncio.sleep

        async def retry_delay(seconds):
            if boundary == "retry_delay" and len(calls) == billed_replies:
                await real_sleep(60)

        monkeypatch.setattr(llm_module, "query_async", provider)
        monkeypatch.setattr(llm_module.asyncio, "sleep", retry_delay)
        client = AsyncLLMClient(model_names="gpt-5-mini", verbose=False)
        task = asyncio.create_task(client.query("msg", "sys"))
        await asyncio.wait_for(reached_boundary.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled()
        assert isinstance(caught.value, BilledCancellationError)
        assert caught.value.cost == pytest.approx(0.5 * billed_replies)
        assert caught.value.cost_results["input_tokens"] == 1000 * billed_replies
        assert caught.value.cost_results["output_tokens"] == 20 * billed_replies
        assert caught.value.cost_results["num_total_queries"] == billed_replies
        assert len(calls) == billed_replies + (boundary == "next_request")

    asyncio.run(run())


def test_cancellation_without_received_usage_keeps_original_exception(monkeypatch):
    async def run():
        started = asyncio.Event()

        async def provider(**kwargs):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(llm_module, "query_async", provider)
        task = asyncio.create_task(
            AsyncLLMClient(model_names="gpt-5-mini", verbose=False).query("msg", "sys")
        )
        await asyncio.wait_for(started.wait(), 2)
        task.cancel("stop requested")
        with pytest.raises(asyncio.CancelledError, match="stop requested") as caught:
            await task
        assert not isinstance(caught.value, BilledCancellationError)

    asyncio.run(run())
