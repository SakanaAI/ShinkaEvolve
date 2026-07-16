from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from shinka.llm.rate_limit import (
    AsyncProviderRateLimiter,
    DailyQuotaExceeded,
    estimate_minimum_request_demand,
    validate_daily_quota_feasibility,
)
from shinka.llm.route_health import RouteHealthCircuitBreaker
from shinka.llm.llm import AsyncLLMClient
import shinka.llm.llm as llm_module


def test_route_health_circuit_breaker_is_separate_from_quality_selection():
    breaker = RouteHealthCircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    routes = ["headless/cursor@test", "headless/antigravity@test"]

    breaker.record_failure(routes[0], "authentication_error")
    assert breaker.available_routes(routes) == routes
    breaker.record_failure(routes[0], "authentication_error")

    assert breaker.available_routes(routes) == [routes[1]]
    assert breaker.snapshot()[routes[0]]["last_failure_class"] == "authentication_error"
    breaker.record_success(routes[0])
    assert breaker.available_routes(routes) == routes


def test_provider_rate_limit_is_scoped_by_request_class():
    limiter = AsyncProviderRateLimiter(
        limits={"google:meta": {"requests_per_minute": 600, "max_concurrency": 1}}
    )

    async def exercise():
        starts = []

        async def meta_call():
            async with limiter.limit("gemini-3-flash", "meta"):
                starts.append(time.monotonic())

        await asyncio.gather(meta_call(), meta_call())
        mutation_started = time.monotonic()
        async with limiter.limit("gemini-3-flash", "mutation"):
            mutation_delay = time.monotonic() - mutation_started
        return starts, mutation_delay

    starts, mutation_delay = asyncio.run(exercise())
    assert starts[1] - starts[0] >= 0.09
    assert mutation_delay < 0.05


def test_daily_quota_is_enforced():
    limiter = AsyncProviderRateLimiter(daily_quotas={"google:meta": 1})

    async def exercise():
        async with limiter.limit("gemini-3-flash", "meta"):
            pass
        with pytest.raises(DailyQuotaExceeded):
            async with limiter.limit("gemini-3-flash", "meta"):
                pass

    asyncio.run(exercise())


def test_meta_daily_demand_is_rejected_before_launch():
    config = SimpleNamespace(
        num_generations=150,
        meta_rec_interval=10,
        meta_llm_models=["gemini-3.5-flash"],
        embedding_model=None,
        llm_daily_quotas={"google:meta": 20},
    )

    assert estimate_minimum_request_demand(config)["google:meta"] == 180
    with pytest.raises(ValueError, match="needs at least 180, quota 20"):
        validate_daily_quota_feasibility(config)


def test_async_batch_kwargs_query_forwards_message_history(monkeypatch):
    captured = []

    async def fake_query_async(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(cost=0.0, content="ok")

    monkeypatch.setattr(llm_module, "query_async", fake_query_async)
    client = AsyncLLMClient(model_names=["gpt-5.1"], verbose=False)
    history = [{"role": "assistant", "content": "prior"}]

    responses = asyncio.run(
        client.batch_kwargs_query(
            num_samples=1,
            msg="next",
            system_msg="system",
            msg_history=[history],
        )
    )

    assert len(responses) == 1
    assert captured[0]["msg_history"] == history
