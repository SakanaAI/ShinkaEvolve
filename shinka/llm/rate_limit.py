from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

from .providers.model_resolver import resolve_model_backend


def _provider_for_model(model: str) -> str:
    try:
        return resolve_model_backend(model).provider
    except ValueError:
        if model.startswith("gemini-"):
            return "google"
        raise


class DailyQuotaExceeded(RuntimeError):
    pass


@dataclass
class _LimitState:
    lock: asyncio.Lock
    semaphore: asyncio.Semaphore
    next_start_at: float = 0.0
    quota_date: date | None = None
    requests_today: int = 0


class AsyncProviderRateLimiter:
    """Shared provider/model/request-class limiter for one evolution run."""

    def __init__(
        self,
        *,
        limits: dict[str, dict[str, float]] | None = None,
        daily_quotas: dict[str, int] | None = None,
    ) -> None:
        self.limits = dict(limits or {})
        self.daily_quotas = dict(daily_quotas or {})
        self._states: dict[str, _LimitState] = {}

    def _matching_key(self, model: str, request_class: str) -> str | None:
        provider = _provider_for_model(model)
        candidates = (
            f"{model}:{request_class}",
            f"{provider}:{request_class}",
            model,
            provider,
        )
        for key in candidates:
            if key in self.limits or key in self.daily_quotas:
                return key
        return None

    def _state(self, key: str) -> _LimitState:
        if key not in self._states:
            max_concurrency = int(self.limits.get(key, {}).get("max_concurrency", 1))
            if max_concurrency < 1:
                raise ValueError(f"{key}.max_concurrency must be >= 1")
            self._states[key] = _LimitState(
                lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(max_concurrency),
            )
        return self._states[key]

    @asynccontextmanager
    async def limit(self, model: str, request_class: str):
        key = self._matching_key(model, request_class)
        if key is None:
            yield
            return

        state = self._state(key)
        async with state.semaphore:
            async with state.lock:
                today = date.today()
                if state.quota_date != today:
                    state.quota_date = today
                    state.requests_today = 0
                quota = self.daily_quotas.get(key)
                if quota is not None and state.requests_today >= int(quota):
                    raise DailyQuotaExceeded(
                        f"Daily request quota exhausted for {key}: {quota}"
                    )

                rpm = float(self.limits.get(key, {}).get("requests_per_minute", 0))
                if rpm < 0:
                    raise ValueError(f"{key}.requests_per_minute must be >= 0")
                if rpm:
                    wait_for = max(0.0, state.next_start_at - time.monotonic())
                    if wait_for:
                        await asyncio.sleep(wait_for)
                    state.next_start_at = time.monotonic() + 60.0 / rpm
                state.requests_today += 1
            yield


def estimate_minimum_request_demand(evo_config: Any) -> dict[str, int]:
    """Estimate unavoidable requests for quota-gated auxiliary services."""
    demand: dict[str, int] = {}
    target = int(getattr(evo_config, "num_generations", 0) or 0)

    meta_interval = getattr(evo_config, "meta_rec_interval", None)
    meta_models = list(getattr(evo_config, "meta_llm_models", None) or [])
    if meta_interval and meta_models and target:
        updates = target // int(meta_interval)
        # The current three-step summarizer makes one Step-1 request per
        # interval program, then one request each for Steps 2 and 3.
        requests = updates * (int(meta_interval) + 2)
        providers = {_provider_for_model(model) for model in meta_models}
        if len(providers) == 1:
            key = f"{next(iter(providers))}:meta"
            demand[key] = demand.get(key, 0) + requests

    embedding_model = getattr(evo_config, "embedding_model", None)
    if embedding_model and target:
        provider = _provider_for_model(embedding_model)
        demand[f"{provider}:embedding"] = target

    return demand


def validate_daily_quota_feasibility(evo_config: Any) -> dict[str, int]:
    demand = estimate_minimum_request_demand(evo_config)
    quotas = dict(getattr(evo_config, "llm_daily_quotas", {}) or {})
    infeasible = {
        key: (requests, int(quotas[key]))
        for key, requests in demand.items()
        if key in quotas and requests > int(quotas[key])
    }
    if infeasible:
        detail = ", ".join(
            f"{key}: needs at least {need}, quota {quota}"
            for key, (need, quota) in sorted(infeasible.items())
        )
        raise ValueError(f"Configured run exceeds known daily quota: {detail}")
    return demand
