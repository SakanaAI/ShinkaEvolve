"""Request admission and completed charges for async evolution stages."""

import asyncio
import math
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, TypeVar

from shinka.local_openai_config import parse_local_openai_model
from shinka.pricing.catalog import ModelKind, get_catalog

_T = TypeVar("_T")
api_cost_lock = threading.Lock()


class APIBudgetExceeded(RuntimeError):
    def __init__(self) -> None:
        super().__init__("API cost budget exhausted before request")


@dataclass(eq=False)
class ApiCostScope:
    """Charges received by a stage before it posts them to the runner total."""

    exhausted: Callable[[], bool] | None
    pending: float = 0.0
    admission_closed: Callable[[], bool] = lambda: False

    def acknowledge(self, cost: float) -> None:
        if math.isclose(self.pending, cost, rel_tol=1e-12):
            self.pending = 0.0
        else:
            self.pending = max(0.0, self.pending - cost)

    def record(self, cost: float) -> None:
        with api_cost_lock:
            self.pending += cost


cost_scope: ContextVar[ApiCostScope | None] = ContextVar("api_cost_scope", default=None)


def may_charge(model: str, kind: ModelKind = "llm") -> bool:
    if model.startswith("openrouter/") and model.endswith(":free"):
        return False
    local_model = parse_local_openai_model(model) if model.startswith("local/") else None
    if local_model is not None and kind == "llm":
        model = local_model.api_model_name
    catalog = get_catalog().catalog
    try:
        price = catalog.get_by_name(model, kind=kind)
    except KeyError:
        try:
            price = catalog.find_by_api_name(model, kind=kind)
        except KeyError:
            return local_model is None
    if kind == "embedding":
        return price.input_price != 0.0
    if local_model is not None and (price.input_price is None or price.output_price is None):
        return False
    rates = (*price.rates(), *price.rates(math.inf))
    return any(rate != 0.0 for rate in rates)


@dataclass
class _PaidRequest:
    exhausted: Callable[[], bool]
    cancelled: bool = False


_paid_request: ContextVar[_PaidRequest | None] = ContextVar(
    "paid_request", default=None
)


def check_provider_attempt() -> bool:
    """Recheck admission before a transport attempt, including provider retries."""
    request = _paid_request.get()
    if request is None:
        return False
    if request.cancelled or request.exhausted():
        raise APIBudgetExceeded()
    return True


async def run_request(
    operation: Callable[[], Awaitable[_T]],
    get_cost: Callable[[_T], float],
    model: str,
    kind: ModelKind = "llm",
    *,
    cancellation: Literal["propagate", "drain"] = "propagate",
) -> _T:
    scope = cost_scope.get()
    if scope is None:
        return await operation()

    async def receive() -> _T:
        check_provider_attempt()
        try:
            result = await operation()
        except (Exception, asyncio.CancelledError) as exc:
            scope.record(getattr(exc, "cost", 0.0))
            raise
        scope.record(get_cost(result))
        return result

    exhausted = scope.exhausted
    request = None
    if may_charge(model, kind):
        def blocked():
            return scope.admission_closed() or (exhausted is not None and exhausted())

        if blocked():
            raise APIBudgetExceeded()
        request = _PaidRequest(blocked)

    if cancellation == "drain" or (request is not None and exhausted is not None):
        return await _receive_shielded(receive, request)
    token = _paid_request.set(request)
    try:
        return await receive()
    finally:
        _paid_request.reset(token)


async def _receive_shielded(
    receive: Callable[[], Awaitable[_T]], request: _PaidRequest | None
) -> _T:
    token = _paid_request.set(request)
    try:
        task = asyncio.ensure_future(receive())
    finally:
        _paid_request.reset(token)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Keep the started request alive to receive its bill. Provider retry
        # admission closes as soon as its caller cancels.
        if request is not None:
            request.cancelled = True
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # The scope already captured the charge, even if the response was lost
        # to cancellation. Retrieve the outcome to avoid an unhandled exception.
        if not task.cancelled():
            task.exception()
        raise
