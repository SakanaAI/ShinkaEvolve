from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class RouteHealthState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    last_failure_class: str | None = None


class RouteHealthCircuitBreaker:
    """Temporarily removes unhealthy transport routes from model sampling."""

    def __init__(self, *, failure_threshold: int = 3, cooldown_seconds: float = 900):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._states: dict[str, RouteHealthState] = {}

    def record_failure(self, route: str, failure_class: str) -> None:
        state = self._states.setdefault(route, RouteHealthState())
        state.consecutive_failures += 1
        state.last_failure_class = failure_class
        if state.consecutive_failures >= self.failure_threshold:
            state.open_until = time.monotonic() + self.cooldown_seconds

    def record_success(self, route: str) -> None:
        self._states[route] = RouteHealthState()

    def available_routes(self, routes: Iterable[str]) -> list[str]:
        now = time.monotonic()
        available = []
        for route in routes:
            state = self._states.get(route)
            if state is None or state.open_until <= now:
                if state is not None and state.open_until:
                    self._states[route] = RouteHealthState()
                available.append(route)
        return available

    def selectable_routes(self, routes: Iterable[str]) -> list[str]:
        routes = list(routes)
        available = self.available_routes(routes)
        if available or not routes:
            return available
        # Avoid a deadlock when every route is cooling down. Permit only the
        # route closest to recovery, rather than silently reopening all routes.
        return [min(routes, key=lambda route: self._states[route].open_until)]

    def snapshot(self) -> dict[str, dict[str, float | int | str | None]]:
        now = time.monotonic()
        return {
            route: {
                "consecutive_failures": state.consecutive_failures,
                "cooldown_remaining_seconds": max(0.0, state.open_until - now),
                "last_failure_class": state.last_failure_class,
            }
            for route, state in self._states.items()
        }
