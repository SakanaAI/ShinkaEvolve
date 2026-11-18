"""Shared retry policy: fail fast on HTTP statuses that cannot succeed."""

import functools
import inspect
from typing import Optional

from .errors import LLMEndpointError, NonRetryableLLMError

# 4xx statuses that describe a transient condition and are worth retrying.
RETRYABLE_CLIENT_STATUS_CODES = frozenset({408, 409, 429})
# Statuses that mean the endpoint, credentials or model are wrong for every request.
ENDPOINT_FAILURE_STATUS_CODES = frozenset({401, 403, 404})


def _status_code(exc: Exception) -> Optional[int]:
    # OpenAI and Anthropic expose `status_code`; google-genai exposes `code`.
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _server_retry_directive(exc: Exception) -> Optional[bool]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    directive = headers.get("x-should-retry") if headers is not None else None
    return {"true": True, "false": False}.get(str(directive))


def _error_code(exc: Exception) -> str:
    for attribute in ("code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value:
            return value
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    if isinstance(error_type, str) and error_type:
        return error_type
    return type(exc).__name__


def is_permanent_status_error(exc: Exception) -> bool:
    """Whether a provider status error would fail again on every retry."""
    status_code = _status_code(exc)
    if status_code is None:
        return False
    directive = _server_retry_directive(exc)
    if directive is not None:
        return not directive
    return 400 <= status_code < 500 and status_code not in RETRYABLE_CLIENT_STATUS_CODES


def permanent_status_error(
    exc: Exception, provider: str
) -> Optional[NonRetryableLLMError]:
    """Return the non-retryable form of a status error, or None to keep retrying."""
    if not is_permanent_status_error(exc):
        return None
    status_code = _status_code(exc)
    assert status_code is not None
    code = _error_code(exc)
    if status_code in ENDPOINT_FAILURE_STATUS_CODES:
        # Do not persist headers, credentials, request URLs or raw response bodies.
        return LLMEndpointError(status_code, code)
    return NonRetryableLLMError(
        f"{provider} rejected the request with HTTP {status_code} ({code}): {exc}"
    )


def fail_fast_on_permanent_status(
    provider: str, *status_error_types: type[Exception]
):
    """Raise `NonRetryableLLMError` instead of a status error that cannot succeed.

    Apply it beneath `backoff.on_exception`: the converted error is outside the
    retried types, so it leaves the backoff loop on the first attempt.
    """

    def decorate(query_fn):
        if inspect.iscoroutinefunction(query_fn):

            @functools.wraps(query_fn)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await query_fn(*args, **kwargs)
                except status_error_types as exc:
                    permanent = permanent_status_error(exc, provider)
                    if permanent is None:
                        raise
                    raise permanent from exc

            return async_wrapper

        @functools.wraps(query_fn)
        def wrapper(*args, **kwargs):
            try:
                return query_fn(*args, **kwargs)
            except status_error_types as exc:
                permanent = permanent_status_error(exc, provider)
                if permanent is None:
                    raise
                raise permanent from exc

        return wrapper

    return decorate
