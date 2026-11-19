"""Shared retry policy: fail fast on HTTP statuses that cannot succeed."""

import functools
import inspect
from typing import Any, List, Optional

from .errors import LLMEndpointError, NonRetryableLLMError, RequestRejectedError

# 4xx statuses that describe a transient condition and are worth retrying:
# timeout, conflict, too early, rate limit, and a request cancelled in transit
# (499, which Google returns for CANCELLED and proxies for a closed connection).
RETRYABLE_CLIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 499})
# Statuses that mean the endpoint, credentials, model or account are wrong for
# every request.
ENDPOINT_FAILURE_STATUS_CODES = frozenset({401, 402, 403, 404})
# Provider error codes that mean the same but arrive under a status shared with
# per-request errors: Google reports a bad key as 400, OpenAI an exhausted
# quota as 429.
ACCOUNT_FAILURE_CODES = frozenset(
    {"API_KEY_INVALID", "invalid_api_key", "insufficient_quota"}
)
# Hosts whose 403 refuses one request, not the endpoint: OpenRouter answers
# 403 when its moderation flags the input.
REQUEST_SCOPED_403_HOSTS = frozenset({"openrouter.ai"})


# The endpoint chooses these strings and they are stored in run artifacts.
MAX_ERROR_CODE_LENGTH = 120
MAX_ERROR_DETAIL_LENGTH = 1000


def _clipped(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


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


def _nested(value: Any, *keys: str) -> Any:
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _error_codes(exc: Exception) -> List[str]:
    """Every machine-readable code a provider attached to a status error."""
    google_details = _nested(getattr(exc, "details", None), "error", "details")
    candidates = [
        getattr(exc, "code", None),  # OpenAI and compatible servers
        *(
            _nested(detail, "reason")  # google.rpc.ErrorInfo
            for detail in (google_details if isinstance(google_details, list) else [])
        ),
        getattr(exc, "status", None),  # Google
        _nested(getattr(exc, "body", None), "error", "type"),  # Anthropic
    ]
    return [code for code in candidates if isinstance(code, str) and code]


def _request_host(exc: Exception) -> Optional[str]:
    url = getattr(getattr(exc, "request", None), "url", None)
    return getattr(url, "host", None)


def _is_endpoint_failure(exc: Exception) -> bool:
    if any(code in ACCOUNT_FAILURE_CODES for code in _error_codes(exc)):
        return True
    status_code = _status_code(exc)
    if status_code == 403 and _request_host(exc) in REQUEST_SCOPED_403_HOSTS:
        return False
    return status_code in ENDPOINT_FAILURE_STATUS_CODES


def _error_code(exc: Exception) -> str:
    codes = _error_codes(exc)
    account_codes = [code for code in codes if code in ACCOUNT_FAILURE_CODES]
    return (account_codes or codes or [type(exc).__name__])[0]


def is_permanent_status_error(exc: Exception) -> bool:
    """Whether a provider status error would fail again on every retry."""
    status_code = _status_code(exc)
    if status_code is None:
        return False
    if _is_endpoint_failure(exc):
        return True
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
    code = _clipped(_error_code(exc), MAX_ERROR_CODE_LENGTH)
    if _is_endpoint_failure(exc):
        # Do not persist headers, credentials, request URLs or raw response bodies.
        return LLMEndpointError(status_code, code)
    detail = _clipped(str(exc), MAX_ERROR_DETAIL_LENGTH)
    return RequestRejectedError(
        f"{provider} rejected the request with HTTP {status_code} ({code}): {detail}"
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
