from __future__ import annotations

import asyncio
import atexit
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import anthropic
import instructor
import openai

from shinka.azure_openai_config import azure_openai_api_key, azure_v1_base_url
from shinka.env import load_shinka_dotenv
from shinka.google_genai import (
    _google_genai_timeout_ms,
    build_google_genai_client,
    configure_google_genai_network,
    google_genai_auth_mode,
)
from shinka.local_openai_config import resolve_local_openai_api_key

from .constants import OPENAI_MAX_RETRIES, TIMEOUT
from .providers.errors import StructuredOutputNotSupportedError
from .providers.model_resolver import ResolvedModel, resolve_model_backend

load_shinka_dotenv()

logger = logging.getLogger(__name__)

_ASYNC_CACHE_ATTRIBUTE = "_shinka_async_client_cache"
_ASYNC_CACHE_USERS_ATTRIBUTE = "_shinka_async_client_cache_users"
_TRANSPORT_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SHINKA_GOOGLE_GENAI_IP_FAMILY",
    "SHINKA_GOOGLE_GENAI_IP_PROBE_TIMEOUT",
)


@dataclass(frozen=True)
class _ClientSpec:
    provider: str
    structured_output: bool
    constructor: Any
    constructor_kwargs: tuple[tuple[str, Any], ...]
    ambient_environment: tuple[str | None, ...]
    cacheable: bool


ClientBuilder = Callable[[_ClientSpec], Any]
ClientCacheKey = tuple[str, bool, Any, str]

_SYNC_CLIENT_CACHE: dict[ClientCacheKey, Any] = {}
_SYNC_CLIENT_LOCK = threading.Lock()


def _environment_values(*names: str) -> tuple[str | None, ...]:
    return tuple(os.getenv(name) for name in names)


def _transport_environment() -> tuple[str | None, ...]:
    return _environment_values(*_TRANSPORT_ENVIRONMENT_VARIABLES)


def _ambient_environment(provider: str) -> tuple[str | None, ...]:
    provider_variables = {
        "anthropic": ("ANTHROPIC_CUSTOM_HEADERS",),
        "bedrock": ("AWS_BEARER_TOKEN_BEDROCK",),
        "openai": ("OPENAI_CUSTOM_HEADERS",),
        "azure_openai": ("OPENAI_CUSTOM_HEADERS",),
        "deepseek": ("OPENAI_CUSTOM_HEADERS",),
        "openrouter": ("OPENAI_CUSTOM_HEADERS",),
        "local_openai": ("OPENAI_CUSTOM_HEADERS",),
    }.get(provider, ())
    return _transport_environment() + _environment_values(*provider_variables)


def _custom_headers(raw_headers: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in (raw_headers or "").splitlines():
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip()] = value.strip()
    return headers


def _openai_constructor_kwargs(api_key: str, base_url: str) -> dict[str, Any]:
    return {
        "api_key": api_key,
        "admin_api_key": os.getenv("OPENAI_ADMIN_KEY") or "",
        "organization": os.getenv("OPENAI_ORG_ID") or "",
        "project": os.getenv("OPENAI_PROJECT_ID") or "",
        "webhook_secret": os.getenv("OPENAI_WEBHOOK_SECRET") or "",
        "base_url": base_url,
        "default_headers": _custom_headers(os.getenv("OPENAI_CUSTOM_HEADERS")),
        "timeout": TIMEOUT,
        "max_retries": OPENAI_MAX_RETRIES,
    }


def _identity_digest(spec: _ClientSpec) -> str:
    values = (spec.constructor_kwargs, spec.ambient_environment)
    serialized = json.dumps(values, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sync_constructors() -> dict[str, Any]:
    return {
        "anthropic": anthropic.Anthropic,
        "bedrock": anthropic.AnthropicBedrock,
        "openai": openai.OpenAI,
        "azure_openai": openai.OpenAI,
        "deepseek": openai.OpenAI,
        "openrouter": openai.OpenAI,
        "local_openai": openai.OpenAI,
        "google": build_google_genai_client,
    }


def _async_constructors() -> dict[str, Any]:
    return {
        "anthropic": anthropic.AsyncAnthropic,
        "bedrock": anthropic.AsyncAnthropicBedrock,
        "openai": openai.AsyncOpenAI,
        "azure_openai": openai.AsyncOpenAI,
        "deepseek": openai.AsyncOpenAI,
        "openrouter": openai.AsyncOpenAI,
        "local_openai": openai.AsyncOpenAI,
        "google": build_google_genai_client,
    }


def _resolve_client_spec(
    provider: str,
    structured_output: bool,
    resolved: ResolvedModel,
    constructors: dict[str, Any],
) -> _ClientSpec:
    if provider == "google" and structured_output:
        raise StructuredOutputNotSupportedError(
            "Gemini does not support structured output."
        )
    constructor = constructors.get(provider)
    if provider != "headless" and constructor is None:
        raise ValueError(f"Model {resolved.original_model_name} not supported.")
    kwargs, cacheable = _constructor_configuration(provider, resolved)
    return _ClientSpec(
        provider=provider,
        structured_output=structured_output,
        constructor=constructor,
        constructor_kwargs=tuple(kwargs.items()),
        ambient_environment=_ambient_environment(provider),
        cacheable=cacheable,
    )


def _constructor_configuration(
    provider: str, resolved: ResolvedModel
) -> tuple[dict[str, Any], bool]:
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        if not api_key and not auth_token:
            return {"timeout": TIMEOUT}, False
        return (
            {
                "api_key": api_key,
                "auth_token": auth_token,
                "base_url": os.getenv("ANTHROPIC_BASE_URL")
                or "https://api.anthropic.com",
                "webhook_key": os.getenv("ANTHROPIC_WEBHOOK_SIGNING_KEY") or "",
                "default_headers": _custom_headers(
                    os.getenv("ANTHROPIC_CUSTOM_HEADERS")
                ),
                "timeout": TIMEOUT,
            },
            True,
        )

    if provider == "bedrock":
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_REGION")
        base_url = os.getenv("ANTHROPIC_BEDROCK_BASE_URL")
        resolved_base_url = base_url or (
            f"https://bedrock-runtime.{region}.amazonaws.com" if region else None
        )
        stable_auth = bool(bearer_token or (access_key and secret_key))
        return (
            {
                "aws_access_key": access_key,
                "aws_secret_key": secret_key,
                "aws_session_token": session_token,
                "aws_region": region,
                "api_key": bearer_token,
                "base_url": resolved_base_url,
                "timeout": TIMEOUT,
            },
            stable_auth and bool(region),
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        admin_key = os.getenv("OPENAI_ADMIN_KEY")
        if not api_key and not admin_key:
            return {"timeout": TIMEOUT, "max_retries": OPENAI_MAX_RETRIES}, False
        kwargs = _openai_constructor_kwargs(
            api_key or "", os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        kwargs["admin_api_key"] = admin_key or ""
        return kwargs, True

    if provider == "azure_openai":
        return (
            _openai_constructor_kwargs(
                azure_openai_api_key(), azure_v1_base_url()
            ),
            True,
        )

    if provider == "deepseek":
        return (
            _openai_constructor_kwargs(
                os.environ["DEEPSEEK_API_KEY"], "https://api.deepseek.com"
            ),
            True,
        )

    if provider == "google":
        auth_mode = google_genai_auth_mode()
        if auth_mode == "vertexai":
            return (
                {
                    "timeout_ms": _google_genai_timeout_ms(TIMEOUT),
                    "auth_mode": auth_mode,
                    "project": os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(),
                    "location": os.getenv("GOOGLE_CLOUD_LOCATION", "").strip(),
                },
                False,
            )
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        return (
            {
                "timeout_ms": _google_genai_timeout_ms(TIMEOUT),
                "auth_mode": auth_mode,
                "api_key": api_key,
            },
            bool(api_key),
        )

    if provider == "openrouter":
        return (
            _openai_constructor_kwargs(
                os.environ["OPENROUTER_API_KEY"], "https://openrouter.ai/api/v1"
            ),
            True,
        )

    if provider == "local_openai":
        return (
            _openai_constructor_kwargs(
                resolve_local_openai_api_key(resolved.api_key_env_name),
                resolved.base_url or "",
            ),
            True,
        )

    return {}, False


def _client_cache_key(
    spec: _ClientSpec, builder: ClientBuilder
) -> ClientCacheKey | None:
    if not spec.cacheable:
        return None
    builder_identity = builder, spec.constructor
    return spec.provider, spec.structured_output, builder_identity, _identity_digest(spec)


def _store_cached_client(
    cache: dict[ClientCacheKey, Any], key: ClientCacheKey, client: Any
) -> None:
    for existing_key in tuple(cache):
        if existing_key[:2] == key[:2]:
            cache.pop(existing_key)
    cache[key] = client


def _build_sync_client(spec: _ClientSpec) -> Any:
    return _build_client(spec)


def _build_async_client(spec: _ClientSpec) -> Any:
    return _build_client(spec)


def _build_client(spec: _ClientSpec) -> Any:
    if spec.provider == "headless":
        return None
    client = spec.constructor(**dict(spec.constructor_kwargs))
    if not spec.structured_output:
        return client
    return _wrap_structured_client(spec.provider, client)


def _wrap_structured_client(provider: str, client: Any) -> Any:
    if provider in {"anthropic", "bedrock"}:
        return instructor.from_anthropic(
            client, mode=instructor.mode.Mode.ANTHROPIC_JSON
        )
    if provider in {"openai", "azure_openai"}:
        return instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    if provider in {"deepseek", "openrouter"}:
        return instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    return client


def _prepare_provider_runtime(provider: str) -> None:
    if provider == "google":
        configure_google_genai_network()


def get_client_llm(
    model_name: str, structured_output: bool = False
) -> tuple[Any, str, str]:
    """Return a process-local synchronous client and resolved model metadata."""
    resolved = resolve_model_backend(model_name)
    spec = _resolve_client_spec(
        resolved.provider, structured_output, resolved, _sync_constructors()
    )
    cache_key = _client_cache_key(spec, _build_sync_client)
    if cache_key is None:
        client = _build_sync_client(spec)
    else:
        cache_hit = False
        with _SYNC_CLIENT_LOCK:
            client = _SYNC_CLIENT_CACHE.get(cache_key)
            if client is None:
                client = _build_sync_client(spec)
                if spec.ambient_environment == _ambient_environment(spec.provider):
                    _store_cached_client(_SYNC_CLIENT_CACHE, cache_key, client)
            else:
                cache_hit = True
        if cache_hit:
            _prepare_provider_runtime(spec.provider)
    return client, resolved.api_model_name, resolved.provider


def get_async_client_llm(
    model_name: str, structured_output: bool = False
) -> tuple[Any, str, str]:
    """Return an event-loop-local asynchronous client and model metadata."""
    resolved = resolve_model_backend(model_name)
    spec = _resolve_client_spec(
        resolved.provider, structured_output, resolved, _async_constructors()
    )
    cache_key = _client_cache_key(spec, _build_async_client)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if cache_key is None or loop is None:
        client = _build_async_client(spec)
    else:
        cache: dict[ClientCacheKey, Any] | None = getattr(
            loop, _ASYNC_CACHE_ATTRIBUTE, None
        )
        if cache is None:
            cache = {}
            setattr(loop, _ASYNC_CACHE_ATTRIBUTE, cache)
        client = cache.get(cache_key)
        if client is None:
            client = _build_async_client(spec)
            if spec.ambient_environment == _ambient_environment(spec.provider):
                _store_cached_client(cache, cache_key, client)
        else:
            _prepare_provider_runtime(spec.provider)
    return client, resolved.api_model_name, resolved.provider


def _underlying_transport(client: Any) -> Any:
    if isinstance(client, (instructor.Instructor, instructor.AsyncInstructor)):
        return client.client
    return client


def _unique_transports(clients: list[Any]) -> list[Any]:
    transports: list[Any] = []
    seen: set[int] = set()
    for client in clients:
        transport = _underlying_transport(client)
        if transport is None or id(transport) in seen:
            continue
        seen.add(id(transport))
        transports.append(transport)
    return transports


def _log_close_failure(kind: str, transport: Any, error: Exception) -> None:
    logger.warning(
        "Failed to close cached %s LLM client transport=%s error=%s",
        kind,
        type(transport).__name__,
        type(error).__name__,
    )


def close_sync_client_cache() -> None:
    """Close and remove every cached synchronous provider transport."""
    with _SYNC_CLIENT_LOCK:
        clients = list(_SYNC_CLIENT_CACHE.values())
        _SYNC_CLIENT_CACHE.clear()
    for transport in _unique_transports(clients):
        close = getattr(transport, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception as error:  # noqa: BLE001
            _log_close_failure("sync", transport, error)


async def _close_async_view(transport: Any) -> None:
    async_view = getattr(transport, "aio", None)
    async_close = getattr(async_view, "aclose", None)
    if async_close is None:
        return
    try:
        await async_close()
    except Exception as error:  # noqa: BLE001
        _log_close_failure("async", transport, error)


async def _close_transport(transport: Any) -> None:
    close = getattr(transport, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as error:  # noqa: BLE001
        _log_close_failure("async", transport, error)


async def close_async_client_cache() -> None:
    """Close and remove cached clients belonging to the current event loop."""
    loop = asyncio.get_running_loop()
    cache: dict[ClientCacheKey, Any] | None = getattr(
        loop, _ASYNC_CACHE_ATTRIBUTE, None
    )
    if not cache:
        return
    clients = list(cache.values())
    cache.clear()
    for transport in _unique_transports(clients):
        await _close_async_view(transport)
        await _close_transport(transport)


@asynccontextmanager
async def async_client_cache_scope() -> AsyncIterator[None]:
    """Retain the current loop cache until its last overlapping user exits."""
    loop = asyncio.get_running_loop()
    users = int(getattr(loop, _ASYNC_CACHE_USERS_ATTRIBUTE, 0)) + 1
    setattr(loop, _ASYNC_CACHE_USERS_ATTRIBUTE, users)
    try:
        yield
    finally:
        remaining_users = int(getattr(loop, _ASYNC_CACHE_USERS_ATTRIBUTE, 1)) - 1
        setattr(loop, _ASYNC_CACHE_USERS_ATTRIBUTE, remaining_users)
        if remaining_users == 0:
            await close_async_client_cache()


def _reset_sync_cache_after_fork() -> None:
    global _SYNC_CLIENT_CACHE, _SYNC_CLIENT_LOCK
    _SYNC_CLIENT_CACHE = {}
    _SYNC_CLIENT_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_sync_cache_after_fork)
atexit.register(close_sync_client_cache)
