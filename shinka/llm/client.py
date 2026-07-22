from typing import Any, Optional, Tuple
import asyncio
import hashlib
import os
import anthropic
import openai
import instructor
from shinka.azure_openai_config import azure_openai_api_key, azure_v1_base_url
from shinka.env import load_shinka_dotenv
from shinka.google_genai import (
    _google_genai_timeout_ms,
    build_google_genai_client,
    google_genai_auth_mode,
)
from shinka.local_openai_config import resolve_local_openai_api_key
from .constants import OPENAI_MAX_RETRIES, TIMEOUT
from .providers.errors import StructuredOutputNotSupportedError
from .providers.model_resolver import ResolvedModel, resolve_model_backend

load_shinka_dotenv()


# Constructing an SDK client (anthropic.Anthropic, openai.OpenAI, ...) spins up
# a fresh httpx connection pool and pays TLS + pool setup on first use. Building
# a brand-new client on every query — and every retry — throws that pool away
# immediately, so sequential queries never share HTTP keep-alive. We memoize the
# constructed clients so repeated queries with identical parameters reuse one
# client (and its pool).
#
# Sync clients live in a plain dict. Async clients (httpx.AsyncClient) bind to
# the event loop that first drives them, so their cache lives on that loop. This
# avoids a process-global cache retaining closed loops through client transports.
_SYNC_CLIENT_CACHE: dict = {}
_ASYNC_CLIENT_CACHE_ATTRIBUTE = "_shinka_async_client_cache"


def _fingerprint(secret: Optional[str]) -> Optional[str]:
    """Stable, non-reversible fingerprint of a secret for use in a cache key.

    Different secrets map to different fingerprints (so two callers with
    distinct keys never share a client), while the raw secret never enters the
    key itself.
    """
    if not secret:
        return None
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


def _has_stable_credentials(provider: str) -> bool:
    """Whether SDK credential discovery is stable enough to cache safely."""
    if provider == "anthropic":
        return bool(
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        )
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY"))
    if provider == "bedrock":
        static_credentials = os.getenv("AWS_ACCESS_KEY_ID") and os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        return bool(static_credentials or os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
    return True


def _client_identity_extras(provider: str, resolved: ResolvedModel) -> Tuple:
    """Dynamic, env-derived config that changes which client gets built.

    Everything here is beyond (provider, api_model_name, structured_output):
    the base URL and API key(s) a provider reads at construction time. Secrets
    are fingerprinted, not stored raw. Two calls that would build clients
    talking to different endpoints or authenticating with different keys land
    on different cache entries.
    """
    if provider == "anthropic":
        return (
            _fingerprint(os.getenv("ANTHROPIC_API_KEY")),
            _fingerprint(os.getenv("ANTHROPIC_AUTH_TOKEN")),
            os.getenv("ANTHROPIC_BASE_URL"),
            _fingerprint(os.getenv("ANTHROPIC_CUSTOM_HEADERS")),
        )
    if provider == "bedrock":
        return (
            _fingerprint(os.getenv("AWS_ACCESS_KEY_ID")),
            _fingerprint(os.getenv("AWS_SECRET_ACCESS_KEY")),
            _fingerprint(os.getenv("AWS_SESSION_TOKEN")),
            _fingerprint(os.getenv("AWS_BEARER_TOKEN_BEDROCK")),
            os.getenv("AWS_REGION_NAME"),
            os.getenv("AWS_REGION"),
            os.getenv("ANTHROPIC_BEDROCK_BASE_URL"),
        )
    if provider == "openai":
        return (
            _fingerprint(os.getenv("OPENAI_API_KEY")),
            _fingerprint(os.getenv("OPENAI_ADMIN_KEY")),
            os.getenv("OPENAI_BASE_URL"),
            os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION"),
            os.getenv("OPENAI_PROJECT_ID"),
            _fingerprint(os.getenv("OPENAI_CUSTOM_HEADERS")),
        )
    if provider == "azure_openai":
        # Azure endpoint + key come from the environment, not from `resolved`.
        return (azure_v1_base_url(), _fingerprint(azure_openai_api_key()))
    if provider == "deepseek":
        return (_fingerprint(os.getenv("DEEPSEEK_API_KEY")),)
    if provider == "google":
        auth_mode = google_genai_auth_mode()
        if auth_mode == "vertexai":
            return (
                auth_mode,
                os.getenv("GOOGLE_CLOUD_PROJECT"),
                os.getenv("GOOGLE_CLOUD_LOCATION"),
            )
        return (
            auth_mode,
            _fingerprint(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        )
    if provider == "openrouter":
        return (_fingerprint(os.getenv("OPENROUTER_API_KEY")),)
    if provider == "local_openai":
        return (
            resolved.base_url,
            resolved.api_key_env_name,
            _fingerprint(resolve_local_openai_api_key(resolved.api_key_env_name)),
        )
    return ()


def _sync_client_builder(provider: str) -> Any:
    """The constructor a sync client is built from, resolved at call time.

    Read live from the module globals so a monkeypatched constructor (e.g. a
    test swapping ``openai.OpenAI``) is a different object and never collides in
    the cache with the real one.
    """
    return {
        "anthropic": anthropic.Anthropic,
        "bedrock": anthropic.AnthropicBedrock,
        "openai": openai.OpenAI,
        "azure_openai": openai.OpenAI,
        "deepseek": openai.OpenAI,
        "openrouter": openai.OpenAI,
        "local_openai": openai.OpenAI,
        "google": build_google_genai_client,
    }.get(provider)


def _async_client_builder(provider: str) -> Any:
    """The constructor an async client is built from, resolved at call time."""
    return {
        "anthropic": anthropic.AsyncAnthropic,
        "bedrock": anthropic.AsyncAnthropicBedrock,
        "openai": openai.AsyncOpenAI,
        "azure_openai": openai.AsyncOpenAI,
        "deepseek": openai.AsyncOpenAI,
        "openrouter": openai.AsyncOpenAI,
        "local_openai": openai.AsyncOpenAI,
        "google": build_google_genai_client,
    }.get(provider)


def _client_cache_key(
    provider: str,
    structured_output: bool,
    resolved: ResolvedModel,
    builder: Any,
) -> Optional[Tuple]:
    """Key that uniquely identifies a client's construction parameters.

    Distinct clients are forced apart by: provider (openai vs azure share a
    constructor but hit different endpoints), api_model_name, the
    structured_output flag (instructor-wrapped clients must never be handed to
    plain callers), the actual constructor object (guards against monkeypatched
    builders), and the env-derived base URL / key fingerprints.
    """
    if not _has_stable_credentials(provider):
        return None
    return (
        provider,
        resolved.api_model_name,
        bool(structured_output),
        builder,
        _client_identity_extras(provider, resolved),
    )


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _build_sync_client(
    provider: str, structured_output: bool, resolved: ResolvedModel
) -> Any:
    """Construct a fresh sync client for the resolved provider."""
    if provider == "anthropic":
        client = anthropic.Anthropic(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "bedrock":
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "openai":
        client = openai.OpenAI(timeout=TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "azure_openai":
        client = openai.OpenAI(
            api_key=azure_openai_api_key(),
            base_url=azure_v1_base_url(),
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "deepseek":
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "google":
        client = build_google_genai_client(timeout_ms=_google_genai_timeout_ms(TIMEOUT))
        if structured_output:
            client = instructor.from_openai(
                client,
                mode=instructor.Mode.GEMINI_JSON,
            )
    elif provider == "openrouter":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "local_openai":
        client = openai.OpenAI(
            api_key=resolve_local_openai_api_key(resolved.api_key_env_name),
            base_url=resolved.base_url,
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
    elif provider == "headless":
        client = None
    else:
        raise ValueError(f"Model {resolved.original_model_name} not supported.")

    return client


def _build_async_client(
    provider: str, structured_output: bool, resolved: ResolvedModel
) -> Any:
    """Construct a fresh async client for the resolved provider."""
    if provider == "anthropic":
        client = anthropic.AsyncAnthropic(timeout=TIMEOUT)
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "bedrock":
        client = anthropic.AsyncAnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "openai":
        client = openai.AsyncOpenAI(timeout=TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "azure_openai":
        client = openai.AsyncOpenAI(
            api_key=azure_openai_api_key(),
            base_url=azure_v1_base_url(),
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "deepseek":
        client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "google":
        client = build_google_genai_client(timeout_ms=_google_genai_timeout_ms(TIMEOUT))
        if structured_output:
            raise StructuredOutputNotSupportedError(
                "Gemini does not support structured output."
            )
    elif provider == "openrouter":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "local_openai":
        client = openai.AsyncOpenAI(
            api_key=resolve_local_openai_api_key(resolved.api_key_env_name),
            base_url=resolved.base_url,
            timeout=TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
        )
    elif provider == "headless":
        client = None
    else:
        raise ValueError(f"Model {resolved.original_model_name} not supported.")

    return client


def get_client_llm(
    model_name: str, structured_output: bool = False
) -> Tuple[Any, str, str]:
    """Get the client and model for the given model name.

    Clients are memoized: repeated calls with identical parameters reuse one
    client (and its connection pool) instead of rebuilding it per query.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str, str]: (client, API model name, resolved provider).
    """
    resolved = resolve_model_backend(model_name)
    provider = resolved.provider
    builder = _sync_client_builder(provider)
    cache_key = _client_cache_key(provider, structured_output, resolved, builder)
    if cache_key is None:
        client = _build_sync_client(provider, structured_output, resolved)
        return client, resolved.api_model_name, provider
    if cache_key not in _SYNC_CLIENT_CACHE:
        _SYNC_CLIENT_CACHE[cache_key] = _build_sync_client(
            provider, structured_output, resolved
        )
    return _SYNC_CLIENT_CACHE[cache_key], resolved.api_model_name, provider


def get_async_client_llm(
    model_name: str, structured_output: bool = False
) -> Tuple[Any, str, str]:
    """Get the async client and model for the given model name.

    Async clients are memoized per running event loop, so repeated queries on
    the same loop reuse one client (and its connection pool), while a client is
    never shared across event loops.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str, str]: (async client, API model name, resolved provider).
    """
    resolved = resolve_model_backend(model_name)
    provider = resolved.provider
    builder = _async_client_builder(provider)
    cache_key = _client_cache_key(provider, structured_output, resolved, builder)
    loop = _running_loop()
    if loop is None or cache_key is None:
        client = _build_async_client(provider, structured_output, resolved)
        return client, resolved.api_model_name, provider

    per_loop = getattr(loop, _ASYNC_CLIENT_CACHE_ATTRIBUTE, None)
    if per_loop is None:
        per_loop = {}
        setattr(loop, _ASYNC_CLIENT_CACHE_ATTRIBUTE, per_loop)
    if cache_key not in per_loop:
        per_loop[cache_key] = _build_async_client(
            provider, structured_output, resolved
        )
    return per_loop[cache_key], resolved.api_model_name, provider
