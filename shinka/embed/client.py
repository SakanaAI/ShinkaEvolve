from dataclasses import dataclass
import os
import re
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

from google import genai
import openai

from shinka.env import load_shinka_dotenv

from .providers.pricing import get_provider

load_shinka_dotenv()

TIMEOUT = 600
_LOCAL_MODEL_PATTERN = re.compile(r"^local/(?P<model>[^@]+)@(?P<url>https?://.+)$")
_OPENROUTER_PREFIX = "openrouter/"


@dataclass(frozen=True)
class ResolvedEmbeddingModel:
    original_model_name: str
    api_model_name: str
    provider: str
    base_url: Optional[str] = None


def resolve_embedding_backend(model_name: str) -> ResolvedEmbeddingModel:
    """Resolve runtime backend info for embedding model identifiers."""
    provider = get_provider(model_name)
    if provider == "azure":
        api_model_name = model_name.split("azure-", 1)[-1]
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider=provider,
            base_url=None,
        )
    if provider is not None:
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=model_name,
            provider=provider,
            base_url=None,
        )
    if model_name.startswith(_OPENROUTER_PREFIX):
        api_model_name = model_name.split(_OPENROUTER_PREFIX, 1)[-1]
        if not api_model_name:
            raise ValueError(
                "OpenRouter embedding model is missing after 'openrouter/'."
            )
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="openrouter",
            base_url=None,
        )

    local_match = _LOCAL_MODEL_PATTERN.match(model_name)
    if local_match:
        api_model_name = local_match.group("model")
        base_url = local_match.group("url")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"Invalid local model URL '{base_url}'. Expected http(s)://host[:port]/..."
            )
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="local_openai",
            base_url=base_url,
        )

    raise ValueError(
        f"Embedding model {model_name} not supported. "
        "Use a known pricing.csv model, 'openrouter/<model>', "
        "or 'local/<model>@http(s)://host[:port]/v1'."
    )


def get_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the client and model for the given embedding model name."""
    resolved = resolve_embedding_backend(model_name)
    provider = resolved.provider

    if provider == "openai":
        client = openai.OpenAI(timeout=TIMEOUT)
    elif provider == "azure":
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=TIMEOUT,
        )
    elif provider == "google":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider == "openrouter":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    elif provider == "local_openai":
        client = openai.OpenAI(
            api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
            base_url=resolved.base_url,
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, resolved.api_model_name


def get_async_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the async client and model for the given embedding model name."""
    resolved = resolve_embedding_backend(model_name)
    provider = resolved.provider

    if provider == "openai":
        client = openai.AsyncOpenAI()
    elif provider == "azure":
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif provider == "google":
        # Gemini doesn't have async client yet, will use thread pool in embedding.py
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    elif provider == "openrouter":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    elif provider == "local_openai":
        client = openai.AsyncOpenAI(
            api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
            base_url=resolved.base_url,
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, resolved.api_model_name
