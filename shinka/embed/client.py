from dataclasses import dataclass
import os
from typing import Any, Tuple

from google import genai
import openai

from shinka.env import load_shinka_dotenv

from .providers.pricing import get_provider

load_shinka_dotenv()

TIMEOUT = 600
_OPENROUTER_PREFIX = "openrouter/"


@dataclass(frozen=True)
class ResolvedEmbeddingModel:
    original_model_name: str
    api_model_name: str
    provider: str


def resolve_embedding_backend(model_name: str) -> ResolvedEmbeddingModel:
    """Resolve runtime backend info for embedding model identifiers."""
    provider = get_provider(model_name)
    if provider == "azure":
        api_model_name = model_name.split("azure-", 1)[-1]
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider=provider,
        )
    if provider is not None:
        return ResolvedEmbeddingModel(
            original_model_name=model_name,
            api_model_name=model_name,
            provider=provider,
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
        )
    raise ValueError(f"Embedding model {model_name} not supported.")


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
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, resolved.api_model_name
