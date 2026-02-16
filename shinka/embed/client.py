from typing import Any, Tuple
import os
import openai
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from .providers.pricing import get_provider

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the client and model for the given embedding model name.

    Args:
        model_name (str): The name of the embedding model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str]: The client and model name for the given model.
    """
    provider = get_provider(model_name)

    if provider == "openai":
        client = openai.OpenAI(timeout=600)
    elif provider == "azure":
        # Strip azure- prefix from model name
        model_name = model_name.split("azure-")[-1]
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            timeout=600,
        )
    elif provider == "google":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, model_name


def get_async_client_embed(model_name: str) -> Tuple[Any, str]:
    """Get the async client and model for the given embedding model name.

    Args:
        model_name (str): The name of the embedding model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str]: The async client and model name for the given model.
    """
    provider = get_provider(model_name)

    if provider == "openai":
        client = openai.AsyncOpenAI()
    elif provider == "azure":
        # Strip azure- prefix from model name
        model_name = model_name.split("azure-")[-1]
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif provider == "google":
        # Gemini doesn't have async client yet, will use thread pool in embedding.py
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    else:
        raise ValueError(f"Embedding model {model_name} not supported.")

    return client, model_name
