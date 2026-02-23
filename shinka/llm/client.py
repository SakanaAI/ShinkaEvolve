from typing import Any, Tuple
import os
import anthropic
import openai
import instructor
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from .providers.model_resolver import resolve_model_backend

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TIMEOUT = 600


def _build_azure_endpoint() -> str:
    endpoint = os.getenv("AZURE_API_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_API_ENDPOINT is required for Azure OpenAI models.")
    if not endpoint.endswith("/"):
        endpoint += "/"
    return endpoint + "openai/v1/"


def get_client_llm(
    model_name: str, structured_output: bool = False
) -> Tuple[Any, str, str]:
    """Get the client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str, str]: (client, API model name, resolved provider).
    """
    resolved = resolve_model_backend(model_name)
    provider = resolved.provider
    api_model_name = resolved.api_model_name

    if provider == "anthropic":
        client = anthropic.Anthropic(timeout=TIMEOUT)  # 10 minutes
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "bedrock":
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif provider == "openai":
        client = openai.OpenAI(timeout=TIMEOUT)  # 10 minutes
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "azure_openai":
        # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle?view=foundry-classic&tabs=python#api-evolution
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=_build_azure_endpoint(),
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "deepseek":
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if structured_output:
            client = instructor.from_openai(
                client,
                mode=instructor.Mode.GEMINI_JSON,
            )
    elif provider == "fugu":
        client = openai.OpenAI(
            api_key=os.environ["FUGU_API_KEY"],
            base_url=os.environ["FUGU_BASE_URL"],
            timeout=TIMEOUT,
        )
        if structured_output:
            raise ValueError("Fugu does not support structured output.")
    elif provider == "openrouter":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,  # 10 minutes
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "local_openai":
        client = openai.OpenAI(
            api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
            base_url=resolved.base_url,
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, api_model_name, provider


def get_async_client_llm(
    model_name: str, structured_output: bool = False
) -> Tuple[Any, str, str]:
    """Get the async client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Any, str, str]: (async client, API model name, resolved provider).
    """
    resolved = resolve_model_backend(model_name)
    provider = resolved.provider
    api_model_name = resolved.api_model_name

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
        client = openai.AsyncOpenAI()
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "azure_openai":
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=_build_azure_endpoint(),
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif provider == "deepseek":
        client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if structured_output:
            raise ValueError("Gemini does not support structured output.")
    elif provider == "fugu":
        client = openai.AsyncOpenAI(
            api_key=os.environ["FUGU_API_KEY"],
            base_url=os.environ["FUGU_BASE_URL"],
            timeout=60000,
        )
        if structured_output:
            raise ValueError("Fugu does not support structured output.")
    elif provider == "openrouter":
        client = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif provider == "local_openai":
        client = openai.AsyncOpenAI(
            api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
            base_url=resolved.base_url,
            timeout=TIMEOUT,
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, api_model_name, provider
