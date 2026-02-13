from typing import Any, Tuple
import os
import anthropic
import openai
import instructor
from pathlib import Path
from dotenv import load_dotenv
import re
from .models.pricing import (
    CLAUDE_MODELS,
    BEDROCK_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
)

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str]:
    """Get the client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        The client and model for the given model name.
    """
    # print(f"Getting client for model {model_name}")

    if model_name.startswith("local-"):
        # Extract URL from model name
        pattern = r"https?://"
        match = re.search(pattern, model_name)
        if match:
            start_index = match.start()
            url = model_name[start_index:]
        else:
            raise ValueError(f"Invalid URL in model name: {model_name}")
        
        # Create OpenAI-compatible client
        client = openai.OpenAI(
            api_key="filler",
            base_url=url
        )

        # Structured output mode (if required)
        if structured_output:
            client = instructor.from_openai(
                client,
                mode=instructor.Mode.JSON,
            )
    elif model_name in CLAUDE_MODELS.keys():
        client = anthropic.Anthropic()
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif model_name in BEDROCK_MODELS.keys():
        model_name = model_name.split("/")[-1]
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
        if structured_output:
            client = instructor.from_anthropic(
                client, mode=instructor.mode.Mode.ANTHROPIC_JSON
            )
    elif model_name in OPENAI_MODELS.keys():
        client = openai.OpenAI()
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif model_name.startswith("azure-"):
        # get rid of the azure- prefix
        model_name = model_name.split("azure-")[-1]
        # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle?view=foundry-classic&tabs=python#api-evolution
        client = openai.OpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_API_ENDPOINT"] + 'openai/v1/'
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.TOOLS_STRICT)
    elif model_name in DEEPSEEK_MODELS.keys():
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    elif model_name in GEMINI_MODELS.keys():
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        if structured_output:
            client = instructor.from_openai(
                client,
                mode=instructor.Mode.GEMINI_JSON,
            )

    elif "/" in model_name and not model_name.startswith("bedrock/"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"Model '{model_name}' detected as OpenRouter model, "
                "but OPENROUTER_API_KEY is not set in environment variables."
            )
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.getenv(
                    "SHINKA_SITE_URL", "https://github.com/SakanaAI/ShinkaEvolve"
                ),
                "X-Title": os.getenv("SHINKA_APP_NAME", "ShinkaEvolve"),
            },
        )
        if structured_output:
            client = instructor.from_openai(client, mode=instructor.Mode.JSON)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model_name
