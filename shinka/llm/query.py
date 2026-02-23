from typing import List, Optional, Dict
from pydantic import BaseModel
from .client import get_client_llm, get_async_client_llm
from .providers import (
    query_anthropic,
    query_openai,
    query_deepseek,
    query_gemini,
    query_local_openai,
    query_anthropic_async,
    query_openai_async,
    query_deepseek_async,
    query_gemini_async,
    query_local_openai_async,
    QueryResult,
)
import logging

logger = logging.getLogger(__name__)


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM."""
    client, model_name, provider = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    if provider in ("anthropic", "bedrock"):
        query_fn = query_anthropic
    elif provider in ("openai", "azure_openai", "fugu", "openrouter"):
        query_fn = query_openai
    elif provider == "deepseek":
        query_fn = query_deepseek
    elif provider == "google":
        query_fn = query_gemini
    elif provider == "local_openai":
        query_fn = query_local_openai
    else:
        raise ValueError(f"Model {model_name} not supported.")
    result = query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
    return result


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM asynchronously."""
    client, model_name, provider = get_async_client_llm(
        model_name, structured_output=output_model is not None
    )
    if provider in ("anthropic", "bedrock"):
        query_fn = query_anthropic_async
    elif provider in ("openai", "azure_openai", "fugu", "openrouter"):
        query_fn = query_openai_async
    elif provider == "deepseek":
        query_fn = query_deepseek_async
    elif provider == "google":
        query_fn = query_gemini_async
    elif provider == "local_openai":
        query_fn = query_local_openai_async
    else:
        raise ValueError(f"Model {model_name} not supported.")
    result = await query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
    return result
