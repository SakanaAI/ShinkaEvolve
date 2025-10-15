from typing import List, Union, Optional, Dict
import random
from pydantic import BaseModel
from .client import get_client_llm
from .models.pricing import (
    CLAUDE_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    GROQ_MODELS,
    BEDROCK_MODELS,
    REASONING_OAI_MODELS,
    REASONING_CLAUDE_MODELS,
    REASONING_DEEPSEEK_MODELS,
    REASONING_GEMINI_MODELS,
    REASONING_AZURE_MODELS,
    REASONING_BEDROCK_MODELS,
    REASONING_GROQ_MODELS,
)
from .models import (
    query_anthropic,
    query_openai,
    query_deepseek,
    query_gemini,
    query_groq,
    QueryResult,
)
import logging

logger = logging.getLogger(__name__)


THINKING_TOKENS = {
    "auto": 0,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}


def sample_batch_kwargs(
    num_samples: int,
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
    unique_filter: bool = False,
):
    """Sample a dictionary of kwargs for a given model."""
    all_kwargs = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops

    while len(all_kwargs) < num_samples and attempts < max_attempts:
        kwargs_dict = sample_model_kwargs(
            model_names=model_names,
            temperatures=temperatures,
            max_tokens=max_tokens,
            reasoning_efforts=reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

        if unique_filter:
            if kwargs_dict not in all_kwargs:
                all_kwargs.append(kwargs_dict)
        else:
            all_kwargs.append(kwargs_dict)

        attempts += 1

    if len(all_kwargs) < num_samples:
        logger.info(
            f"Could not generate {num_samples} unique kwargs combinations "
            f"after {max_attempts} attempts"
        )
        logger.info(f"Returning {len(all_kwargs)} unique kwargs combinations.")

    return all_kwargs


def sample_model_kwargs(
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
):
    """Sample a dictionary of kwargs for a given model."""
    # Make all inputs lists
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(temperatures, float):
        temperatures = [temperatures]
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens]
    if isinstance(reasoning_efforts, str):
        reasoning_efforts = [reasoning_efforts]

    kwargs_dict = {}
    # perform model sampling if list provided
    if model_sample_probs is not None:
        if len(model_sample_probs) != len(model_names):
            raise ValueError(
                "model_sample_probs must have the same length as model_names"
            )
        if not abs(sum(model_sample_probs) - 1.0) < 1e-9:
            raise ValueError("model_sample_probs must sum to 1")
        kwargs_dict["model_name"] = random.choices(
            model_names, weights=model_sample_probs, k=1
        )[0]
    else:
        kwargs_dict["model_name"] = random.choice(model_names)

    # perform temperature sampling if list provided
    # set temperature to 1.0 for reasoning models
    if kwargs_dict["model_name"] in (
        REASONING_OAI_MODELS
        + REASONING_CLAUDE_MODELS
        + REASONING_DEEPSEEK_MODELS
        + REASONING_GEMINI_MODELS
        + REASONING_AZURE_MODELS
        + REASONING_BEDROCK_MODELS
        + REASONING_GROQ_MODELS
    ):
        kwargs_dict["temperature"] = 1.0
    else:
        kwargs_dict["temperature"] = random.choice(temperatures)

    # perform reasoning effort sampling if list provided
    # set max_completion_tokens for OAI reasoning models
    if kwargs_dict["model_name"] in (REASONING_OAI_MODELS + REASONING_AZURE_MODELS):
        kwargs_dict["max_output_tokens"] = random.choice(max_tokens)
        r_effort = random.choice(reasoning_efforts)
        if r_effort != "auto":
            kwargs_dict["reasoning"] = {"effort": r_effort}

    if kwargs_dict["model_name"] in (REASONING_GEMINI_MODELS):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        r_effort = random.choice(reasoning_efforts)
        think_bool = r_effort != "auto"
        if think_bool:
            thinking_tokens = [
                t
                for t in THINKING_TOKENS.values()
                if t < kwargs_dict["max_tokens"] and t >= 1024
            ]
            kwargs_dict["extra_body"] = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_budget": random.choice(thinking_tokens),
                            "include_thoughts": True,
                        }
                    }
                }
            }

    elif kwargs_dict["model_name"] in (
        REASONING_CLAUDE_MODELS + REASONING_BEDROCK_MODELS
    ):
        kwargs_dict["max_tokens"] = min(random.choice(max_tokens), 16384)
        think_bool = random.choice(reasoning_efforts) != "auto"
        if think_bool:
            # filter thinking tokens to be smaller than max_tokens
            # not auto THINKING_TOKENS
            thinking_tokens = [
                t
                for t in THINKING_TOKENS.values()
                if t < kwargs_dict["max_tokens"] and t >= 1024
            ]
            # sample only from thinking tokens that are valid
            kwargs_dict["thinking"] = {
                "type": "enabled",
                "budget_tokens": random.choice(thinking_tokens),
            }

    else:
        if (
            kwargs_dict["model_name"] in CLAUDE_MODELS
            or kwargs_dict["model_name"] in BEDROCK_MODELS
            or kwargs_dict["model_name"] in REASONING_CLAUDE_MODELS
            or kwargs_dict["model_name"] in REASONING_BEDROCK_MODELS
            or kwargs_dict["model_name"] in DEEPSEEK_MODELS
            or kwargs_dict["model_name"] in REASONING_DEEPSEEK_MODELS
            or kwargs_dict["model_name"] in GROQ_MODELS
        ):
            kwargs_dict["max_tokens"] = random.choice(max_tokens)
        else:
            kwargs_dict["max_output_tokens"] = random.choice(max_tokens)

    return kwargs_dict


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
    original_model_name = model_name
    client, model_name = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    if original_model_name in CLAUDE_MODELS.keys() or "anthropic" in original_model_name:
        query_fn = query_anthropic
    elif original_model_name in OPENAI_MODELS.keys():
        query_fn = query_openai
    elif original_model_name in DEEPSEEK_MODELS.keys():
        query_fn = query_deepseek
    elif original_model_name in GEMINI_MODELS.keys():
        query_fn = query_gemini
    elif original_model_name in GROQ_MODELS.keys():
        query_fn = query_groq
    else:
        raise ValueError(f"Model {original_model_name} not supported.")
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
