import backoff
import logging
from typing import Any, cast
from google.genai import types
from shinka.llm.constants import BACKOFF_MAX_TIME, BACKOFF_MAX_TRIES, BACKOFF_MAX_VALUE
from .pricing import calculate_cost
from .result import QueryResult

logger = logging.getLogger(__name__)


MAX_TRIES = BACKOFF_MAX_TRIES
MAX_VALUE = BACKOFF_MAX_VALUE
MAX_TIME = BACKOFF_MAX_TIME

DEFAULT_THINKING_BUDGET = 1024


class GeminiStructuredOutputError(ValueError):
    """Structured output is unsupported and cannot succeed on retry."""


def _giveup_gemini(exc: Exception) -> bool:
    return isinstance(exc, GeminiStructuredOutputError)


def build_gemini_thinking_config(thinking_budget: int):
    """Build Gemini ThinkingConfig across SDK versions.

    Newer google-genai versions only expose include_thoughts/includeThoughts.
    Older versions also support thinking_budget/thinkingBudget.
    """
    model_fields = getattr(types.ThinkingConfig, "model_fields", {})
    config_kwargs: dict[str, object] = {"include_thoughts": True}

    if "thinking_budget" in model_fields:
        config_kwargs["thinking_budget"] = int(thinking_budget)
    elif "thinkingBudget" in model_fields:
        config_kwargs["thinkingBudget"] = int(thinking_budget)

    thinking_config_cls = cast(Any, types.ThinkingConfig)
    return thinking_config_cls(**config_kwargs)


def build_gemini_afc_config():
    """Build Gemini automatic function-calling config without SDK warnings."""
    model_fields = getattr(types.AutomaticFunctionCallingConfig, "model_fields", {})
    config_kwargs: dict[str, object] = {"disable": True}

    # Avoid warning about disable=True with default positive remote call budget.
    if "maximum_remote_calls" in model_fields:
        config_kwargs["maximum_remote_calls"] = None
    elif "maximumRemoteCalls" in model_fields:
        config_kwargs["maximumRemoteCalls"] = None

    afc_config_cls = cast(Any, types.AutomaticFunctionCallingConfig)
    return afc_config_cls(**config_kwargs)


def get_gemini_costs(response, model):
    """Get the costs for the given response and model."""
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        in_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", 0) or 0
    else:
        in_tokens = 0
        candidates_tokens = 0
        thoughts_tokens = 0

    out_tokens = candidates_tokens
    thinking_tokens = thoughts_tokens

    input_cost, output_cost = calculate_cost(
        model, in_tokens, out_tokens + thinking_tokens
    )
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
    }


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Gemini - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def gemini_build_contents(msg_history, msg):
    """Build structured contents from message history and current message.

    Based on: https://ai.google.dev/gemini-api/docs/text-generation
    """
    contents = []
    # Add message history in structured format
    for hist_msg in msg_history:
        role = hist_msg["role"]
        content = hist_msg["content"]

        # Map role names to Gemini's expected format
        gemini_role = "model" if role == "assistant" else role

        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    # Add current user message
    contents.append({"role": "user", "parts": [{"text": msg}]})
    return contents


def gemini_extract_thoughts_and_content(response):
    """Extract thoughts and content from response parts.

    Based on: https://ai.google.dev/gemini-api/docs/thinking
    """
    thoughts = []
    content_parts = []

    # Access the first candidate's content parts
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        has_content = hasattr(candidate, "content")
        has_parts = has_content and hasattr(candidate.content, "parts")
        if has_parts:
            for part in candidate.content.parts:
                # Check if part has text
                part_text = getattr(part, "text", "")
                if not part_text:
                    continue

                # Check if this part is a thought
                if getattr(part, "thought", False):
                    thoughts.append(part_text)
                else:
                    content_parts.append(part_text)

    # Combine thoughts and content
    thought = "\n".join(thoughts) if thoughts else ""
    content = "\n".join(content_parts) if content_parts else ""

    # Fallback to response.text if no parts found
    if not content and hasattr(response, "text"):
        content = response.text or ""

    return thought, content


def validate_gemini_response(response, content: str) -> None:
    """Raise if a Gemini response is empty or was truncated/blocked.

    Empty content is reachable via safety blocks or prompt feedback, and a
    ``finish_reason`` of ``MAX_TOKENS`` means the candidate was cut off
    mid-generation. Either way the generation is incomplete and must not be
    treated as a finished program (mirrors ``openai._extract_response_text``
    and ``headless`` which already raise on empty content).
    """
    candidates = getattr(response, "candidates", None)
    finish_reason = None
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
    # ``finish_reason`` is a str-based FinishReason enum; ``.name`` yields e.g.
    # "MAX_TOKENS". Fall back to str() if a raw string is returned instead.
    finish_name = getattr(finish_reason, "name", None)
    if finish_name is None and finish_reason is not None:
        finish_name = str(finish_reason)

    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)

    if not content or not content.strip():
        raise ValueError(
            "Gemini response contained no text output; "
            f"finish_reason={finish_name}; block_reason={block_reason}."
        )

    # MAX_TOKENS => the candidate was cut off mid-generation.
    if finish_name == "MAX_TOKENS":
        raise ValueError(
            "Gemini response was truncated (finish_reason=MAX_TOKENS); "
            "treating as an incomplete generation."
        )


@backoff.on_exception(
    backoff.expo,
    (Exception,),  # Catch all exceptions for Gemini API errors
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
    giveup=_giveup_gemini,
)
def query_gemini(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Gemini model.

    Based on: https://ai.google.dev/gemini-api/docs/text-generation
    """
    if output_model is not None:
        raise GeminiStructuredOutputError(
            "Gemini does not support structured output."
        )

    # Build structured contents
    contents = gemini_build_contents(msg_history, msg)

    # Extract kwargs for generation config
    temperature = kwargs.get("temperature", 0.8)
    top_p = kwargs.get("top_p", 1.0)
    max_tokens = kwargs.get("max_tokens", 2048)
    thinking_budget = kwargs.get("thinking_budget", DEFAULT_THINKING_BUDGET)

    generation_config = types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        max_output_tokens=int(max_tokens),
        system_instruction=system_msg if system_msg else None,
        automatic_function_calling=build_gemini_afc_config(),
        thinking_config=build_gemini_thinking_config(thinking_budget),
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config,
    )

    # Extract thoughts and content from response parts
    thought, content = gemini_extract_thoughts_and_content(response)

    # Reject empty/truncated generations instead of returning content=""
    validate_gemini_response(response, content)

    # Use content (without thoughts) for message history
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    # Get token counts and costs
    cost_results = get_gemini_costs(response, model)

    # Collect all results
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        **cost_results,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result


@backoff.on_exception(
    backoff.expo,
    (Exception,),  # Catch all exceptions for Gemini API errors
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
    giveup=_giveup_gemini,
)
async def query_gemini_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Gemini model asynchronously.

    Based on: https://ai.google.dev/gemini-api/docs/text-generation
    """
    if output_model is not None:
        raise GeminiStructuredOutputError(
            "Gemini does not support structured output."
        )

    # Build structured contents
    contents = gemini_build_contents(msg_history, msg)

    # Extract kwargs for generation config
    temperature = kwargs.get("temperature", 0.8)
    top_p = kwargs.get("top_p", 1.0)
    max_tokens = kwargs.get("max_tokens", 2048)
    thinking_budget = kwargs.get("thinking_budget", DEFAULT_THINKING_BUDGET)

    generation_config = types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        max_output_tokens=int(max_tokens),
        system_instruction=system_msg if system_msg else None,
        automatic_function_calling=build_gemini_afc_config(),
        thinking_config=build_gemini_thinking_config(thinking_budget),
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config,
    )

    # Extract thoughts and content from response parts
    thought, content = gemini_extract_thoughts_and_content(response)

    # Reject empty/truncated generations instead of returning content=""
    validate_gemini_response(response, content)

    # Use content (without thoughts) for message history
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    # Get token counts and costs
    cost_results = get_gemini_costs(response, model)

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        **cost_results,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
