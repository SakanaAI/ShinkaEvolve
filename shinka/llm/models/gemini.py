import backoff
import openai
import re
from .pricing import GEMINI_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Gemini - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
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
    """Query Gemini model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            text = response.choices[0].message.content
        except Exception:
            # Reasoning models - ResponseOutputMessage
            text = response.output[1].content[0].text
        new_msg_history.append({"role": "assistant", "content": text})
    else:
        raise ValueError("Gemini does not support structured output.")

    # Handle None content gracefully (can happen with reasoning models)
    raw_content = text if text else ""

    # Extract thought if present
    thought_match = re.search(
        r"<thought>(.*?)</thought>", raw_content, re.DOTALL
    )
    thought = thought_match.group(1) if thought_match else ""

    # Content is everything outside thought tags
    content = re.sub(r"<thought>.*?</thought>", "", raw_content, flags=re.DOTALL).strip()

    input_cost = GEMINI_MODELS[model]["input_price"] * response.usage.prompt_tokens
    output_tokens = response.usage.total_tokens - response.usage.prompt_tokens
    output_cost = GEMINI_MODELS[model]["output_price"] * output_tokens
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
