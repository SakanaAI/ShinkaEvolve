import backoff
import anthropic
from shinka.llm.constants import BACKOFF_MAX_TIME, BACKOFF_MAX_TRIES, BACKOFF_MAX_VALUE
from .pricing import calculate_cost, model_exists
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


MAX_TRIES = BACKOFF_MAX_TRIES
MAX_VALUE = BACKOFF_MAX_VALUE
MAX_TIME = BACKOFF_MAX_TIME


def get_anthropic_costs(response, model):
    """Return billed costs with thinking separated from visible output."""
    input_tokens = response.usage.input_tokens
    all_out_tokens = response.usage.output_tokens
    output_details = getattr(response.usage, "output_tokens_details", None)
    reported_thinking = getattr(output_details, "thinking_tokens", 0) or 0
    thinking_tokens = min(max(int(reported_thinking), 0), all_out_tokens)
    output_tokens = all_out_tokens - thinking_tokens
    # Fall back to a zero cost (with a warning) on an unknown model instead of
    # raising, mirroring openai/local so a pricing-catalog miss never aborts a
    # completed generation.
    if model_exists(model):
        input_cost, output_cost = calculate_cost(model, input_tokens, all_out_tokens)
    else:
        logger.warning(
            "Model '%s' has no pricing entry; defaulting query cost to 0.", model
        )
        input_cost, output_cost = 0.0, 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
    }


def split_content_blocks(response):
    """Split response blocks into visible text and thinking, keyed on block type.

    Anthropic returns a list of typed blocks (text, thinking, redacted_thinking,
    tool_use, ...) whose attributes differ per type, and the ordering and count
    are not guaranteed. Unknown types are skipped.
    """
    text_parts = []
    thought_parts = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "thinking":
            thought_parts.append(block.thinking)
        elif block_type == "redacted_thinking":
            thought_parts.append(getattr(block, "data", ""))
    return "\n".join(text_parts), "\n".join(thought_parts)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_anthropic(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model."""
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]
    if output_model is None:
        response = client.messages.create(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            **kwargs,
        )
        content, thought = split_content_blocks(response)
    else:
        raise NotImplementedError("Structured output not supported for Anthropic.")
    new_msg_history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    )
    cost_results = get_anthropic_costs(response, model)
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
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_anthropic_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model asynchronously."""
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]
    if output_model is None:
        response = await client.messages.create(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            **kwargs,
        )
        content, thought = split_content_blocks(response)
    else:
        raise NotImplementedError("Structured output not supported for Anthropic.")
    new_msg_history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    )
    cost_results = get_anthropic_costs(response, model)
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
