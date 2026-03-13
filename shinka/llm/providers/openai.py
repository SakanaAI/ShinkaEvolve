import backoff
import openai
from .pricing import calculate_cost, model_exists
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


def get_openai_costs(response, model):
    # Get token counts and costs
    in_tokens = response.usage.input_tokens
    try:
        thinking_tokens = response.usage.output_tokens_details.reasoning_tokens
    except Exception:
        thinking_tokens = 0
    all_out_tokens = response.usage.output_tokens
    out_tokens = response.usage.output_tokens - thinking_tokens

    # Get actual costs from OpenRouter API if available -- if not use OAI
    cost_details = getattr(response.usage, "cost_details", None)
    if cost_details:
        if isinstance(cost_details, dict):
            input_cost = float(cost_details.get("upstream_inference_input_cost", 0.0))
            output_cost = float(cost_details.get("upstream_inference_output_cost", 0.0))
        else:
            input_cost = float(
                getattr(cost_details, "upstream_inference_input_cost", 0.0) or 0.0
            )
            output_cost = float(
                getattr(cost_details, "upstream_inference_output_cost", 0.0) or 0.0
            )
    elif model_exists(model):
        input_cost, output_cost = calculate_cost(model, in_tokens, all_out_tokens)
    else:
        logger.warning(
            "Model '%s' has no pricing entry and response cost metadata is absent. "
            "Defaulting query cost to 0.",
            model,
        )
        input_cost, output_cost = 0.0, 0.0
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
    }


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    thought = ""
    if output_model is None:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            content = response.output[0].content[0].text
        except Exception:
            # Reasoning models - ResponseOutputMessage
            content = response.output[1].content[0].text

        try:
            thought = response.output[0].summary[0].text
        except Exception:
            pass
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            text_format=output_model,
            **kwargs,
        )
        content = response.output_parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})

    # Get token counts and costs
    cost_results = get_openai_costs(response, model)

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
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_openai_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model asynchronously."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    thought = ""
    if output_model is None:
        response = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            content = response.output[0].content[0].text
        except Exception:
            # Reasoning models - ResponseOutputMessage
            content = response.output[1].content[0].text
        try:
            thought = response.output[0].summary[0].text
        except Exception:
            pass
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response = await client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            text_format=output_model,
            **kwargs,
        )
        content = response.output_parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})
    cost_results = get_openai_costs(response, model)
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
