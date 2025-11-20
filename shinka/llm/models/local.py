import backoff
import openai
from .pricing import LOCAL_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Local LLM - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=5,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_local(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query local OpenAI-compatible model."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for local models.")
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    # Convert max_output_tokens to max_tokens for OpenAI-compatible API
    local_kwargs = kwargs.copy()
    if "max_output_tokens" in local_kwargs:
        local_kwargs["max_tokens"] = local_kwargs.pop("max_output_tokens")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ],
        **local_kwargs,
        n=1,
        stop=None,
    )
    content = response.choices[0].message.content
    try:
        thought = response.choices[0].message.reasoning_content
    except:
        thought = ""
    new_msg_history.append({"role": "assistant", "content": content})
    
    # Get token usage, defaulting to 0 if not available
    input_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
    output_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
    
    input_cost = LOCAL_MODELS[model]["input_price"] * input_tokens
    output_cost = LOCAL_MODELS[model]["output_price"] * output_tokens
    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )

