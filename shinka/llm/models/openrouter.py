import backoff
import openai
from .result import QueryResult
from .pricing import get_openrouter_model_price 
import logging
import os

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenRouter Query - Retry {details['tries']} error: {exc}. Wait {details['wait']:0.1f}s..."
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
def query_openrouter(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    
    unsupported_params = ['max_output_tokens', 'reasoning', 'thinking', 'extra_body']
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
    
    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_msg}, *new_msg_history],
            **filtered_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response = client.chat.completions.create(
            model=model,
            response_model=output_model,
            messages=[{"role": "system", "content": system_msg}, *new_msg_history],
            **filtered_kwargs,
        )
        content = response.model_dump_json()
        new_msg_history.append({"role": "assistant", "content": content})

    try:
        input_price, output_price = get_openrouter_model_price(model, os.getenv("OPENROUTER_API_KEY"))
    except Exception as e:
        logger.error(f"CRITICAL: Failed to get pricing for {model} after retries: {e}")
        
        input_price, output_price = 1.0 / 1_000_000, 1.0 / 1_000_000 

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    input_cost = input_price * input_tokens
    output_cost = output_price * output_tokens
    total_cost = input_cost + output_cost

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=total_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )