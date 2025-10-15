import backoff
import groq
from .pricing import GROQ_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"Groq - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        groq.APIConnectionError,
        groq.APIStatusError,
        groq.RateLimitError,
        groq.APITimeoutError,
    ),
    max_tries=5,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_groq(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Groq model."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for Groq.")

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ],
        **kwargs,
        n=1,
        stop=None,
    )

    content = response.choices[0].message.content
    new_msg_history.append({"role": "assistant", "content": content})

    # Add groq/ prefix back for pricing lookup (client.py strips one groq/ prefix)
    pricing_key = f"groq/{model}"
    input_cost = GROQ_MODELS[pricing_key]["input_price"] * response.usage.prompt_tokens
    output_cost = GROQ_MODELS[pricing_key]["output_price"] * response.usage.completion_tokens

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=pricing_key,  # Use the full groq/ prefixed name
        kwargs=kwargs,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
