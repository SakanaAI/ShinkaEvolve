import os
import backoff
import google.generativeai as genai
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
    Exception,  # Broader exception for genai library
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_gemini(
    client, # client is not used but kept for compatibility with the dispatcher
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Gemini model using the native google-generativeai library."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)

    gemini_history = []
    for message in msg_history:
        role = "user" if message["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [message["content"]]})
    gemini_history.append({"role": "user", "parts": [msg]})

    if output_model is not None:
        raise ValueError("Gemini does not support structured output in this implementation.")

    generation_config = genai.types.GenerationConfig(
        temperature=kwargs.get("temperature"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
    )

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_msg,
        generation_config=generation_config
    )

    response = gemini_model.generate_content(gemini_history)

    try:
        text = response.text
    except (ValueError, IndexError) as e:
        logger.error(f"Error extracting text from Gemini response: {e}. Full response: {response}")
        text = ""

    new_msg_history = msg_history + [{"role": "user", "content": msg}, {"role": "assistant", "content": text}]

    thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    if thought_match:
        content = (text[:thought_match.start()] + text[thought_match.end():]).strip()
    else:
        content = text

    prompt_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count

    input_cost = GEMINI_MODELS[model]["input_price"] * prompt_tokens
    output_cost = GEMINI_MODELS[model]["output_price"] * output_tokens

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result