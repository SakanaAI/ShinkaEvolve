import os
import backoff
import google.generativeai as genai
from .pricing import GEMINI_EMBEDDING_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)

def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Gemini Embedding - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_gemini_embedding(
    model,
    text,
    **kwargs,
) -> tuple:
    """Query Gemini embedding model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)

    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"
    )

    # Gemini API doesn't provide token usage directly for embeddings in the same way as OpenAI.
    # We will estimate cost based on characters, assuming 4 chars = 1 token as a rough heuristic.
    # This is not perfect but provides a cost estimate.
    estimated_input_tokens = len(text) / 4
    cost = GEMINI_EMBEDDING_MODELS[model]["input_price"] * estimated_input_tokens

    return result["embedding"], cost