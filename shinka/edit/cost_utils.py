"""Cost calculation utilities for CLI backends.

Provides shared cost calculation using pricing tables from shinka/llm/models/pricing.py.
Used by gemini_cli.py and codex_cli.py to calculate costs from estimated tokens.
"""

from typing import Optional

from shinka.llm.models.pricing import GEMINI_MODELS, OPENAI_MODELS


def calculate_cost(
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    backend: str = "auto",
) -> float:
    """Calculate cost from tokens using pricing tables.

    Args:
        model: Model name (e.g., "gemini-2.5-flash", "gpt-4o").
        input_tokens: Number of input tokens (can be estimated).
        output_tokens: Number of output tokens (can be estimated).
        backend: Backend hint ("gemini", "codex", or "auto" to detect).

    Returns:
        Estimated cost in USD.
    """
    if not model:
        # No model specified - use conservative fallback
        return (input_tokens + output_tokens) * 0.000002  # $0.002/1K tokens

    # Try to find model in pricing tables
    pricing = None

    if backend == "gemini":
        pricing = GEMINI_MODELS.get(model)
    elif backend == "codex":
        pricing = OPENAI_MODELS.get(model)
    else:
        # Auto-detect: try both tables
        pricing = GEMINI_MODELS.get(model) or OPENAI_MODELS.get(model)

    if not pricing:
        # Model not found in pricing tables - use conservative fallback
        # This handles unknown models gracefully
        return (input_tokens + output_tokens) * 0.000002  # $0.002/1K tokens

    return (
        input_tokens * pricing["input_price"] + output_tokens * pricing["output_price"]
    )
