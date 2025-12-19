"""Cost calculation utilities for CLI backends.

Provides shared cost calculation using pricing tables from shinka/llm/models/pricing.py.
Used by gemini_cli.py and codex_cli.py to calculate costs from estimated tokens.
"""

import logging
from typing import Optional

from shinka.llm.models.pricing import GEMINI_MODELS, OPENAI_MODELS

logger = logging.getLogger(__name__)

# Fallback rate when model pricing is unknown
# Set conservatively high so users notice something is wrong
FALLBACK_RATE_PER_TOKEN = 0.00001  # $10/1M tokens (high to be noticeable)


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
        Estimated cost in USD. Returns fallback estimate with warning if model unknown.
    """
    if not model:
        logger.warning(
            "No model specified for cost calculation - using fallback rate. "
            "Cost estimate will be inaccurate. Configure model explicitly."
        )
        return (input_tokens + output_tokens) * FALLBACK_RATE_PER_TOKEN

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
        logger.warning(
            f"Model '{model}' not found in pricing tables (backend={backend}). "
            f"Using fallback rate. Add model to shinka/llm/models/pricing.py."
        )
        return (input_tokens + output_tokens) * FALLBACK_RATE_PER_TOKEN

    return (
        input_tokens * pricing["input_price"] + output_tokens * pricing["output_price"]
    )
