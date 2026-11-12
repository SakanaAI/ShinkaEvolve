import asyncio
from typing import Dict, Optional


class BilledCancellationError(asyncio.CancelledError):
    """Cancellation carrying provider usage already received by the caller."""

    def __init__(self, cost_results: Dict):
        super().__init__("LLM work cancelled after billed replies")
        self.cost_results = dict(cost_results)

    @property
    def cost(self) -> float:
        return float(self.cost_results.get("cost", 0.0) or 0.0)


class NonRetryableLLMError(ValueError):
    """A deterministic request failure that cannot succeed when retried."""


class LLMEndpointError(NonRetryableLLMError):
    """A permanent endpoint/authentication failure, safe to persist in run artifacts."""

    def __init__(self, status_code: int, code: str):
        self.status_code = status_code
        self.code = code
        super().__init__(f"LLM endpoint returned HTTP {status_code} ({code})")

    def __reduce__(self):
        # Sync batch queries return exceptions through multiprocessing workers.
        return type(self), (self.status_code, self.code)


class StructuredOutputNotSupportedError(NonRetryableLLMError):
    """The selected provider cannot satisfy a structured-output request."""


class EmptyResponseError(ValueError):
    """The provider billed a reply that contains no usable text.

    ``cost_results`` holds the token counts and costs of that reply so the
    caller can still account for it.
    """

    def __init__(
        self,
        message: str,
        incomplete_reason: Optional[str] = None,
        cost_results: Optional[Dict] = None,
    ):
        super().__init__(message)
        self.incomplete_reason = incomplete_reason
        self.cost_results = cost_results or {}

    @property
    def cost(self) -> float:
        return float(self.cost_results.get("cost", 0.0) or 0.0)

    @property
    def hit_output_limit(self) -> bool:
        # Sending the same request again would hit the same limit.
        return self.incomplete_reason == "max_output_tokens"
