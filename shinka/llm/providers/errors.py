from typing import Dict, Optional


class NonRetryableLLMError(ValueError):
    """A deterministic request failure that cannot succeed when retried."""


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
