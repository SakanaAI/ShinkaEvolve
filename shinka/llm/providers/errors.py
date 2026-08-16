class NonRetryableLLMError(ValueError):
    """A deterministic request failure that cannot succeed when retried."""


class StructuredOutputNotSupportedError(NonRetryableLLMError):
    """The selected provider cannot satisfy a structured-output request."""
