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
