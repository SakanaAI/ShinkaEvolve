from __future__ import annotations


class LLMRouteError(RuntimeError):
    """A provider or agent-route failure, separate from candidate quality."""

    failure_class = "transport_error"
    retryable = False

    def __init__(self, message: str, *, artifacts: dict[str, str] | None = None):
        super().__init__(message)
        self.artifacts = dict(artifacts or {})


class LLMAuthenticationError(LLMRouteError):
    failure_class = "authentication_error"


class LLMModelUnavailableError(LLMRouteError):
    failure_class = "model_unavailable"


class LLMTimeoutError(LLMRouteError):
    failure_class = "proposal_timeout"


class LLMExtractionError(LLMRouteError):
    failure_class = "output_extraction_error"


class LLMProcessError(LLMRouteError):
    failure_class = "agent_process_error"
