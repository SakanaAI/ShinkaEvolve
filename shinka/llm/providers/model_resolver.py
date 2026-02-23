from dataclasses import dataclass
import re
from typing import Optional
from urllib.parse import urlparse

from .pricing import get_provider

_LOCAL_MODEL_PATTERN = re.compile(
    r"^local/(?P<model>[^@]+)@(?P<url>https?://.+)$"
)
_OPENROUTER_PREFIX = "openrouter/"


@dataclass(frozen=True)
class ResolvedModel:
    original_model_name: str
    api_model_name: str
    provider: str
    base_url: Optional[str] = None


def resolve_model_backend(model_name: str) -> ResolvedModel:
    """Resolve runtime backend info for known and dynamic model identifiers."""
    provider = get_provider(model_name)
    if provider is not None:
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=model_name,
            provider=provider,
            base_url=None,
        )

    if model_name.startswith("azure-"):
        api_model_name = model_name.split("azure-", 1)[-1]
        if not api_model_name:
            raise ValueError("Azure model name is missing after 'azure-' prefix.")
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="azure_openai",
            base_url=None,
        )

    if model_name.startswith(_OPENROUTER_PREFIX):
        api_model_name = model_name.split(_OPENROUTER_PREFIX, 1)[-1]
        if not api_model_name:
            raise ValueError("OpenRouter model name is missing after 'openrouter/'.")
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="openrouter",
            base_url=None,
        )

    local_match = _LOCAL_MODEL_PATTERN.match(model_name)
    if local_match:
        api_model_name = local_match.group("model")
        base_url = local_match.group("url")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"Invalid local model URL '{base_url}'. Expected http(s)://host[:port]/..."
            )
        return ResolvedModel(
            original_model_name=model_name,
            api_model_name=api_model_name,
            provider="local_openai",
            base_url=base_url,
        )

    raise ValueError(
        f"Model '{model_name}' is not supported. "
        "Use a known pricing.csv model, 'openrouter/<model>', "
        "or 'local/<model>@http(s)://host[:port]/v1'."
    )
