from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from shinka.embed.client import resolve_embedding_backend
from shinka.embed.providers.pricing import (
    get_all_providers as get_all_embedding_providers,
)
from shinka.embed.providers.pricing import (
    get_models_by_provider as get_embedding_models_by_provider,
)
from shinka.google_genai import google_genai_auth_mode
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers.pricing import get_all_providers, get_models_by_provider


PROVIDER_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_API_ENDPOINT", "AZURE_API_VERSION"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "vertexai": (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ),
}


@dataclass(frozen=True)
class ModelEnvAccessIssue:
    model_kind: str
    model_name: str
    provider: str
    missing_env_vars: tuple[str, ...]


def env_var_status(env_var_names: tuple[str, ...]) -> dict[str, bool]:
    return {
        env_var_name: bool(os.getenv(env_var_name, "").strip())
        for env_var_name in sorted(env_var_names)
    }


def provider_env_requirements(provider: str) -> tuple[str, ...] | None:
    normalized_provider = "azure" if provider == "azure_openai" else provider
    if normalized_provider == "google" and google_genai_auth_mode() == "vertexai":
        return PROVIDER_ENV_REQUIREMENTS["vertexai"]
    return PROVIDER_ENV_REQUIREMENTS.get(normalized_provider)


def missing_env_vars_for_provider(provider: str) -> tuple[str, ...]:
    env_var_names = provider_env_requirements(provider)
    if env_var_names is None:
        return ()
    return tuple(
        env_var_name
        for env_var_name, is_present in env_var_status(env_var_names).items()
        if not is_present
    )


def build_provider_availability_entry(provider: str) -> dict[str, Any] | None:
    env_var_names = provider_env_requirements(provider)
    if env_var_names is None:
        return None

    env_vars = env_var_status(env_var_names)
    if not all(env_vars.values()):
        return None

    llm_models = sorted(get_models_by_provider(provider))
    embedding_models = sorted(get_embedding_models_by_provider(provider))
    if not llm_models and not embedding_models:
        return None

    return {
        "provider": provider,
        "env_vars": env_vars,
        "llm_models": llm_models,
        "embedding_models": embedding_models,
    }


def build_model_availability_payload() -> dict[str, Any]:
    all_providers = sorted(
        set(get_all_providers()) | set(get_all_embedding_providers())
    )
    available_providers = [
        provider_entry
        for provider in all_providers
        if (provider_entry := build_provider_availability_entry(provider)) is not None
    ]
    llm_models = sorted(
        model
        for provider_entry in available_providers
        for model in provider_entry["llm_models"]
    )
    embedding_models = sorted(
        model
        for provider_entry in available_providers
        for model in provider_entry["embedding_models"]
    )
    return {
        "available_providers": available_providers,
        "embedding": embedding_models,
        "llm": llm_models,
    }


def find_model_env_access_issues(
    *,
    llm_models: Iterable[str] = (),
    embedding_models: Iterable[str] = (),
) -> list[ModelEnvAccessIssue]:
    issues: list[ModelEnvAccessIssue] = []

    for model_name in llm_models:
        resolved = resolve_model_backend(model_name)
        issues.extend(
            _model_env_access_issues(
                model_kind="llm",
                model_name=model_name,
                provider=resolved.provider,
                api_key_env_name=resolved.api_key_env_name,
            )
        )

    for model_name in embedding_models:
        resolved = resolve_embedding_backend(model_name)
        issues.extend(
            _model_env_access_issues(
                model_kind="embedding",
                model_name=model_name,
                provider=resolved.provider,
                api_key_env_name=resolved.api_key_env_name,
            )
        )

    return issues


def validate_model_env_access(
    *,
    llm_models: Iterable[str] = (),
    embedding_models: Iterable[str] = (),
) -> None:
    issues = find_model_env_access_issues(
        llm_models=llm_models,
        embedding_models=embedding_models,
    )
    if not issues:
        return

    issue_text = "; ".join(
        (
            f"{issue.model_kind} model '{issue.model_name}' "
            f"(provider: {issue.provider}) missing "
            f"{', '.join(issue.missing_env_vars)}"
        )
        for issue in issues
    )
    raise ValueError(
        "Requested model(s) are unavailable because required environment "
        f"variables are missing: {issue_text}. Run `shinka_models --verbose` "
        "to inspect provider availability."
    )


def _model_env_access_issues(
    *,
    model_kind: str,
    model_name: str,
    provider: str,
    api_key_env_name: str | None,
) -> list[ModelEnvAccessIssue]:
    missing_env_vars = missing_env_vars_for_provider(provider)

    if provider == "local_openai" and api_key_env_name:
        missing_env_vars = (
            (api_key_env_name,) if not os.getenv(api_key_env_name, "").strip() else ()
        )

    if not missing_env_vars:
        return []

    return [
        ModelEnvAccessIssue(
            model_kind=model_kind,
            model_name=model_name,
            provider=provider,
            missing_env_vars=missing_env_vars,
        )
    ]
