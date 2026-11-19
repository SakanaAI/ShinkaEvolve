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
from shinka.llm.providers.headless import check_headless_available


EnvVarSet = tuple[str, ...]

# A provider is usable when every variable of at least one set is present. The
# sets mirror what `shinka.llm.client._constructor_configuration` accepts; the
# first set is the one named in error messages unless another is partly set.
# Ambient AWS credential sources (profiles, instance roles) cannot be checked
# from the environment.
PROVIDER_ENV_ALTERNATIVES: dict[str, tuple[EnvVarSet, ...]] = {
    "anthropic": (("ANTHROPIC_API_KEY",), ("ANTHROPIC_AUTH_TOKEN",)),
    "azure": (("AZURE_OPENAI_API_KEY", "AZURE_API_ENDPOINT", "AZURE_API_VERSION"),),
    "bedrock": (
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION_NAME"),
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"),
    ),
    "deepseek": (("DEEPSEEK_API_KEY",),),
    "google": (("GEMINI_API_KEY",),),
    # OPENAI_ADMIN_KEY alone builds a client, but the SDK refuses to send model
    # requests without OPENAI_API_KEY, so it is not an alternative.
    "openai": (("OPENAI_API_KEY",),),
    "openrouter": (("OPENROUTER_API_KEY",),),
    "vertexai": (
        (
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        ),
    ),
}
PROVIDER_ENV_REQUIREMENTS: dict[str, EnvVarSet] = {
    provider: alternatives[0]
    for provider, alternatives in PROVIDER_ENV_ALTERNATIVES.items()
}
AZURE_LLM_ENV_REQUIREMENTS = ("AZURE_OPENAI_API_KEY", "AZURE_API_ENDPOINT")


@dataclass(frozen=True)
class ModelEnvAccessIssue:
    model_kind: str
    model_name: str
    provider: str
    missing_env_vars: tuple[str, ...]
    alternative_env_vars: tuple[tuple[str, ...], ...] = ()


def env_var_status(env_var_names: tuple[str, ...]) -> dict[str, bool]:
    return {
        env_var_name: bool(os.getenv(env_var_name, "").strip())
        for env_var_name in sorted(env_var_names)
    }


def provider_env_alternatives(provider: str) -> tuple[EnvVarSet, ...] | None:
    normalized_provider = "azure" if provider == "azure_openai" else provider
    if normalized_provider == "google" and google_genai_auth_mode() == "vertexai":
        return PROVIDER_ENV_ALTERNATIVES["vertexai"]
    return PROVIDER_ENV_ALTERNATIVES.get(normalized_provider)


def provider_env_requirements(provider: str) -> EnvVarSet | None:
    """Return the primary env var set; see `provider_env_alternatives` for all."""
    alternatives = provider_env_alternatives(provider)
    return None if alternatives is None else alternatives[0]


def missing_env_vars_for_provider(provider: str) -> EnvVarSet:
    alternatives = provider_env_alternatives(provider)
    if alternatives is None:
        return ()
    return _missing_env_vars(_closest_alternative(alternatives))


def _missing_env_vars(env_var_names: EnvVarSet) -> EnvVarSet:
    return tuple(
        env_var_name
        for env_var_name, is_present in env_var_status(env_var_names).items()
        if not is_present
    )


def _closest_alternative(alternatives: tuple[EnvVarSet, ...]) -> EnvVarSet:
    """Return the satisfied set, else the started set that misses the fewest.

    A set counts once one of its variables is present; with none started, the
    primary set is the one to ask for.
    """
    started = [
        names for names in alternatives if len(_missing_env_vars(names)) < len(names)
    ]
    return min(
        started or alternatives[:1], key=lambda names: len(_missing_env_vars(names))
    )


def model_env_alternatives(
    provider: str, model_kind: str
) -> tuple[EnvVarSet, ...] | None:
    normalized_provider = "azure" if provider == "azure_openai" else provider
    if normalized_provider == "azure" and model_kind == "llm":
        return (AZURE_LLM_ENV_REQUIREMENTS,)
    return provider_env_alternatives(provider)


def model_env_requirements(provider: str, model_kind: str) -> EnvVarSet | None:
    """Return the primary env var set; see `model_env_alternatives` for all."""
    alternatives = model_env_alternatives(provider, model_kind)
    return None if alternatives is None else alternatives[0]


def build_provider_availability_entry(provider: str) -> dict[str, Any] | None:
    llm_alternatives = model_env_alternatives(provider, "llm")
    embedding_alternatives = model_env_alternatives(provider, "embedding")
    if llm_alternatives is None and embedding_alternatives is None:
        return None

    llm_models = _available_models(get_models_by_provider(provider), llm_alternatives)
    embedding_models = _available_models(
        get_embedding_models_by_provider(provider), embedding_alternatives
    )
    if not llm_models and not embedding_models:
        return None

    env_var_names = tuple(
        sorted(
            {
                env_var_name
                for alternatives in (llm_alternatives, embedding_alternatives)
                if alternatives is not None
                for env_var_name in _closest_alternative(alternatives)
            }
        )
    )
    return {
        "provider": provider,
        "env_vars": env_var_status(env_var_names),
        "llm_models": llm_models,
        "embedding_models": embedding_models,
    }


def _available_models(
    models: Iterable[str], alternatives: tuple[EnvVarSet, ...] | None
) -> list[str]:
    if alternatives is None or _missing_env_vars(_closest_alternative(alternatives)):
        return []
    return sorted(models)


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
    llm_models = list(llm_models)
    embedding_models = list(embedding_models)
    headless_models = [
        model_name
        for model_name in llm_models
        if resolve_model_backend(model_name).provider == "headless"
    ]
    if headless_models:
        try:
            check_headless_available()
        except ValueError as exc:
            models = ", ".join(headless_models)
            raise ValueError(
                f"Requested headless model(s) are unavailable: {models}. {exc}"
            ) from exc

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
            f"{_alternatives_hint(issue)}"
        )
        for issue in issues
    )
    raise ValueError(
        "Requested model(s) are unavailable because required environment "
        f"variables are missing: {issue_text}. Run `shinka_models --verbose` "
        "to inspect provider availability."
    )


def _alternatives_hint(issue: ModelEnvAccessIssue) -> str:
    if not issue.alternative_env_vars:
        return ""
    options = " or ".join(" + ".join(names) for names in issue.alternative_env_vars)
    return f" (or instead set {options})"


def _model_env_access_issues(
    *,
    model_kind: str,
    model_name: str,
    provider: str,
    api_key_env_name: str | None,
) -> list[ModelEnvAccessIssue]:
    alternatives = model_env_alternatives(provider, model_kind)
    missing_env_vars: EnvVarSet = ()
    alternative_env_vars: tuple[EnvVarSet, ...] = ()
    if alternatives is not None:
        closest = _closest_alternative(alternatives)
        missing_env_vars = _missing_env_vars(closest)
        alternative_env_vars = tuple(
            names for names in alternatives if names != closest
        )

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
            alternative_env_vars=alternative_env_vars,
        )
    ]
