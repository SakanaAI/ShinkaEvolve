"""Startup env validation must accept every auth form the LLM client accepts."""

import json
from pathlib import Path

import pytest

import shinka.cli.models as cli_models
from shinka.llm.client import _constructor_configuration
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers.pricing import get_models_by_provider
from shinka.model_availability import (
    AZURE_LLM_ENV_REQUIREMENTS,
    PROVIDER_ENV_ALTERNATIVES,
    PROVIDER_ENV_REQUIREMENTS,
    validate_model_env_access,
)
from shinka.pricing.catalog import PricingConfig, PricingMode, refresh_model_catalog

_UNLISTED_AUTH_ENV_VARS = (
    "OPENAI_ADMIN_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "GOOGLE_GENAI_USE_VERTEXAI",
)
# Client provider name -> the env var sets its constructor configuration accepts.
_CLIENT_ALTERNATIVES = {
    "anthropic": PROVIDER_ENV_ALTERNATIVES["anthropic"],
    "bedrock": PROVIDER_ENV_ALTERNATIVES["bedrock"],
    "openai": PROVIDER_ENV_ALTERNATIVES["openai"],
    "deepseek": PROVIDER_ENV_ALTERNATIVES["deepseek"],
    "openrouter": PROVIDER_ENV_ALTERNATIVES["openrouter"],
    "google": PROVIDER_ENV_ALTERNATIVES["google"],
    "azure_openai": (AZURE_LLM_ENV_REQUIREMENTS,),
}
_CLIENT_CASES = [
    (provider, env_var_names)
    for provider, alternatives in _CLIENT_ALTERNATIVES.items()
    for env_var_names in alternatives
]


@pytest.fixture(autouse=True)
def _bundled_catalog_without_provider_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    snapshot = refresh_model_catalog(
        PricingConfig(mode=PricingMode.OFFLINE, cache_dir=tmp_path)
    )
    monkeypatch.setattr(cli_models, "refresh_model_catalog", lambda: snapshot)
    monkeypatch.setattr(cli_models, "load_shinka_dotenv", lambda: ())
    listed = {
        env_var_name
        for alternatives in PROVIDER_ENV_ALTERNATIVES.values()
        for env_var_names in alternatives
        for env_var_name in env_var_names
    }
    for env_var_name in (*listed, *_UNLISTED_AUTH_ENV_VARS):
        monkeypatch.delenv(env_var_name, raising=False)


def _set_env(monkeypatch: pytest.MonkeyPatch, env_var_names: tuple[str, ...]) -> None:
    for env_var_name in env_var_names:
        value = (
            "https://example-resource.openai.azure.com"
            if env_var_name == "AZURE_API_ENDPOINT"
            else "test-value"
        )
        monkeypatch.setenv(env_var_name, value)


def _client_accepts_environment(provider: str) -> bool:
    resolved = resolve_model_backend(get_models_by_provider("openai")[0])
    try:
        _, auth_is_configured = _constructor_configuration(provider, resolved)
    except (KeyError, ValueError):
        return False
    return auth_is_configured


def _llm_model(provider: str) -> str:
    return sorted(get_models_by_provider(provider))[0]


@pytest.mark.parametrize(
    "provider,env_var_names",
    [
        ("anthropic", ("ANTHROPIC_AUTH_TOKEN",)),
        ("bedrock", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")),
        ("bedrock", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION_NAME")),
        ("bedrock", ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")),
    ],
)
def test_validation_accepts_alternative_auth_the_client_supports(
    monkeypatch: pytest.MonkeyPatch, provider, env_var_names
):
    _set_env(monkeypatch, env_var_names)

    validate_model_env_access(llm_models=[_llm_model(provider)])


def test_validation_names_the_alternatives_when_none_is_complete(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")

    with pytest.raises(ValueError) as bedrock_error:
        validate_model_env_access(llm_models=[_llm_model("bedrock")])
    with pytest.raises(ValueError) as anthropic_error:
        validate_model_env_access(llm_models=[_llm_model("anthropic")])

    assert "missing AWS_REGION_NAME (or instead set" in str(bedrock_error.value)
    assert "AWS_BEARER_TOKEN_BEDROCK + AWS_REGION" in str(bedrock_error.value)
    assert "missing ANTHROPIC_API_KEY (or instead set ANTHROPIC_AUTH_TOKEN)" in str(
        anthropic_error.value
    )


def test_openai_admin_key_alone_is_still_rejected(monkeypatch: pytest.MonkeyPatch):
    """The SDK builds a client from it but refuses to send model requests."""
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "test-admin-key")

    with pytest.raises(ValueError, match="missing OPENAI_API_KEY"):
        validate_model_env_access(llm_models=[_llm_model("openai")])


def test_shinka_models_lists_models_reachable_through_alternative_auth(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-west-2")

    assert cli_models.main(["--verbose"]) == 0
    payload = json.loads(capsys.readouterr().out)

    providers = {
        entry["provider"]: entry for entry in payload["available_providers"]
    }
    assert sorted(providers) == ["anthropic", "bedrock"]
    assert providers["anthropic"]["env_vars"] == {"ANTHROPIC_AUTH_TOKEN": True}
    assert providers["anthropic"]["llm_models"] == sorted(
        get_models_by_provider("anthropic")
    )
    assert providers["bedrock"]["env_vars"] == {
        "AWS_BEARER_TOKEN_BEDROCK": True,
        "AWS_REGION": True,
    }
    assert providers["bedrock"]["llm_models"] == sorted(
        get_models_by_provider("bedrock")
    )


def test_primary_requirement_sets_are_unchanged():
    assert PROVIDER_ENV_REQUIREMENTS["anthropic"] == ("ANTHROPIC_API_KEY",)
    assert PROVIDER_ENV_REQUIREMENTS["openai"] == ("OPENAI_API_KEY",)
    assert PROVIDER_ENV_REQUIREMENTS["bedrock"] == (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION_NAME",
    )


@pytest.mark.parametrize("provider,env_var_names", _CLIENT_CASES)
def test_client_configuration_accepts_each_listed_alternative(
    monkeypatch: pytest.MonkeyPatch, provider, env_var_names
):
    _set_env(monkeypatch, env_var_names)

    assert _client_accepts_environment(provider)


@pytest.mark.parametrize("provider,env_var_names", _CLIENT_CASES)
def test_client_configuration_needs_every_variable_of_an_alternative(
    monkeypatch: pytest.MonkeyPatch, provider, env_var_names
):
    for omitted in env_var_names:
        with monkeypatch.context() as partial_environment:
            _set_env(
                partial_environment,
                tuple(name for name in env_var_names if name != omitted),
            )

            assert not _client_accepts_environment(provider), omitted
