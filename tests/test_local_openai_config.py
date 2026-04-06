import pytest

from shinka.local_openai_config import resolve_local_openai_api_key


def test_resolve_local_openai_api_key_from_custom_env(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "test-custom-key")

    assert (
        resolve_local_openai_api_key(api_key_env_name="CUSTOM_API_KEY")
        == "test-custom-key"
    )


def test_resolve_local_openai_api_key_uses_local_fallback(monkeypatch):
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "test-local-key")

    assert resolve_local_openai_api_key(api_key_env_name=None) == "test-local-key"


def test_resolve_local_openai_api_key_defaults_to_local(monkeypatch):
    monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)

    assert resolve_local_openai_api_key(api_key_env_name=None) == "local"


def test_resolve_local_openai_api_key_missing_custom_env_raises(monkeypatch):
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CUSTOM_API_KEY"):
        resolve_local_openai_api_key(api_key_env_name="CUSTOM_API_KEY")
