import pytest

import shinka.llm.client as llm_client_module
from shinka.google_genai import _google_genai_timeout_ms
from shinka.llm.client import get_async_client_llm, get_client_llm
from shinka.llm.constants import TIMEOUT


def test_google_genai_timeout_is_in_milliseconds():
    assert _google_genai_timeout_ms(TIMEOUT) == TIMEOUT * 1000


def test_get_client_llm_dynamic_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    client, model_name, provider = get_client_llm("openrouter/qwen/qwen3-coder")

    assert provider == "openrouter"
    assert model_name == "qwen/qwen3-coder"
    assert "openrouter.ai" in str(client.base_url)


def test_get_client_llm_local_openai_inline_url(monkeypatch):
    monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
    client, model_name, provider = get_client_llm(
        "local/qwen2.5-coder@http://localhost:11434/v1"
    )

    assert provider == "local_openai"
    assert model_name == "qwen2.5-coder"
    assert str(client.base_url).startswith("http://localhost:11434")


def test_get_async_client_llm_local_openai_inline_url(monkeypatch):
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "test-local-key")
    client, model_name, provider = get_async_client_llm(
        "local/qwen2.5-coder@http://localhost:11434/v1"
    )

    assert provider == "local_openai"
    assert model_name == "qwen2.5-coder"
    assert str(client.base_url).startswith("http://localhost:11434")


def test_get_async_client_llm_openai_sets_timeout(monkeypatch):
    captured_kwargs = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(llm_client_module.openai, "AsyncOpenAI", _FakeAsyncOpenAI)

    _client, model_name, provider = get_async_client_llm("gpt-5.4-mini")

    assert provider == "openai"
    assert model_name == "gpt-5.4-mini"
    assert captured_kwargs["timeout"] == llm_client_module.TIMEOUT


def test_get_client_llm_gemini_sets_timeout(monkeypatch):
    captured_kwargs = {}
    fake_client = object()

    def _fake_build_google_genai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        llm_client_module,
        "build_google_genai_client",
        _fake_build_google_genai_client,
    )

    client, model_name, provider = get_client_llm("gemini-2.5-flash")

    assert client is fake_client
    assert provider == "google"
    assert model_name == "gemini-2.5-flash"
    assert captured_kwargs == {"timeout_ms": TIMEOUT * 1000}


def test_get_async_client_llm_gemini_sets_timeout(monkeypatch):
    captured_kwargs = {}
    fake_client = object()

    def _fake_build_google_genai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        llm_client_module,
        "build_google_genai_client",
        _fake_build_google_genai_client,
    )

    client, model_name, provider = get_async_client_llm("gemini-2.5-flash")

    assert client is fake_client
    assert provider == "google"
    assert model_name == "gemini-2.5-flash"
    assert captured_kwargs == {"timeout_ms": TIMEOUT * 1000}


def test_get_client_llm_local_openai_uses_api_key_env_query_param(monkeypatch):
    captured_kwargs = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.base_url = kwargs["base_url"]

    monkeypatch.setenv("CUSTOM_API_KEY", "test-custom-key")
    monkeypatch.setattr(llm_client_module.openai, "OpenAI", _FakeOpenAI)

    client, model_name, provider = get_client_llm(
        "local/dummy-model@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
    )

    assert provider == "local_openai"
    assert model_name == "dummy-model"
    assert str(client.base_url).startswith("https://api.example.test/v1")
    assert captured_kwargs["api_key"] == "test-custom-key"


def test_get_async_client_llm_local_openai_missing_api_key_env_raises(monkeypatch):
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CUSTOM_API_KEY"):
        get_async_client_llm(
            "local/dummy-model@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
        )
