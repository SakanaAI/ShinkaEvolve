import shinka.llm.client as llm_client_module
from shinka.llm.client import get_async_client_llm, get_client_llm


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
