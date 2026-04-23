import pytest

import shinka.google_genai as google_genai_module


def test_google_genai_timeout_is_in_milliseconds():
    assert google_genai_module._google_genai_timeout_ms(1200) == 1_200_000


@pytest.mark.parametrize("flag_value", ["1", "true", "yes", "on", "TRUE"])
def test_google_genai_auth_mode_uses_vertexai_for_truthy_flag(
    monkeypatch: pytest.MonkeyPatch, flag_value: str
):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", flag_value)

    assert google_genai_module.google_genai_auth_mode() == "vertexai"


def test_google_genai_auth_mode_defaults_to_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    assert google_genai_module.google_genai_auth_mode() == "api_key"


def test_build_google_genai_client_uses_api_key(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs = {}
    fake_client = object()

    class _FakeHttpOptions:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]

    def _fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(google_genai_module.genai.types, "HttpOptions", _FakeHttpOptions)
    monkeypatch.setattr(google_genai_module.genai, "Client", _fake_client)

    client = google_genai_module.build_google_genai_client(timeout_ms=1234)

    assert client is fake_client
    assert captured_kwargs["api_key"] == "test-gemini-key"
    assert captured_kwargs["http_options"].timeout == 1234


def test_build_google_genai_client_uses_vertexai(monkeypatch: pytest.MonkeyPatch):
    captured_kwargs = {}
    fake_client = object()

    class _FakeHttpOptions:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]

    def _fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(google_genai_module.genai.types, "HttpOptions", _FakeHttpOptions)
    monkeypatch.setattr(google_genai_module.genai, "Client", _fake_client)

    client = google_genai_module.build_google_genai_client(timeout_ms=5678)

    assert client is fake_client
    assert captured_kwargs["vertexai"] is True
    assert captured_kwargs["project"] == "test-project"
    assert captured_kwargs["location"] == "us-central1"
    assert captured_kwargs["http_options"].timeout == 5678


def test_build_google_genai_client_requires_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY") as exc_info:
        google_genai_module.build_google_genai_client()

    error_message = str(exc_info.value)
    assert "Gemini API mode" in error_message
    assert "GOOGLE_GENAI_USE_VERTEXAI" in error_message
    assert "GOOGLE_CLOUD_PROJECT" in error_message
    assert "GOOGLE_CLOUD_LOCATION" in error_message


def test_build_google_genai_client_requires_vertex_project(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        google_genai_module.build_google_genai_client()


def test_build_google_genai_client_requires_vertex_location(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_LOCATION"):
        google_genai_module.build_google_genai_client()
