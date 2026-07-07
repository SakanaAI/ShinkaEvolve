import socket

import pytest

import shinka.google_genai as google_genai_module


@pytest.fixture(autouse=True)
def _use_system_google_genai_network(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHINKA_GOOGLE_GENAI_IP_FAMILY", "system")


def _addr(family: socket.AddressFamily, host: str):
    sockaddr = (host, 443) if family == socket.AF_INET else (host, 443, 0, 0)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)


def test_google_api_host_detection_is_scoped():
    assert google_genai_module._is_google_api_host("generativelanguage.googleapis.com")
    assert google_genai_module._is_google_api_host(
        "us-central1-aiplatform.googleapis.com"
    )
    assert not google_genai_module._is_google_api_host("api.openai.com")
    assert not google_genai_module._is_google_api_host("notgoogleapis.com")


def test_configure_google_genai_network_ipv4_first_reorders_only_google_hosts(
    monkeypatch: pytest.MonkeyPatch,
):
    mixed_addresses = [
        _addr(socket.AF_INET6, "2001:db8::1"),
        _addr(socket.AF_INET6, "2001:db8::2"),
        _addr(socket.AF_INET, "203.0.113.1"),
        _addr(socket.AF_INET, "203.0.113.2"),
    ]

    def _fake_getaddrinfo(*args, **kwargs):
        return list(mixed_addresses)

    monkeypatch.setenv("SHINKA_GOOGLE_GENAI_IP_FAMILY", "ipv4_first")
    monkeypatch.setattr(google_genai_module, "_ORIGINAL_GETADDRINFO", _fake_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    policy = google_genai_module.configure_google_genai_network()

    assert policy == "ipv4_first"
    google_addresses = socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    non_google_addresses = socket.getaddrinfo("api.openai.com", 443)
    assert [address[0] for address in google_addresses] == [
        socket.AF_INET,
        socket.AF_INET,
        socket.AF_INET6,
        socket.AF_INET6,
    ]
    assert non_google_addresses == mixed_addresses


def test_configure_google_genai_network_ipv4_filters_google_hosts(
    monkeypatch: pytest.MonkeyPatch,
):
    mixed_addresses = [
        _addr(socket.AF_INET6, "2001:db8::1"),
        _addr(socket.AF_INET, "203.0.113.1"),
    ]

    def _fake_getaddrinfo(*args, **kwargs):
        return list(mixed_addresses)

    monkeypatch.setenv("SHINKA_GOOGLE_GENAI_IP_FAMILY", "ipv4")
    monkeypatch.setattr(google_genai_module, "_ORIGINAL_GETADDRINFO", _fake_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    policy = google_genai_module.configure_google_genai_network()

    assert policy == "ipv4"
    addresses = socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    assert [address[0] for address in addresses] == [socket.AF_INET]


def test_configure_google_genai_network_respects_explicit_family(
    monkeypatch: pytest.MonkeyPatch,
):
    mixed_addresses = [
        _addr(socket.AF_INET6, "2001:db8::1"),
        _addr(socket.AF_INET, "203.0.113.1"),
    ]

    def _fake_getaddrinfo(*args, **kwargs):
        return list(mixed_addresses)

    monkeypatch.setenv("SHINKA_GOOGLE_GENAI_IP_FAMILY", "ipv4")
    monkeypatch.setattr(google_genai_module, "_ORIGINAL_GETADDRINFO", _fake_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    google_genai_module.configure_google_genai_network()

    addresses = socket.getaddrinfo(
        "generativelanguage.googleapis.com",
        443,
        family=socket.AF_INET6,
    )
    assert addresses == mixed_addresses


def test_configure_google_genai_network_auto_installs_detected_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    mixed_addresses = [
        _addr(socket.AF_INET6, "2001:db8::1"),
        _addr(socket.AF_INET, "203.0.113.1"),
    ]

    def _fake_getaddrinfo(*args, **kwargs):
        return list(mixed_addresses)

    monkeypatch.setenv("SHINKA_GOOGLE_GENAI_IP_FAMILY", "auto")
    monkeypatch.setattr(google_genai_module, "_ORIGINAL_GETADDRINFO", _fake_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr(
        google_genai_module,
        "_detect_google_genai_ip_policy",
        lambda: "ipv4_first",
    )

    policy = google_genai_module.configure_google_genai_network()

    assert policy == "ipv4_first"
    addresses = socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    assert [address[0] for address in addresses] == [socket.AF_INET, socket.AF_INET6]


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
    configured_policies = []

    class _FakeHttpOptions:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]

    def _fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    def _fake_configure_google_genai_network():
        configured_policies.append("system")
        return "system"

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(google_genai_module.genai.types, "HttpOptions", _FakeHttpOptions)
    monkeypatch.setattr(google_genai_module.genai, "Client", _fake_client)
    monkeypatch.setattr(
        google_genai_module,
        "configure_google_genai_network",
        _fake_configure_google_genai_network,
    )

    client = google_genai_module.build_google_genai_client(timeout_ms=1234)

    assert client is fake_client
    assert configured_policies == ["system"]
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
    configured_policies = []

    def _fake_configure_google_genai_network():
        configured_policies.append("system")
        return "system"

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        google_genai_module,
        "configure_google_genai_network",
        _fake_configure_google_genai_network,
    )

    with pytest.raises(ValueError, match="GEMINI_API_KEY") as exc_info:
        google_genai_module.build_google_genai_client()

    error_message = str(exc_info.value)
    assert "Gemini API mode" in error_message
    assert "GOOGLE_GENAI_USE_VERTEXAI" in error_message
    assert "GOOGLE_CLOUD_PROJECT" in error_message
    assert "GOOGLE_CLOUD_LOCATION" in error_message
    assert configured_policies == []


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
