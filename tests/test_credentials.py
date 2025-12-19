import json

from shinka.tools.credentials import get_api_key


def test_get_api_key_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps({"OPENAI_API_KEY": "file-key"}))
    assert get_api_key("codex", credentials_path=credentials_path) == "env-key"


def test_get_api_key_from_credentials_env_var_name(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps({"OPENAI_API_KEY": "file-key"}))
    assert get_api_key("codex", credentials_path=credentials_path) == "file-key"


def test_get_api_key_from_credentials_provider_name(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps({"codex": "file-key"}))
    assert get_api_key("codex", credentials_path=credentials_path) == "file-key"
