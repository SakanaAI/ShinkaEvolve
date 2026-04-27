from __future__ import annotations

from pathlib import Path

import pytest

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.model_availability import validate_model_env_access


_PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_API_ENDPOINT",
    "AZURE_API_VERSION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION_NAME",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "LOCAL_OPENAI_API_KEY",
)


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var_name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var_name, raising=False)


def test_validate_model_env_access_rejects_missing_llm_provider_key(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_provider_env(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        validate_model_env_access(llm_models=["gpt-5-mini"])

    error = str(exc_info.value)
    assert "gpt-5-mini" in error
    assert "OPENAI_API_KEY" in error


def test_validate_model_env_access_allows_local_models_without_env_key(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_provider_env(monkeypatch)

    validate_model_env_access(
        llm_models=["local/qwen2.5-coder@http://localhost:11434/v1"],
        embedding_models=["local/BAAI/bge-small-en-v1.5@http://localhost:8080/v1"],
    )


def test_validate_model_env_access_rejects_local_model_with_missing_key_env(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_provider_env(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        validate_model_env_access(
            llm_models=[
                "local/qwen2.5-coder@https://api.example.test/v1?api_key_env=CUSTOM_API_KEY"
            ],
        )

    error = str(exc_info.value)
    assert "CUSTOM_API_KEY" in error
    assert "local/qwen2.5-coder" in error


def test_validate_model_env_access_rejects_missing_embedding_provider_key(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_provider_env(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        validate_model_env_access(embedding_models=["text-embedding-3-small"])

    error = str(exc_info.value)
    assert "text-embedding-3-small" in error
    assert "OPENAI_API_KEY" in error


def test_validate_model_env_access_rejects_incomplete_vertex_env(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    with pytest.raises(ValueError) as exc_info:
        validate_model_env_access(llm_models=["gemini-3-flash-preview"])

    error = str(exc_info.value)
    assert "gemini-3-flash-preview" in error
    assert "GOOGLE_CLOUD_LOCATION" in error


def test_async_runner_fails_fast_when_requested_model_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _clear_provider_env(monkeypatch)
    results_dir = tmp_path / "results"

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        ShinkaEvolveRunner(
            evo_config=EvolutionConfig(
                llm_models=["gpt-5-mini"],
                llm_dynamic_selection=None,
                meta_rec_interval=None,
                embedding_model=None,
                num_generations=1,
                results_dir=str(results_dir),
            ),
            job_config=LocalJobConfig(),
            db_config=DatabaseConfig(),
            verbose=False,
        )

    assert not results_dir.exists()
