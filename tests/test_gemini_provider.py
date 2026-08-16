import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest
from google.genai import types

from shinka.llm.providers import gemini


def test_build_gemini_thinking_config_omits_budget_when_not_supported(monkeypatch):
    captured = {}

    class ThinkingConfigNoBudget:
        model_fields = {"include_thoughts": object()}

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gemini.types, "ThinkingConfig", ThinkingConfigNoBudget)

    gemini.build_gemini_thinking_config(thinking_budget=0)

    assert captured == {"include_thoughts": True}


def test_build_gemini_thinking_config_includes_budget_when_supported(monkeypatch):
    captured = {}

    class ThinkingConfigWithBudget:
        model_fields = {
            "include_thoughts": object(),
            "thinking_budget": object(),
        }

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gemini.types, "ThinkingConfig", ThinkingConfigWithBudget)

    gemini.build_gemini_thinking_config(thinking_budget=256)

    assert captured == {"include_thoughts": True, "thinking_budget": 256}


def test_build_gemini_afc_config_sets_max_remote_calls_none(monkeypatch):
    captured = {}

    class AutomaticFunctionCallingConfig:
        model_fields = {
            "disable": object(),
            "maximum_remote_calls": object(),
        }

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        gemini.types,
        "AutomaticFunctionCallingConfig",
        AutomaticFunctionCallingConfig,
    )

    gemini.build_gemini_afc_config()

    assert captured == {"disable": True, "maximum_remote_calls": None}


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("LOW", types.ThinkingLevel.LOW),
        ("low", types.ThinkingLevel.LOW),
        ("MEDIUM", types.ThinkingLevel.MEDIUM),
        ("medium", types.ThinkingLevel.MEDIUM),
        ("HIGH", types.ThinkingLevel.HIGH),
        ("high", types.ThinkingLevel.HIGH),
        (None, None),
    ],
)
def test_build_gemini_thinking_level_config_uses_sdk_enum(
    monkeypatch,
    level: str | None,
    expected: types.ThinkingLevel | None,
):
    captured = {}

    class ThinkingConfig:
        model_fields: ClassVar[dict[str, object]] = {
            "include_thoughts": object(),
            "thinking_level": object(),
        }

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gemini.types, "ThinkingConfig", ThinkingConfig)

    gemini.build_gemini_thinking_level_config(level)

    assert captured["include_thoughts"] is True
    if expected is None:
        assert "thinking_level" not in captured
    else:
        assert captured["thinking_level"] == expected


def test_build_gemini_thinking_level_config_rejects_unknown_level():
    with pytest.raises(ValueError, match="NOT_A_LEVEL"):
        gemini.build_gemini_thinking_level_config("NOT_A_LEVEL")


@pytest.mark.parametrize("model_name", ["gemini-3.6-flash", "gemini-3.7-flash"])
def test_latest_gemini_generation_config_omits_unsupported_fields(
    monkeypatch,
    model_name: str,
):
    captured = {}

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        gemini.types,
        "GenerateContentConfig",
        GenerateContentConfig,
    )
    monkeypatch.setattr(gemini, "build_gemini_afc_config", lambda: "afc")
    monkeypatch.setattr(
        gemini,
        "build_gemini_thinking_level_config",
        lambda level: {"thinking_level": level},
    )

    gemini.build_gemini_generation_config(
        model=model_name,
        temperature=0.25,
        top_p=0.9,
        max_tokens=4096,
        system_instruction="system",
        thinking_budget=1024,
        thinking_level="MEDIUM",
    )

    assert captured == {
        "max_output_tokens": 4096,
        "system_instruction": "system",
        "automatic_function_calling": "afc",
        "thinking_config": {"thinking_level": "MEDIUM"},
    }
    for unsupported_field in (
        "temperature",
        "top_p",
        "top_k",
        "candidate_count",
        "thinking_budget",
    ):
        assert unsupported_field not in captured


def test_legacy_gemini_generation_config_keeps_sampling_and_budget(monkeypatch):
    captured = {}

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        gemini.types,
        "GenerateContentConfig",
        GenerateContentConfig,
    )
    monkeypatch.setattr(gemini, "build_gemini_afc_config", lambda: "afc")
    monkeypatch.setattr(
        gemini,
        "build_gemini_thinking_config",
        lambda budget: {"thinking_budget": budget},
    )

    gemini.build_gemini_generation_config(
        model="gemini-2.5-flash",
        temperature=0.25,
        top_p=0.9,
        max_tokens=4096,
        system_instruction="system",
        thinking_budget=1024,
        thinking_level=None,
    )

    assert captured == {
        "temperature": 0.25,
        "top_p": 0.9,
        "max_output_tokens": 4096,
        "system_instruction": "system",
        "automatic_function_calling": "afc",
        "thinking_config": {"thinking_budget": 1024},
    }


def test_sync_query_uses_shared_generation_config_builder(monkeypatch):
    captured = {}
    response = SimpleNamespace(candidates=[], text="answer", usage_metadata=None)

    class Models:
        def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return response

    def build_config(**kwargs):
        captured["config_kwargs"] = kwargs
        return "config"

    monkeypatch.setattr(gemini, "build_gemini_generation_config", build_config)

    result = gemini.query_gemini(
        SimpleNamespace(models=Models()),
        "gemini-3.6-flash",
        "prompt",
        "system",
        [],
        None,
        temperature=0.25,
        top_p=0.9,
        top_k=40,
        candidate_count=2,
        max_tokens=4096,
        thinking_budget=2048,
        thinking_level="HIGH",
    )

    assert result.content == "answer"
    assert captured["config_kwargs"] == {
        "model": "gemini-3.6-flash",
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 4096,
        "system_instruction": "system",
        "thinking_budget": 2048,
        "thinking_level": "HIGH",
    }
    assert captured["request"]["config"] == "config"


def test_async_query_uses_shared_generation_config_builder(monkeypatch):
    captured = {}
    response = SimpleNamespace(candidates=[], text="answer", usage_metadata=None)

    class Models:
        async def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return response

    def build_config(**kwargs):
        captured["config_kwargs"] = kwargs
        return "config"

    monkeypatch.setattr(gemini, "build_gemini_generation_config", build_config)

    result = asyncio.run(
        gemini.query_gemini_async(
            SimpleNamespace(aio=SimpleNamespace(models=Models())),
            "gemini-3.6-flash",
            "prompt",
            "system",
            [],
            None,
            max_tokens=4096,
            thinking_level="LOW",
        )
    )

    assert result.content == "answer"
    assert captured["config_kwargs"] == {
        "model": "gemini-3.6-flash",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 4096,
        "system_instruction": "system",
        "thinking_budget": 1024,
        "thinking_level": "LOW",
    }
    assert captured["request"]["config"] == "config"
