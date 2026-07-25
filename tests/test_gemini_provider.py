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


def test_gemini_3_6_generation_config_omits_sampling_params(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gemini.types, "GenerateContentConfig", FakeConfig)
    monkeypatch.setattr(gemini, "build_gemini_afc_config", lambda: "afc")
    monkeypatch.setattr(
        gemini,
        "build_gemini_thinking_config",
        lambda budget: {"budget": budget},
    )

    gemini.build_gemini_generation_config(
        model="gemini-3.6-flash",
        temperature=0.0,
        top_p=1.0,
        max_tokens=4096,
        system_instruction="system",
        thinking_budget=1024,
    )

    assert "temperature" not in captured
    assert "top_p" not in captured
    assert captured["max_output_tokens"] == 4096
