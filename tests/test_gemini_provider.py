import asyncio

import pytest

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


def test_gemini_adc_error_is_non_retryable():
    assert gemini.is_non_retryable_gemini_error(
        Exception("Your default credentials were not found")
    )


def test_gemini_api_key_error_is_non_retryable():
    assert gemini.is_non_retryable_gemini_error(Exception("API key not valid."))


def test_gemini_timeout_error_remains_retryable():
    assert not gemini.is_non_retryable_gemini_error(Exception("deadline exceeded"))


def test_query_gemini_does_not_retry_non_retryable_auth_errors():
    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Your default credentials were not found")

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    client = FakeClient()

    with pytest.raises(RuntimeError, match="default credentials"):
        gemini.query_gemini(
            client=client,
            model="gemini-3-flash-preview",
            msg="hello",
            system_msg="",
            msg_history=[],
            output_model=None,
        )

    assert client.models.calls == 1


def test_query_gemini_async_does_not_retry_non_retryable_auth_errors():
    class FakeModels:
        def __init__(self):
            self.calls = 0

        async def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Your default credentials were not found")

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self):
            self.aio = FakeAio()

    async def run_test():
        client = FakeClient()

        with pytest.raises(RuntimeError, match="default credentials"):
            await gemini.query_gemini_async(
                client=client,
                model="gemini-3-flash-preview",
                msg="hello",
                system_msg="",
                msg_history=[],
                output_model=None,
            )

        assert client.aio.models.calls == 1

    asyncio.run(run_test())
