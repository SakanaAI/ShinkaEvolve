from pathlib import Path

import pytest

from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers.pricing import get_model_prices
from shinka.pricing.catalog import (
    PricingConfig,
    activate_model_catalog,
    catalog_from_models_dev_payload,
    get_catalog,
    refresh_model_catalog,
)
from shinka.tools.pricing.generate_csvs import TargetModel, generate_llm_rows

CURRENT_MODELS = [
    ("openai", "gpt-6-astra", 10.0, 50.0, True),
    ("openai", "gpt-6.1-sol", 2.0, 10.0, True),
    ("openai", "gpt-6.1-terra", 1.0, 5.0, False),
    ("openai", "gpt-6-luna", 0.1, 0.5, False),
    ("anthropic", "claude-fable-5-1", 10.0, 50.0, True),
    ("anthropic", "claude-fable-5-2", 8.0, 40.0, True),
    ("anthropic", "claude-opus-5-5", 4.0, 20.0, True),
    ("anthropic", "claude-sonnet-5-5", 2.0, 10.0, False),
    ("anthropic", "claude-haiku-5-5", 0.1, 0.5, False),
    ("google", "gemini-4-argon", 2.0, 10.0, False),
    ("google", "gemini-4-hydrogen", 1.5, 7.5, False),
    ("google", "gemini-3.8-flash", 0.75, 3.75, False),
    ("google", "gemini-3.5-flash-lite", 0.3, 2.5, False),
]


@pytest.fixture
def offline_catalog(tmp_path: Path):
    previous = get_catalog()
    snapshot = refresh_model_catalog(PricingConfig(mode="offline", cache_dir=tmp_path))
    assert snapshot.source == "bundled"
    try:
        yield snapshot.catalog
    finally:
        activate_model_catalog(previous)


@pytest.mark.parametrize(
    ("provider", "model_name", "input_price", "output_price", "requires_reasoning"),
    CURRENT_MODELS,
)
def test_current_models_resolve_with_bundled_prices_and_reasoning_metadata(
    offline_catalog,
    provider: str,
    model_name: str,
    input_price: float,
    output_price: float,
    requires_reasoning: bool,
):
    resolved = resolve_model_backend(model_name)
    assert resolved.provider == provider
    assert resolved.api_model_name == model_name
    assert get_model_prices(model_name) == {
        "input_price": input_price / 1_000_000,
        "output_price": output_price / 1_000_000,
    }
    entry = offline_catalog.get(provider, model_name)
    assert entry.is_reasoning
    assert entry.requires_reasoning is requires_reasoning


@pytest.mark.parametrize(
    ("provider", "model_name", "threshold", "input_price", "output_price"),
    [
        ("openai", "gpt-6-astra", 272_000, 20.0, 75.0),
        ("openai", "gpt-6.1-sol", 272_000, 4.0, 15.0),
        ("openai", "gpt-6.1-terra", 272_000, 2.0, 7.5),
        ("openai", "gpt-6-luna", 272_000, 0.2, 0.75),
        ("anthropic", "claude-haiku-5-5", 100_000, 0.5, 2.5),
    ],
)
def test_current_models_charge_context_tier_only_above_threshold(
    offline_catalog,
    provider: str,
    model_name: str,
    threshold: int,
    input_price: float,
    output_price: float,
):
    entry = offline_catalog.get(provider, model_name)
    assert entry.rates(threshold) == entry.rates()
    assert entry.rates(threshold + 1) == (
        input_price / 1_000_000,
        output_price / 1_000_000,
    )


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("openai", "gpt-6-astra"),
        ("openai", "gpt-6.1-sol"),
        ("anthropic", "claude-fable-5-1"),
        ("anthropic", "claude-fable-5-2"),
        ("anthropic", "claude-opus-5-5"),
    ],
)
def test_live_refresh_preserves_required_reasoning_without_pinning_prices(
    provider: str, model_name: str
):
    payload = {
        provider: {
            "models": {
                model_name: {
                    "cost": {"input": 1.23, "output": 4.56},
                    "reasoning": True,
                    "temperature": False,
                }
            }
        }
    }
    catalog = catalog_from_models_dev_payload(payload, include_bundled=False)

    entry = catalog.get(provider, model_name)
    assert entry.requires_reasoning
    assert entry.rates() == (1.23 / 1_000_000, 4.56 / 1_000_000)


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("openai", "gpt-5.1-codex"),
        ("openai", "gpt-5.1-codex-max"),
        ("openai", "gpt-5.1-codex-mini"),
        ("openai", "gpt-5.2-codex"),
        ("deepseek", "deepseek-chat"),
        ("deepseek", "deepseek-reasoner"),
        ("google", "gemini-3-pro-preview"),
    ],
)
def test_legacy_prices_remain_reproducible_after_upstream_removes_model(
    offline_catalog, provider: str, model_name: str
):
    rows = generate_llm_rows({}, [TargetModel("llm", model_name, provider)])

    entry = offline_catalog.get(provider, model_name)
    assert float(rows[0]["input_price"]) / 1_000_000 == entry.input_price
    assert float(rows[0]["output_price"]) / 1_000_000 == entry.output_price
    assert rows[0]["is_reasoning"] == str(entry.is_reasoning)
    assert rows[0]["requires_reasoning"] == str(int(entry.requires_reasoning))
