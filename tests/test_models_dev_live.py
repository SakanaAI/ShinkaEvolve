from __future__ import annotations

from pathlib import Path

import pytest

from shinka.pricing.catalog import PricingConfig, PricingMode, refresh_model_catalog


@pytest.mark.models_dev_live
def test_live_models_dev_catalog_prices_representative_models(tmp_path: Path) -> None:
    snapshot = refresh_model_catalog(
        PricingConfig(mode=PricingMode.REQUIRED, cache_dir=tmp_path)
    )

    assert snapshot.source == "live"
    for model_name in (
        "gpt-5-mini",
        "gemini-3-flash-preview",
        "claude-sonnet-4-6",
        "deepseek-chat",
        "openrouter/qwen/qwen3-coder",
        "us.anthropic.claude-sonnet-4-6-v1:0",
    ):
        price = snapshot.catalog.get_by_name(model_name)
        assert price.input_price >= 0
        assert price.output_price is not None and price.output_price >= 0

    embedding = snapshot.catalog.get_by_name("text-embedding-3-small", kind="embedding")
    assert embedding.input_price > 0
