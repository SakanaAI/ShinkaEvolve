"""Regression tests for runtime embedding-price overlays."""

from shinka.pricing.catalog import ModelPrice
from shinka.pricing.normalization import MILLION, _apply_embedding_overrides


def _embedding_price(input_price: float) -> ModelPrice:
    return ModelPrice(
        model_name="gemini-embedding-exp-03-07",
        api_model_name="gemini-embedding-exp-03-07",
        provider="google",
        kind="embedding",
        input_price=input_price,
        output_price=0.0,
    )


def test_embedding_overrides_replace_runtime_discovery_price():
    key = ("embedding", "google", "gemini-embedding-exp-03-07")
    entries = {key: _embedding_price(input_price=1.5 / MILLION)}

    _apply_embedding_overrides(
        entries,
        [
            {
                "provider": "google",
                "model_name": "gemini-embedding-exp-03-07",
                "input_price": 0.0,
            }
        ],
    )

    assert entries[key].input_price == 0.0


def test_embedding_overrides_ignore_unknown_and_malformed_entries():
    key = ("embedding", "google", "gemini-embedding-exp-03-07")
    entries = {key: _embedding_price(input_price=2.0 / MILLION)}
    _apply_embedding_overrides(entries, "not-a-list")
    _apply_embedding_overrides(entries, [{"provider": "google"}])
    _apply_embedding_overrides(
        entries,
        [{"provider": "unknown", "model_name": "unknown", "input_price": 0.0}],
    )

    assert entries[key].input_price == 2.0 / MILLION
