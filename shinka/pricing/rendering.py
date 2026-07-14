"""Render runtime catalog entries in the bundled CSV schema."""

from __future__ import annotations

from .catalog import ModelKind, ModelPrice, PricingCatalog
from .normalization import MILLION


def render_llm_row(
    catalog: PricingCatalog, model_name: str, provider: str
) -> dict[str, str]:
    entry = _catalog_entry(catalog, "llm", model_name, provider)
    if entry.input_price is None or entry.output_price is None:
        raise KeyError(f"Missing pricing for {provider}/{model_name}")
    return {
        "model_name": model_name,
        "provider": provider,
        "input_price": _format_price(entry.input_price * MILLION),
        "output_price": _format_price(entry.output_price * MILLION),
        "input_price_tier2": _format_optional_price(entry.input_price_tier2),
        "output_price_tier2": _format_optional_price(entry.output_price_tier2),
        "tier_threshold": (
            "" if entry.tier_threshold is None else str(entry.tier_threshold)
        ),
        "is_reasoning": str(entry.is_reasoning),
        "think_temp_fixed": _format_bool(entry.think_temp_fixed),
        "requires_reasoning": _format_bool(entry.requires_reasoning),
    }


def render_embedding_row(
    catalog: PricingCatalog, model_name: str, provider: str
) -> dict[str, str]:
    entry = _catalog_entry(catalog, "embedding", model_name, provider)
    if entry.input_price is None:
        raise KeyError(f"Missing pricing for {provider}/{model_name}")
    return {
        "model_name": model_name,
        "provider": provider,
        "input_price": _format_price(entry.input_price * MILLION),
    }


def _catalog_entry(
    catalog: PricingCatalog, kind: ModelKind, model_name: str, provider: str
) -> ModelPrice:
    try:
        return catalog.get(provider, model_name, kind=kind)
    except KeyError:
        return catalog.get_by_name(model_name, kind=kind)


def _format_optional_price(value: float | None) -> str:
    return "" if value is None else _format_price(value * MILLION)


def _format_price(value: float) -> str:
    return str(float(value))


def _format_bool(value: bool) -> str:
    return "1" if value else "0"
