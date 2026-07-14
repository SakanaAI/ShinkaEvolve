"""Generate the bundled offline pricing fallback from models.dev.

Runtime startup uses the same upstream catalog directly. This maintainer tool
keeps release artifacts current for offline and outage fallback behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

from shinka.pricing.catalog import PricingCatalog, catalog_from_models_dev_payload
from shinka.pricing.rendering import render_embedding_row, render_llm_row


MODELS_DEV_API_URL = "https://models.dev/api.json"
MODELS_DEV_USER_AGENT = "ShinkaEvolve pricing refresh"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_JSON = Path(__file__).resolve().parent / "models_dev_overlay.json"
LLM_PRICING_CSV = PACKAGE_ROOT / "llm" / "providers" / "pricing.csv"
EMBEDDING_PRICING_CSV = PACKAGE_ROOT / "embed" / "providers" / "pricing.csv"
FIXED_TEMPERATURE_WHEN_REASONING_PROVIDERS = {"anthropic", "bedrock", "openrouter"}

LLM_HEADERS = [
    "model_name",
    "provider",
    "input_price",
    "output_price",
    "input_price_tier2",
    "output_price_tier2",
    "tier_threshold",
    "is_reasoning",
    "think_temp_fixed",
    "requires_reasoning",
]
EMBEDDING_HEADERS = ["model_name", "provider", "input_price"]


@dataclass(frozen=True)
class TargetModel:
    kind: str
    model_name: str
    provider: str


@dataclass(frozen=True)
class SourceModel:
    provider: str
    model_name: str


@dataclass(frozen=True)
class LLMOverride:
    input_price: float | str | None = None
    output_price: float | str | None = None
    is_reasoning: bool | None = None
    think_temp_fixed: bool | None = None
    requires_reasoning: bool | None = None
    input_price_tier2: float | None = None
    output_price_tier2: float | None = None
    tier_threshold: int | None = None


@dataclass(frozen=True)
class EmbeddingOverride:
    input_price: float | str | None = None


@dataclass(frozen=True)
class PricingOverlay:
    provider_aliases: dict[str, str]
    model_aliases: dict[tuple[str, str, str], SourceModel]
    llm_overrides: dict[tuple[str, str], LLMOverride]
    embedding_overrides: dict[tuple[str, str], EmbeddingOverride]
    fixed_temperature_when_reasoning_providers: set[str]
    strip_prefix_aliases: dict[tuple[str, str], str]


def load_models_dev_payload(source: str) -> dict[str, Any]:
    """Load models.dev JSON from a local path or URL."""
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(
            source,
            headers={"User-Agent": MODELS_DEV_USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    with Path(source).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_pricing_overlay(path: Path = OVERLAY_JSON) -> PricingOverlay:
    with path.open(encoding="utf-8") as handle:
        raw_overlay = json.load(handle)

    return PricingOverlay(
        provider_aliases=dict(raw_overlay.get("provider_aliases", {})),
        model_aliases={
            _target_key(entry): SourceModel(
                provider=entry["source_provider"],
                model_name=entry["source_model_name"],
            )
            for entry in raw_overlay.get("model_aliases", [])
        },
        llm_overrides={
            (entry["provider"], entry["model_name"]): LLMOverride(
                input_price=entry.get("input_price"),
                output_price=entry.get("output_price"),
                is_reasoning=entry.get("is_reasoning"),
                think_temp_fixed=entry.get("think_temp_fixed"),
                requires_reasoning=entry.get("requires_reasoning"),
                input_price_tier2=entry.get("input_price_tier2"),
                output_price_tier2=entry.get("output_price_tier2"),
                tier_threshold=entry.get("tier_threshold"),
            )
            for entry in raw_overlay.get("llm_overrides", [])
        },
        embedding_overrides={
            (entry["provider"], entry["model_name"]): EmbeddingOverride(
                input_price=entry.get("input_price"),
            )
            for entry in raw_overlay.get("embedding_overrides", [])
        },
        fixed_temperature_when_reasoning_providers=set(
            raw_overlay.get(
                "fixed_temperature_when_reasoning_providers",
                FIXED_TEMPERATURE_WHEN_REASONING_PROVIDERS,
            )
        ),
        strip_prefix_aliases={
            (entry["kind"], entry["provider"]): entry["prefix"]
            for entry in raw_overlay.get("strip_prefix_aliases", [])
        },
    )


def read_targets(csv_path: Path, kind: str, headers: list[str]) -> list[TargetModel]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != headers:
            raise ValueError(
                f"{csv_path} headers do not match expected contract: {reader.fieldnames}"
            )
        return [
            TargetModel(
                kind=kind,
                model_name=row["model_name"].strip(),
                provider=row["provider"].strip(),
            )
            for row in reader
        ]


def generate_llm_rows(
    payload: dict[str, Any],
    targets: Iterable[TargetModel],
    overlay: PricingOverlay | None = None,
) -> list[dict[str, str]]:
    overlay = load_pricing_overlay() if overlay is None else overlay
    rows = []
    for target in targets:
        if target.kind != "llm":
            raise ValueError(f"Expected LLM target, got {target.kind}")
        rows.append(_generate_llm_row(payload, target, overlay))
    return rows


def generate_embedding_rows(
    payload: dict[str, Any],
    targets: Iterable[TargetModel],
    overlay: PricingOverlay | None = None,
) -> list[dict[str, str]]:
    overlay = load_pricing_overlay() if overlay is None else overlay
    rows = []
    for target in targets:
        if target.kind != "embedding":
            raise ValueError(f"Expected embedding target, got {target.kind}")
        rows.append(_generate_embedding_row(payload, target, overlay))
    return rows


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_csv(headers, rows), encoding="utf-8")


def render_csv(headers: list[str], rows: list[dict[str, str]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _generate_llm_row(
    payload: dict[str, Any], target: TargetModel, overlay: PricingOverlay
) -> dict[str, str]:
    llm_override = overlay.llm_overrides.get((target.provider, target.model_name))
    upstream_model = _lookup_model(payload, target, overlay)

    if upstream_model is None and llm_override is None:
        raise ValueError(
            f"No models.dev model or manual LLM overlay for "
            f"{target.provider}/{target.model_name}"
        )

    if upstream_model is None:
        assert llm_override is not None
        return _llm_override_row(target, llm_override)

    cost = upstream_model.get("cost") or {}
    input_price = _override_or_default(llm_override, "input_price", cost.get("input"))
    output_price = _override_or_default(
        llm_override, "output_price", cost.get("output")
    )
    if input_price is None or output_price is None:
        if llm_override is None:
            raise ValueError(
                f"Missing input/output cost for {target.provider}/{target.model_name}"
            )
        return _llm_override_row(target, llm_override)

    tier = _override_tier(llm_override) or _first_context_tier(cost)
    is_reasoning = _override_or_default(
        llm_override, "is_reasoning", bool(upstream_model.get("reasoning", False))
    )
    think_temp_fixed = _override_or_default(
        llm_override,
        "think_temp_fixed",
        _requires_fixed_temperature(target, upstream_model, is_reasoning, overlay),
    )
    requires_reasoning = _override_or_default(llm_override, "requires_reasoning", False)

    return {
        "model_name": target.model_name,
        "provider": target.provider,
        "input_price": _format_price(input_price),
        "output_price": _format_price(output_price),
        "input_price_tier2": _format_optional_price(
            None if tier is None else tier.get("input")
        ),
        "output_price_tier2": _format_optional_price(
            None if tier is None else tier.get("output")
        ),
        "tier_threshold": "" if tier is None else str(tier["size"]),
        "is_reasoning": str(is_reasoning),
        "think_temp_fixed": _format_bool_int(think_temp_fixed),
        "requires_reasoning": _format_bool_int(requires_reasoning),
    }


def _generate_embedding_row(
    payload: dict[str, Any], target: TargetModel, overlay: PricingOverlay
) -> dict[str, str]:
    embedding_override = overlay.embedding_overrides.get(
        (target.provider, target.model_name)
    )
    upstream_model = _lookup_model(payload, target, overlay)

    if upstream_model is None and embedding_override is None:
        raise ValueError(
            f"No models.dev model or manual embedding overlay for "
            f"{target.provider}/{target.model_name}"
        )

    if upstream_model is None:
        assert embedding_override is not None
        return {
            "model_name": target.model_name,
            "provider": target.provider,
            "input_price": _format_price(embedding_override.input_price),
        }

    input_price = (upstream_model.get("cost") or {}).get("input")
    if input_price is None:
        if embedding_override is None or embedding_override.input_price is None:
            raise ValueError(
                f"Missing input cost for {target.provider}/{target.model_name}"
            )
        input_price = embedding_override.input_price
    elif embedding_override is not None and embedding_override.input_price is not None:
        input_price = embedding_override.input_price

    return {
        "model_name": target.model_name,
        "provider": target.provider,
        "input_price": _format_price(input_price),
    }


def _lookup_model(
    payload: dict[str, Any], target: TargetModel, overlay: PricingOverlay
) -> dict[str, Any] | None:
    source_provider, source_model = _source_model(target, overlay)
    provider_payload = payload.get(source_provider) or {}
    models = provider_payload.get("models") or {}
    model = models.get(source_model)
    if model is not None:
        return model

    strip_prefix = overlay.strip_prefix_aliases.get((target.kind, target.provider))
    if strip_prefix and target.model_name.startswith(strip_prefix):
        return models.get(target.model_name.removeprefix(strip_prefix))

    return None


def _source_model(target: TargetModel, overlay: PricingOverlay) -> tuple[str, str]:
    alias = overlay.model_aliases.get((target.kind, target.provider, target.model_name))
    if alias is not None:
        return alias.provider, alias.model_name
    return (
        overlay.provider_aliases.get(target.provider, target.provider),
        target.model_name,
    )


def _first_context_tier(cost: dict[str, Any]) -> dict[str, Any] | None:
    for tier in cost.get("tiers") or []:
        tier_info = tier.get("tier") or {}
        if tier_info.get("type") == "context" and tier_info.get("size") is not None:
            return {
                "input": tier.get("input"),
                "output": tier.get("output"),
                "size": int(tier_info["size"]),
            }
    return None


def _override_tier(override: LLMOverride | None) -> dict[str, Any] | None:
    if override is None:
        return None
    if override.input_price_tier2 is None or override.output_price_tier2 is None:
        return None
    if override.tier_threshold is None:
        return None
    return {
        "input": override.input_price_tier2,
        "output": override.output_price_tier2,
        "size": override.tier_threshold,
    }


def _requires_fixed_temperature(
    target: TargetModel,
    upstream_model: dict[str, Any],
    is_reasoning: bool,
    overlay: PricingOverlay,
) -> bool:
    if upstream_model.get("temperature") is False:
        return True
    return (
        target.provider in overlay.fixed_temperature_when_reasoning_providers
        and is_reasoning
    )


def _llm_override_row(target: TargetModel, override: LLMOverride) -> dict[str, str]:
    if override.input_price is None or override.output_price is None:
        raise ValueError(
            f"Missing manual prices for {target.provider}/{target.model_name}"
        )

    return {
        "model_name": target.model_name,
        "provider": target.provider,
        "input_price": _format_price(override.input_price),
        "output_price": _format_price(override.output_price),
        "input_price_tier2": _format_optional_price(override.input_price_tier2),
        "output_price_tier2": _format_optional_price(override.output_price_tier2),
        "tier_threshold": (
            "" if override.tier_threshold is None else str(override.tier_threshold)
        ),
        "is_reasoning": str(bool(override.is_reasoning)),
        "think_temp_fixed": _format_bool_int(bool(override.think_temp_fixed)),
        "requires_reasoning": _format_bool_int(bool(override.requires_reasoning)),
    }


def _target_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return entry["kind"], entry["provider"], entry["model_name"]


def _override_or_default(
    override: LLMOverride | None, field_name: str, default: Any
) -> Any:
    if override is None:
        return default
    value = getattr(override, field_name)
    return default if value is None else value


def _format_price(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    return str(float(value))


def _format_optional_price(value: float | int | str | None) -> str:
    if value is None:
        return ""
    return _format_price(value)


def _format_bool_int(value: bool) -> str:
    return "1" if value else "0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Shinka pricing CSV files from models.dev metadata.",
    )
    parser.add_argument(
        "--api-json",
        default=MODELS_DEV_API_URL,
        help="Path or URL for the models.dev API JSON payload.",
    )
    parser.add_argument(
        "--llm-output",
        type=Path,
        default=LLM_PRICING_CSV,
        help="Output path for the LLM pricing CSV.",
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=EMBEDDING_PRICING_CSV,
        help="Output path for the embedding pricing CSV.",
    )
    parser.add_argument(
        "--overlay-json",
        type=Path,
        default=OVERLAY_JSON,
        help="Path for Shinka compatibility overlays.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated CSVs differ from committed outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = load_models_dev_payload(args.api_json)
    overlay = load_pricing_overlay(args.overlay_json)
    llm_targets = read_targets(args.llm_output, "llm", LLM_HEADERS)
    embedding_targets = read_targets(
        args.embedding_output, "embedding", EMBEDDING_HEADERS
    )
    catalog = catalog_from_models_dev_payload(
        payload, args.overlay_json, include_bundled=False
    )
    llm_csv = render_csv(
        LLM_HEADERS,
        [
            _generated_llm_row(catalog, payload, target, overlay)
            for target in llm_targets
        ],
    )
    embedding_csv = render_csv(
        EMBEDDING_HEADERS,
        [
            _generated_embedding_row(catalog, payload, target, overlay)
            for target in embedding_targets
        ],
    )

    if args.check:
        mismatches = []
        if args.llm_output.read_text(encoding="utf-8") != llm_csv:
            mismatches.append(str(args.llm_output))
        if args.embedding_output.read_text(encoding="utf-8") != embedding_csv:
            mismatches.append(str(args.embedding_output))
        if mismatches:
            print(
                "Generated pricing CSVs differ: " + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 1
        return 0

    args.llm_output.write_text(llm_csv, encoding="utf-8")
    args.embedding_output.write_text(embedding_csv, encoding="utf-8")
    return 0


def _generated_llm_row(
    catalog: PricingCatalog,
    payload: dict[str, Any],
    target: TargetModel,
    overlay: PricingOverlay,
) -> dict[str, str]:
    try:
        return render_llm_row(catalog, target.model_name, target.provider)
    except KeyError:
        return _generate_llm_row(payload, target, overlay)


def _generated_embedding_row(
    catalog: PricingCatalog,
    payload: dict[str, Any],
    target: TargetModel,
    overlay: PricingOverlay,
) -> dict[str, str]:
    try:
        return render_embedding_row(catalog, target.model_name, target.provider)
    except KeyError:
        return _generate_embedding_row(payload, target, overlay)


if __name__ == "__main__":
    raise SystemExit(main())
