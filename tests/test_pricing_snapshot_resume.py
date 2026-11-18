"""The per-run pricing snapshot must reload on resume, embeddings included."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from shinka.pricing.catalog import (
    CatalogSnapshot,
    ModelPrice,
    PricingCatalog,
    _catalog_digest,
    catalog_from_models_dev_payload,
    load_run_pricing_snapshot,
    write_run_pricing_snapshot,
)

EMBEDDING_MODEL = "text-embedding-3-small"
MODELS_DEV_PAYLOAD = {
    "openai": {
        "models": {
            EMBEDDING_MODEL: {
                "family": "text-embedding",
                "cost": {"input": 0.02, "output": 0},
            },
            "gpt-snapshot-test": {"cost": {"input": 1.25, "output": 10.0}},
        }
    }
}


def _snapshot(catalog: PricingCatalog) -> CatalogSnapshot:
    return CatalogSnapshot(
        catalog=catalog,
        source="live",
        fetched_at="2026-07-14T09:00:00+00:00",
        etag='"catalog-v1"',
        sha256="abc123",
    )


def _empty_overlay(tmp_path: Path) -> Path:
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text("{}", encoding="utf-8")
    return overlay_path


def test_models_dev_embedding_entry_survives_snapshot_round_trip(tmp_path: Path):
    catalog = catalog_from_models_dev_payload(
        MODELS_DEV_PAYLOAD,
        overlay_path=_empty_overlay(tmp_path),
        include_bundled=False,
    )
    written = catalog.get("openai", EMBEDDING_MODEL, kind="embedding")
    assert written.output_price is None

    write_run_pricing_snapshot(_snapshot(catalog), tmp_path)
    restored = load_run_pricing_snapshot(tmp_path)

    assert restored is not None
    assert restored.catalog.entries == catalog.entries
    assert restored.catalog.get(
        "openai", EMBEDDING_MODEL, kind="embedding"
    ).input_price == pytest.approx(0.02 / 1_000_000)


def test_full_models_dev_catalog_survives_snapshot_round_trip(tmp_path: Path):
    """Whatever the normaliser writes, bundled entries included, must reload."""
    catalog = catalog_from_models_dev_payload(MODELS_DEV_PAYLOAD)

    write_run_pricing_snapshot(_snapshot(catalog), tmp_path)
    restored = load_run_pricing_snapshot(tmp_path)

    assert restored is not None
    assert restored.catalog.entries == catalog.entries


def test_legacy_snapshot_with_zero_embedding_output_price_still_loads(
    tmp_path: Path,
):
    """Snapshots already on disk hold ``output_price: 0.0`` for embeddings."""
    legacy_entries = (
        ModelPrice(
            model_name=EMBEDDING_MODEL,
            api_model_name=EMBEDDING_MODEL,
            provider="openai",
            kind="embedding",
            input_price=0.02 / 1_000_000,
            output_price=0.0,
        ),
        ModelPrice(
            model_name="gpt-snapshot-test",
            api_model_name="gpt-snapshot-test",
            provider="openai",
            kind="llm",
            input_price=1.25 / 1_000_000,
            output_price=10.0 / 1_000_000,
        ),
    )
    legacy_snapshot = _snapshot(PricingCatalog(legacy_entries))
    snapshot_path = tmp_path / "pricing_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                **legacy_snapshot.metadata(),
                "catalog": [asdict(entry) for entry in legacy_entries],
                "catalog_sha256": _catalog_digest(legacy_entries),
            }
        ),
        encoding="utf-8",
    )
    on_disk = snapshot_path.read_text(encoding="utf-8")

    restored = load_run_pricing_snapshot(tmp_path)

    assert restored is not None
    assert restored.sha256 == legacy_snapshot.sha256
    embedding = restored.catalog.get("openai", EMBEDDING_MODEL, kind="embedding")
    assert embedding.input_price == pytest.approx(0.02 / 1_000_000)
    assert embedding.output_price is None
    assert restored.catalog.get("openai", "gpt-snapshot-test").output_price == (
        pytest.approx(10.0 / 1_000_000)
    )

    # A resumed run must keep the prices it started with, not replace the file.
    fresh_catalog = PricingCatalog(
        (
            ModelPrice(
                model_name="gpt-snapshot-test",
                api_model_name="gpt-snapshot-test",
                provider="openai",
                kind="llm",
                input_price=9.0 / 1_000_000,
                output_price=90.0 / 1_000_000,
            ),
        )
    )
    write_run_pricing_snapshot(_snapshot(fresh_catalog), tmp_path)
    assert snapshot_path.read_text(encoding="utf-8") == on_disk
