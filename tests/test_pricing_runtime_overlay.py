"""Regression tests for runtime embedding-price overlays."""

import json
from pathlib import Path

from shinka.pricing.catalog import catalog_from_models_dev_payload


MODEL_NAME = "gemini-embedding-exp-03-07"


def _runtime_catalog(tmp_path: Path, overrides):
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(
        json.dumps({"embedding_overrides": overrides}), encoding="utf-8"
    )
    payload = {
        "google": {
            "models": {
                MODEL_NAME: {
                    "family": "embedding",
                    "cost": {"input": 1.5},
                }
            }
        }
    }
    return catalog_from_models_dev_payload(
        payload,
        overlay_path=overlay_path,
        include_bundled=False,
    )


def test_embedding_overrides_replace_runtime_discovery_price(tmp_path: Path):
    catalog = _runtime_catalog(
        tmp_path,
        [
            {
                "provider": "google",
                "model_name": MODEL_NAME,
                "input_price": 0.0,
                "output_price": 999.0,
            }
        ],
    )

    entry = catalog.get("google", MODEL_NAME, kind="embedding")
    assert entry.input_price == 0.0
    assert entry.output_price is None


def test_embedding_overrides_ignore_unknown_and_malformed_entries(tmp_path: Path):
    catalog = _runtime_catalog(
        tmp_path,
        [
            {"provider": "google"},
            {"provider": "unknown", "model_name": "unknown", "input_price": 0.0},
        ],
    )

    assert catalog.get("google", MODEL_NAME, kind="embedding").input_price == 1.5e-6
