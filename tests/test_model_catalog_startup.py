from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import shinka.cli.models as cli_models
import shinka.core.async_runner as async_runner
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers.pricing import calculate_cost, get_model_prices
from shinka.pricing.catalog import (
    CatalogSnapshot,
    ModelPrice,
    PricingCatalog,
    PricingConfig,
    PricingMode,
    activate_model_catalog,
    get_catalog,
    load_run_pricing_snapshot,
    refresh_model_catalog,
    write_run_pricing_snapshot,
)


def _models_dev_payload(model_name: str = "models-dev-only") -> dict:
    return {
        "openai": {
            "id": "openai",
            "name": "OpenAI",
            "models": {
                model_name: {
                    "id": model_name,
                    "name": "Runtime-discovered model",
                    "cost": {"input": 1.25, "output": 5.0},
                    "reasoning": False,
                }
            },
        }
    }


def _live_client(payload: dict, etag: str = '"catalog-v1"') -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://models.dev/api.json")
        return httpx.Response(
            200,
            headers={"ETag": etag},
            json=payload,
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _offline_config(cache_dir: Path) -> PricingConfig:
    return PricingConfig(mode=PricingMode.OFFLINE, cache_dir=cache_dir)


@pytest.fixture(autouse=True)
def _restore_bundled_catalog(tmp_path: Path):
    yield
    refresh_model_catalog(_offline_config(tmp_path / "restore-cache"))


def test_runner_refreshes_catalog_before_model_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    events: list[str] = []
    refreshed_snapshot = object()

    class ValidationStopped(Exception):
        pass

    def refresh() -> object:
        events.append("refresh")
        return refreshed_snapshot

    def validate(_config: EvolutionConfig) -> None:
        events.append("validate")
        raise ValidationStopped

    monkeypatch.setattr(async_runner, "refresh_model_catalog", refresh)
    monkeypatch.setattr(
        async_runner,
        "_validate_evo_config_model_env_access",
        validate,
    )

    with pytest.raises(ValidationStopped):
        ShinkaEvolveRunner(
            evo_config=EvolutionConfig(
                llm_models=["models-dev-only"],
                llm_dynamic_selection=None,
                meta_rec_interval=None,
                embedding_model=None,
                num_generations=1,
                results_dir=str(tmp_path / "results"),
            ),
            job_config=LocalJobConfig(),
            db_config=DatabaseConfig(),
            verbose=False,
        )

    assert events == ["refresh", "validate"]


def test_runner_resume_uses_snapshot_without_network_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    events: list[str] = []
    snapshot = get_catalog()

    class ValidationStopped(Exception):
        pass

    def load_snapshot(_results_dir: Path) -> CatalogSnapshot:
        events.append("load")
        return snapshot

    def refresh() -> CatalogSnapshot:
        raise AssertionError("resume must not refresh models.dev")

    def validate(_config: EvolutionConfig) -> None:
        events.append("validate")
        raise ValidationStopped

    monkeypatch.setattr(async_runner, "load_run_pricing_snapshot", load_snapshot)
    monkeypatch.setattr(async_runner, "refresh_model_catalog", refresh)
    monkeypatch.setattr(
        async_runner,
        "_validate_evo_config_model_env_access",
        validate,
    )

    with pytest.raises(ValidationStopped):
        ShinkaEvolveRunner(
            evo_config=EvolutionConfig(
                llm_models=["gpt-5-mini"],
                llm_dynamic_selection=None,
                meta_rec_interval=None,
                embedding_model=None,
                num_generations=1,
                results_dir=str(tmp_path / "results"),
            ),
            job_config=LocalJobConfig(),
            db_config=DatabaseConfig(),
            verbose=False,
        )

    assert events == ["load", "validate"]


def test_runner_catalog_context_propagates_to_worker_threads() -> None:
    snapshot = CatalogSnapshot(
        catalog=PricingCatalog(),
        source="bundled",
        fetched_at=None,
        etag=None,
        sha256="context-test",
    )
    activate_model_catalog(snapshot)

    observed = asyncio.run(asyncio.to_thread(get_catalog))

    assert observed is snapshot


def test_runtime_discovered_model_resolves_and_has_nonzero_cost(tmp_path: Path):
    cache_dir = tmp_path / "catalog-cache"
    config = PricingConfig(mode=PricingMode.AUTO, cache_dir=cache_dir)

    with _live_client(_models_dev_payload()) as client:
        snapshot = refresh_model_catalog(config=config, client=client)

    resolved = resolve_model_backend("models-dev-only")
    prices = get_model_prices("models-dev-only")
    input_cost, output_cost = calculate_cost("models-dev-only", 2_000, 500)

    assert snapshot.source == "live"
    assert resolved.provider == "openai"
    assert resolved.api_model_name == "models-dev-only"
    assert prices == {
        "input_price": pytest.approx(1.25 / 1_000_000),
        "output_price": pytest.approx(5.0 / 1_000_000),
    }
    assert input_cost == pytest.approx(0.0025)
    assert output_cost == pytest.approx(0.0025)


def test_shinka_models_verbose_refreshes_and_reports_catalog_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[str] = []
    snapshot = CatalogSnapshot(
        catalog=PricingCatalog(),
        source="live",
        fetched_at="2026-07-14T09:00:00+00:00",
        etag='"catalog-v1"',
        sha256="abc123",
        stale=False,
    )

    def refresh() -> CatalogSnapshot:
        calls.append("refresh")
        return snapshot

    def build_payload() -> dict[str, list[object]]:
        calls.append("payload")
        return {"available_providers": [], "embedding": [], "llm": []}

    monkeypatch.setattr(cli_models, "load_shinka_dotenv", lambda: ())
    monkeypatch.setattr(cli_models, "refresh_model_catalog", refresh)
    monkeypatch.setattr(cli_models, "_build_payload", build_payload)

    exit_code = cli_models.main(["--verbose"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["refresh", "payload"]
    assert output["pricing_catalog"] == {
        "source": "live",
        "fetched_at": "2026-07-14T09:00:00+00:00",
        "etag": '"catalog-v1"',
        "sha256": "abc123",
        "stale": False,
    }


def test_run_pricing_snapshot_is_not_replaced_when_resuming(tmp_path: Path):
    results_dir = tmp_path / "results"
    first_snapshot = CatalogSnapshot(
        catalog=PricingCatalog(
            entries=(
                ModelPrice(
                    model_name="models-dev-only",
                    api_model_name="models-dev-only",
                    provider="openai",
                    kind="llm",
                    input_price=1.25 / 1_000_000,
                    output_price=5.0 / 1_000_000,
                ),
            )
        ),
        source="live",
        fetched_at="2026-07-14T09:00:00+00:00",
        etag='"catalog-v1"',
        sha256="abc123",
        stale=False,
    )

    write_run_pricing_snapshot(first_snapshot, results_dir)
    snapshot_path = results_dir / "pricing_snapshot.json"
    original_snapshot = snapshot_path.read_text(encoding="utf-8")
    original_payload = json.loads(original_snapshot)

    later_snapshot = replace(
        first_snapshot,
        source="cache",
        fetched_at="2026-07-15T09:00:00+00:00",
        etag='"catalog-v2"',
        stale=True,
    )
    write_run_pricing_snapshot(later_snapshot, results_dir)

    assert snapshot_path.read_text(encoding="utf-8") == original_snapshot
    assert original_payload["source"] == "live"
    assert original_payload["etag"] == '"catalog-v1"'
    assert original_payload["sha256"] == first_snapshot.sha256
    assert original_payload["catalog"]

    restored = load_run_pricing_snapshot(results_dir)
    assert restored is not None
    assert restored.sha256 == first_snapshot.sha256
    assert get_catalog().sha256 == first_snapshot.sha256


def test_corrupt_run_pricing_snapshot_is_replaced(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    snapshot_path = results_dir / "pricing_snapshot.json"
    snapshot_path.write_text("not-json", encoding="utf-8")

    snapshot = refresh_model_catalog(_offline_config(tmp_path / "cache"))
    write_run_pricing_snapshot(snapshot, results_dir)

    restored = load_run_pricing_snapshot(results_dir)
    assert restored is not None
    assert restored.sha256 == snapshot.sha256
