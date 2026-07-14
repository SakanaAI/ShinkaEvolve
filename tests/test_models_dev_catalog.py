from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import shinka.pricing.catalog as catalog_module
from shinka.pricing.catalog import (
    MAX_CATALOG_BYTES,
    PricingConfig,
    PricingMode,
    catalog_from_models_dev_payload,
    refresh_model_catalog,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "models_dev_catalog.json"
MODELS_DEV_URL = "https://models.dev/api.json"


@pytest.fixture
def models_dev_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _restore_bundled_catalog(tmp_path: Path):
    yield
    refresh_model_catalog(
        PricingConfig(
            mode=PricingMode.OFFLINE,
            cache_dir=tmp_path / "restore-cache",
        )
    )


def _config(cache_dir: Path) -> PricingConfig:
    return PricingConfig(mode="auto", url=MODELS_DEV_URL, cache_dir=cache_dir)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _live_response(
    request: httpx.Request,
    payload: dict,
    *,
    etag: str = '"catalog-v1"',
) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        headers={"ETag": etag},
        request=request,
    )


def test_live_catalog_keeps_native_and_openrouter_prices_separate(
    tmp_path: Path, models_dev_payload: dict
):
    def handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    with _client(handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    native = snapshot.catalog.get("openai", "gpt-catalog-test")
    routed = snapshot.catalog.get("openrouter", "openai/gpt-catalog-test")

    assert snapshot.source == "live"
    assert snapshot.etag == '"catalog-v1"'
    assert snapshot.stale is False
    assert native.input_price == pytest.approx(2.0 / 1_000_000)
    assert native.output_price == pytest.approx(8.0 / 1_000_000)
    assert routed.input_price == pytest.approx(0.5 / 1_000_000)
    assert routed.output_price == pytest.approx(0.75 / 1_000_000)
    assert snapshot.catalog.get_by_name("gpt-catalog-test") == native
    assert snapshot.catalog.get_by_name("openrouter/openai/gpt-catalog-test") == routed


def test_models_dev_prices_are_converted_to_per_token_and_context_tiers_apply(
    tmp_path: Path, models_dev_payload: dict
):
    def handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    with _client(handler) as client:
        catalog = refresh_model_catalog(_config(tmp_path), client=client).catalog

    tiered = catalog.get("openai", "gpt-tiered-test")
    at_threshold = tiered.rates(input_tokens=200_000)
    above_threshold = tiered.rates(input_tokens=200_001)

    assert at_threshold == pytest.approx((1.25 / 1_000_000, 10.0 / 1_000_000))
    assert above_threshold == pytest.approx((2.5 / 1_000_000, 15.0 / 1_000_000))


def test_200_response_is_cached_with_etag_and_304_reuses_it(
    tmp_path: Path, models_dev_payload: dict
):
    requests: list[httpx.Request] = []

    def live_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _live_response(request, models_dev_payload, etag='"catalog-v2"')

    with _client(live_handler) as client:
        first = refresh_model_catalog(_config(tmp_path), client=client)

    def not_modified_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(304, request=request)

    with _client(not_modified_handler) as client:
        second = refresh_model_catalog(_config(tmp_path), client=client)

    assert first.source == "live"
    assert second.source == "cache"
    assert second.etag == '"catalog-v2"'
    assert second.stale is False
    assert requests[1].headers["If-None-Match"] == '"catalog-v2"'
    assert second.sha256 == first.sha256
    assert second.catalog.get("openai", "gpt-catalog-test") == first.catalog.get(
        "openai", "gpt-catalog-test"
    )


@pytest.mark.parametrize("failure", ["timeout", "malformed"])
def test_network_failure_keeps_last_known_good_cache(
    tmp_path: Path, models_dev_payload: dict, failure: str
):
    def live_handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    with _client(live_handler) as client:
        expected = refresh_model_catalog(_config(tmp_path), client=client)

    def failed_handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ConnectTimeout("models.dev did not respond", request=request)
        return httpx.Response(
            200,
            content=b"not-json",
            headers={"ETag": '"broken"'},
            request=request,
        )

    with _client(failed_handler) as client:
        fallback = refresh_model_catalog(_config(tmp_path), client=client)

    assert fallback.source == "cache"
    assert fallback.stale is True
    assert fallback.etag == expected.etag
    assert fallback.sha256 == expected.sha256
    assert fallback.catalog.get("openai", "gpt-catalog-test") == expected.catalog.get(
        "openai", "gpt-catalog-test"
    )


def test_empty_cache_and_failed_request_use_packaged_fallback(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("models.dev did not respond", request=request)

    with _client(handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    bundled = snapshot.catalog.get("openai", "gpt-4.1")

    assert snapshot.source == "bundled"
    assert snapshot.stale is True
    assert bundled.input_price == pytest.approx(2.0 / 1_000_000)
    assert bundled.output_price == pytest.approx(8.0 / 1_000_000)


def test_models_with_missing_costs_are_not_registered_as_free(
    tmp_path: Path, models_dev_payload: dict
):
    def handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    with _client(handler) as client:
        catalog = refresh_model_catalog(_config(tmp_path), client=client).catalog

    with pytest.raises((KeyError, ValueError), match="gpt-missing-output-test"):
        catalog.get("openai", "gpt-missing-output-test")


def test_non_text_output_models_are_not_registered_as_llms(
    tmp_path: Path, models_dev_payload: dict
):
    def handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    with _client(handler) as client:
        catalog = refresh_model_catalog(_config(tmp_path), client=client).catalog

    with pytest.raises(KeyError, match="gpt-image-test"):
        catalog.get("openai", "gpt-image-test")


def test_generator_catalog_does_not_mask_missing_upstream_models(
    models_dev_payload: dict,
):
    catalog = catalog_from_models_dev_payload(
        models_dev_payload, include_bundled=False
    )

    with pytest.raises(KeyError, match="gpt-4.1"):
        catalog.get("openai", "gpt-4.1")


def test_offline_mode_skips_network_and_uses_bundled_catalog(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("offline mode must not make an HTTP request")

    with _client(handler) as client:
        snapshot = refresh_model_catalog(
            PricingConfig(mode=PricingMode.OFFLINE, cache_dir=tmp_path),
            client=client,
        )

    assert snapshot.source == "bundled"
    assert snapshot.stale is False


def test_required_mode_fails_when_live_catalog_is_unavailable(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("models.dev did not respond", request=request)

    with _client(handler) as client:
        with pytest.raises(RuntimeError, match="required models.dev pricing"):
            refresh_model_catalog(
                PricingConfig(mode=PricingMode.REQUIRED, cache_dir=tmp_path),
                client=client,
            )


def test_oversized_response_is_rejected_without_replacing_fallback(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{" + b" " * MAX_CATALOG_BYTES + b"}",
            request=request,
        )

    with _client(handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    assert snapshot.source == "bundled"
    assert snapshot.stale is True
    assert not (tmp_path / "models-dev-cache.json").exists()


def test_invalid_cache_is_ignored_when_network_is_unavailable(tmp_path: Path):
    (tmp_path / "models-dev-cache.json").write_text(
        '{"payload": "invalid"}', encoding="utf-8"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("models.dev did not respond", request=request)

    with _client(handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    assert snapshot.source == "bundled"
    assert snapshot.stale is True


def test_non_finite_cache_is_ignored(tmp_path: Path):
    (tmp_path / "models-dev-cache.json").write_text(
        '{"payload": {"cost": Infinity}}', encoding="utf-8"
    )

    snapshot = refresh_model_catalog(
        PricingConfig(mode=PricingMode.OFFLINE, cache_dir=tmp_path)
    )

    assert snapshot.source == "bundled"


def test_live_catalog_survives_cache_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    models_dev_payload: dict,
):
    def handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    def fail_cache_write(_path: Path, _document: dict) -> None:
        raise OSError("read-only cache")

    monkeypatch.setattr(catalog_module, "_write_json_atomic", fail_cache_write)
    with _client(handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    assert snapshot.source == "live"
    assert snapshot.catalog.get("openai", "gpt-catalog-test").input_price > 0


def test_non_finite_json_uses_bundled_fallback(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'{"openai":{"models":{"bad":{"cost":{"input":1,'
                b'"output":2},"cost":{"tiers":[{"tier":{"type":'
                b'"context","size":Infinity}}]}}}}}'
            ),
            request=request,
        )

    with _client(handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    assert snapshot.source == "bundled"
    assert snapshot.stale is True


def test_tampered_cache_is_not_used_after_network_failure(
    tmp_path: Path, models_dev_payload: dict
):
    def live_handler(request: httpx.Request) -> httpx.Response:
        return _live_response(request, models_dev_payload)

    with _client(live_handler) as client:
        refresh_model_catalog(_config(tmp_path), client=client)

    cache_path = tmp_path / "models-dev-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["payload"]["openai"]["models"]["gpt-catalog-test"]["cost"]["input"] = 999
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    def failed_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("models.dev did not respond", request=request)

    with _client(failed_handler) as client:
        snapshot = refresh_model_catalog(_config(tmp_path), client=client)

    assert snapshot.source == "bundled"
