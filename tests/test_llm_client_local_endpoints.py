"""Several local endpoints in one model pool must each keep a cached client."""

import asyncio

import pytest

import shinka.llm.client as llm_client

ENDPOINT_A = "local/model-a@http://localhost:9001/v1"
ENDPOINT_B = "local/model-b@http://localhost:9002/v1"


@pytest.fixture(autouse=True)
def _empty_sync_cache():
    llm_client.close_sync_client_cache()
    yield
    llm_client.close_sync_client_cache()


def _count_constructions(monkeypatch: pytest.MonkeyPatch, builder: str) -> list[str]:
    built: list[str] = []

    def build(spec):
        built.append(dict(spec.constructor_kwargs)["base_url"])
        return object()

    monkeypatch.setattr(llm_client, builder, build)
    return built


def test_sync_clients_for_two_local_endpoints_do_not_evict_each_other(
    monkeypatch: pytest.MonkeyPatch,
):
    built = _count_constructions(monkeypatch, "_build_sync_client")

    clients = [
        llm_client.get_client_llm(model)[0]
        for _ in range(5)
        for model in (ENDPOINT_A, ENDPOINT_B)
    ]

    assert built == ["http://localhost:9001/v1", "http://localhost:9002/v1"]
    assert len({id(client) for client in clients}) == 2
    assert len(llm_client._SYNC_CLIENT_CACHE) == 2


def test_async_clients_for_two_local_endpoints_do_not_evict_each_other(
    monkeypatch: pytest.MonkeyPatch,
):
    built = _count_constructions(monkeypatch, "_build_async_client")

    async def scenario() -> int:
        for _ in range(5):
            llm_client.get_async_client_llm(ENDPOINT_A)
            llm_client.get_async_client_llm(ENDPOINT_B)
        loop = asyncio.get_running_loop()
        return len(getattr(loop, llm_client._ASYNC_CACHE_ATTRIBUTE))

    assert asyncio.run(scenario()) == 2
    assert built == ["http://localhost:9001/v1", "http://localhost:9002/v1"]


def test_rotated_key_still_replaces_the_client_of_its_own_endpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    built = _count_constructions(monkeypatch, "_build_sync_client")
    keyed_endpoint = ENDPOINT_A + "?api_key_env=LOCAL_ENDPOINT_KEY"

    llm_client.get_client_llm(ENDPOINT_B)
    for index in range(5):
        monkeypatch.setenv("LOCAL_ENDPOINT_KEY", f"rotated-key-{index}")
        llm_client.get_client_llm(keyed_endpoint)
    llm_client.get_client_llm(ENDPOINT_B)

    assert built.count("http://localhost:9002/v1") == 1
    assert built.count("http://localhost:9001/v1") == 5
    assert len(llm_client._SYNC_CLIENT_CACHE) == 2
