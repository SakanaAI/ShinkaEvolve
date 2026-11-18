"""Every name a package exports must exist."""

import importlib

import pytest


@pytest.mark.parametrize("package_name", ["shinka.llm", "shinka.embed"])
def test_star_import_resolves_every_exported_name(package_name):
    package = importlib.import_module(package_name)

    missing = [name for name in package.__all__ if not hasattr(package, name)]

    assert missing == []
    namespace: dict[str, object] = {}
    exec(f"from {package_name} import *", namespace)
    assert set(package.__all__) <= set(namespace)


def test_embedding_clients_are_exported_by_the_embed_package_only():
    llm = importlib.import_module("shinka.llm")
    embed = importlib.import_module("shinka.embed")

    assert {"EmbeddingClient", "AsyncEmbeddingClient"} <= set(embed.__all__)
    assert not {"EmbeddingClient", "AsyncEmbeddingClient"} & set(llm.__all__)
