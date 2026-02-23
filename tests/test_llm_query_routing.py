import asyncio

import shinka.llm.query as query_module


def test_query_routes_local_openai(monkeypatch):
    monkeypatch.setattr(
        query_module,
        "get_client_llm",
        lambda model_name, structured_output=False: ("client", "local-model", "local_openai"),
    )
    called = {}

    def _fake_local_query(
        client,
        model,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=None,
        **kwargs,
    ):
        called["provider"] = "local_openai"
        called["model"] = model
        return "ok"

    monkeypatch.setattr(query_module, "query_local_openai", _fake_local_query)

    result = query_module.query(
        model_name="local/foo@http://localhost:11434/v1",
        msg="hello",
        system_msg="sys",
    )

    assert result == "ok"
    assert called["provider"] == "local_openai"
    assert called["model"] == "local-model"


def test_query_async_routes_local_openai(monkeypatch):
    monkeypatch.setattr(
        query_module,
        "get_async_client_llm",
        lambda model_name, structured_output=False: ("client", "local-model", "local_openai"),
    )
    called = {}

    async def _fake_local_query_async(
        client,
        model,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=None,
        **kwargs,
    ):
        called["provider"] = "local_openai"
        called["model"] = model
        return "ok-async"

    monkeypatch.setattr(query_module, "query_local_openai_async", _fake_local_query_async)

    result = asyncio.run(
        query_module.query_async(
            model_name="local/foo@http://localhost:11434/v1",
            msg="hello",
            system_msg="sys",
        )
    )

    assert result == "ok-async"
    assert called["provider"] == "local_openai"
    assert called["model"] == "local-model"
