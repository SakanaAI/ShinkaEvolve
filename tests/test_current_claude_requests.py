"""Exercise current thinking controls through real SDK serialization, offline."""

import asyncio
import inspect
import json

import anthropic
from anthropic import _base_client
from anthropic.resources.messages import Messages
import pytest

from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers.anthropic import query_anthropic, query_anthropic_async
from shinka.llm.providers.pricing import get_models_by_provider

# SDK 1.x uses httpx2; 2.x uses httpx. Match the installed transport backend.
httpx = getattr(_base_client, "httpx", None) or getattr(_base_client, "httpx2")


def _send(model: str, kwargs: dict, mode: str):
    """Send one query through the real SDK; return the result and the request body."""
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    transport = httpx.MockTransport(respond)

    async def run_async():
        async with anthropic.AsyncAnthropic(
            api_key="test-only",
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            return await query_anthropic_async(
                client, model, "Reply OK", "Follow instructions", [], None, **kwargs
            )

    if mode == "async":
        result = asyncio.run(run_async())
    else:
        with anthropic.Anthropic(
            api_key="test-only",
            http_client=httpx.Client(transport=transport),
        ) as client:
            result = query_anthropic(
                client, model, "Reply OK", "Follow instructions", [], None, **kwargs
            )

    assert len(captured) == 1
    return result, captured[0]


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-1",
        "claude-fable-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    ],
)
@pytest.mark.parametrize("effort", ["disabled", "max"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_current_claude_request_serializes_through_sdk(model, effort, mode):
    kwargs = sample_model_kwargs(
        model_names=model,
        reasoning_efforts=effort,
        temperatures=[0.25],
        max_tokens=4096,
    )
    kwargs.pop("model_name")

    result, body = _send(model, kwargs, mode)

    assert result.content == "OK"
    assert body["thinking"] == kwargs["thinking"]
    assert body["output_config"] == kwargs["output_config"]
    assert body["max_tokens"] == 4096
    assert "temperature" not in body
    assert result.cost > 0


# Budget-based thinking, and no thinking at all.
@pytest.mark.parametrize("model", ["claude-haiku-4-5-20251001", "claude-opus-4-7"])
@pytest.mark.parametrize("effort", ["disabled", "high"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_older_claude_request_serializes_through_sdk(model, effort, mode):
    """SDK 1.0 dropped the `temperature` argument; API 2026-10-21 rejects the field."""
    kwargs = sample_model_kwargs(
        model_names=model,
        reasoning_efforts=effort,
        temperatures=[0.25],
        max_tokens=4096,
    )
    kwargs.pop("model_name")

    result, body = _send(model, kwargs, mode)

    assert result.content == "OK"
    assert body["max_tokens"] == 4096
    assert body.get("thinking") == kwargs.get("thinking")
    assert "temperature" not in body


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
@pytest.mark.parametrize("effort", ["disabled", "high"])
def test_every_claude_model_gets_only_arguments_the_sdk_accepts(provider, effort):
    accepted = set(inspect.signature(Messages.create).parameters)
    models = get_models_by_provider(provider)
    assert models

    for model in models:
        kwargs = sample_model_kwargs(
            model_names=model,
            reasoning_efforts=effort,
            temperatures=[0.25],
            max_tokens=4096,
        )
        kwargs.pop("model_name")
        assert set(kwargs) <= accepted, (model, sorted(set(kwargs) - accepted))
