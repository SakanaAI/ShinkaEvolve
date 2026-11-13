"""Exercise current thinking controls through real SDK serialization, offline."""

import asyncio
import json

import anthropic
from anthropic import _base_client
import pytest

from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers.anthropic import query_anthropic, query_anthropic_async

# SDK 1.x uses httpx2; 2.x uses httpx. Match the installed transport backend.
httpx = getattr(_base_client, "httpx", None) or getattr(_base_client, "httpx2")


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
    captured = []
    kwargs = sample_model_kwargs(
        model_names=model,
        reasoning_efforts=effort,
        temperatures=[0.25],
        max_tokens=4096,
    )
    kwargs.pop("model_name")

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

    assert result.content == "OK"
    assert len(captured) == 1
    assert captured[0]["thinking"] == kwargs["thinking"]
    assert captured[0]["output_config"] == kwargs["output_config"]
    assert captured[0]["max_tokens"] == 4096
    assert "temperature" not in captured[0]
    assert result.cost > 0
