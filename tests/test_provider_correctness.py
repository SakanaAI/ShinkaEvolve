"""Regression tests for provider/correctness fixes.

Covers four verified correctness bugs:

* Q18 - ``QueryResult.__str__`` divided by ``output_tokens`` before guarding on
  it, crashing on degenerate (0-output-token) responses.
* Q11 - Gemini / local-OpenAI providers accepted empty or truncated output as a
  finished program instead of raising.
* Q12 - ``extract_between`` returned the truthy string ``"none"`` on failure,
  so ``if result:`` callers treated a failed extraction as success.
* Q17 - the novelty judge fails closed on completed empty responses while
  provider failures remain fail-open.
"""

import asyncio
import logging
import traceback
from types import SimpleNamespace

import pytest

from google.genai import types

from shinka.llm import AsyncLLMClient, LLMClient, LLMQueryError, extract_between
from shinka.llm.llm import query_fn
from shinka.llm.providers.result import IncompleteResponseError, QueryResult
from shinka.llm.providers.gemini import (
    IncompleteGeminiResponseError,
    query_gemini,
    query_gemini_async,
    validate_gemini_response,
)
from shinka.llm.providers.local_openai import (
    query_local_openai,
    _extract_local_openai_content,
)
from shinka.core.novelty_judge import NoveltyJudge
from shinka.core.async_novelty_judge import AsyncNoveltyJudge
from shinka.database import Program


# ---------------------------------------------------------------------------
# Q18 - QueryResult.__str__ must not divide by zero output tokens
# ---------------------------------------------------------------------------


def _make_result(*, output_tokens: int, thinking_tokens: int) -> QueryResult:
    return QueryResult(
        content="print('hi')",
        msg="m",
        system_msg="s",
        new_msg_history=[],
        model_name="test-model",
        kwargs={},
        input_tokens=5,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
    )


def test_query_result_str_survives_zero_output_tokens():
    # Reachable via Gemini safety-blocks and local-openai max(out-think, 0).
    result = _make_result(output_tokens=0, thinking_tokens=3)
    text = str(result)  # must not raise ZeroDivisionError
    assert "n/a" in text
    assert "Thinking tokens: 3" in text


def test_query_result_str_reports_ratio_when_output_tokens_present():
    result = _make_result(output_tokens=10, thinking_tokens=5)
    assert "(0.50)" in str(result)


# ---------------------------------------------------------------------------
# Q12 - extract_between returns None (not "none") on failure
# ---------------------------------------------------------------------------


def test_extract_between_returns_none_when_no_match():
    assert extract_between("no tags here", return_dict=False) is None
    assert extract_between("no tags here", return_dict=True) is None


def test_extract_between_failure_is_falsy_not_string_none():
    # The old sentinel "none" was truthy; callers using `if result:` silently
    # accepted a failed extraction. None must be falsy and not the str "none".
    result = extract_between("```\nno python fence\n```", "```python", "```", False)
    assert result is None
    assert not result


def test_extract_between_still_extracts_matches():
    assert extract_between('<json>{"a": 1}</json>') == {"a": 1}
    fenced = "```python\nprint(1)\n```"
    assert extract_between(fenced, "```python", "```", False) == "print(1)"


# ---------------------------------------------------------------------------
# Q11 - local-OpenAI provider rejects empty / truncated completions
# ---------------------------------------------------------------------------


def _local_response(content, *, finish_reason="stop", reasoning_content=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


class _FakeLocalClient:
    def __init__(self, response, *, is_async=False):
        # Distinct names (not a conditional redefinition of one name) so mypy
        # doesn't flag mismatched sync/async signatures.
        async def _acreate(**kwargs):
            return response

        def _create(**kwargs):
            return response

        create = _acreate if is_async else _create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_extract_local_openai_content_raises_on_empty():
    with pytest.raises(IncompleteResponseError, match="no text output"):
        _extract_local_openai_content(_local_response(""))
    with pytest.raises(IncompleteResponseError, match="no text output"):
        _extract_local_openai_content(_local_response(None))


def test_extract_local_openai_content_raises_on_truncation():
    with pytest.raises(IncompleteResponseError, match="truncated"):
        _extract_local_openai_content(
            _local_response("partial code", finish_reason="length")
        )


def test_extract_local_openai_content_returns_valid_text():
    assert _extract_local_openai_content(_local_response("ok")) == "ok"


def test_query_local_openai_raises_on_empty_content():
    response = _local_response("")
    response.usage = SimpleNamespace(
        prompt_tokens=7,
        completion_tokens=5,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    client = _FakeLocalClient(response)
    with pytest.raises(
        IncompleteResponseError, match="no text output"
    ) as exc_info:
        query_local_openai(client, "dummy-model", "msg", "sys", [], None)

    rejected = exc_info.value.query_result
    assert rejected is not None
    assert rejected.content == ""
    assert rejected.input_tokens == 7
    assert rejected.output_tokens == 3
    assert rejected.thinking_tokens == 2


def test_query_local_openai_returns_result_for_valid_content():
    client = _FakeLocalClient(_local_response("solution"))
    result = query_local_openai(client, "dummy-model", "msg", "sys", [], None)
    assert result.content == "solution"
    assert result.new_msg_history[-1] == {"role": "assistant", "content": "solution"}


# ---------------------------------------------------------------------------
# Q11 - Gemini provider rejects empty / truncated responses
# ---------------------------------------------------------------------------


def _gemini_part(text, *, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def _gemini_response(*, parts=None, text=None, finish_reason=None):
    content = SimpleNamespace(parts=parts or [])
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], text=text)


class _FakeGeminiClient:
    def __init__(self, response):
        self.call_count = 0

        def generate_content(**kwargs):
            self.call_count += 1
            return response

        self.models = SimpleNamespace(
            generate_content=generate_content
        )


class _FakeAsyncGeminiClient:
    def __init__(self, response):
        self.call_count = 0

        async def generate_content(**kwargs):
            self.call_count += 1
            return response

        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )


def test_validate_gemini_response_raises_on_empty_content():
    response = _gemini_response(parts=[], text=None)
    with pytest.raises(ValueError, match="no text output"):
        validate_gemini_response(response, "")


def test_validate_gemini_response_raises_on_max_tokens_truncation():
    response = _gemini_response(
        parts=[_gemini_part("partial")],
        finish_reason=types.FinishReason.MAX_TOKENS,
    )
    with pytest.raises(ValueError, match="truncated"):
        validate_gemini_response(response, "partial")


@pytest.mark.parametrize(
    "finish_reason",
    [types.FinishReason.SAFETY, types.FinishReason.RECITATION],
)
def test_validate_gemini_response_rejects_non_success_finish(
    finish_reason,
):
    response = _gemini_response(
        parts=[_gemini_part("partial")],
        finish_reason=finish_reason,
    )

    with pytest.raises(IncompleteGeminiResponseError, match="incomplete"):
        validate_gemini_response(response, "partial")


def test_validate_gemini_response_accepts_completed_content():
    response = _gemini_response(
        parts=[_gemini_part("done")],
        finish_reason=types.FinishReason.STOP,
    )
    validate_gemini_response(response, "done")  # must not raise


def test_validate_gemini_response_accepts_unspecified_block_reason():
    response = _gemini_response(
        parts=[_gemini_part("done")],
        finish_reason=types.FinishReason.STOP,
    )
    response.prompt_feedback = SimpleNamespace(
        block_reason=types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
    )

    validate_gemini_response(response, "done")


def test_query_gemini_raises_on_empty_response():
    # A Gemini safety-block returns candidates with no text parts.
    response = _gemini_response(parts=[], text=None)
    client = _FakeGeminiClient(response)
    with pytest.raises(
        IncompleteGeminiResponseError, match="no text output"
    ) as exc_info:
        query_gemini(
            client,
            "gemini-2.5-flash",
            "msg",
            "sys",
            [],
            None,
            max_tokens=128,
        )
    assert client.call_count == 1
    assert exc_info.value.query_result is not None
    assert exc_info.value.query_result.content == ""


def test_query_gemini_preserves_rejected_response_usage(monkeypatch):
    response = _gemini_response(parts=[], text=None)
    response.usage_metadata = SimpleNamespace(
        prompt_token_count=12,
        candidates_token_count=3,
        thoughts_token_count=4,
    )
    client = _FakeGeminiClient(response)
    monkeypatch.setattr(
        "shinka.llm.providers.gemini.calculate_cost",
        lambda model, input_tokens, output_tokens: (0.1, 0.2),
    )

    with pytest.raises(IncompleteGeminiResponseError) as exc_info:
        query_gemini(
            client,
            "gemini-2.5-flash",
            "msg",
            "sys",
            [],
            None,
            max_tokens=128,
        )

    rejected = exc_info.value.query_result
    assert rejected is not None
    assert rejected.input_tokens == 12
    assert rejected.output_tokens == 3
    assert rejected.thinking_tokens == 4
    assert rejected.cost == pytest.approx(0.3)


def test_query_gemini_async_rejects_without_retry():
    client = _FakeAsyncGeminiClient(
        _gemini_response(parts=[], text=None)
    )

    async def query():
        with pytest.raises(IncompleteGeminiResponseError):
            await query_gemini_async(
                client,
                "gemini-2.5-flash",
                "msg",
                "sys",
                [],
                None,
                max_tokens=128,
            )

    asyncio.run(query())

    assert client.call_count == 1


def test_async_llm_client_returns_rejected_usage_without_retry(monkeypatch):
    rejected = _make_result(output_tokens=3, thinking_tokens=4)
    rejected.content = ""
    rejected.cost = 0.3
    error = IncompleteGeminiResponseError("blocked")
    error.query_result = rejected
    calls = 0

    async def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("shinka.llm.llm.query_async", fail_once)
    monkeypatch.setattr("shinka.llm.llm.asyncio.sleep", no_sleep)
    client = AsyncLLMClient("gemini-2.5-flash", verbose=False)

    result = asyncio.run(
        client.query(
            "msg",
            "sys",
            llm_kwargs={"model_name": "gemini-2.5-flash"},
        )
    )

    assert calls == 1
    assert result is rejected
    assert result.cost == pytest.approx(0.3)


def test_llm_client_raises_after_provider_retries_are_exhausted(monkeypatch):
    calls = 0

    def fail_query(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("network down")

    monkeypatch.setattr("shinka.llm.llm.query", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 2)
    client = LLMClient("gpt-5", verbose=False)

    with pytest.raises(LLMQueryError, match="gpt-5") as exc_info:
        client.query_or_raise("msg", "sys", llm_kwargs={"model_name": "gpt-5"})

    assert calls == 2
    assert exc_info.value.provider_error_type == "RuntimeError"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_llm_client_preserves_none_on_provider_retry_exhaustion(monkeypatch):
    calls = 0

    def fail_query(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("network down")

    monkeypatch.setattr("shinka.llm.llm.query", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 2)
    client = LLMClient("gpt-5", verbose=False)

    result = client.query(
        "msg", "sys", llm_kwargs={"model_name": "gpt-5"}
    )

    assert result is None
    assert calls == 2


def test_async_llm_client_raises_after_provider_retries_are_exhausted(monkeypatch):
    calls = 0

    async def fail_query(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("network down")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("shinka.llm.llm.query_async", fail_query)
    monkeypatch.setattr("shinka.llm.llm.asyncio.sleep", no_sleep)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 2)
    client = AsyncLLMClient("gpt-5", verbose=False)

    async def query():
        with pytest.raises(LLMQueryError, match="gpt-5") as exc_info:
            await client.query_or_raise(
                "msg", "sys", llm_kwargs={"model_name": "gpt-5"}
            )
        assert exc_info.value.provider_error_type == "RuntimeError"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    asyncio.run(query())

    assert calls == 2


def test_llm_client_returns_completed_empty_response_without_retry(monkeypatch):
    calls = 0

    def empty_response(**kwargs):
        nonlocal calls
        calls += 1
        raise IncompleteResponseError("no text output")

    monkeypatch.setattr("shinka.llm.llm.query", empty_response)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 3)
    client = LLMClient("gpt-5", verbose=False)

    result = client.query_or_raise(
        "msg", "sys", llm_kwargs={"model_name": "gpt-5"}
    )

    assert result is None
    assert calls == 1


def test_sync_batch_query_preserves_completed_response_usage(monkeypatch):
    rejected = _make_result(output_tokens=3, thinking_tokens=2)
    rejected.content = ""
    rejected.cost = 0.4
    error = IncompleteResponseError("no text output")
    error.query_result = rejected

    def empty_response(**kwargs):
        raise error

    monkeypatch.setattr("shinka.llm.llm.query", empty_response)

    index, result = query_fn(
        4,
        "msg",
        "sys",
        kwargs={"model_name": "gpt-5"},
    )

    assert index == 4
    assert result is rejected
    assert result.cost == pytest.approx(0.4)


def test_sync_batch_logs_redact_credentials(caplog, monkeypatch):
    def fail_query(**kwargs):
        raise RuntimeError("request failed with token=error-secret")

    model_name = (
        "local/model@https://user:url-secret@example.test/v1?token=url-secret"
    )
    monkeypatch.setattr("shinka.llm.llm.query", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 1)
    caplog.set_level(logging.INFO)

    query_fn(
        0,
        "msg",
        "sys",
        kwargs={
            "model_name": model_name,
            "extra_headers": {"Authorization": "Bearer header-secret"},
        },
        verbose=True,
    )

    assert "local_openai/model" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "url-secret" not in caplog.text
    assert "header-secret" not in caplog.text
    assert "error-secret" not in caplog.text


def test_async_batch_logs_redact_credentials(caplog, monkeypatch):
    async def fail_query(**kwargs):
        raise RuntimeError("request failed with token=error-secret")

    model_name = (
        "local/model@https://user:url-secret@example.test/v1?token=url-secret"
    )
    monkeypatch.setattr("shinka.llm.llm.query_async", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 1)
    caplog.set_level(logging.INFO)
    client = AsyncLLMClient(model_name, verbose=True)

    index, result = asyncio.run(
        client._query_async_with_retry(
            0,
            "msg",
            "sys",
            kwargs={
                "model_name": model_name,
                "extra_headers": {"Authorization": "Bearer header-secret"},
            },
        )
    )

    assert index == 0
    assert result is None
    assert "local_openai/model" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "url-secret" not in caplog.text
    assert "header-secret" not in caplog.text
    assert "error-secret" not in caplog.text


@pytest.mark.parametrize("client_class", [LLMClient, AsyncLLMClient])
def test_get_kwargs_logs_redact_model_credentials(client_class, caplog):
    model_name = (
        "local/model@https://user:url-secret@example.test/v1?token=url-secret"
    )
    client = client_class(model_name, verbose=True)
    caplog.set_level(logging.INFO)

    client.get_kwargs()

    assert "local_openai/model" in caplog.text
    assert "url-secret" not in caplog.text
    assert "example.test" not in caplog.text


def test_sync_public_batch_logs_redact_model_credentials(caplog, monkeypatch):
    class FakeAsyncResult:
        def get(self):
            return 0, None

    class FakePool:
        def __init__(self, processes):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def apply_async(self, function, args):
            return FakeAsyncResult()

    model_name = (
        "local/model@https://user:url-secret@example.test/v1?token=url-secret"
    )
    monkeypatch.setattr("shinka.llm.llm.mp.Pool", FakePool)
    caplog.set_level(logging.INFO)
    client = LLMClient(model_name, verbose=True)

    client.batch_kwargs_query(1, "msg", "sys")

    assert "local_openai/model" in caplog.text
    assert "url-secret" not in caplog.text
    assert "example.test" not in caplog.text


def test_async_public_batch_logs_redact_model_credentials(caplog, monkeypatch):
    model_name = (
        "local/model@https://user:url-secret@example.test/v1?token=url-secret"
    )
    client = AsyncLLMClient(model_name, verbose=True)

    async def completed_query(index, *args, **kwargs):
        return index, None

    monkeypatch.setattr(
        client,
        "_sample_kwargs_query_async_with_retry",
        completed_query,
    )
    caplog.set_level(logging.INFO)

    asyncio.run(client.batch_kwargs_query(1, "msg", "sys"))

    assert "local_openai/model" in caplog.text
    assert "url-secret" not in caplog.text
    assert "example.test" not in caplog.text


def test_llm_query_error_redacts_local_endpoint_credentials(monkeypatch):
    def fail_query(**kwargs):
        raise RuntimeError(
            "request failed: https://user:secret@example.test/v1?token=abc"
        )

    model_name = (
        "local/model@https://user:secret@example.test/v1?token=abc"
    )
    monkeypatch.setattr("shinka.llm.llm.query", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 1)
    client = LLMClient(model_name, verbose=False)

    with pytest.raises(LLMQueryError) as exc_info:
        client.query_or_raise(
            "msg", "sys", llm_kwargs={"model_name": model_name}
        )

    error_text = str(exc_info.value)
    traceback_text = "".join(
        traceback.format_exception(exc_info.value)
    )
    assert "local_openai/model" in error_text
    assert "secret" not in traceback_text
    assert "token=abc" not in traceback_text
    assert "example.test" not in traceback_text


def test_llm_query_error_sanitizes_headless_model_label(monkeypatch):
    def fail_query(**kwargs):
        raise RuntimeError("provider failed")

    model_name = "headless/co\ndex@gpt-5?token=secret"
    monkeypatch.setattr("shinka.llm.llm.query", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 1)
    client = LLMClient(model_name, verbose=False)

    with pytest.raises(LLMQueryError) as exc_info:
        client.query_or_raise(
            "msg", "sys", llm_kwargs={"model_name": model_name}
        )

    error_text = str(exc_info.value)
    assert "headless/co?dex" in error_text
    assert "\n" not in error_text
    assert "token=secret" not in error_text


def test_async_batch_preserves_failed_response_position(monkeypatch):
    valid = _make_result(output_tokens=3, thinking_tokens=0)
    client = AsyncLLMClient("gemini-2.5-flash", verbose=False)

    async def fake_query(idx, *args, **kwargs):
        return idx, None if idx == 0 else valid

    monkeypatch.setattr(
        client,
        "_sample_kwargs_query_async_with_retry",
        fake_query,
    )

    results = asyncio.run(
        client.batch_kwargs_query(
            num_samples=2,
            msg=["first", "second"],
            system_msg=["sys", "sys"],
        )
    )

    assert results == [None, valid]


def test_async_batch_preserves_cancelled_response_position(monkeypatch):
    valid = _make_result(output_tokens=3, thinking_tokens=0)
    client = AsyncLLMClient("gemini-2.5-flash", verbose=False)

    async def fake_query(idx, *args, **kwargs):
        if idx == 0:
            raise asyncio.CancelledError
        return idx, valid

    monkeypatch.setattr(client, "_query_async_with_retry", fake_query)
    results = asyncio.run(
        client.batch_query(
            num_samples=2,
            msg=["first", "second"],
            system_msg=["sys", "sys"],
            llm_kwargs=[
                {"model_name": "gemini-2.5-flash"},
                {"model_name": "gemini-2.5-flash"},
            ],
        )
    )

    assert results == [None, valid]

    monkeypatch.setattr(
        client,
        "_sample_kwargs_query_async_with_retry",
        fake_query,
    )
    results = asyncio.run(
        client.batch_kwargs_query(
            num_samples=2,
            msg=["first", "second"],
            system_msg=["sys", "sys"],
        )
    )

    assert results == [None, valid]


def test_query_gemini_returns_result_for_valid_content():
    response = _gemini_response(
        parts=[_gemini_part("hello world")],
        finish_reason=types.FinishReason.STOP,
    )
    client = _FakeGeminiClient(response)
    result = query_gemini(
        client,
        "gemini-2.5-flash",
        "msg",
        "sys",
        [],
        None,
        max_tokens=128,
    )
    assert result.content == "hello world"


# ---------------------------------------------------------------------------
# Q17 - novelty judge fails CLOSED on empty response
# ---------------------------------------------------------------------------


class _SyncNoveltyLLM:
    """Minimal sync novelty LLM stub returning a fixed (possibly empty) reply."""

    def __init__(self, response):
        self._response = response

    def get_kwargs(self):
        return {}

    def query(self, msg, system_msg, llm_kwargs):
        return self._response


class _AsyncNoveltyLLM:
    def __init__(self, response):
        self._response = response

    async def query(self, msg, system_msg):
        return self._response


def _similar_program():
    return Program(id="existing", code="def f():\n    return 1\n", language="python")


def test_check_llm_novelty_fails_closed_when_response_is_none():
    judge = NoveltyJudge(novelty_llm_client=_SyncNoveltyLLM(None))
    is_novel, explanation, cost = judge.check_llm_novelty(
        proposed_code="def g():\n    return 2\n",
        most_similar_program=_similar_program(),
    )
    assert is_novel is False  # fail closed => reject as not novel
    assert cost == 0.0
    assert "empty" in explanation.lower()


def test_check_llm_novelty_fails_closed_when_content_is_none():
    response = SimpleNamespace(content=None, cost=0.0)
    judge = NoveltyJudge(novelty_llm_client=_SyncNoveltyLLM(response))
    is_novel, _explanation, cost = judge.check_llm_novelty(
        proposed_code="def g():\n    return 2\n",
        most_similar_program=_similar_program(),
    )
    assert is_novel is False
    assert cost == 0.0


def test_async_check_llm_novelty_fails_closed_when_response_is_none():
    async_judge = AsyncNoveltyJudge(
        NoveltyJudge(), async_llm_client=_AsyncNoveltyLLM(None)
    )
    is_novel, explanation, cost = asyncio.run(
        async_judge._check_llm_novelty_async(
            "def g():\n    return 2\n", _similar_program()
        )
    )
    assert is_novel is False
    assert cost == 0.0
    assert "empty" in explanation.lower()


def test_novelty_judge_fails_open_after_provider_retries_are_exhausted(monkeypatch):
    def fail_query(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("shinka.llm.llm.query", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 1)
    judge = NoveltyJudge(
        novelty_llm_client=LLMClient("gpt-5", verbose=False),
    )

    is_novel, explanation, cost = judge.check_llm_novelty(
        proposed_code="def g():\n    return 2\n",
        most_similar_program=_similar_program(),
    )

    assert is_novel is True
    assert "Query failed" in explanation
    assert cost == 0.0


def test_async_novelty_judge_fails_open_after_provider_retries_are_exhausted(
    monkeypatch,
):
    async def fail_query(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("shinka.llm.llm.query_async", fail_query)
    monkeypatch.setattr("shinka.llm.llm.MAX_RETRIES", 1)
    async_judge = AsyncNoveltyJudge(
        NoveltyJudge(),
        async_llm_client=AsyncLLMClient("gpt-5", verbose=False),
    )

    is_novel, explanation, cost = asyncio.run(
        async_judge._check_llm_novelty_async(
            "def g():\n    return 2\n",
            _similar_program(),
        )
    )

    assert is_novel is True
    assert "Query failed" in explanation
    assert cost == 0.0
