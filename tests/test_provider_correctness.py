"""Regression tests for provider/correctness fixes.

Covers four verified correctness bugs:

* Q18 - ``QueryResult.__str__`` divided by ``output_tokens`` before guarding on
  it, crashing on degenerate (0-output-token) responses.
* Q11 - Gemini / local-OpenAI providers accepted empty or truncated output as a
  finished program instead of raising.
* Q12 - ``extract_between`` returned the truthy string ``"none"`` on failure,
  so ``if result:`` callers treated a failed extraction as success.
* Q17 - the novelty judge failed OPEN (accepted the candidate as novel) on an
  empty/transient LLM response instead of failing closed.
"""

import asyncio
from types import SimpleNamespace

import pytest

from google.genai import types

from shinka.llm import extract_between
from shinka.llm.providers.result import QueryResult
from shinka.llm.providers.gemini import query_gemini, validate_gemini_response
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
        if is_async:

            async def _create(**kwargs):
                return response

        else:

            def _create(**kwargs):
                return response

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_create)
        )


def test_extract_local_openai_content_raises_on_empty():
    with pytest.raises(ValueError, match="no text output"):
        _extract_local_openai_content(_local_response(""))
    with pytest.raises(ValueError, match="no text output"):
        _extract_local_openai_content(_local_response(None))


def test_extract_local_openai_content_raises_on_truncation():
    with pytest.raises(ValueError, match="truncated"):
        _extract_local_openai_content(
            _local_response("partial code", finish_reason="length")
        )


def test_extract_local_openai_content_returns_valid_text():
    assert _extract_local_openai_content(_local_response("ok")) == "ok"


def test_query_local_openai_raises_on_empty_content():
    client = _FakeLocalClient(_local_response(""))
    with pytest.raises(ValueError, match="no text output"):
        query_local_openai(client, "dummy-model", "msg", "sys", [], None)


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
        self.models = SimpleNamespace(
            generate_content=lambda **kwargs: response
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


def test_validate_gemini_response_accepts_completed_content():
    response = _gemini_response(
        parts=[_gemini_part("done")],
        finish_reason=types.FinishReason.STOP,
    )
    validate_gemini_response(response, "done")  # must not raise


def test_query_gemini_raises_on_empty_response():
    # A Gemini safety-block returns candidates with no text parts.
    response = _gemini_response(parts=[], text=None)
    client = _FakeGeminiClient(response)
    with pytest.raises(ValueError, match="no text output"):
        query_gemini(
            client,
            "gemini-2.5-flash",
            "msg",
            "sys",
            [],
            None,
            max_tokens=128,
        )


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
