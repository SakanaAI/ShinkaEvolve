from __future__ import annotations

import json
from pathlib import Path

import pytest

from shinka.llm.client import get_client_llm
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.llm.providers import wolfram_llm as wlm
from shinka.llm.providers.wolfram_llm import (
    BRIDGE_SCRIPT,
    WolframLlmError,
    _build_argv,
    _build_spec,
    _ensure_bridge_available,
    _parse_model_name,
    _raise_if_error,
    _read_response,
    _result_to_query,
    query_wolfram_llm,
)
from shinka.utils.wolfram import (
    DEFAULT_WOLFRAMSCRIPT_BIN,
    check_wolframscript_available,
    escape_wolfram_string,
    is_shell_script,
    is_wolframscript_available,
    is_wsl,
    resolve_wolframscript_bin,
    wolframscript_bin,
)

# ---------------------------------------------------------------------------
# Model name parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("wolfram-llm/OpenAI/gpt-4o", ("OpenAI", "gpt-4o")),
        ("wolfram-llm/Anthropic/claude-sonnet-4-5", ("Anthropic", "claude-sonnet-4-5")),
        (
            "wolfram-llm/GoogleGemini/gemini-2.5-flash",
            ("GoogleGemini", "gemini-2.5-flash"),
        ),
        ("wolfram-llm/DeepSeek/deepseek-chat", ("DeepSeek", "deepseek-chat")),
        (
            "wolfram-llm/Groq/llama-3.3-70b-versatile",
            ("Groq", "llama-3.3-70b-versatile"),
        ),
    ],
)
def test_parse_model_name(name, expected):
    assert _parse_model_name(name) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "wolfram-llm/OpenAI",  # missing model
        "wolfram-llm//gpt-4",  # empty service
        "wolfram-llm/OpenAI/",  # empty model
        "claude-cli/sonnet",  # different prefix
        "wolfram-llm",  # no slash at all
    ],
)
def test_parse_model_name_rejects_invalid(bad):
    with pytest.raises(WolframLlmError):
        _parse_model_name(bad)


# ---------------------------------------------------------------------------
# Resolver / client construction
# ---------------------------------------------------------------------------


def test_resolver_recognizes_wolfram_llm():
    resolved = resolve_model_backend("wolfram-llm/Anthropic/claude-sonnet-4-5")
    assert resolved.provider == "wolfram_llm"
    # api_model_name is the bare <Model> so pricing.csv lookups
    # (is_reasoning_model, etc.) resolve the same way as for any direct-API
    # model; original_model_name keeps the full composite for the bridge.
    assert resolved.api_model_name == "claude-sonnet-4-5"
    assert resolved.original_model_name == "wolfram-llm/Anthropic/claude-sonnet-4-5"


def test_resolver_rejects_malformed_wolfram_llm():
    with pytest.raises(ValueError):
        resolve_model_backend("wolfram-llm/OpenAI")  # missing model


def test_get_client_llm_returns_none_for_wolfram_llm():
    client, name, provider = get_client_llm("wolfram-llm/OpenAI/gpt-4o")
    assert client is None
    assert provider == "wolfram_llm"
    assert name == "wolfram-llm/OpenAI/gpt-4o"


# ---------------------------------------------------------------------------
# Spec building
# ---------------------------------------------------------------------------


def test_build_spec_basic():
    spec = _build_spec(
        "wolfram-llm/OpenAI/gpt-4o",
        "you are terse",
        "hello",
        {},
    )
    assert spec["service"] == "OpenAI"
    assert spec["model"] == "gpt-4o"
    # System prompt is the first turn, current user message the last;
    # the bridge forwards this list to ServiceExecute as-is.
    assert spec["messages"] == [
        {"role": "system", "content": "you are terse"},
        {"role": "user", "content": "hello"},
    ]
    assert "temperature" not in spec  # only set when explicitly passed
    assert "maxTokens" not in spec


def test_build_spec_preserves_msg_history():
    """Prior turns are forwarded as role-tagged messages, not collapsed."""
    history = [
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "first reply"},
        # Anthropic-style block list should be flattened to plain text:
        {"role": "user", "content": [{"type": "text", "text": "second ask"}]},
        {"role": "assistant", "content": "second reply"},
    ]
    spec = _build_spec(
        "wolfram-llm/OpenAI/gpt-4o",
        "be terse",
        "current ask",
        {},
        msg_history=history,
    )
    assert spec["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second ask"},
        {"role": "assistant", "content": "second reply"},
        {"role": "user", "content": "current ask"},
    ]


def test_build_spec_no_system_prompt_omits_system_turn():
    spec = _build_spec(
        "wolfram-llm/OpenAI/gpt-4o",
        "",
        "just ask",
        {},
    )
    assert spec["messages"] == [{"role": "user", "content": "just ask"}]


def test_build_spec_with_kwargs():
    spec = _build_spec(
        "wolfram-llm/Anthropic/claude-sonnet-4-5",
        "sys",
        "msg",
        {"temperature": 0.7, "max_tokens": 4096},
    )
    assert spec["temperature"] == 0.7
    assert spec["maxTokens"] == 4096


def test_build_spec_max_output_tokens_alias():
    """ShinkaEvolve's kwargs builder uses max_output_tokens for non-anthropic
    providers; wolfram_llm should accept both spellings."""
    spec = _build_spec(
        "wolfram-llm/OpenAI/gpt-4o",
        "",
        "hi",
        {"max_output_tokens": 2048},
    )
    assert spec["maxTokens"] == 2048


# ---------------------------------------------------------------------------
# Bridge script presence
# ---------------------------------------------------------------------------


def test_bridge_script_present():
    assert BRIDGE_SCRIPT.exists()
    assert BRIDGE_SCRIPT.suffix == ".wl"
    text = BRIDGE_SCRIPT.read_text(encoding="utf-8")
    # Bridge talks to Wolfram's service framework via the Chat endpoint.
    assert "ServiceConnect" in text
    assert "ServiceExecute" in text
    assert '"Chat"' in text


# ---------------------------------------------------------------------------
# End-to-end with mocked subprocess
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _mock_subprocess_run(canned_output: dict, returncode: int = 0, stderr: str = ""):
    """Build a side_effect that writes `canned_output` to the bridge's
    output file and returns a fake CompletedProcess. Handles both
    direct-argv and bash-c-wrapped argv forms."""

    def _side_effect(argv, **_kwargs):
        if len(argv) >= 3 and argv[0] == "bash" and argv[1] == "-c":
            # argv = ["bash", "-c", "<bin> -file <bridge> <in> <out>"]
            import shlex

            tokens = shlex.split(argv[2])
            out_path = tokens[-1]
        else:
            # argv = [bin, -file, bridge, in_path, out_path]
            out_path = argv[-1]
        Path(out_path).write_text(json.dumps(canned_output), encoding="utf-8")
        return _FakeCompletedProcess(returncode=returncode, stderr=stderr)

    return _side_effect


def test_query_wolfram_llm_success(monkeypatch):
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/local/bin/wolframscript",
    )
    monkeypatch.setattr(
        "shinka.llm.providers.wolfram_llm.subprocess.run",
        _mock_subprocess_run(
            {
                "content": "Hello from Wolfram!",
                "service": "OpenAI",
                "model": "gpt-4.1",
                "inputTokens": 11,
                "outputTokens": 5,
            }
        ),
    )

    result = query_wolfram_llm(
        client=None,
        model="wolfram-llm/OpenAI/gpt-4.1",
        msg="say hi",
        system_msg="be terse",
        msg_history=[],
        output_model=None,
    )
    assert result.content == "Hello from Wolfram!"
    assert result.model_name == "wolfram-llm/OpenAI/gpt-4.1"
    assert result.input_tokens == 11
    assert result.output_tokens == 5
    # gpt-4.1 is in pricing.csv so cost is real, not 0.0:
    assert result.cost > 0.0
    assert result.input_cost > 0.0
    assert result.output_cost > 0.0
    # History closed with assistant turn at the end:
    assert result.new_msg_history[-1]["role"] == "assistant"


# The error-path tests below target the helper functions directly rather than
# the @backoff-wrapped query function (which would retry 20 times under
# BACKOFF_MAX_TRIES — fine in production but unusable in unit tests).


def test_raise_if_error_with_error():
    with pytest.raises(WolframLlmError, match="credentials missing"):
        _raise_if_error(
            {"error": "ServiceConnect: OpenAI credentials missing", "raw": "$Failed"},
            stderr_tail="",
        )


def test_raise_if_error_passthrough():
    """No error key — no raise."""
    _raise_if_error({"content": "ok"}, stderr_tail="")


def test_read_response_missing_file(tmp_path):
    with pytest.raises(WolframLlmError, match="no output file"):
        _read_response(str(tmp_path / "does_not_exist.json"))


def test_read_response_invalid_json(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("this is not json", encoding="utf-8")
    with pytest.raises(WolframLlmError, match="parse"):
        _read_response(str(p))


def test_ensure_bridge_available_missing_binary(monkeypatch):
    monkeypatch.setattr("shinka.utils.wolfram.shutil.which", lambda _bin: None)
    monkeypatch.setattr("shinka.utils.wolfram.Path.is_file", lambda _self: False)
    with pytest.raises(WolframLlmError, match="not found on PATH"):
        _ensure_bridge_available()


def test_result_to_query_shape():
    """Verify QueryResult assembly is correct.

    _result_to_query takes the *original* msg_history and appends both the
    user and the assistant turn.
    """
    qr = _result_to_query(
        data={
            "content": "the answer",
            "inputTokens": 17,
            "outputTokens": 4,
        },
        msg="ask",
        system_msg="be helpful",
        model_name="wolfram-llm/OpenAI/gpt-4.1",
        msg_history=[],
        kwargs={},
        model_posteriors=None,
    )
    assert qr.content == "the answer"
    assert qr.model_name == "wolfram-llm/OpenAI/gpt-4.1"
    assert qr.new_msg_history[0]["role"] == "user"
    assert qr.new_msg_history[0]["content"][0]["text"] == "ask"
    assert qr.new_msg_history[-1]["role"] == "assistant"
    assert qr.new_msg_history[-1]["content"][0]["text"] == "the answer"
    assert qr.input_tokens == 17
    assert qr.output_tokens == 4
    # gpt-4.1 is in pricing.csv; cost is calculated from the bare model name.
    assert qr.cost > 0.0
    assert qr.input_cost > 0.0
    assert qr.output_cost > 0.0


def test_result_to_query_no_usage_no_cost():
    """When the bridge omits token counts (e.g. service paclet doesn't
    surface Usage), tokens stay at 0 and cost stays at 0 rather than
    making up a fake nonzero value."""
    qr = _result_to_query(
        data={"content": "ok"},
        msg="ask",
        system_msg="",
        model_name="wolfram-llm/OpenAI/gpt-4.1",
        msg_history=[],
        kwargs={},
        model_posteriors=None,
    )
    assert qr.input_tokens == 0
    assert qr.output_tokens == 0
    assert qr.cost == 0.0
    assert qr.input_cost == 0.0
    assert qr.output_cost == 0.0


def test_result_to_query_unknown_bare_model_no_cost():
    """If the bare model is not in pricing.csv (e.g. some Groq or
    AlephAlpha models), token counts are still surfaced honestly but cost
    is 0 rather than crashing on the unknown-model pricing lookup."""
    qr = _result_to_query(
        data={
            "content": "ok",
            "inputTokens": 50,
            "outputTokens": 5,
        },
        msg="ask",
        system_msg="",
        model_name="wolfram-llm/Groq/no-such-groq-model-in-pricing-csv",
        msg_history=[],
        kwargs={},
        model_posteriors=None,
    )
    assert qr.input_tokens == 50
    assert qr.output_tokens == 5
    assert qr.cost == 0.0


# ---------------------------------------------------------------------------
# Cross-platform argv construction
# ---------------------------------------------------------------------------


def test_resolve_wolframscript_bin_returns_which_result(monkeypatch):
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/opt/Wolfram/wolframscript",
    )
    assert resolve_wolframscript_bin() == "/opt/Wolfram/wolframscript"


def test_resolve_wolframscript_bin_falls_back_to_raw_when_not_found(monkeypatch):
    monkeypatch.setattr("shinka.utils.wolfram.shutil.which", lambda _bin: None)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.delenv("WOLFRAMSCRIPT_BIN", raising=False)
    assert resolve_wolframscript_bin() == DEFAULT_WOLFRAMSCRIPT_BIN


def test_wolframscript_bin_reads_env_at_call_time(monkeypatch):
    monkeypatch.delenv("WOLFRAMSCRIPT_BIN", raising=False)
    assert wolframscript_bin() == DEFAULT_WOLFRAMSCRIPT_BIN
    monkeypatch.setenv("WOLFRAMSCRIPT_BIN", "/opt/wolfram/wolframscript")
    assert wolframscript_bin() == "/opt/wolfram/wolframscript"


def test_is_wolframscript_available_when_findable(monkeypatch):
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/local/bin/wolframscript",
    )
    assert is_wolframscript_available() is True


def test_is_wolframscript_available_when_missing(monkeypatch):
    monkeypatch.setattr("shinka.utils.wolfram.shutil.which", lambda _bin: None)
    monkeypatch.setattr("shinka.utils.wolfram.Path.is_file", lambda _self: False)
    assert is_wolframscript_available() is False


def test_check_wolframscript_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr("shinka.utils.wolfram.shutil.which", lambda _bin: None)
    monkeypatch.setattr("shinka.utils.wolfram.Path.is_file", lambda _self: False)
    with pytest.raises(ValueError, match="not found on PATH"):
        check_wolframscript_available()


def test_check_wolframscript_available_passes_when_findable(monkeypatch):
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/local/bin/wolframscript",
    )
    check_wolframscript_available()  # must not raise


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain", "plain"),
        (r'name"with"quotes', r"name\"with\"quotes"),
        (r"path\with\backslash", r"path\\with\\backslash"),
        (
            r'C:\Program Files\"wolfram"\file.wl',
            r"C:\\Program Files\\\"wolfram\"\\file.wl",
        ),
        ("", ""),
    ],
)
def test_escape_wolfram_string(raw, expected):
    assert escape_wolfram_string(raw) == expected


def test_looks_like_shell_script_detects_shebang(tmp_path):
    p = tmp_path / "wolframscript"
    p.write_bytes(b'#!/bin/bash\nexec /mnt/c/wolframscript.exe "$@"\n')
    assert is_shell_script(str(p)) is True


def test_looks_like_shell_script_rejects_binary(tmp_path):
    p = tmp_path / "wolframscript"
    p.write_bytes(b"\x7fELF\x02\x01\x01\x00")
    assert is_shell_script(str(p)) is False


def test_looks_like_shell_script_handles_missing(tmp_path):
    assert is_shell_script(str(tmp_path / "nope")) is False


def test_build_argv_direct_when_not_wsl(monkeypatch, tmp_path):
    monkeypatch.setattr("shinka.utils.wolfram.is_wsl", lambda: False)
    monkeypatch.setattr(
        "shinka.utils.wolfram.shutil.which",
        lambda _bin: "/usr/bin/wolframscript",
    )
    argv = _build_argv(str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    assert argv[0] == "/usr/bin/wolframscript"
    assert "bash" not in argv[:2]


def test_build_argv_direct_when_wsl_but_binary_native(monkeypatch, tmp_path):
    """WSL plus a real (non-shebang) binary still goes direct, no bash wrap."""
    binary = tmp_path / "wolframscript"
    binary.write_bytes(b"\x7fELF\x02\x01")
    monkeypatch.setattr("shinka.utils.wolfram.is_wsl", lambda: True)
    monkeypatch.setattr("shinka.utils.wolfram.shutil.which", lambda _b: str(binary))
    argv = _build_argv(str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    assert argv[0] == str(binary)
    assert argv[:2] != ["bash", "-c"]


def test_build_argv_wraps_on_wsl_when_shell_wrapper(monkeypatch, tmp_path):
    """WSL plus a shell-script wrapper requires the bash -c wrap."""
    binary = tmp_path / "wolframscript.sh"
    binary.write_bytes(b'#!/bin/bash\nexec /mnt/c/wolframscript.exe "$@"\n')
    monkeypatch.setattr("shinka.utils.wolfram.is_wsl", lambda: True)
    monkeypatch.setattr("shinka.utils.wolfram.shutil.which", lambda _b: str(binary))
    argv = _build_argv(str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    assert argv[:2] == ["bash", "-c"]
    assert str(binary) in argv[2]


def test_is_wsl_detects_env(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-24.04")
    assert is_wsl() is True


# ---------------------------------------------------------------------------
# Error message hint pattern matching
# ---------------------------------------------------------------------------


def test_raise_if_error_appends_llmkit_hint():
    with pytest.raises(WolframLlmError) as exc:
        _raise_if_error(
            {"error": "Cannot check LLMKit subscription status", "raw": "$Failed"},
            stderr_tail="",
        )
    assert "wolframscript -authenticate" in str(exc.value)
    assert "API key env var" in str(exc.value)


def test_raise_if_error_no_hint_for_unrelated_error():
    with pytest.raises(WolframLlmError) as exc:
        _raise_if_error(
            {"error": "Service quota exhausted", "raw": ""},
            stderr_tail="",
        )
    assert "wolframscript -authenticate" not in str(exc.value)


# ---------------------------------------------------------------------------
# Env-var config is read at call time
# ---------------------------------------------------------------------------


def test_wolfram_llm_timeout_reads_env_at_call_time(monkeypatch):
    monkeypatch.delenv("WOLFRAM_LLM_TIMEOUT_SEC", raising=False)
    assert wlm._wolfram_llm_timeout() == 600
    monkeypatch.setenv("WOLFRAM_LLM_TIMEOUT_SEC", "900")
    assert wlm._wolfram_llm_timeout() == 900


def test_use_llmsynthesize_reads_env_at_call_time(monkeypatch):
    monkeypatch.delenv("WOLFRAM_LLM_USE_LLMSYNTHESIZE", raising=False)
    assert wlm._use_llmsynthesize() is False
    monkeypatch.setenv("WOLFRAM_LLM_USE_LLMSYNTHESIZE", "1")
    assert wlm._use_llmsynthesize() is True


def test_build_spec_includes_llmsynthesize_when_enabled(monkeypatch):
    monkeypatch.setenv("WOLFRAM_LLM_USE_LLMSYNTHESIZE", "1")
    spec = _build_spec("wolfram-llm/OpenAI/gpt-4o", "sys", "msg", {})
    assert spec["useLLMSynthesize"] is True


def test_build_spec_omits_llmsynthesize_by_default(monkeypatch):
    monkeypatch.delenv("WOLFRAM_LLM_USE_LLMSYNTHESIZE", raising=False)
    spec = _build_spec("wolfram-llm/OpenAI/gpt-4o", "sys", "msg", {})
    assert "useLLMSynthesize" not in spec
