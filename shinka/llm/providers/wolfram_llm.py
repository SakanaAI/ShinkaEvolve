"""Wolfram-language LLM provider routed through ``wolframscript``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import backoff

from shinka.llm.constants import BACKOFF_MAX_TIME, BACKOFF_MAX_TRIES, BACKOFF_MAX_VALUE
from shinka.utils.wolfram import (
    build_wolframscript_argv,
    is_wolframscript_available,
    wolframscript_bin,
)
from .pricing import calculate_cost, model_exists
from .result import QueryResult

logger = logging.getLogger(__name__)


MAX_TRIES = BACKOFF_MAX_TRIES
MAX_VALUE = BACKOFF_MAX_VALUE
MAX_TIME = BACKOFF_MAX_TIME

# A run that times out is almost always going to time out again; cap retries
# to keep one bad subprocess from monopolizing the caller for hours.
TIMEOUT_MAX_TRIES = 2
TIMEOUT_MAX_VALUE = 10

WOLFRAM_LLM_TIMEOUT_ENV = "WOLFRAM_LLM_TIMEOUT_SEC"
WOLFRAM_LLM_USE_LLMSYNTHESIZE_ENV = "WOLFRAM_LLM_USE_LLMSYNTHESIZE"
BRIDGE_SCRIPT = Path(__file__).parent / "wolfram_llm_bridge.wl"


def _wolfram_llm_timeout() -> int:
    """Per-call subprocess timeout in seconds (default 600)."""
    return int(os.environ.get(WOLFRAM_LLM_TIMEOUT_ENV, "600"))


def _use_llmsynthesize() -> bool:
    """Whether to route the bridge through ``LLMSynthesize`` instead of the
    default ``ServiceConnect`` + ``ServiceExecute["ChatService"]`` path."""
    return os.environ.get(WOLFRAM_LLM_USE_LLMSYNTHESIZE_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Per-service env vars the bridge forwards to ServiceConnect as the
# Authentication APIKey. Standard names so existing keys work as-is.
SERVICE_API_KEY_ENV = {
    "OpenAI": ("OPENAI_API_KEY",),
    "Anthropic": ("ANTHROPIC_API_KEY",),
    "GoogleGemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "DeepSeek": ("DEEPSEEK_API_KEY",),
    "Groq": ("GROQ_API_KEY",),
    "MistralAI": ("MISTRAL_API_KEY",),
    "Cohere": ("COHERE_API_KEY",),
    "AlephAlpha": ("ALEPH_ALPHA_API_KEY", "ALEPHALPHA_API_KEY"),
    "TogetherAI": ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
}


class WolframLlmError(RuntimeError):
    pass


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Wolfram LLM - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


def _api_key_for(service):
    for var in SERVICE_API_KEY_ENV.get(service, ()):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return None


def _parse_model_name(model_name):
    """Split wolfram-llm/<Service>/<Model> into (service, model)."""
    prefix = "wolfram-llm/"
    if not model_name.startswith(prefix):
        raise WolframLlmError(
            f"Expected model name starting with 'wolfram-llm/', got {model_name!r}"
        )
    rest = model_name[len(prefix) :]
    if "/" not in rest:
        raise WolframLlmError(
            f"Model name {model_name!r} must be wolfram-llm/<service>/<model>"
        )
    service, model = rest.split("/", 1)
    if not service or not model:
        raise WolframLlmError(
            f"Both <service> and <model> are required in {model_name!r}"
        )
    return service, model


def _turn_content_to_text(content):
    """Extract plain text from either a string or an Anthropic-style list of
    content blocks (``[{"type": "text", "text": ...}, ...]``)."""
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
    if content is None:
        return ""
    return str(content)


def _build_messages(system_msg, msg_history, user_msg):
    """Build the role-tagged message list the bridge forwards verbatim to
    ``ServiceExecute[..., "ChatService", ...]``. Preserves the multi-turn
    structure the Wolfram chat APIs natively accept instead of collapsing
    history into a single user string."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    for turn in msg_history:
        messages.append(
            {
                "role": turn.get("role", "user"),
                "content": _turn_content_to_text(turn.get("content")),
            }
        )
    messages.append({"role": "user", "content": user_msg})
    return messages


def _build_spec(model_name, system_msg, user_msg, kwargs, msg_history=()):
    service, model = _parse_model_name(model_name)
    spec = {
        "service": service,
        "model": model,
        "messages": _build_messages(system_msg, msg_history, user_msg),
    }
    if "temperature" in kwargs:
        spec["temperature"] = float(kwargs["temperature"])
    if "max_tokens" in kwargs:
        spec["maxTokens"] = int(kwargs["max_tokens"])
    elif "max_output_tokens" in kwargs:
        spec["maxTokens"] = int(kwargs["max_output_tokens"])
    key = _api_key_for(service)
    if key:
        spec["apiKey"] = key
    if _use_llmsynthesize():
        spec["useLLMSynthesize"] = True
    return spec


def _ensure_bridge_available():
    if not is_wolframscript_available():
        raise WolframLlmError(
            f"`{wolframscript_bin()}` not found on PATH. Install Wolfram Engine "
            "or Mathematica, or set WOLFRAMSCRIPT_BIN to the absolute path of the binary."
        )
    if not BRIDGE_SCRIPT.exists():
        raise WolframLlmError(f"Wolfram bridge script not found at {BRIDGE_SCRIPT}.")


def _build_argv(in_path, out_path):
    return build_wolframscript_argv(["-file", str(BRIDGE_SCRIPT), in_path, out_path])


@contextlib.contextmanager
def _scratch_paths():
    tmp = Path(tempfile.gettempdir())
    tag = uuid.uuid4().hex[:12]
    in_path = tmp / f"shinka_wlm_{tag}_in.json"
    out_path = tmp / f"shinka_wlm_{tag}_out.json"
    try:
        yield str(in_path), str(out_path)
    finally:
        for p in (in_path, out_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def _read_response(out_path):
    p = Path(out_path)
    if not p.exists():
        raise WolframLlmError("wolframscript produced no output file.")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WolframLlmError(f"Could not parse bridge output: {exc}")


_LLMKIT_HINT = (
    "\n\nHint: this looks like the Wolfram LLMKit subscription gate. Either "
    "(a) set the service-specific API key env var (e.g. OPENAI_API_KEY, "
    "GEMINI_API_KEY, ANTHROPIC_API_KEY) so the bridge passes it through to "
    "ServiceConnect directly, or (b) run `wolframscript -authenticate` once "
    "to sign in your Wolfram ID."
)


def _raise_if_error(data, stderr_tail):
    if "error" in data:
        msg = data["error"]
        if data.get("raw"):
            msg += f" (raw: {str(data['raw'])[:200]})"
        if stderr_tail:
            msg += f" stderr: {stderr_tail}"
        haystack = msg.lower()
        if "llmkit" in haystack or (
            "subscription" in haystack and "wolfram" in haystack
        ):
            msg += _LLMKIT_HINT
        raise WolframLlmError(msg)


def _result_to_query(
    data, msg, system_msg, model_name, msg_history, kwargs, model_posteriors
):
    content = str(data.get("content", ""))
    new_msg_history = msg_history + [
        {"role": "user", "content": [{"type": "text", "text": msg}]},
        {"role": "assistant", "content": [{"type": "text", "text": content}]},
    ]
    input_tokens = int(data.get("inputTokens", 0))
    output_tokens = int(data.get("outputTokens", 0))
    _, bare_model = _parse_model_name(model_name)
    if model_exists(bare_model):
        input_cost, output_cost = calculate_cost(
            bare_model, input_tokens, output_tokens
        )
    else:
        input_cost = output_cost = 0.0
    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model_name,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=0,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )


@backoff.on_exception(
    backoff.expo,
    subprocess.TimeoutExpired,
    max_tries=TIMEOUT_MAX_TRIES,
    max_value=TIMEOUT_MAX_VALUE,
    on_backoff=backoff_handler,
)
@backoff.on_exception(
    backoff.expo,
    WolframLlmError,
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_wolfram_llm(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query an LLM via wolframscript's service framework."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for wolfram_llm.")
    _ensure_bridge_available()

    spec = _build_spec(model, system_msg, msg, kwargs, msg_history)

    with _scratch_paths() as (in_path, out_path):
        Path(in_path).write_text(json.dumps(spec), encoding="utf-8")
        proc = subprocess.run(
            _build_argv(in_path, out_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_wolfram_llm_timeout(),
        )
        stderr_tail = (proc.stderr or "").strip()[-400:]
        if proc.returncode != 0 and not Path(out_path).exists():
            raise WolframLlmError(
                f"wolframscript exited rc={proc.returncode}; stderr: {stderr_tail}"
            )
        data = _read_response(out_path)

    _raise_if_error(data, stderr_tail)
    return _result_to_query(
        data,
        msg,
        system_msg,
        model,
        msg_history,
        kwargs,
        model_posteriors,
    )


@backoff.on_exception(
    backoff.expo,
    asyncio.TimeoutError,
    max_tries=TIMEOUT_MAX_TRIES,
    max_value=TIMEOUT_MAX_VALUE,
    on_backoff=backoff_handler,
)
@backoff.on_exception(
    backoff.expo,
    WolframLlmError,
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_wolfram_llm_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Wolfram LLM bridge asynchronously."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for wolfram_llm.")
    _ensure_bridge_available()

    spec = _build_spec(model, system_msg, msg, kwargs, msg_history)

    with _scratch_paths() as (in_path, out_path):
        Path(in_path).write_text(json.dumps(spec), encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            *_build_argv(in_path, out_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=_wolfram_llm_timeout()
            )
        except (asyncio.TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            proc.kill()
            try:
                await proc.wait()
            except BaseException:
                pass
            raise
        stderr_tail = stderr_b.decode("utf-8", errors="replace").strip()[-400:]
        if proc.returncode != 0 and not Path(out_path).exists():
            raise WolframLlmError(
                f"wolframscript exited rc={proc.returncode}; stderr: {stderr_tail}"
            )
        data = _read_response(out_path)

    _raise_if_error(data, stderr_tail)
    return _result_to_query(
        data,
        msg,
        system_msg,
        model,
        msg_history,
        kwargs,
        model_posteriors,
    )
