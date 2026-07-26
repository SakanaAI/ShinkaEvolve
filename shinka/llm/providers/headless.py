from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs

from pydantic import BaseModel

from shinka.llm.constants import TIMEOUT

from .result import QueryResult
from .errors import (
    LLMAuthenticationError,
    LLMExtractionError,
    LLMModelUnavailableError,
    LLMProcessError,
    LLMTimeoutError,
)

DEFAULT_HEADLESS_COMMAND = "npx -y @roberttlange/headless"
HEADLESS_COMMAND_ENV = "SHINKA_HEADLESS_COMMAND"
HEADLESS_TIMEOUT_ENV = "SHINKA_HEADLESS_TIMEOUT"
HEADLESS_AGENT_TIMEOUT_ENV_PREFIX = "SHINKA_HEADLESS_TIMEOUT_"
HEADLESS_CLEANUP_GRACE_ENV = "SHINKA_HEADLESS_CLEANUP_GRACE"
HEADLESS_OUTPUT_MODE_ENV = "SHINKA_HEADLESS_OUTPUT_MODE"
DEFAULT_HEADLESS_PROPOSAL_TIMEOUT = 7200.0
DEFAULT_HEADLESS_TEXT_TIMEOUT = 900.0
DEFAULT_HEADLESS_CLEANUP_GRACE = 60.0
HeadlessResponseMode = Literal["worktree", "text"]

_VALID_EFFORTS = {"low", "medium", "high", "xhigh"}
_THREAD_LOCK = threading.Lock()
_CLAUDE_TRANSIENT_RETRIES = 2
_CLAUDE_TRANSIENT_BACKOFF_SECONDS = 3.0


@dataclass(frozen=True)
class HeadlessModel:
    agent: str
    agent_model: str | None = None
    effort: str | None = None


def headless_command_prefix() -> list[str]:
    raw_command = os.getenv(HEADLESS_COMMAND_ENV, DEFAULT_HEADLESS_COMMAND).strip()
    if not raw_command:
        raise ValueError(f"{HEADLESS_COMMAND_ENV} cannot be empty.")
    return shlex.split(raw_command)


def _parse_timeout_env(*, env_name: str, raw_timeout: str) -> float:
    timeout = float(raw_timeout)
    if timeout <= 0:
        raise ValueError(f"{env_name} must be > 0.")
    return timeout


def _agent_timeout_env_name(model: HeadlessModel | None) -> str | None:
    if model is None:
        return None
    agent_key = "".join(
        char.upper() if char.isalnum() else "_" for char in model.agent
    )
    return f"{HEADLESS_AGENT_TIMEOUT_ENV_PREFIX}{agent_key}"


def headless_timeout(
    model: HeadlessModel | None = None,
    configured_timeout: float | None = None,
) -> float:
    if configured_timeout is not None:
        if float(configured_timeout) <= 0:
            raise ValueError("headless proposal timeout must be > 0.")
        return float(configured_timeout)
    agent_timeout_env = _agent_timeout_env_name(model)
    if agent_timeout_env:
        raw_agent_timeout = os.getenv(agent_timeout_env)
        if raw_agent_timeout is not None:
            return _parse_timeout_env(
                env_name=agent_timeout_env,
                raw_timeout=raw_agent_timeout,
            )

    raw_timeout = os.getenv(HEADLESS_TIMEOUT_ENV)
    if raw_timeout is None:
        return max(float(TIMEOUT), DEFAULT_HEADLESS_PROPOSAL_TIMEOUT)
    return _parse_timeout_env(env_name=HEADLESS_TIMEOUT_ENV, raw_timeout=raw_timeout)


def headless_cleanup_grace(configured_grace: float | None = None) -> float:
    if configured_grace is not None:
        grace = float(configured_grace)
    else:
        grace = float(
            os.getenv(HEADLESS_CLEANUP_GRACE_ENV, DEFAULT_HEADLESS_CLEANUP_GRACE)
        )
    if grace <= 0:
        raise ValueError("headless cleanup grace must be > 0.")
    return grace


def headless_output_mode(configured_mode: str | None = None) -> str:
    mode = (configured_mode or os.getenv(HEADLESS_OUTPUT_MODE_ENV, "json")).lower()
    if mode not in {"json", "usage"}:
        raise ValueError("headless output mode must be 'json' or 'usage'.")
    return mode


def _thread_cli_lock() -> threading.Lock:
    return _THREAD_LOCK


async def _acquire_cli_lock_async() -> None:
    await asyncio.to_thread(_THREAD_LOCK.acquire)


def parse_headless_model(model_name: str) -> HeadlessModel:
    if not model_name.startswith("headless/"):
        raise ValueError(f"Headless model must start with 'headless/': {model_name}")

    body = model_name.split("headless/", 1)[1]
    route, _, query = body.partition("?")
    agent, separator, agent_model = route.partition("@")
    if not agent:
        raise ValueError("Headless agent name is required.")

    parsed_query = parse_qs(query, keep_blank_values=True)
    unknown_keys = sorted(set(parsed_query) - {"effort"})
    if unknown_keys:
        raise ValueError(
            f"Unsupported headless model query parameter(s): {unknown_keys}"
        )

    effort_values = parsed_query.get("effort", [])
    if len(effort_values) > 1:
        raise ValueError("Headless model may specify effort only once.")
    effort = effort_values[0] if effort_values else None
    if effort is not None and effort not in _VALID_EFFORTS:
        raise ValueError(
            f"Unsupported headless effort '{effort}'. "
            f"Expected one of {sorted(_VALID_EFFORTS)}."
        )

    return HeadlessModel(
        agent=agent,
        agent_model=agent_model if separator else None,
        effort=effort,
    )


def check_headless_available() -> None:
    cmd = [*headless_command_prefix(), "--check"]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=min(headless_timeout(), 60.0),
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Headless command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Headless availability check timed out.") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise ValueError(f"Headless availability check failed: {detail}")


def _normalize_response_mode(response_mode: str | None) -> HeadlessResponseMode:
    mode = (response_mode or "worktree").lower()
    if mode not in {"worktree", "text"}:
        raise ValueError(
            "headless_response_mode must be 'worktree' or 'text', "
            f"got {response_mode!r}"
        )
    return mode  # type: ignore[return-value]


def _render_prompt(
    *,
    work_dir: Path,
    msg: str,
    system_msg: str,
    msg_history: list[dict],
    response_mode: HeadlessResponseMode = "worktree",
) -> str:
    history_text = json.dumps(msg_history, indent=2, ensure_ascii=False)
    if response_mode == "text":
        return (
            "# System Instructions\n\n"
            f"{system_msg}\n\n"
            "# Previous Messages\n\n"
            f"{history_text}\n\n"
            "# User Request\n\n"
            f"{msg}\n\n"
            "# Response Contract\n\n"
            "Treat this as a text-only chat completion. Put your full answer in "
            "the final assistant message. Do not edit, create, or delete files "
            "unless the user request explicitly requires it. An empty scratch "
            f"directory is available at `{work_dir}` if a working directory is "
            "required by the agent runtime.\n"
        )
    return (
        "# Active Repository\n\n"
        f"The proposal repository is `{work_dir}`. Begin every filesystem and "
        "terminal operation in this exact directory. Do not place candidate "
        "files in a provider scratch directory; only changes under this path "
        "are captured by Shinka.\n\n"
        "# System Instructions\n\n"
        f"{system_msg}\n\n"
        "# Previous Messages\n\n"
        f"{history_text}\n\n"
        "# User Request\n\n"
        f"{msg}\n"
    )


def _write_prompt_file(
    *,
    work_dir: str | None,
    msg: str,
    system_msg: str,
    msg_history: list[dict],
    response_mode: HeadlessResponseMode = "worktree",
) -> Path:
    resolved_work_dir = Path(work_dir or os.getcwd()).absolute()
    prompt_dir = resolved_work_dir / ".shinka"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"headless_prompt_{uuid.uuid4().hex[:8]}.md"
    prompt_path.write_text(
        _render_prompt(
            work_dir=resolved_work_dir,
            msg=msg,
            system_msg=system_msg,
            msg_history=msg_history,
            response_mode=response_mode,
        ),
        encoding="utf-8",
    )
    return prompt_path


def _prepare_headless_work_dir(
    *,
    headless_work_dir: str | None,
    response_mode: HeadlessResponseMode,
) -> tuple[Path, bool]:
    """Return (work_dir, owns_directory).

    Text mode always uses an ephemeral empty scratch directory so meta/novelty/
    prompt calls do not need a real repository. The optional headless_work_dir
    is treated as a parent for those scratch dirs.

    When a parent under the run results is provided, the scratch is retained for
    audit (owns_directory=False). System temp scratches are cleaned up.
    """
    if response_mode == "text":
        if headless_work_dir:
            parent = Path(headless_work_dir).absolute()
            parent.mkdir(parents=True, exist_ok=True)
            scratch = Path(
                tempfile.mkdtemp(prefix="headless_chat_", dir=str(parent))
            )
            return scratch, False
        scratch = Path(tempfile.mkdtemp(prefix="headless_chat_"))
        return scratch, True
    return Path(headless_work_dir or os.getcwd()).absolute(), False


def _headless_log_paths(*, work_dir: Path) -> tuple[Path, Path]:
    shinka_dir = work_dir / ".shinka"
    shinka_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    return (
        shinka_dir / f"headless_{run_id}.stdout.log",
        shinka_dir / f"headless_{run_id}.stderr.log",
    )


def _build_headless_command(
    *,
    model: HeadlessModel,
    prompt_path: Path,
    work_dir: str | None,
    session_name: str | None = None,
    proposal_timeout: float | None = None,
    output_mode: str | None = None,
) -> list[str]:
    resolved_work_dir = str(Path(work_dir or os.getcwd()).absolute())
    cmd = [
        *headless_command_prefix(),
        model.agent,
        "--prompt-file",
        str(prompt_path),
        "--work-dir",
        resolved_work_dir,
        "--allow",
        "yolo",
        "--timeout",
        str(max(1, math.ceil(headless_timeout(model, proposal_timeout)))),
    ]
    if headless_output_mode(output_mode) == "json":
        cmd.append("--json")
    else:
        cmd.append("--usage")
    if model.agent_model:
        cmd.extend(["--model", model.agent_model])
    if model.effort:
        cmd.extend(["--reasoning-effort", model.effort])
    if session_name:
        cmd.extend(["--session", session_name])
    return cmd


def _uses_shell_invocation(model: HeadlessModel) -> bool:
    return model.agent == "claude"


def _subprocess_env(model: HeadlessModel) -> dict[str, str] | None:
    if model.agent != "claude":
        return None
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    return env


def _is_transient_claude_credit_error(
    *,
    model: HeadlessModel,
    completed: subprocess.CompletedProcess,
) -> bool:
    if model.agent != "claude" or completed.returncode == 0:
        return False
    output = f"{completed.stderr}\n{completed.stdout}"
    return "Credit balance is too low" in output and '"pricingSource":"models.dev"' in output


def _headless_failure(*, completed: subprocess.CompletedProcess) -> RuntimeError:
    detail = (completed.stderr.strip() or completed.stdout.strip()).strip()
    normalized = detail.lower()
    if any(
        marker in normalized
        for marker in (
            "authentication required",
            "not authenticated",
            "unauthorized",
            "could not create cursor session",
            "run 'agent login'",
        )
    ):
        return LLMAuthenticationError(detail or "Headless authentication failed")
    if any(
        marker in normalized
        for marker in ("model not found", "unknown model", "model unavailable")
    ):
        return LLMModelUnavailableError(detail or "Headless model unavailable")
    if "could not extract final message" in normalized:
        return LLMExtractionError(detail)
    if completed.returncode == 124 or "timed out" in normalized:
        return LLMTimeoutError(detail or "Headless proposal timed out")
    return LLMProcessError(detail or f"Headless exited {completed.returncode}")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group_sync(
    process: subprocess.Popen,
    *,
    grace_seconds: float,
) -> None:
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and _process_group_exists(process_group_id):
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def _terminate_process_group_async(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> None:
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and _process_group_exists(process_group_id):
        await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.returncode is None:
        await process.wait()


def _run_subprocess_with_timeout(
    *,
    model: HeadlessModel,
    command: list[str],
    stdout_file,
    stderr_file,
    proposal_timeout: float,
    cleanup_grace: float,
) -> subprocess.CompletedProcess:
    if _uses_shell_invocation(model):
        popen_args = {
            "args": shlex.join(command),
            "shell": True,
            "executable": "/bin/sh",
        }
    else:
        popen_args = {"args": command}

    process = subprocess.Popen(
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        env=_subprocess_env(model),
        start_new_session=True,
        **popen_args,
    )
    try:
        process.wait(timeout=proposal_timeout + cleanup_grace)
    except subprocess.TimeoutExpired:
        _terminate_process_group_sync(process, grace_seconds=cleanup_grace)
        raise
    if process.returncode != 0:
        _terminate_process_group_sync(process, grace_seconds=cleanup_grace)
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
    )


def _run_headless_command_sync(
    *,
    model: HeadlessModel,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    proposal_timeout: float,
    cleanup_grace: float,
) -> subprocess.CompletedProcess:
    attempts = _CLAUDE_TRANSIENT_RETRIES + 1
    for attempt in range(attempts):
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            completed = _run_subprocess_with_timeout(
                model=model,
                command=command,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                proposal_timeout=proposal_timeout,
                cleanup_grace=cleanup_grace,
            )
        completed.stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        completed.stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

        if not _is_transient_claude_credit_error(model=model, completed=completed):
            return completed
        if attempt < attempts - 1:
            time.sleep(_CLAUDE_TRANSIENT_BACKOFF_SECONDS * (attempt + 1))
    return completed


async def _run_headless_command_async(
    *,
    model: HeadlessModel,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    proposal_timeout: float,
    cleanup_grace: float,
):
    attempts = _CLAUDE_TRANSIENT_RETRIES + 1
    for attempt in range(attempts):
        stdout_file = stdout_path.open("w", encoding="utf-8")
        stderr_file = stderr_path.open("w", encoding="utf-8")
        try:
            if _uses_shell_invocation(model):
                process = await asyncio.create_subprocess_shell(
                    shlex.join(command),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    executable="/bin/sh",
                    env=_subprocess_env(model),
                    start_new_session=True,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=_subprocess_env(model),
                    start_new_session=True,
                )
            try:
                await asyncio.wait_for(
                    process.communicate(),
                    timeout=proposal_timeout + cleanup_grace,
                )
            except asyncio.TimeoutError:
                await _terminate_process_group_async(
                    process,
                    grace_seconds=cleanup_grace,
                )
                raise
            except asyncio.CancelledError:
                await _terminate_process_group_async(
                    process,
                    grace_seconds=cleanup_grace,
                )
                raise
        finally:
            stdout_file.close()
            stderr_file.close()
        completed = subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout_path.read_text(encoding="utf-8", errors="replace"),
            stderr=stderr_path.read_text(encoding="utf-8", errors="replace"),
        )
        if completed.returncode != 0:
            await _terminate_process_group_async(
                process,
                grace_seconds=cleanup_grace,
            )
        if not _is_transient_claude_credit_error(model=model, completed=completed):
            return completed
        if attempt < attempts - 1:
            await asyncio.sleep(_CLAUDE_TRANSIENT_BACKOFF_SECONDS * (attempt + 1))
    return completed


def _usage_int(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if value is not None:
            if isinstance(value, dict):
                return int(
                    sum(
                        item
                        for item in _numeric_values(value)
                        if float(item).is_integer()
                    )
                )
            return int(value)
    return 0


def _usage_float(usage: dict[str, Any], *names: str) -> float:
    for name in names:
        value = usage.get(name)
        if value is not None:
            if isinstance(value, dict):
                if isinstance(value.get("total"), (int, float)):
                    return float(value["total"])
                return sum(_numeric_values(value))
            return float(value)
    return 0.0


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        numbers: list[float] = []
        for item in value.values():
            numbers.extend(_numeric_values(item))
        return numbers
    if isinstance(value, list):
        numbers = []
        for item in value:
            numbers.extend(_numeric_values(item))
        return numbers
    return []


_HEADLESS_USAGE_KEYS = {
    "inputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
    "input_tokens",
    "output_tokens",
}


def _headless_usage_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in reversed(value):
            usage = _headless_usage_payload(item)
            if usage is not None:
                return usage
        return None
    if not isinstance(value, dict):
        return None

    nested_usage = value.get("usage")
    if isinstance(nested_usage, dict):
        return nested_usage
    if _HEADLESS_USAGE_KEYS.intersection(value):
        return value
    return None


def _extract_headless_usage(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            continue
        usage = _headless_usage_payload(parsed)
        if usage is not None:
            return usage

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    usage = _headless_usage_payload(parsed)
    return usage or {}


def _query_usage_from_headless(usage: dict[str, Any]) -> dict[str, Any]:
    if not usage:
        return {}

    query_usage = dict(usage)
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens", "inputTokens")
    cache_read_tokens = _usage_int(
        usage,
        "cache_read_tokens",
        "cacheReadTokens",
        "cache_read_input_tokens",
        "cached_input_tokens",
    )
    cache_write_tokens = _usage_int(
        usage,
        "cache_write_tokens",
        "cacheWriteTokens",
        "cache_creation_input_tokens",
    )
    output_tokens = _usage_int(
        usage, "output_tokens", "completion_tokens", "outputTokens"
    )
    thinking_tokens = _usage_int(
        usage,
        "thinking_tokens",
        "reasoning_tokens",
        "reasoning_output_tokens",
        "reasoningOutputTokens",
    )
    total_tokens = _usage_int(usage, "total_tokens", "totalTokens")
    input_side_tokens = input_tokens + cache_read_tokens + cache_write_tokens

    if thinking_tokens and total_tokens == input_side_tokens + output_tokens:
        output_tokens = max(output_tokens - thinking_tokens, 0)

    query_usage["input_tokens"] = input_side_tokens
    query_usage["output_tokens"] = output_tokens
    query_usage["thinking_tokens"] = thinking_tokens
    query_usage["cache_read_tokens"] = cache_read_tokens
    query_usage["cache_write_tokens"] = cache_write_tokens
    query_usage["total_tokens"] = total_tokens

    nested_cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
    input_cost = (
        _usage_float(nested_cost, "input")
        + _usage_float(nested_cost, "cacheRead", "cache_read")
        + _usage_float(nested_cost, "cacheWrite", "cache_write")
    )
    output_cost = _usage_float(nested_cost, "output", "completion")
    if input_cost:
        query_usage["input_cost"] = input_cost
    if output_cost:
        query_usage["output_cost"] = output_cost

    return query_usage


def _worktree_completion_content(work_dir: Path) -> str:
    status_text = ""
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            status_text = completed.stdout.strip()
    except FileNotFoundError:
        status_text = ""

    if status_text:
        return (
            "Headless agent completed and modified the worktree directly.\n\n"
            "Git status:\n"
            f"{status_text}"
        )
    return "Headless agent completed. Inspect the worktree for generated changes."


def _extract_message_from_json_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if isinstance(payload, list):
        for item in reversed(payload):
            extracted = _extract_message_from_json_payload(item)
            if extracted:
                return extracted
        return None
    if not isinstance(payload, dict):
        return None

    for key in (
        "final_message",
        "finalMessage",
        "message",
        "content",
        "text",
        "output_text",
        "outputText",
        "result",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            extracted = _extract_message_from_json_payload(value)
            if extracted:
                return extracted
    return None


def _extract_headless_text_content(stdout: str) -> str:
    """Extract the final assistant message from Headless stdout.

    Prefer plain text (default / --usage modes). When --json is used, attempt to
    recover a final message field from the streamed JSON payload.
    """
    text = (stdout or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    content_end = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        trimmed = lines[index].strip()
        if not trimmed:
            content_end = index
            continue
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            break
        if _headless_usage_payload(parsed) is not None:
            content_end = index
            continue
        extracted = _extract_message_from_json_payload(parsed)
        if extracted:
            return extracted
        break

    content = "\n".join(lines[:content_end]).strip()
    if content:
        return content

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _extract_message_from_json_payload(parsed) or text


def _resolve_text_output_mode(configured_mode: str | None) -> str:
    """Prefer modes that emit the final assistant message for text responses."""
    if configured_mode is None:
        return "usage"
    mode = headless_output_mode(configured_mode)
    # --json streams native traces without a guaranteed extracted final message.
    if mode == "json":
        return "usage"
    return mode


def _query_result(
    *,
    content: str,
    usage: dict[str, Any],
    model: str,
    msg: str,
    system_msg: str,
    msg_history: list[dict],
    kwargs: dict[str, Any],
    model_posteriors: dict[str, float] | None,
) -> QueryResult:
    nested_cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
    input_cost = _usage_float(usage, "input_cost", "prompt_cost")
    if input_cost == 0.0:
        input_cost = _usage_float(nested_cost, "input", "prompt")
    output_cost = _usage_float(usage, "output_cost", "completion_cost")
    if output_cost == 0.0:
        output_cost = _usage_float(nested_cost, "output", "completion")
    cost = _usage_float(usage, "cost", "total_cost")
    if cost == 0.0:
        cost = input_cost + output_cost

    return QueryResult(
        msg=msg,
        system_msg=system_msg,
        content=content,
        new_msg_history=[
            *msg_history,
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content},
        ],
        model_name=model,
        kwargs=kwargs,
        input_tokens=_usage_int(usage, "input_tokens", "prompt_tokens", "inputTokens"),
        output_tokens=_usage_int(
            usage, "output_tokens", "completion_tokens", "outputTokens"
        ),
        thinking_tokens=_usage_int(
            usage,
            "thinking_tokens",
            "reasoning_tokens",
            "reasoningOutputTokens",
        ),
        cost=cost,
        input_cost=input_cost,
        output_cost=output_cost,
        model_posteriors=model_posteriors,
        num_total_queries=_usage_int(usage, "num_total_queries", "total_queries") or 1,
    )


def query_headless(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model: BaseModel | None,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    if output_model is not None:
        raise ValueError("Headless does not support structured output.")

    headless_work_dir = kwargs.pop("headless_work_dir", None)
    headless_session_name = kwargs.pop("headless_session", None) or kwargs.pop(
        "headless_session_name", None
    )
    configured_timeout = kwargs.pop("headless_timeout_seconds", None)
    configured_grace = kwargs.pop("headless_cleanup_grace_seconds", None)
    configured_output_mode = kwargs.pop("headless_output_mode", None)
    response_mode = _normalize_response_mode(kwargs.pop("headless_response_mode", None))
    keep_scratch = bool(kwargs.pop("headless_keep_scratch", False))
    parsed_model = parse_headless_model(model)
    if response_mode == "text" and configured_timeout is None:
        configured_timeout = DEFAULT_HEADLESS_TEXT_TIMEOUT
    proposal_timeout = headless_timeout(parsed_model, configured_timeout)
    cleanup_grace = headless_cleanup_grace(configured_grace)
    if response_mode == "text":
        configured_output_mode = _resolve_text_output_mode(configured_output_mode)
    resolved_work_dir, owns_work_dir = _prepare_headless_work_dir(
        headless_work_dir=headless_work_dir,
        response_mode=response_mode,
    )
    prompt_path = _write_prompt_file(
        work_dir=str(resolved_work_dir),
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        response_mode=response_mode,
    )
    command = _build_headless_command(
        model=parsed_model,
        prompt_path=prompt_path,
        work_dir=str(resolved_work_dir),
        session_name=headless_session_name,
        proposal_timeout=proposal_timeout,
        output_mode=configured_output_mode,
    )
    stdout_path, stderr_path = _headless_log_paths(work_dir=resolved_work_dir)

    try:
        if _uses_shell_invocation(parsed_model):
            _thread_cli_lock().acquire()
            lock_acquired = True
        else:
            lock_acquired = False
        try:
            completed = _run_headless_command_sync(
                model=parsed_model,
                command=command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                proposal_timeout=proposal_timeout,
                cleanup_grace=cleanup_grace,
            )
        finally:
            if lock_acquired:
                _THREAD_LOCK.release()

        if completed.returncode != 0:
            failure = _headless_failure(completed=completed)
            failure.artifacts.update(
                {
                    "headless_prompt_path": str(prompt_path),
                    "headless_stdout_path": str(stdout_path),
                    "headless_stderr_path": str(stderr_path),
                }
            )
            raise failure

        headless_usage = _extract_headless_usage(completed.stdout)
        query_usage = _query_usage_from_headless(headless_usage)
        if response_mode == "text":
            content = _extract_headless_text_content(completed.stdout)
            if not content:
                raise LLMExtractionError(
                    "Headless text response was empty; expected a final assistant message."
                )
        else:
            content = _worktree_completion_content(resolved_work_dir)
        result_kwargs = {
            **kwargs,
            "model_name": model,
            "headless_work_dir": str(resolved_work_dir),
            "headless_prompt_path": str(prompt_path),
            "headless_stdout_path": str(stdout_path),
            "headless_stderr_path": str(stderr_path),
            "headless_session_name": headless_session_name,
            "headless_session_id": headless_session_name,
            "headless_usage": headless_usage or None,
            "headless_usage_unknown": not bool(headless_usage),
            "headless_timeout_seconds": proposal_timeout,
            "headless_cleanup_grace_seconds": cleanup_grace,
            "headless_output_mode": headless_output_mode(configured_output_mode),
            "headless_response_mode": response_mode,
        }
        return _query_result(
            content=content,
            usage=query_usage,
            model=model,
            msg=msg,
            system_msg=system_msg,
            msg_history=msg_history,
            kwargs=result_kwargs,
            model_posteriors=model_posteriors,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMTimeoutError(
            f"Headless query timed out after {proposal_timeout}s "
            f"with {cleanup_grace}s cleanup grace.",
            artifacts={
                "headless_prompt_path": str(prompt_path),
                "headless_stdout_path": str(stdout_path),
                "headless_stderr_path": str(stderr_path),
            },
        ) from exc
    finally:
        if owns_work_dir and not keep_scratch:
            shutil.rmtree(resolved_work_dir, ignore_errors=True)


async def query_headless_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model: BaseModel | None,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    if output_model is not None:
        raise ValueError("Headless does not support structured output.")

    headless_work_dir = kwargs.pop("headless_work_dir", None)
    headless_session_name = kwargs.pop("headless_session", None) or kwargs.pop(
        "headless_session_name", None
    )
    configured_timeout = kwargs.pop("headless_timeout_seconds", None)
    configured_grace = kwargs.pop("headless_cleanup_grace_seconds", None)
    configured_output_mode = kwargs.pop("headless_output_mode", None)
    response_mode = _normalize_response_mode(kwargs.pop("headless_response_mode", None))
    keep_scratch = bool(kwargs.pop("headless_keep_scratch", False))
    parsed_model = parse_headless_model(model)
    if response_mode == "text" and configured_timeout is None:
        configured_timeout = DEFAULT_HEADLESS_TEXT_TIMEOUT
    proposal_timeout = headless_timeout(parsed_model, configured_timeout)
    cleanup_grace = headless_cleanup_grace(configured_grace)
    if response_mode == "text":
        configured_output_mode = _resolve_text_output_mode(configured_output_mode)
    resolved_work_dir, owns_work_dir = _prepare_headless_work_dir(
        headless_work_dir=headless_work_dir,
        response_mode=response_mode,
    )
    prompt_path = _write_prompt_file(
        work_dir=str(resolved_work_dir),
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        response_mode=response_mode,
    )
    command = _build_headless_command(
        model=parsed_model,
        prompt_path=prompt_path,
        work_dir=str(resolved_work_dir),
        session_name=headless_session_name,
        proposal_timeout=proposal_timeout,
        output_mode=configured_output_mode,
    )
    stdout_path, stderr_path = _headless_log_paths(work_dir=resolved_work_dir)

    lock_acquired = False
    try:
        if _uses_shell_invocation(parsed_model):
            await _acquire_cli_lock_async()
            lock_acquired = True
        try:
            completed = await _run_headless_command_async(
                model=parsed_model,
                command=command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                proposal_timeout=proposal_timeout,
                cleanup_grace=cleanup_grace,
            )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(
                f"Headless query timed out after {proposal_timeout}s "
                f"with {cleanup_grace}s cleanup grace.",
                artifacts={
                    "headless_prompt_path": str(prompt_path),
                    "headless_stdout_path": str(stdout_path),
                    "headless_stderr_path": str(stderr_path),
                },
            ) from exc
        finally:
            if lock_acquired:
                _THREAD_LOCK.release()
                lock_acquired = False

        if completed.returncode != 0:
            failure = _headless_failure(completed=completed)
            failure.artifacts.update(
                {
                    "headless_prompt_path": str(prompt_path),
                    "headless_stdout_path": str(stdout_path),
                    "headless_stderr_path": str(stderr_path),
                }
            )
            raise failure

        headless_usage = _extract_headless_usage(completed.stdout)
        query_usage = _query_usage_from_headless(headless_usage)
        if response_mode == "text":
            content = _extract_headless_text_content(completed.stdout)
            if not content:
                raise LLMExtractionError(
                    "Headless text response was empty; expected a final assistant message."
                )
        else:
            content = _worktree_completion_content(resolved_work_dir)
        result_kwargs = {
            **kwargs,
            "model_name": model,
            "headless_work_dir": str(resolved_work_dir),
            "headless_prompt_path": str(prompt_path),
            "headless_stdout_path": str(stdout_path),
            "headless_stderr_path": str(stderr_path),
            "headless_session_name": headless_session_name,
            "headless_session_id": headless_session_name,
            "headless_usage": headless_usage or None,
            "headless_usage_unknown": not bool(headless_usage),
            "headless_timeout_seconds": proposal_timeout,
            "headless_cleanup_grace_seconds": cleanup_grace,
            "headless_output_mode": headless_output_mode(configured_output_mode),
            "headless_response_mode": response_mode,
        }
        return _query_result(
            content=content,
            usage=query_usage,
            model=model,
            msg=msg,
            system_msg=system_msg,
            msg_history=msg_history,
            kwargs=result_kwargs,
            model_posteriors=model_posteriors,
        )
    finally:
        if lock_acquired:
            _THREAD_LOCK.release()
        if owns_work_dir and not keep_scratch:
            shutil.rmtree(resolved_work_dir, ignore_errors=True)
