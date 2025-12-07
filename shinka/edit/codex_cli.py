"""Helpers for interacting with the Codex CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from shinka.tools.codex_session_registry import (
    register_session_process,
    remove_session_process,
    update_session_process,
)
from shinka.edit.cost_utils import calculate_cost


class CodexUnavailableError(RuntimeError):
    """Raised when the Codex CLI binary cannot be located."""


class CodexExecutionError(RuntimeError):
    """Raised when a Codex run fails or exceeds configured limits."""


def ensure_codex_available(codex_path: Optional[str] = None) -> Path:
    """Return the resolved path to the Codex CLI binary.

    Args:
        codex_path: Optional override pointing directly to the CLI executable.

    Raises:
        CodexUnavailableError: If the binary cannot be found or executed.

    Returns:
        Path: Absolute path to the Codex CLI binary.
    """

    candidate = codex_path or shutil.which("codex")
    if not candidate:
        raise CodexUnavailableError(
            "Codex CLI not found. Install it with `npm install -g @openai/codex` "
            "or add it to PATH, then authenticate via `codex login`."
        )

    resolved = Path(candidate)
    if not resolved.exists() or not resolved.is_file():
        raise CodexUnavailableError(
            f"Codex CLI binary not found at resolved path: {resolved}"
        )

    return resolved


def _format_extra_config(extra: Dict[str, object]) -> Iterable[str]:
    """Yield CLI `-c key=value` pairs from a dictionary."""

    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, str):
            yield "-c"
            yield f"{key}={value}"
        else:
            yield "-c"
            yield f"{key}={json.dumps(value)}"


def run_codex_task(
    user_prompt: str,
    workdir: Path,
    *,
    system_prompt: Optional[str] = None,
    profile: Optional[str],
    sandbox: str,
    approval_mode: str,
    max_seconds: int,
    max_events: int,
    extra_cli_config: Dict[str, object],
    codex_path: Optional[str] = None,
    cli_path: Optional[str] = None,  # Alias for codex_path
    resume_session_id: Optional[str] = None,
    session_kind: str = "unknown",
    # Metadata params (unused but accepted for API compat with agentic.py)
    parent_id: Optional[str] = None,
    generation: Optional[int] = None,
    patch_type: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    """Execute a Codex CLI task and stream its JSON events.

    Args:
        user_prompt: Natural language instruction for Codex.
        workdir: Workspace directory Codex should modify.
        system_prompt: Optional system instructions (prepended to prompt).
        profile: Optional Codex profile name (selects model/settings).
        sandbox: Sandbox policy passed to `--sandbox`.
        approval_mode: Either `full-auto` or values accepted by
            `--ask-for-approval`.
        max_seconds: Wall-clock guardrail for the Codex process.
        max_events: Maximum number of JSON events to yield before aborting.
        extra_cli_config: Additional key/value overrides forwarded via `-c`.
        codex_path: Optional explicit path to the CLI binary.
        cli_path: Alias for codex_path (for backend-agnostic calls).
        resume_session_id: Optional session UUID to resume via
            `codex exec resume`.

    Raises:
        CodexExecutionError: If Codex fails, times out, or exceeds limits.
        CodexUnavailableError: If the CLI binary cannot be located.

    Yields:
        Parsed JSON events emitted by the CLI.
    """

    # Use cli_path if provided, fall back to codex_path for backward compat
    binary = ensure_codex_available(cli_path or codex_path)

    cmd = [str(binary), "exec"]
    if resume_session_id:
        cmd.append("resume")
    cmd.extend(["--json", "--skip-git-repo-check", "-C", str(workdir)])

    if profile:
        cmd.extend(["--profile", profile])

    if sandbox:
        cmd.extend(["--sandbox", sandbox])

    if approval_mode == "full-auto":
        cmd.append("--full-auto")
    elif approval_mode:
        cmd.extend(["--ask-for-approval", approval_mode])

    cmd.extend(_format_extra_config(extra_cli_config))

    if resume_session_id:
        cmd.append(resume_session_id)

    # NOTE: Codex CLI does not support a separate system prompt flag.
    # In agentic mode, the harness owns the system prompt entirely - task-specific
    # context (task_sys_msg) is included in the user prompt by the sampler.
    # The system_prompt param here contains only operational instructions (AGENTIC_SYS_FORMAT)
    # which we prepend to the user prompt since Codex has no system prompt mechanism.
    full_prompt = user_prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

    cmd.append(full_prompt)

    start_time = time.monotonic()
    events_emitted = 0

    # Token estimation for cost tracking (Codex CLI doesn't emit usage data)
    estimated_input_tokens = len(full_prompt) // 4 if full_prompt else 0
    estimated_output_tokens = 0
    model_name = profile or "gpt-4.1-mini"  # Default Codex model (in pricing.py)
    session_id: Optional[str] = None

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    prompt_preview = full_prompt.strip().splitlines()[0][:160] if full_prompt else ""
    register_session_process(
        process.pid,
        prompt_preview=prompt_preview,
        workdir=workdir,
        session_kind=session_kind,
        parent_id=parent_id,
        generation=generation,
        patch_type=patch_type,
        results_dir=results_dir,
    )

    try:
        if not process.stdout:
            raise CodexExecutionError("Codex CLI did not provide stdout pipe.")

        while True:
            if max_seconds > 0 and time.monotonic() - start_time > max_seconds:
                process.kill()
                raise CodexExecutionError(
                    f"Codex task exceeded {max_seconds}s timeout."
                )

            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise CodexExecutionError(
                    f"Failed to parse Codex event: {line}"
                ) from exc

            events_emitted += 1
            if max_events and events_emitted > max_events:
                process.kill()
                raise CodexExecutionError(
                    "Codex emitted more events than allowed (max_events)."
                )

            if isinstance(event, dict):
                extracted_sid = _extract_session_id(event)
                if extracted_sid:
                    session_id = extracted_sid
                    update_session_process(process.pid, session_id=extracted_sid)

                # Track output content for token estimation
                content = (
                    event.get("content")
                    or event.get("text")
                    or ""
                )
                # Also check nested message content
                msg = event.get("message")
                if isinstance(msg, dict):
                    msg_content = msg.get("content")
                    if isinstance(msg_content, str):
                        content = msg_content
                    elif isinstance(msg_content, list):
                        # Handle content blocks
                        for block in msg_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                content += block.get("text", "")

                if isinstance(content, str) and content:
                    estimated_output_tokens += len(content) // 4

            yield event

        # Emit usage event at session end
        total_tokens = estimated_input_tokens + estimated_output_tokens
        yield {
            "type": "usage",
            "session_id": session_id,
            "usage": {
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": calculate_cost(
                    model_name,
                    estimated_input_tokens,
                    estimated_output_tokens,
                    "codex",
                ),
            },
            "model": model_name,
        }

        returncode = process.wait(timeout=1)
        if returncode != 0:
            stderr_out = process.stderr.read() if process.stderr else ""
            raise CodexExecutionError(
                f"Codex CLI exited with status {returncode}: {stderr_out.strip()}"
            )
    finally:
        if process.poll() is None:
            process.kill()
        remove_session_process(process.pid)


def _extract_session_id(event: Dict[str, object]) -> Optional[str]:
    """Attempt to pull a session/thread id from a Codex CLI event."""

    if not isinstance(event, dict):
        return None
    event_type = event.get("type")
    if isinstance(event_type, str) and event_type.startswith("thread."):
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    session_id = event.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    session_obj = event.get("session")
    if isinstance(session_obj, dict):
        candidate = session_obj.get("id") or session_obj.get("session_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None
