"""Helpers for interacting with the Codex CLI."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, Iterable, Iterator, Literal, Optional

from shinka.edit.cost_utils import calculate_cost
from shinka.edit.event_utils import extract_session_id
from shinka.tools.codex_session_registry import (
    register_session_process,
    remove_session_process,
    update_session_process,
)
from shinka.tools.credentials import get_api_key


class CodexUnavailableError(RuntimeError):
    """Raised when the Codex CLI binary cannot be located."""


class CodexExecutionError(RuntimeError):
    """Raised when a Codex run fails or exceeds configured limits."""


class CodexAuthError(RuntimeError):
    """Raised when Codex authentication cannot be established."""


def _is_interactive() -> bool:
    """Check if running in interactive context (avoid hanging in CI/background)."""
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _status_looks_authenticated(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    if "not logged" in combined:
        return False
    if "unauthorized" in combined:
        return False
    if "please login" in combined or "please log in" in combined:
        return False
    return True


def _is_codex_authenticated(codex_bin: Path) -> bool:
    """Return True if Codex CLI reports an authenticated session."""
    try:
        result = subprocess.run(
            [str(codex_bin), "login", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return _status_looks_authenticated(result.stdout or "", result.stderr or "")


def _login_with_api_key(codex_bin: Path, api_key: str, *, timeout_seconds: int) -> bool:
    """Attempt a non-interactive login using an API key via stdin."""
    try:
        result = subprocess.run(
            [str(codex_bin), "login", "--with-api-key"],
            input=f"{api_key}\n",
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _login_device_auth(codex_bin: Path, *, timeout_seconds: int) -> bool:
    """Attempt a device auth login, inheriting stdio so the user sees the code."""
    try:
        result = subprocess.run(
            [str(codex_bin), "login", "--device-auth"],
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _ensure_codex_authenticated(
    codex_bin: Path,
    *,
    api_key: Optional[str] = None,
    timeout_seconds: int = 900,
    allow_interactive: Optional[bool] = None,
) -> Literal["status", "device_auth", "api_key"]:
    """Ensure Codex is authenticated, attempting login flows if needed.

    Order of operations:
    1) `codex login status` (fast path)
    2) If not logged in and interactive, attempt `codex login --device-auth`
    3) If still not logged in and api_key provided, attempt `codex login --with-api-key`

    Raises:
        CodexAuthError: If authentication is not available after attempts.
    """
    if _is_codex_authenticated(codex_bin):
        return "status"

    interactive = _is_interactive() if allow_interactive is None else allow_interactive
    if interactive:
        if _login_device_auth(codex_bin, timeout_seconds=timeout_seconds):
            if _is_codex_authenticated(codex_bin):
                return "device_auth"

    if api_key:
        if _login_with_api_key(codex_bin, api_key, timeout_seconds=timeout_seconds):
            if _is_codex_authenticated(codex_bin):
                return "api_key"

    raise CodexAuthError(
        "Codex authentication required. Options:\n"
        "  1. Run `codex login --device-auth` (requires enabling device code auth in ChatGPT Security Settings first)\n"
        "  2. Run `echo $OPENAI_API_KEY | codex login --with-api-key`\n"
        "  3. Set OPENAI_API_KEY environment variable or add to ~/.shinka/credentials.json"
    )


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
            "or add it to PATH, then authenticate via `codex login --device-auth` "
            "(requires enabling device code auth in ChatGPT Security Settings) "
            "or `codex login --with-api-key`."
        )

    resolved = Path(candidate)
    if not resolved.exists() or not resolved.is_file():
        raise CodexUnavailableError(
            f"Codex CLI binary not found at resolved path: {resolved}"
        )

    return resolved


def validate_codex_setup(codex_path: Optional[str] = None) -> None:
    """Validate Codex CLI is installed and authenticated at startup.

    This should be called early (e.g., in EvolutionRunner.__init__) to fail fast
    before evolution starts, rather than failing mid-evolution on the first edit.

    Args:
        codex_path: Optional override pointing directly to the CLI executable.

    Raises:
        CodexUnavailableError: If Codex CLI is not installed.
        CodexAuthError: If Codex CLI is not authenticated.
    """
    # Check binary is available
    codex_bin = ensure_codex_available(codex_path)

    # Check authentication status (without triggering interactive login)
    if not _is_codex_authenticated(codex_bin):
        raise CodexAuthError(
            "Codex CLI is not authenticated. Please run:\n\n"
            "  $ codex login\n\n"
            "This will open your browser for OAuth authentication.\n"
            "After authenticating, verify with: codex login status"
        )


def _to_primitive(obj: object) -> object:
    """Convert OmegaConf DictConfig/ListConfig to primitive Python types."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    return obj


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
            yield f"{key}={json.dumps(_to_primitive(value))}"


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

    # Authentication: prefer an existing Codex CLI login (e.g. ChatGPT subscription),
    # and only fall back to API key auth when no interactive login is available.
    api_key = get_api_key("codex")
    try:
        auth_method = _ensure_codex_authenticated(binary, api_key=api_key)
    except CodexAuthError as exc:
        raise CodexExecutionError(str(exc)) from exc

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

    # Prevent the prompt from being interpreted as extra CLI options when it begins
    # with '-' / '--' (e.g. "--sandbox host") by terminating option parsing.
    cmd.append("--")
    cmd.append(full_prompt)

    start_time = time.monotonic()
    events_emitted = 0

    # Token estimation for cost tracking (Codex CLI doesn't emit usage data)
    estimated_input_tokens = len(full_prompt) // 4 if full_prompt else 0
    estimated_output_tokens = 0
    # Model priority: extra_cli_config["model"] > profile > FAIL
    # We intentionally fail instead of silently falling back to an old model
    model_name = extra_cli_config.get("model") or profile
    if not model_name:
        raise CodexExecutionError(
            "No model configured for Codex CLI. "
            "Set evo_config.agentic.extra_cli_config.model or evo_config.agentic.cli_profile. "
            "Example: evo_config.agentic.extra_cli_config.model=gpt-4.1"
        )
    session_id: Optional[str] = None

    env = dict(os.environ)
    if auth_method == "api_key" and api_key:
        env["OPENAI_API_KEY"] = api_key

    process = subprocess.Popen(
        cmd,
        env=env,
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
                # Don't kill immediately - let this event finish and break gracefully
                logger.warning(
                    f"Codex emitted {events_emitted} events (max: {max_events}) - "
                    "stopping gracefully with results collected so far"
                )
                process.kill()
                break  # Exit loop gracefully instead of raising error

            if isinstance(event, dict):
                extracted_sid = extract_session_id(event)
                if extracted_sid:
                    session_id = extracted_sid
                    update_session_process(process.pid, session_id=extracted_sid)

                # Track output content for token estimation
                content = event.get("content") or event.get("text") or ""
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
            # Don't fail if we have actual results (events processed)
            # Exit code 1 can happen for benign reasons (e.g., hit max_turns)
            if events_emitted > 0:
                logger.warning(
                    f"Codex CLI exited with status {returncode} but produced "
                    f"{events_emitted} events - continuing with results"
                )
            else:
                raise CodexExecutionError(
                    f"Codex CLI exited with status {returncode}: {stderr_out.strip()}"
                )
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        remove_session_process(process.pid)
