"""Native ShinkaAgent backend using shinka/llm/LLMClient.

This module implements a native, model-agnostic agentic editing backend
that uses Shinka's existing LLM infrastructure. Unlike the CLI wrappers
(Codex, Gemini, Claude), ShinkaAgent runs entirely in-process, providing
full control over the agent loop and leveraging existing LLM ensembling.

The design follows the mini-SWE-agent pattern:
- Single bash action per response (enforced via regex)
- Linear message history (no branching)
- subprocess.run() for action execution (stateless)
- Termination via magic output string

Reference: https://github.com/SWE-agent/mini-swe-agent
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from shinka.llm import LLMClient

logger = logging.getLogger(__name__)


class ShinkaUnavailableError(RuntimeError):
    """Raised when no LLM API keys are configured."""


class ShinkaExecutionError(RuntimeError):
    """Raised when the agent loop fails or times out."""


# Regex to extract bash code block (trailing newline optional for robustness)
ACTION_RE = re.compile(r"```bash\s*\n(.*?)(?:\n)?```", re.DOTALL)

# System prompt for bash-only agent
SHINKA_SYSTEM_PROMPT = """You are an expert software engineer working inside a sandboxed repository.

IMPORTANT RULES:
1. You can ONLY interact via bash commands in ```bash...``` blocks
2. You can include multiple bash blocks per response - all will be executed in order
3. Only edit code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers
4. Use standard tools: cat, sed, echo, python, etc.
5. Keep responses concise - avoid lengthy explanations

When your task is complete, include this exact text in your response:
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Example response:
I'll read the current file first.
```bash
cat main.py
```

After seeing the output, make targeted edits to improve the score.
"""

# Observation template
OBSERVATION_TEMPLATE = """OBSERVATION:
Exit code: {exit_code}
{output}"""

# Max characters for observation to avoid context overflow
MAX_OBSERVATION_CHARS = 16000

def ensure_shinka_available() -> bool:
    """Check that at least one LLM provider API key is configured."""
    from shinka.tools.credentials import PROVIDER_ENV_VAR_MAP, get_api_key

    # Check environment variables
    for env_var in set(PROVIDER_ENV_VAR_MAP.values()):
        if os.environ.get(env_var):
            return True

    # Check credential store
    for provider, env_var in PROVIDER_ENV_VAR_MAP.items():
        if key := get_api_key(provider):
            os.environ[env_var] = key
            return True

    raise ShinkaUnavailableError(
        "No LLM API keys found. Set one of: " + ", ".join(set(PROVIDER_ENV_VAR_MAP.values()))
    )


def _truncate_output(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """Truncate output to avoid context overflow."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n... [truncated {len(text) - max_chars} chars] ...\n{text[-half:]}"


def _execute_bash(command: str, cwd: Path, timeout: int = 120) -> tuple[int, str, str]:
    """Execute a bash command and return (exit_code, stdout, stderr)."""
    # Skip empty commands
    if not command.strip():
        return 0, "", "(empty command skipped)"

    # Validate workdir exists and is directory
    if not cwd.exists() or not cwd.is_dir():
        return 1, "", f"Invalid working directory: {cwd}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return 1, "", str(e)


def run_shinka_task(
    user_prompt: str,
    workdir: Path,
    *,
    system_prompt: Optional[str] = None,
    profile: Optional[str],
    sandbox: str,
    approval_mode: str,
    max_seconds: int,
    max_events: int,
    extra_cli_config: Dict[str, Any],
    codex_path: Optional[str] = None,
    cli_path: Optional[str] = None,  # Alias for codex_path (unused for ShinkaAgent)
    resume_session_id: Optional[str] = None,
    session_kind: str = "unknown",
    # Metadata params for session registry tracking
    parent_id: Optional[str] = None,
    generation: Optional[int] = None,
    patch_type: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Execute a ShinkaAgent task and stream JSON events.

    This function implements the AgentRunner protocol for native in-process
    agent execution using shinka/llm/LLMClient.

    Args:
        user_prompt: Natural language instruction for the agent.
        workdir: Workspace directory the agent should modify.
        system_prompt: Optional system instructions (combined with base prompt).
        profile: Optional model name override.
        sandbox: Sandbox policy (ignored for ShinkaAgent - runs locally).
        approval_mode: Approval mode (ignored for ShinkaAgent - full-auto).
        max_seconds: Wall-clock timeout for the session.
        max_events: Maximum number of LLM turns before stopping.
        extra_cli_config: Additional config (model, temperature, etc.).
        codex_path: Ignored for ShinkaAgent.
        resume_session_id: Optional session UUID to resume (future feature).
        session_kind: Session type label for UI tracking.

    Yields:
        Parsed JSON events in the same format as CLI wrappers:
        - init: Session start with session_id, model, timestamp
        - agent_message: LLM response text
        - command_execution: Bash command result
        - usage: Token/cost telemetry at session end

    Raises:
        ShinkaUnavailableError: If no API keys are configured.
        ShinkaExecutionError: If the agent loop fails catastrophically.
    """
    ensure_shinka_available()

    session_id = resume_session_id or str(uuid.uuid4())
    start_time = time.monotonic()

    # Determine model(s) to use
    # Priority: extra_cli_config["model"] > profile > FAIL
    # We intentionally fail instead of silently falling back to an old model
    model_name = extra_cli_config.get("model") or profile
    if not model_name:
        raise ShinkaExecutionError(
            "No model configured for ShinkaAgent. "
            "Set evo_config.agentic.extra_cli_config.model or evo_config.agentic.cli_profile. "
            "Example: evo_config.agentic.extra_cli_config.model=gpt-4.1"
        )
    model_names = [model_name] if isinstance(model_name, str) else list(model_name)

    # Extract LLM kwargs from extra_cli_config with proper key mapping
    # LLMClient uses 'temperatures' (plural) but config often has 'temperature'
    llm_kwargs = {}
    if "temperature" in extra_cli_config:
        llm_kwargs["temperatures"] = extra_cli_config["temperature"]
    if "max_tokens" in extra_cli_config:
        llm_kwargs["max_tokens"] = extra_cli_config["max_tokens"]
    # IMPORTANT: reasoning_efforts controls thinking tokens for reasoning models
    # Without this, Gemini and other reasoning models may return empty responses
    # Default to "auto" (no thinking) for agentic mode to avoid response format issues
    if "reasoning_efforts" in extra_cli_config:
        llm_kwargs["reasoning_efforts"] = extra_cli_config["reasoning_efforts"]
    else:
        # Explicitly set to "auto" to disable thinking tokens in agentic mode
        # This avoids Gemini returning empty/None content due to thinking mode
        llm_kwargs["reasoning_efforts"] = "auto"

    # Initialize LLMClient with configured models
    llm = LLMClient(model_names=model_names, verbose=False, **llm_kwargs)

    # NOTE: ShinkaAgent has its own SHINKA_SYSTEM_PROMPT that defines how the
    # agent operates (bash-only, one block per response, etc.). In agentic mode,
    # task-specific context (task_sys_msg) is included in the user prompt by the
    # sampler. The system_prompt param here contains only operational instructions
    # (AGENTIC_SYS_FORMAT) which we prepend to our SHINKA_SYSTEM_PROMPT.
    base_system = SHINKA_SYSTEM_PROMPT
    if system_prompt:
        base_system = f"{system_prompt}\n\n{SHINKA_SYSTEM_PROMPT}"

    # Message history for multi-turn conversation
    messages: List[Dict[str, str]] = []

    # Cost tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    # Emit init event
    yield {
        "type": "init",
        "session_id": session_id,
        "model": model_names[0],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    # Add initial user message
    current_msg = user_prompt
    turn_count = 0

    while True:
        # Check time limit
        elapsed = time.monotonic() - start_time
        if max_seconds > 0 and elapsed > max_seconds:
            yield {
                "type": "agent_message",
                "item": {
                    "type": "agent_message",
                    "text": f"[Session timed out after {elapsed:.1f}s]",
                },
                "session_id": session_id,
            }
            break

        # Check turn limit
        turn_count += 1
        if max_events > 0 and turn_count > max_events:
            yield {
                "type": "agent_message",
                "item": {
                    "type": "agent_message",
                    "text": f"[Session reached max turns: {max_events}]",
                },
                "session_id": session_id,
            }
            break

        # Query LLM
        llm_call_kwargs = llm.get_kwargs()
        response = llm.query(
            msg=current_msg,
            system_msg=base_system,
            msg_history=messages,
            llm_kwargs=llm_call_kwargs,
        )

        if response is None or response.content is None:
            yield {
                "type": "agent_message",
                "item": {
                    "type": "agent_message",
                    "text": "[LLM returned empty response]",
                },
                "session_id": session_id,
            }
            break

        # Track costs using actual values from QueryResult
        total_cost += response.cost or 0.0
        total_input_tokens += response.input_tokens or 0
        total_output_tokens += response.output_tokens or 0

        # Update message history
        messages.append({"role": "user", "content": current_msg})
        messages.append({"role": "assistant", "content": response.content})

        # Emit agent message event
        yield {
            "type": "agent_message",
            "item": {"type": "agent_message", "text": response.content},
            "session_id": session_id,
        }

        # Parse ALL bash actions - execute all commands before checking termination
        # (Some models output multiple bash blocks in one response)
        action_matches = list(ACTION_RE.finditer(response.content))
        has_termination = (
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response.content
        )

        # Execute ALL bash blocks in sequence
        observations = []
        for action_match in action_matches:
            command = action_match.group(1).strip()

            # Execute command
            exit_code, stdout, stderr = _execute_bash(command, workdir)

            # Format observation
            output = stdout + stderr
            output = _truncate_output(output)
            observation = OBSERVATION_TEMPLATE.format(
                exit_code=exit_code,
                output=output or "(no output)",
            )
            observations.append(observation)

            # Emit command execution event
            yield {
                "type": "command_execution",
                "item": {
                    "type": "command_execution",
                    "command": command,
                    "status": "success" if exit_code == 0 else "error",
                    "exit_code": exit_code,
                    "stdout": _truncate_output(stdout, 8000),
                    "stderr": _truncate_output(stderr, 8000),
                },
                "session_id": session_id,
            }

        # Combine all observations for next message
        if observations:
            current_msg = "\n\n".join(observations)

        # Check for termination AFTER executing any bash commands
        if has_termination:
            logger.info(
                f"ShinkaAgent completed task in {turn_count} turns, "
                f"{elapsed:.1f}s, cost=${total_cost:.4f}"
            )
            break

        # If no bash action and no termination, prompt for one
        if not action_matches:
            current_msg = (
                "Please provide a bash command in ```bash...``` block, "
                "or say COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT if done."
            )

    # Emit usage event at end
    yield {
        "type": "usage",
        "session_id": session_id,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost_usd": total_cost,
        },
    }
