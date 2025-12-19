"""Agentic editing harness with a pluggable backend (Codex default)."""

from __future__ import annotations

import base64
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .codex_cli import run_codex_task
from .event_utils import extract_session_id
from .types import AgentRunner

logger = logging.getLogger(__name__)

MAX_BASE_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_BINARY_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILES_TO_SCAN = 10_000


@dataclass
class CommandResult:
    """Represents a command execution issued by the agent."""

    command: Optional[str]
    status: Optional[str]
    exit_code: Optional[int]
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class AgentResult:
    """Container for the outcome of an agentic editing session."""

    changed_files: Dict[Path, str]
    session_log: List[str]
    commands_run: List[CommandResult]
    final_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    session_log_path: Optional[Path] = None
    session_events: List[Dict[str, Any]] = field(default_factory=list)
    binary_changed_files: Dict[Path, str] = field(default_factory=dict)
    session_id: Optional[str] = None
    model: Optional[str] = None  # Actual model from CLI init event


@dataclass
class AgentContext:
    """Inputs required to run an agentic editing session.

    Note on system_prompt: In agentic mode, the harness (Codex/Gemini/Claude CLI)
    owns the system prompt. This field contains only AGENTIC_SYS_FORMAT (operational
    instructions for sandbox editing), NOT task-specific context. Task context
    (task_sys_msg from config) is included in the user_prompt as "# Task Context".
    This ensures we don't override the CLI's native system behavior.
    """

    user_prompt: str
    language: str
    base_files: Dict[Path, str]
    primary_file: Path
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resume_session_id: Optional[str] = None


class AgenticEditor:
    """Drive an agentic editing session within a dedicated scratch directory.

    Backend is selected by the caller (Codex/Gemini/Claude/ShinkaAgent); Codex
    is only the default runner, not a requirement.
    """

    def __init__(
        self,
        scratch_dir: Path,
        config,
        *,
        runner: AgentRunner = run_codex_task,
        codex_runner: AgentRunner | None = None,  # Deprecated: use runner
    ) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.config = config
        # Accept the legacy codex_runner keyword for backward compatibility
        self.runner = runner if codex_runner is None else codex_runner

    def _prepare_scratch(self, base_files: Dict[Path, str]) -> Dict[Path, str]:
        # Preserve session_meta.json if it exists (written by runner.py for visualization)
        meta_path = self.scratch_dir / "session_meta.json"
        preserved_meta = None
        if meta_path.exists():
            try:
                preserved_meta = meta_path.read_text(encoding="utf-8")
            except Exception:
                pass

        scratch_resolved = self.scratch_dir.resolve()

        if self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Restore session_meta.json
        if preserved_meta is not None:
            try:
                meta_path.write_text(preserved_meta, encoding="utf-8")
            except Exception:
                pass

        baseline: Dict[Path, str] = {}
        for relative_path, content in base_files.items():
            if relative_path.is_absolute():
                raise ValueError("Base file paths must be relative to the scratch root")
            target = self.scratch_dir / relative_path
            try:
                if not target.resolve().is_relative_to(scratch_resolved):
                    raise ValueError(
                        f"Base file path '{relative_path}' escapes scratch directory"
                    )
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"Invalid base file path '{relative_path}': {e}"
                ) from e

            content_bytes = len(content.encode("utf-8"))
            if content_bytes > MAX_BASE_FILE_SIZE:
                raise ValueError(
                    f"Base file {relative_path} exceeds max size "
                    f"({content_bytes} > {MAX_BASE_FILE_SIZE} bytes)"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            baseline[relative_path] = content
        return baseline

    def run_session(self, context: AgentContext) -> AgentResult:
        baseline = self._prepare_scratch(context.base_files)

        session_log: List[str] = []
        commands: List[CommandResult] = []
        start_time = time.monotonic()

        session_log_path = self.scratch_dir / "session_log.jsonl"
        event_count = 0
        session_events: List[Dict[str, Any]] = []
        binary_changed_files: Dict[Path, str] = {}
        session_id: Optional[str] = None
        model_from_event: Optional[str] = None  # Actual model from CLI init event

        # Telemetry aggregation
        usage_metrics: Dict[str, float] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

        with session_log_path.open("w", encoding="utf-8") as event_handle:
            for event in self.runner(
                user_prompt=context.user_prompt,
                system_prompt=context.system_prompt,
                workdir=self.scratch_dir,
                profile=self.config.cli_profile,
                sandbox=self.config.sandbox,
                approval_mode=self.config.approval_mode,
                max_seconds=self.config.max_seconds,
                max_events=self.config.max_events,
                extra_cli_config=self.config.extra_cli_config,
                cli_path=self.config.cli_path,
                resume_session_id=context.resume_session_id,
                session_kind="edit",
                parent_id=context.metadata.get("parent_id"),
                generation=context.metadata.get("generation"),
                patch_type=context.metadata.get("patch_type"),
                results_dir=context.metadata.get("results_dir"),
            ):
                if isinstance(event, dict):
                    json.dump(event, event_handle)
                    event_handle.write("\n")
                    event_count += 1
                    session_events.append(event)
                    if session_id is None:
                        candidate = extract_session_id(event)
                        if candidate:
                            session_id = candidate

                # Handle standard event types
                item = event.get("item") if isinstance(event, dict) else None
                if item:
                    item_type = item.get("type")
                    if item_type == "agent_message":
                        text = item.get("text")
                        if text:
                            session_log.append(text)
                    elif item_type == "command_execution":
                        commands.append(
                            CommandResult(
                                command=item.get("command"),
                                status=item.get("status"),
                                exit_code=item.get("exit_code"),
                                stdout=item.get("stdout"),
                                stderr=item.get("stderr"),
                            )
                        )

                # Handle direct event types
                event_type = event.get("type")

                # Capture model from init event (Claude CLI and ShinkaAgent emit this)
                if event_type == "init" and model_from_event is None:
                    model_candidate = event.get("model")
                    if isinstance(model_candidate, str) and model_candidate:
                        model_from_event = model_candidate

                if event_type == "usage":
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        usage_metrics["input_tokens"] += float(
                            usage.get("input_tokens", 0)
                        )
                        usage_metrics["output_tokens"] += float(
                            usage.get("output_tokens", 0)
                        )
                        usage_metrics["total_tokens"] += float(
                            usage.get("total_tokens", 0)
                        )
                        # Use real cost from Claude CLI if available
                        if "total_cost_usd" in usage:
                            usage_metrics["total_cost_usd"] += float(
                                usage.get("total_cost_usd", 0.0)
                            )

        elapsed = time.monotonic() - start_time

        changed_files: Dict[Path, str] = {}
        files_checked = 0
        scratch_resolved = self.scratch_dir.resolve()

        for file_path in self.scratch_dir.rglob("*"):
            # Prevent unbounded scans in pathological scratch trees.
            if files_checked >= MAX_FILES_TO_SCAN:
                break

            if not file_path.is_file():
                continue

            # Avoid following symlinks/paths that escape the sandbox.
            try:
                if not file_path.resolve().is_relative_to(scratch_resolved):
                    continue
            except (OSError, ValueError):
                continue

            rel_path = file_path.relative_to(self.scratch_dir)

            # Skip internal session files - they shouldn't be part of the program
            if str(rel_path) in ("session_log.jsonl", "session_meta.json"):
                continue

            files_checked += 1
            try:
                new_content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    if file_path.stat().st_size > MAX_BINARY_FILE_SIZE:
                        continue
                except OSError:
                    continue
                raw_bytes = file_path.read_bytes()
                binary_changed_files[rel_path] = base64.b64encode(raw_bytes).decode(
                    "ascii"
                )
                continue

            baseline_content = baseline.get(rel_path)
            if baseline_content is None:
                # New file created
                changed_files[rel_path] = new_content
            elif baseline_content != new_content:
                # Existing file modified
                changed_files[rel_path] = new_content

        if not changed_files and files_checked > 0:
            logger.info(
                "Agentic session completed but no files changed. "
                f"Checked {files_checked} files in {self.scratch_dir}. "
                f"Baseline files: {len(baseline)}"
            )
        elif changed_files:
            logger.info(
                f"Agentic session changed {len(changed_files)} files: {[str(p) for p in changed_files.keys()]}"
            )

        # Use real cost if available (Claude CLI provides total_cost_usd),
        # otherwise fallback to token-based placeholder estimate
        real_cost = usage_metrics.get("total_cost_usd", 0.0)
        fallback_cost = usage_metrics["total_tokens"] / 1000.0  # rough placeholder
        final_cost = real_cost if real_cost > 0 else fallback_cost

        metrics = {
            "elapsed_seconds": elapsed,
            "commands_run": float(len(commands)),
            "messages_logged": float(len(session_log)),
            "events_logged": float(event_count),
            "estimated_input_tokens": usage_metrics["input_tokens"],
            "estimated_output_tokens": usage_metrics["output_tokens"],
            "estimated_total_tokens": usage_metrics["total_tokens"],
            "estimated_total_cost": final_cost,
            "total_cost": final_cost,
            "input_tokens": usage_metrics["input_tokens"],
            "output_tokens": usage_metrics["output_tokens"],
            "total_tokens": usage_metrics["total_tokens"],
            "real_cost_available": real_cost > 0,
        }

        final_message = session_log[-1] if session_log else None

        return AgentResult(
            changed_files=changed_files,
            binary_changed_files=binary_changed_files,
            session_log=session_log,
            commands_run=commands,
            final_message=final_message,
            metrics=metrics,
            session_log_path=session_log_path,
            session_events=session_events,
            session_id=session_id,
            model=model_from_event,
        )
