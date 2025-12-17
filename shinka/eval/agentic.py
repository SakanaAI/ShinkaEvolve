"""Agentic evaluator that uses LLM to assess code and write metrics.

The evaluator can:
1. Run an evaluation command and parse the output
2. Write metrics.json itself with qualitative judgment
3. Use custom evaluation criteria (eval_prompt) for domain-specific assessment
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from shinka.edit.agentic import CommandResult
from shinka.edit.codex_cli import CodexExecutionError, run_codex_task
from shinka.edit.event_utils import extract_session_id
from shinka.edit.types import AgentRunner
from shinka.prompts import AGENTIC_EVAL_SYS, AGENTIC_EVAL_USER

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from shinka.core.runner import AgenticEvaluatorConfig


@dataclass
class AgenticEvaluatorResult:
    """Structured output from an agentic evaluation session."""

    metrics: Dict[str, Any]
    correct: bool
    error_message: Optional[str]
    stdout_log: str
    stderr_log: str
    session_log: List[str]
    commands_run: List[CommandResult]
    session_log_path: Path
    session_events: List[Dict[str, Any]]
    session_id: Optional[str]
    session_dir: Path
    elapsed_seconds: float
    # Prompts used for evaluation (for debugging/UI display)
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None


class AgenticEvaluator:
    """Drive the Codex-based evaluator from the repository root."""

    def __init__(
        self,
        config: "AgenticEvaluatorConfig",
        *,
        codex_runner: AgentRunner = None,
        agent_runner: AgentRunner = None,  # Alias for codex_runner
    ) -> None:
        self.config = config
        # Accept either codex_runner or agent_runner for backward compatibility
        self.codex_runner = codex_runner or agent_runner or run_codex_task

    def evaluate(
        self,
        *,
        repo_root: Path,
        eval_command: Sequence[str],
        program_path: Path,
        results_path: Path,
        metrics_path: Path,
        eval_sessions_root: Path,
        task_name: str,
        results_dir: Optional[str] = None,
        eval_prompt: Optional[str] = None,
        max_score: float = 100.0,
    ) -> AgenticEvaluatorResult:
        session_uuid = uuid.uuid4().hex
        session_dir = eval_sessions_root / session_uuid
        session_dir.mkdir(parents=True, exist_ok=True)
        session_log_path = session_dir / "session_log.jsonl"

        user_prompt, system_prompt = self._build_prompt(
            task_name=task_name,
            eval_command=eval_command,
            program_path=program_path,
            results_path=results_path,
            metrics_path=metrics_path,
            eval_prompt=eval_prompt,
            max_score=max_score,
        )

        session_log: List[str] = []
        commands: List[CommandResult] = []
        session_events: List[Dict[str, Any]] = []
        resolved_session_id: Optional[str] = None

        start_time = time.monotonic()
        with session_log_path.open("w", encoding="utf-8") as handle:
            for event in self.codex_runner(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                workdir=repo_root,
                profile=self.config.cli_profile,
                sandbox=self.config.sandbox,
                approval_mode=self.config.approval_mode,
                max_seconds=self.config.max_seconds,
                max_events=self.config.max_events,
                extra_cli_config=self.config.extra_cli_config,
                cli_path=self.config.cli_path,
                session_kind="eval",
                results_dir=results_dir,
            ):
                if isinstance(event, dict):
                    json.dump(event, handle)
                    handle.write("\n")
                    handle.flush()  # Flush for real-time visibility
                    session_events.append(event)
                    if resolved_session_id is None:
                        resolved_session_id = extract_session_id(event)

                item = event.get("item") if isinstance(event, dict) else None
                if not item:
                    continue
                if item.get("type") == "agent_message":
                    text = item.get("text")
                    if text:
                        session_log.append(text)
                elif item.get("type") == "command_execution":
                    commands.append(
                        CommandResult(
                            command=item.get("command"),
                            status=item.get("status"),
                            exit_code=item.get("exit_code"),
                            stdout=item.get("stdout"),
                            stderr=item.get("stderr"),
                        )
                    )
        elapsed = time.monotonic() - start_time

        # Convert relative metrics_path to absolute path for checking
        # (metrics_path is relative to repo_root, not the current working directory)
        metrics_absolute = repo_root / metrics_path if not metrics_path.is_absolute() else metrics_path

        if not metrics_absolute.exists():
            raise CodexExecutionError(
                f"Agentic evaluator did not produce metrics at {metrics_path}"
            )

        # Parse metrics with error handling for malformed JSON
        try:
            metrics = json.loads(metrics_absolute.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metrics.json: {e}")
            metrics = {"error": f"Invalid JSON in metrics: {e}", "combined_score": 0}

        # Read 'correct' from metrics.json (consolidated schema)
        # Fall back to correct.json for backward compatibility
        if "correct" in metrics:
            correct_flag = bool(metrics.get("correct", False))
            error_msg = metrics.get("details") if not correct_flag else None
        else:
            # Backward compatibility: try reading from separate correct.json
            correct_payload: Dict[str, Any] = {}
            # Convert relative results_path to absolute path for file operations
            results_absolute = repo_root / results_path if not results_path.is_absolute() else results_path
            correct_file = results_absolute / "correct.json"
            if correct_file.exists():
                try:
                    correct_payload = json.loads(
                        correct_file.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse correct.json: {e}")
                    correct_payload = {"correct": False, "error": f"Invalid JSON: {e}"}
            correct_flag = bool(correct_payload.get("correct", False))
            error_msg = correct_payload.get("error")

        stdout_log = "\n".join((cmd.stdout or "") for cmd in commands if cmd.stdout)
        stderr_log = "\n".join((cmd.stderr or "") for cmd in commands if cmd.stderr)

        metrics.setdefault("evaluation_time_seconds", elapsed)

        return AgenticEvaluatorResult(
            metrics=metrics,
            correct=correct_flag,
            error_message=error_msg,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            session_log=session_log,
            commands_run=commands,
            session_log_path=session_log_path,
            session_events=session_events,
            session_id=resolved_session_id,
            session_dir=session_dir,
            elapsed_seconds=elapsed,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_prompt(
        self,
        *,
        task_name: str,
        eval_command: Sequence[str],
        program_path: Path,
        results_path: Path,
        metrics_path: Path,
        eval_prompt: Optional[str] = None,
        max_score: float = 100.0,
    ) -> tuple[str, str]:
        # Build evaluation criteria section if custom prompt provided
        eval_criteria = ""
        if eval_prompt:
            eval_criteria = f"\nEvaluation criteria:\n{eval_prompt.strip()}\n"

        # Program directory is the parent of the program file
        program_dir = program_path.parent if hasattr(program_path, "parent") else Path(program_path).parent

        if eval_command:
            # Standard case: run eval command and write metrics
            command_str = " ".join(eval_command)
            user = AGENTIC_EVAL_USER.format(
                task_name=task_name,
                eval_command=command_str,
                program_dir=program_dir,
                program_path=program_path,
                results_path=results_path,
                metrics_path=metrics_path,
                max_score=max_score,
                eval_criteria=eval_criteria,
            )
        else:
            # No eval command - LLM judges the code directly
            user = f"""# Evaluation Task (no script provided)

- Task: {task_name}
- Working directory: repository root
- Program path: {program_path}
- Results path: {results_path}
- Metrics JSON: {metrics_path}
- Max score: {max_score}

No evaluation command was supplied.
1) Inspect the workspace/program as needed.
2) Judge the submission against the evaluation criteria below.
3) Write a single JSON file at the metrics path with this schema:
   {{"combined_score": <float 0-{max_score}>, "correct": <boolean>, "details": <short reason>}}.
   - combined_score: How well the code performed
   - correct: true if code runs without critical errors (be generous for open-ended tasks)
   - details: Brief explanation of score and any issues
   You may add more fields if useful.
4) If you cannot score, still create the file with fallback values (score=0, correct=false).
{eval_criteria}
Finish after metrics.json is written.
"""

        return user.strip(), AGENTIC_EVAL_SYS.format(max_score=max_score).strip()
