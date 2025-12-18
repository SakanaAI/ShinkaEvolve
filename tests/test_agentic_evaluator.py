"""Comprehensive tests for shinka/eval/agentic.py - Agentic evaluator."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock

import pytest

from shinka.core.runner import AgenticEvaluatorConfig
from shinka.edit.codex_cli import CodexExecutionError
from shinka.eval.agentic import AgenticEvaluator, AgenticEvaluatorResult


@pytest.fixture
def mock_config():
    """Create a mock AgenticEvaluatorConfig."""
    config = MagicMock(spec=AgenticEvaluatorConfig)
    config.cli_profile = "test-profile"
    config.sandbox = True
    config.approval_mode = "auto"
    config.max_seconds = 300
    config.max_events = 100
    config.extra_cli_config = {}
    config.cli_path = None
    return config


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with typical structure."""
    workspace = {
        "repo_root": tmp_path / "repo",
        "program_path": tmp_path / "repo" / "solution.py",
        "results_path": tmp_path / "repo" / "results",
        "metrics_path": tmp_path / "repo" / "results" / "metrics.json",
        "eval_sessions_root": tmp_path / "eval_sessions",
    }
    workspace["repo_root"].mkdir(parents=True)
    workspace["results_path"].mkdir(parents=True)
    workspace["eval_sessions_root"].mkdir(parents=True)
    workspace["program_path"].write_text("# Test program\nprint('Hello')\n")
    return workspace


def make_mock_runner(
    session_events: List[Dict[str, Any]],
    include_metrics: bool = True,
    metrics_data: Optional[Dict[str, Any]] = None,
) -> callable:
    """Create a mock agent runner that yields events and optionally creates metrics.json."""

    def mock_runner(
        user_prompt: str,
        system_prompt: str,
        workdir: Path,
        profile: str,
        sandbox: bool,
        approval_mode: str,
        max_seconds: int,
        max_events: int,
        extra_cli_config: Dict[str, Any],
        cli_path: Optional[str],
        session_kind: str,
        results_dir: Optional[str],
    ) -> Iterator[Dict[str, Any]]:
        """Mock runner that yields session events."""
        # Yield all session events
        for event in session_events:
            yield event

        # Optionally write metrics.json after all events
        if include_metrics:
            metrics_file = workdir / "results" / "metrics.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            data = metrics_data or {
                "combined_score": 85.0,
                "correct": True,
                "details": "Test passed successfully",
            }
            metrics_file.write_text(json.dumps(data))

    return mock_runner


def test_agentic_evaluator_success(mock_config, temp_workspace):
    """Test successful evaluation with metrics written."""
    # Create mock session events
    session_events = [
        {
            "type": "thread.init",
            "thread_id": "test-thread-123",
            "item": {
                "type": "agent_message",
                "text": "Starting evaluation",
            },
        },
        {
            "type": "thread.message",
            "thread_id": "test-thread-123",
            "item": {
                "type": "command_execution",
                "command": "python solution.py",
                "status": "success",
                "exit_code": 0,
                "stdout": "Test output",
                "stderr": "",
            },
        },
        {
            "type": "thread.message",
            "thread_id": "test-thread-123",
            "item": {
                "type": "agent_message",
                "text": "Evaluation complete, metrics written",
            },
        },
    ]

    mock_runner = make_mock_runner(session_events)
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["python", "eval.py"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test_task",
    )

    # Verify result structure
    assert isinstance(result, AgenticEvaluatorResult)
    assert result.correct is True
    assert result.metrics["combined_score"] == 85.0
    assert result.metrics["details"] == "Test passed successfully"
    assert result.error_message is None
    assert result.session_id == "test-thread-123"
    assert len(result.session_log) == 2
    assert len(result.commands_run) == 1
    assert result.commands_run[0].command == "python solution.py"
    assert result.commands_run[0].exit_code == 0
    assert result.stdout_log == "Test output"
    assert result.stderr_log == ""
    assert result.elapsed_seconds > 0
    assert result.session_log_path.exists()
    assert result.system_prompt is not None
    assert result.user_prompt is not None


def test_agentic_evaluator_no_metrics(mock_config, temp_workspace):
    """Test error when metrics.json not produced."""
    # Events that don't write metrics.json
    session_events = [
        {
            "type": "thread.init",
            "thread_id": "test-thread-456",
            "item": {
                "type": "agent_message",
                "text": "Evaluation started but failed",
            },
        },
    ]

    mock_runner = make_mock_runner(session_events, include_metrics=False)
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    with pytest.raises(CodexExecutionError) as exc_info:
        evaluator.evaluate(
            repo_root=temp_workspace["repo_root"],
            eval_command=["python", "eval.py"],
            program_path=temp_workspace["program_path"],
            results_path=temp_workspace["results_path"],
            metrics_path=temp_workspace["metrics_path"],
            eval_sessions_root=temp_workspace["eval_sessions_root"],
            task_name="test_task",
        )

    assert "did not produce metrics" in str(exc_info.value)
    assert str(temp_workspace["metrics_path"]) in str(exc_info.value)


def test_agentic_evaluator_malformed_json(mock_config, temp_workspace):
    """Test handling of invalid JSON in metrics.json."""
    session_events = [
        {
            "type": "thread.message",
            "thread_id": "test-thread-789",
            "item": {
                "type": "agent_message",
                "text": "Writing malformed metrics",
            },
        },
    ]

    def mock_runner_with_bad_json(**kwargs) -> Iterator[Dict[str, Any]]:
        for event in session_events:
            yield event
        # Write invalid JSON
        metrics_file = kwargs["workdir"] / "results" / "metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text("{invalid json content")

    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner_with_bad_json)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["python", "eval.py"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test_task",
    )

    # Should handle gracefully with error in metrics
    assert "error" in result.metrics
    assert "Invalid JSON in metrics" in result.metrics["error"]
    assert result.metrics["combined_score"] == 0


def test_agentic_evaluator_custom_eval_prompt(mock_config, temp_workspace):
    """Test eval_prompt injection into user prompt."""
    custom_eval_prompt = """
    Check for the following:
    - Code quality and readability
    - Proper error handling
    - Performance optimization
    """

    session_events = [
        {
            "type": "thread.message",
            "item": {
                "type": "agent_message",
                "text": "Evaluating with custom criteria",
            },
        },
    ]

    mock_runner = make_mock_runner(session_events)
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["python", "eval.py"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test_task",
        eval_prompt=custom_eval_prompt,
    )

    # Verify custom prompt was included
    assert result.user_prompt is not None
    assert "Evaluation criteria:" in result.user_prompt
    assert "Code quality and readability" in result.user_prompt
    assert "Proper error handling" in result.user_prompt


def test_agentic_evaluator_no_command_mode(mock_config, temp_workspace):
    """Test LLM-as-judge mode with no eval command."""
    session_events = [
        {
            "type": "thread.message",
            "item": {
                "type": "agent_message",
                "text": "Inspecting code directly",
            },
        },
    ]

    mock_runner = make_mock_runner(
        session_events,
        metrics_data={
            "combined_score": 75.0,
            "correct": True,
            "details": "LLM judged the code as good",
        },
    )
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=[],  # Empty command = LLM-as-judge mode
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test_task",
        eval_prompt="Judge code quality",
    )

    # Verify no-command mode prompt
    assert result.user_prompt is not None
    assert "no script provided" in result.user_prompt.lower()
    assert "Inspect the workspace/program" in result.user_prompt
    assert "Judge the submission" in result.user_prompt
    assert result.correct is True
    assert result.metrics["combined_score"] == 75.0


def test_build_prompt_with_eval_criteria(mock_config):
    """Test prompt construction with evaluation criteria."""
    evaluator = AgenticEvaluator(mock_config)

    user_prompt, system_prompt = evaluator._build_prompt(
        task_name="code_quality_check",
        eval_command=["pytest", "tests/"],
        program_path=Path("/repo/solution.py"),
        results_path=Path("/repo/results"),
        metrics_path=Path("/repo/results/metrics.json"),
        eval_prompt="Focus on test coverage and code style",
        max_score=100.0,
    )

    # Verify user prompt includes all components
    assert "code_quality_check" in user_prompt
    assert "pytest tests/" in user_prompt
    assert "/repo/solution.py" in user_prompt
    assert "/repo/results/metrics.json" in user_prompt
    assert "Evaluation criteria:" in user_prompt
    assert "Focus on test coverage and code style" in user_prompt
    assert "Max score: 100.0" in user_prompt

    # Verify system prompt
    assert "autonomous evaluator" in system_prompt.lower()
    assert "metrics JSON file" in system_prompt
    assert "combined_score" in system_prompt


def test_build_prompt_default(mock_config):
    """Test default prompt construction without eval_prompt."""
    evaluator = AgenticEvaluator(mock_config)

    user_prompt, system_prompt = evaluator._build_prompt(
        task_name="basic_test",
        eval_command=["python", "test.py"],
        program_path=Path("/repo/main.py"),
        results_path=Path("/repo/out"),
        metrics_path=Path("/repo/out/metrics.json"),
        eval_prompt=None,
        max_score=50.0,
    )

    # Verify no eval criteria section when none provided
    assert "Evaluation criteria:" not in user_prompt
    assert "basic_test" in user_prompt
    assert "python test.py" in user_prompt
    assert "Max score: 50.0" in user_prompt

    # System prompt should be present
    assert system_prompt
    assert "50.0" in system_prompt


def test_extract_session_id_from_events(mock_config, temp_workspace):
    """Test session ID extraction from various event formats."""
    # Test with thread.init event
    events_thread = [
        {
            "type": "thread.init",
            "thread_id": "thread-abc-123",
            "item": {"type": "agent_message", "text": "Starting"},
        },
    ]

    mock_runner = make_mock_runner(events_thread)
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)
    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["echo", "test"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test",
    )
    assert result.session_id == "thread-abc-123"

    # Test with direct session_id field
    events_session = [
        {
            "type": "custom",
            "session_id": "session-xyz-456",
            "item": {"type": "agent_message", "text": "Starting"},
        },
    ]

    mock_runner2 = make_mock_runner(events_session)
    evaluator2 = AgenticEvaluator(mock_config, codex_runner=mock_runner2)
    result2 = evaluator2.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["echo", "test"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test",
    )
    assert result2.session_id == "session-xyz-456"

    # Test with nested session object
    events_nested = [
        {
            "type": "custom",
            "session": {"id": "nested-session-789"},
            "item": {"type": "agent_message", "text": "Starting"},
        },
    ]

    mock_runner3 = make_mock_runner(events_nested)
    evaluator3 = AgenticEvaluator(mock_config, codex_runner=mock_runner3)
    result3 = evaluator3.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["echo", "test"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="test",
    )
    assert result3.session_id == "nested-session-789"


def test_agentic_evaluator_backward_compatibility_correct_json(
    mock_config, temp_workspace
):
    """Test backward compatibility with separate correct.json file."""
    session_events = [
        {
            "type": "thread.message",
            "item": {"type": "agent_message", "text": "Evaluation done"},
        },
    ]

    def mock_runner_with_legacy(**kwargs) -> Iterator[Dict[str, Any]]:
        for event in session_events:
            yield event
        # Write old-style metrics without 'correct' field
        metrics_file = kwargs["workdir"] / "results" / "metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(
            json.dumps({"combined_score": 90.0, "details": "Legacy format"})
        )
        # Write separate correct.json
        correct_file = kwargs["workdir"] / "results" / "correct.json"
        correct_file.write_text(json.dumps({"correct": True}))

    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner_with_legacy)
    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["python", "eval.py"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="legacy_test",
    )

    # Should read correct flag from correct.json
    assert result.correct is True
    assert result.error_message is None
    assert result.metrics["combined_score"] == 90.0


def test_agentic_evaluator_agent_runner_alias(mock_config, temp_workspace):
    """Test agent_runner parameter alias for backward compatibility."""
    session_events = [
        {
            "type": "thread.message",
            "item": {"type": "agent_message", "text": "Using alias"},
        },
    ]

    mock_runner = make_mock_runner(session_events)
    # Use agent_runner instead of codex_runner
    evaluator = AgenticEvaluator(mock_config, agent_runner=mock_runner)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["echo", "test"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="alias_test",
    )

    assert result.metrics["combined_score"] == 85.0


def test_agentic_evaluator_max_score_propagation(mock_config, temp_workspace):
    """Test that max_score parameter is properly propagated to prompts."""
    session_events = [
        {
            "type": "thread.message",
            "item": {"type": "agent_message", "text": "Custom max score"},
        },
    ]

    mock_runner = make_mock_runner(
        session_events,
        metrics_data={"combined_score": 150.0, "correct": True, "details": "Excellent"},
    )
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["python", "eval.py"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="custom_max_score_test",
        max_score=200.0,
    )

    # Verify max_score in prompts
    assert "200.0" in result.system_prompt
    assert "200.0" in result.user_prompt
    assert result.metrics["combined_score"] == 150.0


def test_agentic_evaluator_session_log_persistence(mock_config, temp_workspace):
    """Test that session logs are properly written to disk."""
    session_events = [
        {
            "type": "thread.message",
            "item": {"type": "agent_message", "text": "First message"},
        },
        {
            "type": "thread.message",
            "item": {"type": "agent_message", "text": "Second message"},
        },
    ]

    mock_runner = make_mock_runner(session_events)
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["echo", "test"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="log_test",
    )

    # Verify session log file exists and contains events
    assert result.session_log_path.exists()
    log_content = result.session_log_path.read_text()
    assert log_content.count("\n") == len(session_events)  # One line per event
    # Verify JSONL format
    for line in log_content.strip().split("\n"):
        assert json.loads(line)  # Should be valid JSON


def test_agentic_evaluator_evaluation_time_in_metrics(mock_config, temp_workspace):
    """Test that evaluation_time_seconds is added to metrics."""
    session_events = [
        {
            "type": "thread.message",
            "item": {"type": "agent_message", "text": "Processing"},
        },
    ]

    mock_runner = make_mock_runner(session_events)
    evaluator = AgenticEvaluator(mock_config, codex_runner=mock_runner)

    start = time.monotonic()
    result = evaluator.evaluate(
        repo_root=temp_workspace["repo_root"],
        eval_command=["echo", "test"],
        program_path=temp_workspace["program_path"],
        results_path=temp_workspace["results_path"],
        metrics_path=temp_workspace["metrics_path"],
        eval_sessions_root=temp_workspace["eval_sessions_root"],
        task_name="timing_test",
    )
    elapsed = time.monotonic() - start

    # Verify evaluation_time_seconds is in metrics
    assert "evaluation_time_seconds" in result.metrics
    assert result.metrics["evaluation_time_seconds"] > 0
    assert result.metrics["evaluation_time_seconds"] <= elapsed + 0.1  # Small tolerance
    assert result.elapsed_seconds == result.metrics["evaluation_time_seconds"]
