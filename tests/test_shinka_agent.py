"""Tests for shinka/edit/shinka_agent.py - Native agentic editing backend."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from shinka.edit.shinka_agent import (
    ACTION_RE,
    MAX_OBSERVATION_CHARS,
    ShinkaExecutionError,
    ShinkaUnavailableError,
    _execute_bash,
    _truncate_output,
    ensure_shinka_available,
    run_shinka_task,
)
from shinka.llm.models.result import QueryResult


# ============================================================================
# Core Functionality Tests - ensure_shinka_available
# ============================================================================


def test_ensure_shinka_available_with_env_var(monkeypatch):
    """Test that ensure_shinka_available returns True when env var is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert ensure_shinka_available() is True


def test_ensure_shinka_available_with_credentials_file(monkeypatch):
    """Test that ensure_shinka_available returns True when credentials file has key."""
    # Clear all env vars
    for var in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "AWS_ACCESS_KEY_ID",
    ]:
        monkeypatch.delenv(var, raising=False)

    # Mock get_api_key to return a key for codex
    # The function imports get_api_key inside, so we patch it at the source
    with patch("shinka.tools.credentials.get_api_key") as mock_get_api_key:
        mock_get_api_key.return_value = "creds-file-key"
        result = ensure_shinka_available()

        assert result is True
        # Verify the key was set in environment
        import os

        assert os.environ.get("OPENAI_API_KEY") == "creds-file-key"


def test_ensure_shinka_available_raises_when_none(monkeypatch):
    """Test that ensure_shinka_available raises when no keys are available."""
    # Clear all env vars
    for var in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "AWS_ACCESS_KEY_ID",
    ]:
        monkeypatch.delenv(var, raising=False)

    # Mock get_api_key to return None
    # The function imports get_api_key inside, so we patch it at the source
    with patch("shinka.tools.credentials.get_api_key") as mock_get_api_key:
        mock_get_api_key.return_value = None

        with pytest.raises(ShinkaUnavailableError) as exc_info:
            ensure_shinka_available()

        assert "No LLM API keys found" in str(exc_info.value)


# ============================================================================
# Bash Execution Tests - _execute_bash
# ============================================================================


def test_execute_bash_success(tmp_path):
    """Test successful bash command execution."""
    workdir = tmp_path
    test_file = workdir / "test.txt"
    test_file.write_text("hello world")

    exit_code, stdout, stderr = _execute_bash(f"cat {test_file}", workdir)

    assert exit_code == 0
    assert "hello world" in stdout
    assert stderr == ""


def test_execute_bash_timeout(tmp_path, monkeypatch):
    """Test bash command timeout handling."""
    workdir = tmp_path

    # Mock subprocess.run to raise TimeoutExpired
    original_run = subprocess.run

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sleep 1000", timeout=1)

    monkeypatch.setattr(subprocess, "run", mock_run)

    exit_code, stdout, stderr = _execute_bash("sleep 1000", workdir, timeout=1)

    assert exit_code == 1
    assert stdout == ""
    assert "timed out after 1s" in stderr


def test_execute_bash_nonzero_exit(tmp_path):
    """Test bash command with non-zero exit code."""
    workdir = tmp_path

    # Run a command that will fail
    exit_code, stdout, stderr = _execute_bash(
        "cat nonexistent_file_12345.txt", workdir
    )

    assert exit_code == 1
    assert "No such file or directory" in stderr or "cannot open" in stderr.lower()


# ============================================================================
# Agent Loop Tests - run_shinka_task with mocked LLM
# ============================================================================


def test_run_shinka_task_single_turn(tmp_path, monkeypatch):
    """Test run_shinka_task with single turn: bash block then termination."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Create a test file
    test_file = workdir / "test.py"
    test_file.write_text("print('hello')")

    # Mock LLMClient
    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # First response: bash command + termination
        response1 = QueryResult(
            content="Let me read the file.\n```bash\ncat test.py\n```\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            msg="test",
            system_msg="sys",
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={},
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
        )

        mock_llm.query.return_value = response1
        mock_llm.get_kwargs.return_value = {}

        # Run the task
        events = list(
            run_shinka_task(
                user_prompt="Read the file",
                workdir=workdir,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=60,
                max_events=10,
                extra_cli_config={},
            )
        )

        # Verify events
        assert len(events) >= 3  # init, agent_message, command_execution, usage
        assert events[0]["type"] == "init"
        assert events[-1]["type"] == "usage"

        # Check that bash command was executed
        command_events = [e for e in events if e["type"] == "command_execution"]
        assert len(command_events) == 1
        assert "cat test.py" in command_events[0]["item"]["command"]
        assert "hello" in command_events[0]["item"]["stdout"]


def test_run_shinka_task_multi_turn(tmp_path, monkeypatch):
    """Test run_shinka_task with multiple turns and observations."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    test_file = workdir / "test.py"
    test_file.write_text("x = 1")

    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Response sequence
        responses = [
            QueryResult(
                content="```bash\ncat test.py\n```",
                msg="test",
                system_msg="sys",
                new_msg_history=[],
                model_name="gpt-4",
                kwargs={},
                input_tokens=100,
                output_tokens=30,
                cost=0.005,
            ),
            QueryResult(
                content="```bash\necho 'y = 2' >> test.py\n```",
                msg="test",
                system_msg="sys",
                new_msg_history=[],
                model_name="gpt-4",
                kwargs={},
                input_tokens=150,
                output_tokens=40,
                cost=0.007,
            ),
            QueryResult(
                content="Done! COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                msg="test",
                system_msg="sys",
                new_msg_history=[],
                model_name="gpt-4",
                kwargs={},
                input_tokens=180,
                output_tokens=20,
                cost=0.003,
            ),
        ]

        mock_llm.query.side_effect = responses
        mock_llm.get_kwargs.return_value = {}

        events = list(
            run_shinka_task(
                user_prompt="Modify the file",
                workdir=workdir,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=120,
                max_events=10,
                extra_cli_config={},
            )
        )

        # Check that we got multiple command executions
        command_events = [e for e in events if e["type"] == "command_execution"]
        assert len(command_events) == 2

        # Check total cost tracking
        usage_event = [e for e in events if e["type"] == "usage"][0]
        assert usage_event["usage"]["total_cost_usd"] == pytest.approx(0.015, rel=1e-5)
        assert usage_event["usage"]["input_tokens"] == 430
        assert usage_event["usage"]["output_tokens"] == 90


def test_run_shinka_task_termination_signal(tmp_path, monkeypatch):
    """Test run_shinka_task properly handles COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        response = QueryResult(
            content="Task is complete. COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            msg="test",
            system_msg="sys",
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={},
            input_tokens=50,
            output_tokens=20,
            cost=0.002,
        )

        mock_llm.query.return_value = response
        mock_llm.get_kwargs.return_value = {}

        events = list(
            run_shinka_task(
                user_prompt="Do nothing",
                workdir=workdir,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=60,
                max_events=10,
                extra_cli_config={},
            )
        )

        # Should terminate after first message
        agent_messages = [e for e in events if e["type"] == "agent_message"]
        # Only one real agent message (no timeout/max turns messages)
        assert len(agent_messages) == 1
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in agent_messages[0]["item"]["text"]


def test_run_shinka_task_max_events(tmp_path, monkeypatch):
    """Test that run_shinka_task respects max_events limit."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Response that never terminates
        response = QueryResult(
            content="```bash\necho 'still working'\n```",
            msg="test",
            system_msg="sys",
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={},
            input_tokens=100,
            output_tokens=30,
            cost=0.005,
        )

        mock_llm.query.return_value = response
        mock_llm.get_kwargs.return_value = {}

        events = list(
            run_shinka_task(
                user_prompt="Keep working",
                workdir=workdir,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=1000,
                max_events=3,  # Limit to 3 turns
                extra_cli_config={},
            )
        )

        # Should stop after max_events
        agent_messages = [e for e in events if e["type"] == "agent_message"]
        # Last message should be about reaching max turns
        timeout_message = [
            m for m in agent_messages if "reached max turns" in m["item"]["text"]
        ]
        assert len(timeout_message) == 1


def test_run_shinka_task_empty_response(tmp_path, monkeypatch):
    """Test handling when LLM returns None or empty response."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Return None response
        mock_llm.query.return_value = None
        mock_llm.get_kwargs.return_value = {}

        events = list(
            run_shinka_task(
                user_prompt="Test empty",
                workdir=workdir,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=60,
                max_events=10,
                extra_cli_config={},
            )
        )

        # Should have an error message
        agent_messages = [e for e in events if e["type"] == "agent_message"]
        error_messages = [
            m for m in agent_messages if "empty response" in m["item"]["text"]
        ]
        assert len(error_messages) == 1


def test_run_shinka_task_no_model_configured(tmp_path, monkeypatch):
    """Test that run_shinka_task raises error when no model is configured."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ShinkaExecutionError) as exc_info:
        list(
            run_shinka_task(
                user_prompt="Test",
                workdir=workdir,
                profile=None,  # No profile
                sandbox="none",
                approval_mode="auto",
                max_seconds=60,
                max_events=10,
                extra_cli_config={},  # No model in config either
            )
        )

    assert "No model configured" in str(exc_info.value)


# ============================================================================
# Utility Tests
# ============================================================================


def test_action_regex_extraction():
    """Test ACTION_RE regex extracts bash blocks correctly."""
    # Test single bash block
    text1 = "Let me run this command:\n```bash\necho 'hello'\n```\nDone!"
    match1 = ACTION_RE.search(text1)
    assert match1 is not None
    assert match1.group(1).strip() == "echo 'hello'"

    # Test multiline bash block
    text2 = """I'll do this:
```bash
cd /tmp
ls -la
pwd
```
That's it."""
    match2 = ACTION_RE.search(text2)
    assert match2 is not None
    extracted = match2.group(1).strip()
    assert "cd /tmp" in extracted
    assert "ls -la" in extracted
    assert "pwd" in extracted

    # Test no bash block
    text3 = "No commands here, just text."
    match3 = ACTION_RE.search(text3)
    assert match3 is None

    # Test first bash block only (should ignore second)
    text4 = "```bash\nfirst\n```\nsome text\n```bash\nsecond\n```"
    match4 = ACTION_RE.search(text4)
    assert match4 is not None
    assert match4.group(1).strip() == "first"


def test_truncate_output():
    """Test _truncate_output respects max_chars limit."""
    # Short text - no truncation
    short_text = "short"
    assert _truncate_output(short_text, 100) == short_text

    # Long text - should truncate
    long_text = "a" * 20000
    truncated = _truncate_output(long_text, MAX_OBSERVATION_CHARS)

    assert len(truncated) < len(long_text)
    assert "truncated" in truncated
    # Should have first half and last half
    assert truncated.startswith("a" * 100)  # First part
    assert truncated.endswith("a" * 100)  # Last part

    # Custom max_chars
    custom_truncated = _truncate_output(long_text, 1000)
    assert len(custom_truncated) < 1100  # Some overhead for truncation message
    assert "truncated" in custom_truncated

    # Edge case: exactly at limit
    exact_text = "x" * 100
    assert _truncate_output(exact_text, 100) == exact_text


# ============================================================================
# Integration-style Tests
# ============================================================================


def test_run_shinka_task_with_system_prompt(tmp_path, monkeypatch):
    """Test that system_prompt is properly combined with base prompt."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        response = QueryResult(
            content="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            msg="test",
            system_msg="sys",
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={},
            input_tokens=50,
            output_tokens=10,
            cost=0.001,
        )

        mock_llm.query.return_value = response
        mock_llm.get_kwargs.return_value = {}

        custom_system = "Custom instructions here."

        list(
            run_shinka_task(
                user_prompt="Test",
                workdir=workdir,
                system_prompt=custom_system,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=60,
                max_events=10,
                extra_cli_config={},
            )
        )

        # Verify system_msg passed to query includes custom prompt
        call_args = mock_llm.query.call_args
        system_msg_used = call_args.kwargs["system_msg"]
        assert custom_system in system_msg_used
        assert "You are an expert software engineer" in system_msg_used


def test_run_shinka_task_bash_then_termination(tmp_path, monkeypatch):
    """Test that bash command is executed even when termination signal is present."""
    workdir = tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    test_file = workdir / "output.txt"

    with patch("shinka.edit.shinka_agent.LLMClient") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Response with both bash and termination
        response = QueryResult(
            content=f"```bash\necho 'test' > {test_file}\n```\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            msg="test",
            system_msg="sys",
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={},
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
        )

        mock_llm.query.return_value = response
        mock_llm.get_kwargs.return_value = {}

        events = list(
            run_shinka_task(
                user_prompt="Create file",
                workdir=workdir,
                profile="gpt-4",
                sandbox="none",
                approval_mode="auto",
                max_seconds=60,
                max_events=10,
                extra_cli_config={},
            )
        )

        # Verify bash was executed
        command_events = [e for e in events if e["type"] == "command_execution"]
        assert len(command_events) == 1
        assert test_file.exists()
        assert test_file.read_text().strip() == "test"
