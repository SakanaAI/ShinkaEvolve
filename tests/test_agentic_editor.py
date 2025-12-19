"""Comprehensive tests for shinka/edit/agentic.py."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from unittest.mock import MagicMock

import pytest

from shinka.edit.agentic import (
    AgentContext,
    AgentResult,
    AgenticEditor,
    CommandResult,
    MAX_BASE_FILE_SIZE,
    MAX_BINARY_FILE_SIZE,
)


@pytest.fixture
def mock_config():
    """Create a mock config for AgenticEditor."""
    config = MagicMock()
    config.cli_profile = "test_profile"
    config.sandbox = "enabled"
    config.approval_mode = "auto"
    config.max_seconds = 300
    config.max_turns = 20
    config.extra_cli_config = {}
    config.cli_path = None
    return config


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    """Create a temporary scratch directory."""
    return tmp_path / "scratch"


# ============================================================================
# Scratch Directory Tests (_prepare_scratch method)
# ============================================================================


def test_prepare_scratch_basic(scratch_dir: Path, mock_config):
    """Test basic file writing to scratch directory."""
    editor = AgenticEditor(scratch_dir, mock_config)

    base_files = {
        Path("main.py"): "def hello():\n    print('world')\n",
        Path("utils.py"): "def helper():\n    return 42\n",
    }

    baseline = editor._prepare_scratch(base_files)

    # Check that scratch directory was created
    assert scratch_dir.exists()
    assert scratch_dir.is_dir()

    # Check that files were written
    assert (scratch_dir / "main.py").exists()
    assert (scratch_dir / "utils.py").exists()

    # Check file contents
    assert (scratch_dir / "main.py").read_text() == "def hello():\n    print('world')\n"
    assert (scratch_dir / "utils.py").read_text() == "def helper():\n    return 42\n"

    # Check baseline return value
    assert baseline == base_files


def test_prepare_scratch_preserves_session_meta(scratch_dir: Path, mock_config):
    """Test that session_meta.json is preserved across prepare_scratch calls."""
    editor = AgenticEditor(scratch_dir, mock_config)

    # Create scratch directory with session_meta.json
    scratch_dir.mkdir(parents=True)
    meta_content = json.dumps({"session_id": "test_123", "parent_id": "parent_456"})
    (scratch_dir / "session_meta.json").write_text(meta_content, encoding="utf-8")
    (scratch_dir / "old_file.py").write_text("old content")

    # Prepare scratch with new files
    base_files = {Path("new_file.py"): "new content"}
    editor._prepare_scratch(base_files)

    # Check that session_meta.json was preserved
    assert (scratch_dir / "session_meta.json").exists()
    assert (scratch_dir / "session_meta.json").read_text(encoding="utf-8") == meta_content

    # Check that old file was removed
    assert not (scratch_dir / "old_file.py").exists()

    # Check that new file was created
    assert (scratch_dir / "new_file.py").exists()


def test_prepare_scratch_rejects_absolute_paths(scratch_dir: Path, mock_config):
    """Test ValueError for absolute paths in base_files."""
    editor = AgenticEditor(scratch_dir, mock_config)

    base_files = {
        Path("/etc/passwd"): "malicious content",
    }

    with pytest.raises(ValueError, match="must be relative"):
        editor._prepare_scratch(base_files)


def test_prepare_scratch_rejects_path_traversal(scratch_dir: Path, mock_config):
    """Test ValueError for ../ path traversal attempts."""
    editor = AgenticEditor(scratch_dir, mock_config)

    base_files = {
        Path("../escape.py"): "escaped content",
    }

    with pytest.raises(ValueError, match="escapes scratch directory"):
        editor._prepare_scratch(base_files)

    # Also test more complex traversal
    base_files = {
        Path("subdir/../../escape.py"): "escaped content",
    }

    with pytest.raises(ValueError, match="escapes scratch directory"):
        editor._prepare_scratch(base_files)


def test_prepare_scratch_file_size_limit(scratch_dir: Path, mock_config):
    """Test MAX_BASE_FILE_SIZE enforcement."""
    editor = AgenticEditor(scratch_dir, mock_config)

    # Create a file that exceeds the size limit
    large_content = "x" * (MAX_BASE_FILE_SIZE + 1)
    base_files = {
        Path("large_file.txt"): large_content,
    }

    with pytest.raises(ValueError, match="exceeds max size"):
        editor._prepare_scratch(base_files)


# ============================================================================
# Session Execution Tests (run_session method with mocked runner)
# ============================================================================


def mock_runner_basic(
    user_prompt: str,
    workdir: Path,
    **kwargs
) -> Iterator[Dict[str, Any]]:
    """Basic mock runner that yields controlled events."""
    # Init event with model
    yield {
        "type": "init",
        "model": "claude-opus-4-5",
        "session_id": "sess_abc123",
    }

    # Agent message
    yield {
        "type": "event",
        "item": {
            "type": "agent_message",
            "text": "I'll help you with that task.",
        },
    }

    # Write a file
    (workdir / "output.py").write_text("def new_function():\n    return 'hello'\n")

    # Usage event
    yield {
        "type": "usage",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "total_cost_usd": 0.0025,
        },
    }

    # Final message
    yield {
        "type": "event",
        "item": {
            "type": "agent_message",
            "text": "Task completed successfully.",
        },
    }


def test_run_session_detects_changed_files(scratch_dir: Path, mock_config):
    """Test that changed files are detected correctly."""
    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_basic)

    base_files = {
        Path("existing.py"): "original content",
    }

    context = AgentContext(
        user_prompt="Create a new function",
        language="python",
        base_files=base_files,
        primary_file=Path("existing.py"),
    )

    result = editor.run_session(context)

    # Check that new file was detected
    assert Path("output.py") in result.changed_files
    assert result.changed_files[Path("output.py")] == "def new_function():\n    return 'hello'\n"

    # Check that existing file wasn't changed
    assert Path("existing.py") not in result.changed_files


def test_run_session_handles_binary_files(scratch_dir: Path, mock_config):
    """Test base64 encoding of binary files."""
    def mock_runner_with_binary(user_prompt: str, workdir: Path, **kwargs):
        # Create a binary file
        binary_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        (workdir / "image.png").write_bytes(binary_data)

        yield {"type": "init", "model": "test-model"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Created image"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_with_binary)

    context = AgentContext(
        user_prompt="Create an image",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Binary file should be in binary_changed_files, not changed_files
    assert Path("image.png") not in result.changed_files
    assert Path("image.png") in result.binary_changed_files

    # Check base64 encoding
    expected_b64 = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR').decode("ascii")
    assert result.binary_changed_files[Path("image.png")] == expected_b64


def test_run_session_skips_internal_files(scratch_dir: Path, mock_config):
    """Test that session_log.jsonl and session_meta.json are not in changed_files."""
    def mock_runner_with_internal_files(user_prompt: str, workdir: Path, **kwargs):
        # Create internal files
        (workdir / "session_meta.json").write_text('{"test": "meta"}')
        (workdir / "real_change.py").write_text("changed code")

        yield {"type": "init"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Done"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_with_internal_files)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Internal files should be excluded
    assert Path("session_log.jsonl") not in result.changed_files
    assert Path("session_meta.json") not in result.changed_files

    # Real changes should be included
    assert Path("real_change.py") in result.changed_files


def test_run_session_cost_metrics(scratch_dir: Path, mock_config):
    """Test usage aggregation from events."""
    def mock_runner_with_multiple_usage(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init", "model": "test-model"}

        # First API call
        yield {
            "type": "usage",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "total_cost_usd": 0.002,
            },
        }

        # Second API call
        yield {
            "type": "usage",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 75,
                "total_tokens": 275,
                "total_cost_usd": 0.003,
            },
        }

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Done"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_with_multiple_usage)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Check aggregated metrics
    assert result.metrics["input_tokens"] == 300.0
    assert result.metrics["output_tokens"] == 125.0
    assert result.metrics["total_tokens"] == 425.0
    assert result.metrics["total_cost"] == 0.005
    assert result.metrics["real_cost_available"] is True


def test_run_session_extracts_model_from_init(scratch_dir: Path, mock_config):
    """Test model extraction from init event."""
    def mock_runner_with_model(user_prompt: str, workdir: Path, **kwargs):
        yield {
            "type": "init",
            "model": "claude-sonnet-4-5",
            "session_id": "test_session",
        }

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Working..."},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_with_model)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Model should be extracted from init event
    assert result.model == "claude-sonnet-4-5"
    assert result.session_id == "test_session"


def test_run_session_command_execution(scratch_dir: Path, mock_config):
    """Test that command executions are captured."""
    def mock_runner_with_commands(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}

        # Command execution event
        yield {
            "type": "event",
            "item": {
                "type": "command_execution",
                "command": "pytest tests/",
                "status": "completed",
                "exit_code": 0,
                "stdout": "All tests passed",
                "stderr": "",
            },
        }

        yield {
            "type": "event",
            "item": {
                "type": "command_execution",
                "command": "pylint code.py",
                "status": "failed",
                "exit_code": 1,
                "stdout": "",
                "stderr": "Linting errors found",
            },
        }

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Commands executed"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_with_commands)

    context = AgentContext(
        user_prompt="Run tests",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Check that commands were captured
    assert len(result.commands_run) == 2

    # First command
    assert result.commands_run[0].command == "pytest tests/"
    assert result.commands_run[0].status == "completed"
    assert result.commands_run[0].exit_code == 0
    assert result.commands_run[0].stdout == "All tests passed"

    # Second command
    assert result.commands_run[1].command == "pylint code.py"
    assert result.commands_run[1].status == "failed"
    assert result.commands_run[1].exit_code == 1
    assert result.commands_run[1].stderr == "Linting errors found"


def test_run_session_session_log_accumulation(scratch_dir: Path, mock_config):
    """Test that agent messages are accumulated in session_log."""
    def mock_runner_with_messages(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Starting task..."},
        }

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Processing files..."},
        }

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Task completed!"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_with_messages)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Check session log
    assert len(result.session_log) == 3
    assert result.session_log[0] == "Starting task..."
    assert result.session_log[1] == "Processing files..."
    assert result.session_log[2] == "Task completed!"

    # Final message should be the last one
    assert result.final_message == "Task completed!"


def test_run_session_fallback_cost_estimate(scratch_dir: Path, mock_config):
    """Test fallback cost estimation when no real cost is provided."""
    def mock_runner_no_cost(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}

        # Usage without cost_usd
        yield {
            "type": "usage",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            },
        }

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Done"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_no_cost)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Should use fallback cost estimate (tokens / 1000)
    assert result.metrics["total_tokens"] == 1500.0
    assert result.metrics["total_cost"] == 1.5  # 1500 / 1000
    assert result.metrics["real_cost_available"] is False


def test_run_session_detects_modified_files(scratch_dir: Path, mock_config):
    """Test that modifications to existing files are detected."""
    def mock_runner_modify(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}

        # Modify existing file
        existing_file = workdir / "existing.py"
        existing_file.write_text("modified content")

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Modified file"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_modify)

    base_files = {
        Path("existing.py"): "original content",
    }

    context = AgentContext(
        user_prompt="Modify file",
        language="python",
        base_files=base_files,
        primary_file=Path("existing.py"),
    )

    result = editor.run_session(context)

    # Modified file should be in changed_files
    assert Path("existing.py") in result.changed_files
    assert result.changed_files[Path("existing.py")] == "modified content"


def test_run_session_with_nested_directories(scratch_dir: Path, mock_config):
    """Test handling of files in nested directories."""
    def mock_runner_nested(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}

        # Create nested structure
        (workdir / "src" / "module").mkdir(parents=True)
        (workdir / "src" / "module" / "code.py").write_text("nested code")

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Created nested files"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_nested)

    context = AgentContext(
        user_prompt="Create nested structure",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Check nested file was detected
    nested_path = Path("src") / "module" / "code.py"
    assert nested_path in result.changed_files
    assert result.changed_files[nested_path] == "nested code"


def test_run_session_events_logged_to_jsonl(scratch_dir: Path, mock_config):
    """Test that all events are logged to session_log.jsonl."""
    def mock_runner_events(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init", "model": "test"}
        yield {"type": "usage", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        yield {"type": "event", "item": {"type": "agent_message", "text": "Done"}}

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_events)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Check that session log file exists
    assert result.session_log_path is not None
    assert result.session_log_path.exists()

    # Read and parse JSONL
    lines = result.session_log_path.read_text().strip().split("\n")
    events = [json.loads(line) for line in lines]

    # Should have 3 events
    assert len(events) == 3
    assert events[0]["type"] == "init"
    assert events[1]["type"] == "usage"
    assert events[2]["type"] == "event"

    # Also check session_events in result
    assert len(result.session_events) == 3


def test_run_session_large_binary_files_skipped(scratch_dir: Path, mock_config):
    """Test that binary files exceeding MAX_BINARY_FILE_SIZE are skipped."""
    def mock_runner_large_binary(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}

        # Create a binary file exceeding the limit with non-UTF8 data
        # Use 0xFF bytes which will fail UTF-8 decoding
        large_binary = b'\xff' * (MAX_BINARY_FILE_SIZE + 1)
        (workdir / "large.bin").write_bytes(large_binary)

        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Created large binary"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_large_binary)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Large binary should be skipped
    assert Path("large.bin") not in result.changed_files
    assert Path("large.bin") not in result.binary_changed_files


def test_run_session_backward_compat_codex_runner(scratch_dir: Path, mock_config):
    """Test backward compatibility with codex_runner parameter."""
    def mock_codex_runner(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init", "model": "codex-model"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Codex runner works"},
        }

    # Use deprecated codex_runner parameter
    editor = AgenticEditor(scratch_dir, mock_config, codex_runner=mock_codex_runner)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Should use the codex_runner
    assert result.model == "codex-model"
    assert "Codex runner works" in result.session_log


def test_agent_context_with_metadata(scratch_dir: Path, mock_config):
    """Test that metadata is passed through to runner."""
    captured_kwargs = {}

    def mock_runner_capture(user_prompt: str, workdir: Path, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "init"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Done"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_capture)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
        metadata={
            "parent_id": "parent_123",
            "generation": 5,
            "patch_type": "full",
            "results_dir": "/tmp/results",
        },
    )

    result = editor.run_session(context)

    # Check that metadata was passed to runner
    assert captured_kwargs["parent_id"] == "parent_123"
    assert captured_kwargs["generation"] == 5
    assert captured_kwargs["patch_type"] == "full"
    assert captured_kwargs["results_dir"] == "/tmp/results"


def test_agent_context_with_system_prompt(scratch_dir: Path, mock_config):
    """Test that system_prompt is passed to runner."""
    captured_kwargs = {}

    def mock_runner_capture(user_prompt: str, workdir: Path, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "init"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Done"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_capture)

    system_prompt = "You are a helpful coding assistant."

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
        system_prompt=system_prompt,
    )

    result = editor.run_session(context)

    # Check that system_prompt was passed to runner
    assert captured_kwargs["system_prompt"] == system_prompt


def test_agent_context_with_resume_session(scratch_dir: Path, mock_config):
    """Test resuming a session with resume_session_id."""
    captured_kwargs = {}

    def mock_runner_capture(user_prompt: str, workdir: Path, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"type": "init", "session_id": "resumed_session_456"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Resumed"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_capture)

    context = AgentContext(
        user_prompt="Continue",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
        resume_session_id="session_to_resume_123",
    )

    result = editor.run_session(context)

    # Check that resume_session_id was passed to runner
    assert captured_kwargs["resume_session_id"] == "session_to_resume_123"
    assert result.session_id == "resumed_session_456"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_run_session_no_changes(scratch_dir: Path, mock_config):
    """Test session that completes without making any changes."""
    def mock_runner_no_changes(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "No changes needed"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_no_changes)

    context = AgentContext(
        user_prompt="Review code",
        language="python",
        base_files={Path("code.py"): "def foo(): pass"},
        primary_file=Path("code.py"),
    )

    result = editor.run_session(context)

    # Should have no changed files
    assert len(result.changed_files) == 0
    assert len(result.binary_changed_files) == 0


def test_run_session_empty_base_files(scratch_dir: Path, mock_config):
    """Test session with no base files."""
    def mock_runner_create(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}
        (workdir / "new.py").write_text("created from scratch")
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Created new file"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_create)

    context = AgentContext(
        user_prompt="Create file",
        language="python",
        base_files={},
        primary_file=Path("new.py"),
    )

    result = editor.run_session(context)

    # New file should be detected
    assert Path("new.py") in result.changed_files


def test_prepare_scratch_creates_parent_directories(scratch_dir: Path, mock_config):
    """Test that parent directories are created for nested files."""
    editor = AgenticEditor(scratch_dir, mock_config)

    base_files = {
        Path("a/b/c/deep.py"): "deep file",
    }

    baseline = editor._prepare_scratch(base_files)

    # Check that nested structure was created
    assert (scratch_dir / "a" / "b" / "c" / "deep.py").exists()
    assert (scratch_dir / "a" / "b" / "c" / "deep.py").read_text() == "deep file"


def test_run_session_metrics_include_elapsed_time(scratch_dir: Path, mock_config):
    """Test that elapsed_seconds is included in metrics."""
    def mock_runner_simple(user_prompt: str, workdir: Path, **kwargs):
        yield {"type": "init"}
        yield {
            "type": "event",
            "item": {"type": "agent_message", "text": "Done"},
        }

    editor = AgenticEditor(scratch_dir, mock_config, runner=mock_runner_simple)

    context = AgentContext(
        user_prompt="Test",
        language="python",
        base_files={},
        primary_file=Path("main.py"),
    )

    result = editor.run_session(context)

    # Should have elapsed_seconds metric
    assert "elapsed_seconds" in result.metrics
    assert result.metrics["elapsed_seconds"] > 0


def test_prepare_scratch_handles_unicode(scratch_dir: Path, mock_config):
    """Test handling of unicode content in base files."""
    editor = AgenticEditor(scratch_dir, mock_config)

    base_files = {
        Path("unicode.py"): "# 日本語コメント\ndef hello():\n    print('こんにちは')\n",
    }

    baseline = editor._prepare_scratch(base_files)

    # Check unicode was preserved
    content = (scratch_dir / "unicode.py").read_text(encoding="utf-8")
    assert "日本語" in content
    assert "こんにちは" in content


def test_command_result_dataclass():
    """Test CommandResult dataclass construction."""
    cmd = CommandResult(
        command="pytest",
        status="completed",
        exit_code=0,
        stdout="All tests passed",
        stderr="",
    )

    assert cmd.command == "pytest"
    assert cmd.status == "completed"
    assert cmd.exit_code == 0
    assert cmd.stdout == "All tests passed"
    assert cmd.stderr == ""


def test_agent_result_default_fields():
    """Test AgentResult default field values."""
    result = AgentResult(
        changed_files={Path("test.py"): "content"},
        session_log=["message1", "message2"],
        commands_run=[],
    )

    assert result.final_message is None
    assert result.metrics == {}
    assert result.session_log_path is None
    assert result.session_events == []
    assert result.binary_changed_files == {}
    assert result.session_id is None
    assert result.model is None
