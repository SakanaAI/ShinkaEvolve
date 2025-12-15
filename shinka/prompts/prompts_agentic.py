"""Prompt fragments specialized for agentic editing sessions.

IMPORTANT ARCHITECTURE NOTE:
In agentic mode, the CLI harness (Codex, Claude CLI, Gemini CLI) owns the system
prompt. These harnesses inject their own instructions for tool use, file editing,
and shell access. Shinka should NOT provide a system prompt - it would conflict
with or duplicate the harness's instructions.

Instead, task context goes in the USER prompt as a "# Task" section. The harness
sees this as the user's request and applies its own system prompt with tool
instructions.
"""

# Empty - CLI harness provides its own system prompt with tool/shell instructions.
# Do NOT add content here; it would conflict with harness prompts.
AGENTIC_SYS_FORMAT = ""


AGENTIC_ITER_MSG = """# Task

{task_context}

# Score

{score_context}
{text_feedback_section}

Explore the workspace and make improvements. When done, explain what you changed and why.
"""
