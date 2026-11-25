"""A full rewrite is the whole fenced block inside <CODE>."""

import pytest

from shinka.edit import apply_full_patch

MD = "# Skill\n<!-- EVOLVE-BLOCK-START -->\nRun it.\n<!-- EVOLVE-BLOCK-END -->\n"
NEW_MD = (
    "# Skill\n<!-- EVOLVE-BLOCK-START -->\n## Run\n\n```bash\nshinka_run --help\n```\n\n"
    "## Output\n\n```\nusage: shinka_run\n```\n<!-- EVOLVE-BLOCK-END -->\n"
)
PY = "# EVOLVE-BLOCK-START\ndef prompt(task):\n    return task\n# EVOLVE-BLOCK-END\n"
NEW_PY = (
    "# EVOLVE-BLOCK-START\ndef prompt(task):\n"
    '    return f"""Solve {task}. Answer in a fence:\n```python\n...\n```\n"""\n'
    "# EVOLVE-BLOCK-END\n"
)


def _reply(
    code: str, tag: str, ticks: str = "```", description: str = "Rewrite."
) -> str:
    return (
        f"<NAME>\nrewrite\n</NAME>\n\n<DESCRIPTION>\n{description}\n</DESCRIPTION>\n\n"
        f"<CODE>\n{ticks}{tag}\n{code}{ticks}\n</CODE>"
    )


def _apply(reply: str, original: str, language: str):
    updated, num_applied, _, error, _, _ = apply_full_patch(
        patch_str=reply, original_str=original, language=language, verbose=False
    )
    return updated, num_applied, error


@pytest.mark.parametrize("ticks", ["```", "````"])
@pytest.mark.parametrize(
    ("original", "new", "language"),
    [(MD, NEW_MD, "markdown"), (PY, NEW_PY, "python")],
    ids=["markdown", "python"],
)
def test_code_fences_inside_the_program_are_kept(original, new, language, ticks):
    updated, num_applied, error = _apply(
        _reply(new, language, ticks), original, language
    )

    assert error is None
    assert num_applied == 1
    assert updated == new


def test_fenced_snippet_in_the_description_is_not_the_program():
    new = PY.replace("return task", "return task.strip()")
    reply = _reply(new, "python", description="Replaces\n```python\nreturn task\n```")

    updated, num_applied, error = _apply(reply, PY, "python")

    assert error is None
    assert num_applied == 1
    assert updated == new


def test_code_tag_mentioned_in_description_is_not_a_section_start():
    new = PY.replace("return task", "return task.strip()")
    reply = _reply(
        new,
        "python",
        description=f"The <CODE> section revises this example:\n```py\n{PY}```",
    )

    assert _apply(reply, PY, "python") == (new, 1, None)


def test_code_tag_mentioned_after_the_rewrite_is_not_a_section_end():
    new = PY.replace("return task", "return task.strip()")
    reply = _reply("def prompt(task):\n    return task.strip()\n", "python")
    reply += "\n\nFor example:\n```python\nprompt(' task ')\n```\nSee `</CODE>` above."

    assert _apply(reply, PY, "python") == (new, 1, None)


def test_unsupported_code_fence_does_not_apply_description_example():
    description = "For example:\n```python\ndef prompt(task):\n    return None\n```"
    reply = _reply(PY, "text", description=description)

    assert _apply(reply, PY, "python") == (
        PY,
        0,
        "Could not extract code from patch string",
    )


def test_shorter_inner_fence_does_not_close_code_section():
    reply = '<CODE>\n````python\ndef prompt(task):\n    return """\n```text\nx\n```\n</CODE>'

    assert _apply(reply, PY, "python") == (
        PY,
        0,
        "Could not extract code from patch string",
    )


def test_first_block_is_used_without_a_code_section():
    new = PY.replace("return task", "return task.strip()")
    reply = f"```python\n{new}```\n\nCheck it with:\n```bash\npython main.py\n```"

    updated, num_applied, error = _apply(reply, PY, "python")

    assert error is None
    assert num_applied == 1
    assert updated == new
