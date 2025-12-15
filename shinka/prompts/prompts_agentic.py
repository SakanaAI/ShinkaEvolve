"""Prompt fragments specialized for agentic Codex editing sessions."""

AGENTIC_SYS_FORMAT = """
You are operating inside a sandboxed checkout of the user's repository. You have
direct shell access and must apply changes by editing the files within this
workspace instead of replying with diffs or entire rewritten files. Run shell
commands such as `apply_patch`, `cat <<'EOF'`, text editors, or project CLI
commands to read and modify files. You may open and change multiple files during
the same edit as long as every change remains within EVOLVE-BLOCK regions for
those files, and you keep the program runnable.

Multi-file edits are expected: helper modules, evaluators, assets, and configs
that live next to the main program are already copied into the workspace for
you. Update them whenever your change requires supporting code, and feel free to
run formatters or tests inside the sandbox to validate your work.

When you are satisfied with the repository state, stop issuing shell commands
and send a single final message formatted exactly like this:

<NAME>
short_snake_case_identifier
</NAME>

<DESCRIPTION>
Reasoning behind the change and which behaviors or metrics it should improve.
</DESCRIPTION>

<SUMMARY>
- main.py: example note about the adjustment you made
- helpers/motifs.py: describe any helper edits (add more bullets as needed)
</SUMMARY>

Do not include raw code or diffs in the final summary—the tooling captures the
actual files automatically. If you forget to modify the files and only describe
a change, the run will be discarded.
"""


AGENTIC_ITER_MSG = """{task_context}
# Current program

Here is the current program snapshot for quick reference. You still need to
inspect and edit the real files in the workspace when making changes.

```{language}
{code_content}
```

Here are the current performance metrics:

{performance_metrics}{text_feedback_section}

# Workspace instructions

1. Treat `main.{language}` as the primary entry point, but feel free to open and
   modify any helper modules (for example, rendering utilities or motif
   libraries) that sit next to it in the workspace.
2. Only change code that lies between the `EVOLVE-BLOCK-START` and
   `EVOLVE-BLOCK-END` markers within each file. Leave scaffold code outside
   those markers untouched.
3. Use shell commands to edit files directly: `apply_patch`, `python - <<'PY'`,
   redirection into files, or other CLI tools are all available. Running tests
   or formatters (e.g., `pytest`, `ruff`, `black`) is encouraged when it helps
   validate your edit.
4. Multi-file edits should stay coherent—if you introduce a function in
   `main.py`, update the relevant helper modules or configs in the same session
   so the evaluator can run without manual fixes.

# Task

Propose and implement a concrete improvement that should increase the
`combined_score`. Think in terms of hill-climbing: inspect the workspace, edit
the files needed for your idea, and make sure the resulting program still runs.
When finished, provide the formatted summary described in the system prompt.
"""
