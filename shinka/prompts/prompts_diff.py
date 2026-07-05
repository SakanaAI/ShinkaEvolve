DIFF_SYS_FORMAT = """
Make targeted repository edits directly in the active working directory.
Do not return SEARCH/REPLACE blocks or a standalone code block for Shinka to apply.
The repository files you edit are the proposed change.
""".rstrip()


DIFF_ITER_MSG = """# Current repository individual

Here is the current repository summary:
{code_content}

Here are the performance metrics of the repository individual:

{performance_metrics}{text_feedback_section}

# Instructions

Make sure that your repository edits are consistent with each other. For example, if you use a new config variable somewhere, also add the definition or wiring it needs.

# Task

Apply a targeted idea to improve performance, inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the repository individual.

IMPORTANT: Do not rewrite the entire repository implementation - focus on targeted improvements.
""".rstrip()
