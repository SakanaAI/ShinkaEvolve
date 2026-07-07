DIFF_SYS_FORMAT = """
<<<<<<< HEAD
Make targeted repository edits directly in the active working directory.
Do not return SEARCH/REPLACE blocks or a standalone code block for Shinka to apply.
The repository files you edit are the proposed change.
=======
You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format shown below to indicate changes:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file, including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.
>>>>>>> wandb-log
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
