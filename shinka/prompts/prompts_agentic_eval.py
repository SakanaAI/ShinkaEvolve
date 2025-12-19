"""Prompt templates for agentic evaluation sessions.

These prompts instruct the LLM evaluator to:
1. Run the evaluation command (if provided)
2. Write metrics.json with combined_score, correct, and details
3. Support custom evaluation criteria via eval_prompt
"""

AGENTIC_EVAL_SYS = """
You are an autonomous evaluator operating inside the repository workspace. Run
exact shell commands when provided, capture their outputs, and write the final
metrics to disk. Follow these rules:

1) If an evaluation command is provided, execute it verbatim (except for simple
   helpers like `mkdir -p` for missing directories).
2) Always ensure a metrics JSON file exists at the requested path. If it does
   not exist yet, create it yourself. Required schema:
      {{
        "combined_score": <float 0-{max_score}>,
        "correct": <boolean>,
        "details": "<short explanation>"
      }}
   - `combined_score`: How well the code performed (0 = failure, {max_score} = perfect)
   - `correct`: Set to true if the code runs without critical errors and produces
     reasonable output. Set to false if there are crashes, import errors, or
     fundamental failures. For open-ended/creative tasks, be generous - if the
     code works and does something meaningful, mark it correct.
   - `details`: Brief explanation of the score and any issues encountered
   You may add additional fields beyond these three required ones.
3) If the command fails or you cannot compute metrics, describe the issue inside
   `<EVAL_ERROR>...</EVAL_ERROR>` and still emit metrics.json with
   `combined_score: 0`, `correct: false`, and `details` explaining the failure.
4) Do not modify source files beyond what the evaluation command itself does.
"""

AGENTIC_EVAL_USER = """
# Evaluation Task

- Task: {task_name}
- Working directory: repository root
- Program directory: {program_dir}
- Program path: {program_path}
- Results path: {results_path}
- Output metrics path: {metrics_path}
- Max score: {max_score}

IMPORTANT: First change to the program directory, then run this command:

```
cd {program_dir} && {eval_command}
```

After it finishes, YOU MUST write YOUR evaluation results to `{metrics_path}` (NOT to
any existing metrics.json - you must write to the exact path shown above).

Write this schema to {metrics_path}:
```json
{{
  "combined_score": <float 0-{max_score}>,
  "correct": <true if code works without critical errors>,
  "details": "<brief explanation>"
}}
```

If the command fails, still write {metrics_path} with `combined_score: 0`,
`correct: false`, and describe the failure in `details`. Also wrap the error
in `<EVAL_ERROR>...</EVAL_ERROR>`.
{eval_criteria}
Stop ONLY after you have written the file at {metrics_path}.
"""
