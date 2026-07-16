from pathlib import Path


def test_runtime_prompts_have_no_merge_conflict_markers():
    prompt_root = Path(__file__).parents[1] / "shinka" / "prompts"
    offenders = []
    for path in prompt_root.glob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith(("<<<<<<<", "=======", ">>>>>>>")):
                offenders.append(f"{path}:{line_number}")
    assert offenders == []
