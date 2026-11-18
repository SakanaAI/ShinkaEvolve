"""Child results from real local evaluator processes are validated end to end."""

import asyncio
from pathlib import Path
from textwrap import dedent

import pytest

from eval_integrity_utils import (
    SEED_ID,
    children,
    make_job,
    make_runner,
    open_database,
)
from shinka.launch import JobScheduler, LocalJobConfig

EVALUATOR_PREAMBLE = (
    "import os\nimport sys\nfrom pathlib import Path\n"
    "results = Path(sys.argv[sys.argv.index('--results_dir') + 1])\n"
    "results.mkdir(parents=True, exist_ok=True)\n"
)

WRAPPED_EVALUATOR = """
import numpy as np
from shinka.core import run_shinka_eval
run_shinka_eval(
    program_path=sys.argv[sys.argv.index('--program_path') + 1],
    results_dir=str(results),
    experiment_fn_name="run_experiment",
    num_runs=1,
    aggregate_metrics_fn=lambda runs: {"combined_score": np.float32(1.5)},
)
"""

CASES = {
    # NumPy metrics used to crash the second file write after a correct verdict.
    "numpy-metrics": (WRAPPED_EVALUATOR, True, 1.5, ""),
    # A kill between the two writes of a hand-written evaluator.
    "verdict-without-metrics": (
        """
        (results / 'correct.json').write_text('{"correct": true}')
        os._exit(0)
        """,
        False,
        0.0,
        "Metrics file not found",
    ),
    "crash-after-correct-result": (
        """
        (results / 'metrics.json').write_text('{"combined_score": 7.0}')
        (results / 'correct.json').write_text('{"correct": true}')
        sys.exit(3)
        """,
        False,
        0.0,
        "exited with status 3",
    ),
    "infinite-score": (
        """
        (results / 'metrics.json').write_text('{"combined_score": Infinity}')
        (results / 'correct.json').write_text('{"correct": true}')
        """,
        False,
        0.0,
        "non-finite 'combined_score'",
    ),
    "string-verdict": (
        """
        (results / 'metrics.json').write_text('{"combined_score": 7.0}')
        (results / 'correct.json').write_text('{"correct": "false"}')
        """,
        False,
        0.0,
        "boolean 'correct' field",
    ),
}


@pytest.mark.parametrize(
    ("body", "correct", "score", "diagnostic"), CASES.values(), ids=CASES.keys()
)
def test_local_child_evaluation_is_validated(
    tmp_path: Path, body: str, correct: bool, score: float, diagnostic: str
) -> None:
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(EVALUATOR_PREAMBLE + dedent(body))
    program = tmp_path / "gen_1" / "main.py"
    program.parent.mkdir()
    program.write_text("def run_experiment(seed):\n    return seed\n")
    results_dir = program.parent / "results"
    scheduler = JobScheduler("local", LocalJobConfig(eval_program_path=str(evaluator)))

    async def run() -> None:
        async with open_database(tmp_path) as (sync_db, async_db):
            runner = make_runner(async_db, scheduler)
            handle = await scheduler.submit_async_nonblocking(
                str(program), str(results_dir)
            )
            job = make_job(
                job_id=handle, exec_fname=str(program), results_dir=str(results_dir)
            )

            assert await runner._process_single_job_safely(job)

            (child,) = children(sync_db)
            assert bool(child.correct) is correct
            assert child.combined_score == score
            assert diagnostic in child.text_feedback
            assert child.metadata.get("evaluation_failed", False) is not correct
            assert child.metadata["source_job_id"].startswith(
                f"ProcessWithLogging(PID: {handle.pid})|gen=1|"
            )
            best = sync_db.get_best_program()
            assert best.id == (child.id if correct else SEED_ID)

    try:
        asyncio.run(run())
    finally:
        scheduler.shutdown()
