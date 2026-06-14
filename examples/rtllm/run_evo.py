"""Quick-start: evolve a single RTLLM design for PPA (defaults to adder_8bit).

    python run_evo.py                       # adder_8bit, 30 generations
    RTLLM_DESIGN=multi_8bit python run_evo.py

Requires problems/rtllm_proto.jsonl + seeds/ (run extract_dataset.py first),
iverilog, yosys (native or hdlc/yosys docker), and a provider key in repo .env.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))          # force THIS fork (has Verilog support)
load_dotenv(REPO_ROOT / ".env", override=True)

from shinka.core import ShinkaEvolveRunner, EvolutionConfig  # noqa: E402
from shinka.database import DatabaseConfig  # noqa: E402
from shinka.launch import LocalJobConfig  # noqa: E402

DESIGN = os.environ.get("RTLLM_DESIGN", "adder_8bit")
os.environ.setdefault("RTLLM_PROBLEM_FILE", "problems/rtllm_proto.jsonl")

job_conf = LocalJobConfig(eval_program_path="evaluate.py")
db_conf = DatabaseConfig(num_islands=1, archive_size=12)
evo_conf = EvolutionConfig(
    init_program_path=str(Path(__file__).resolve().parent / "seeds" / DESIGN / "initial.sv"),
    language="verilog",
    num_generations=30,
    llm_models=["openrouter/anthropic/claude-sonnet-4.5"],
    llm_kwargs=dict(temperatures=[0.3, 0.7], max_tokens=6144),
    embedding_model=None,
    results_dir=f"results/{DESIGN}",
    use_text_feedback=True,
    task_sys_msg=(
        "You are an expert digital design engineer optimizing Verilog RTL for area "
        "and timing. The function is fixed and checked by FORMAL EQUIVALENCE against "
        "a golden reference (identical for ALL inputs, not just a testbench). Minimize "
        "post-synthesis AIG area (cell count) and logic depth. Explore stronger "
        "microarchitectures (parallel-prefix adders, Booth/Wallace multipliers). Keep "
        "the module name and ports unchanged; synthesizable Verilog only; preserve "
        "EVOLVE-BLOCK markers."
    ),
)

runner = ShinkaEvolveRunner(
    evo_config=evo_conf, job_config=job_conf, db_config=db_conf,
    max_evaluation_jobs=2, max_proposal_jobs=2,
)
runner.run()
