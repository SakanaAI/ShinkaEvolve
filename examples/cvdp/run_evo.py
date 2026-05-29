"""Run CVDP benchmark evolution with ShinkaEvolve.

Usage:
    python run_evo.py                                                     # Default problem
    CVDP_PROBLEM_ID=cvdp_copilot_lfsr_0001 python run_evo.py            # Specific problem
    CVDP_PROBLEM_FILE=problems/my_set.jsonl python run_evo.py           # Custom problem set

Requires:
    - Docker installed and running (for CocoTB-based evaluation)
    - CVDP problem JSONL (included in problems/ or download via download_dataset.py)
"""

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_conf = LocalJobConfig(eval_program_path="evaluate.py")

db_conf = DatabaseConfig(
    num_islands=2,
    archive_size=30,
)

evo_conf = EvolutionConfig(
    init_program_path="initial.sv",
    language="verilog",
    num_generations=30,
    llm_models=["azure-gpt-4-1-mini"],
    llm_kwargs=dict(temperatures=[0.3, 0.7], max_tokens=4096),
    embedding_model=None,
    task_sys_msg=(
        "You are an expert digital design engineer specializing in Verilog/SystemVerilog RTL. "
        "You are evolving a hardware module to pass a CocoTB testbench. "
        "Improve the candidate module while preserving the module name and port interface exactly. "
        "Use synthesizable constructs only (no $display, no initial blocks, no delays). "
        "Avoid latches: use default assignments or complete case/if-else coverage. "
        "Use non-blocking assignments (<=) in clocked always blocks. "
        "Pay attention to reset polarity and edge sensitivity specified in the problem. "
        "Preserve EVOLVE-BLOCK markers."
    ),
)

runner = ShinkaEvolveRunner(
    evo_config=evo_conf,
    job_config=job_conf,
    db_config=db_conf,
    max_evaluation_jobs=2,
    max_proposal_jobs=4,
)
runner.run()
