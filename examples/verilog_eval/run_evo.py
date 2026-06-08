"""Run VerilogEval evolution with ShinkaEvolve.

Usage:
    python run_evo.py                           # Default: LFSR32
    VERILOG_PROBLEM=Prob144_conwaylife python run_evo.py  # Conway's Game of Life

Requires:
    - iverilog v12 installed (apt install iverilog / brew install icarus-verilog)
    - verilog-eval repo cloned next to ShinkaEvolve (or set VERILOG_EVAL_DIR)
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
        "Improve the candidate module while preserving the TopModule interface and correctness. "
        "Use synthesizable constructs only. Avoid latches (always use default in case statements). "
        "Use non-blocking assignments (<=) in clocked always blocks. "
        "Be careful with reset polarity (synchronous active-high unless specified otherwise). "
        "Preserve EVOLVE-BLOCK markers."
    ),
)

runner = ShinkaEvolveRunner(
    evo_config=evo_conf,
    job_config=job_conf,
    db_config=db_conf,
    max_evaluation_jobs=4,
    max_proposal_jobs=4,
)
runner.run()
