#!/usr/bin/env python3
"""
Tests for async database additions with non-blocking complexity calculation.

These tests intentionally avoid requiring `pytest-asyncio` by running async
helpers via `asyncio.run(...)` inside synchronous pytest test functions.
"""

import asyncio
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from shinka.database import DatabaseConfig, Program, ProgramDatabase


# Allow running this file directly with `python tests/test_async_complexity_1000.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


NUM_PROGRAMS = 200
EMBEDDING_RECOMPUTE_INTERVAL = 50


def mock_analyze_code_metrics(code: str, language: str) -> Dict[str, Any]:
    """Mock complexity analyzer that simulates bounded CPU work."""
    time.sleep(0.002)
    return {
        "complexity_score": len(code) / 10.0,
        "cyclomatic_complexity": random.randint(1, 5),
        "lines": len(code.splitlines()),
    }


def build_program(prefix: str, idx: int) -> Program:
    return Program(
        id=f"{prefix}-{idx:04d}",
        code=f"def p_{idx}():\n    return {idx}\n",
        language="python",
        generation=idx,
        combined_score=float(idx),
        correct=True,
    )


async def _run_single_additions_with_complexity() -> float:
    from shinka.database.async_dbase import AsyncProgramDatabase
    from shinka.database import async_dbase as async_dbase_module

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "async_single.db"
        sync_db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1),
            embedding_model="",
        )
        async_db = AsyncProgramDatabase(
            sync_db=sync_db,
            embedding_recompute_interval=EMBEDDING_RECOMPUTE_INTERVAL,
        )

        original_analyze = async_dbase_module.analyze_code_metrics
        async_dbase_module.analyze_code_metrics = mock_analyze_code_metrics

        try:
            start_time = time.time()
            for i in range(NUM_PROGRAMS):
                await async_db.add_program_async(program=build_program("single", i))
            total_time = time.time() - start_time

            sample_program = sync_db.get(f"single-{NUM_PROGRAMS // 2:04d}")
            assert sample_program is not None
            assert sample_program.complexity > 0
            assert "code_analysis_metrics" in (sample_program.metadata or {})
            return total_time
        finally:
            async_dbase_module.analyze_code_metrics = original_analyze
            await async_db.close_async()
            sync_db.close()


async def _run_concurrent_additions_with_complexity() -> float:
    from shinka.database.async_dbase import AsyncProgramDatabase
    from shinka.database import async_dbase as async_dbase_module

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "async_concurrent.db"
        sync_db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1),
            embedding_model="",
        )
        async_db = AsyncProgramDatabase(
            sync_db=sync_db,
            embedding_recompute_interval=EMBEDDING_RECOMPUTE_INTERVAL,
        )

        original_analyze = async_dbase_module.analyze_code_metrics
        async_dbase_module.analyze_code_metrics = mock_analyze_code_metrics

        try:
            start_time = time.time()

            async def add_batch(start_idx: int, count: int):
                for i in range(count):
                    idx = start_idx + i
                    await async_db.add_program_async(program=build_program("conc", idx))
                    await asyncio.sleep(0.0005)

            chunk = NUM_PROGRAMS // 4
            tasks = [add_batch(i, chunk) for i in range(0, NUM_PROGRAMS, chunk)]
            await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            sample_program = sync_db.get(f"conc-{(NUM_PROGRAMS * 3) // 4:04d}")
            assert sample_program is not None
            assert sample_program.complexity > 0
            assert "code_analysis_metrics" in (sample_program.metadata or {})
            return total_time
        finally:
            async_dbase_module.analyze_code_metrics = original_analyze
            await async_db.close_async()
            sync_db.close()


def test_single_additions_with_complexity():
    total_time = asyncio.run(_run_single_additions_with_complexity())
    assert total_time > 0.0


def test_concurrent_additions_with_complexity():
    total_time = asyncio.run(_run_concurrent_additions_with_complexity())
    assert total_time > 0.0


async def main():
    print("🧪 Async DB + complexity smoke/perf test")
    print("=" * 60)
    single_time = await _run_single_additions_with_complexity()
    concurrent_time = await _run_concurrent_additions_with_complexity()
    print(f"Single-add time ({NUM_PROGRAMS}): {single_time:.2f}s")
    print(f"Concurrent time ({NUM_PROGRAMS}): {concurrent_time:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
