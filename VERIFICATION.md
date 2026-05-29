# Verification for CVDP Integration PR

This document outlines how to verify the CVDP integration works correctly for upstream review.

## What This PR Adds

1. **Verilog language support** (`shinka/utils/languages.py`, `shinka/edit/async_apply.py`)
2. **VerilogEval example** – Local `iverilog`-based evaluation (tested ✅)
3. **CVDP example** – Docker/CocoTB-based evaluation (requires CVDP image build)

## Verification Steps

### 1. Language Support Tests
```bash
# Run all Verilog language tests
pytest tests/test_edit_verilog.py -v
pytest tests/test_verilog_eval_ci.py -v
pytest tests/test_cvdp_evaluator.py -v
```

**Expected:** All tests pass (12 passed, 1 skipped for Docker).

### 2. VerilogEval Example (Working)
```bash
cd examples/verilog_eval

# Test evaluator with broken initial design
VERILOG_EVAL_DIR=/path/to/verilog-eval/dataset_spec-to-rtl \
  python evaluate.py --program_path initial.sv --results_dir test_results

# Check output
cat test_results/metrics.json | jq .combined_score  # Should be ~0.0 (failing design)
```

**Requirements:** `iverilog` installed locally.

### 3. CVDP Example Readiness Test
```bash
cd examples/cvdp

# Test harness extraction (no Docker)
python -c "
import json
from pathlib import Path
from evaluate import _load_problems, _find_problem, _prepare_workspace

# Load test problem
problems = _load_problems(Path('problems/cvdp_lite.jsonl'))
problem = _find_problem(problems, 'cvdp_copilot_lfsr_0001')
print(f'Problem loaded: {problem[\"id\"]}')
print('✅ Harness loading works')

# Test workspace preparation
import tempfile, shutil
with tempfile.NamedTemporaryFile(mode='w', suffix='.sv') as f:
    f.write('module test(); endmodule')
    workspace = _prepare_workspace(problem, f.name, 'test-image:latest')
    print(f'Workspace created at: {workspace}')
    
    # Verify files
    expected = ['docker-compose.yml', 'Dockerfile', 'rtl/lfsr_8bit.sv', 'src/.env', 'src/test_lfsr.py', 'src/test_runner.py']
    for file in expected:
        if (workspace / file).exists():
            print(f'  ✅ {file}')
        else:
            print(f'  ❌ {file}')
    
    shutil.rmtree(workspace, ignore_errors=True)
print('✅ Workspace preparation works')
"

# Test evaluator (will fail without Docker image)
CVDP_SIM_IMAGE=nonexistent python evaluate.py --program_path initial.sv --results_dir test_cvdp 2>&1 | grep -A5 "CVDP simulation image"
```

**Expected:** Clear error message about missing CVDP image.

### 4. Full CVDP Test (Requires CVDP Image Build)

```bash
# Build CVDP image (takes ~10-15 minutes)
git clone https://github.com/NVlabs/cvdp-benchmark
cd cvdp-benchmark
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .

# Test ShinkaEvolve integration
cd ../ShinkaEvolve/examples/cvdp
python evaluate.py --program_path initial.sv --results_dir results

# Check output
cat results/metrics.json | jq '.combined_score, .public.stage, .public.passed, .public.total'
```

**Expected:** Score ~0.0 (initial design wrong), but Docker runs and pytest executes.

### 5. Evolution Test
```bash
# Quick evolution test (2 generations)
cd examples/cvdp
CVDP_SIM_IMAGE=nvidia/cvdp-sim:v1.0.0 \
  python run_evo.py --generations 2  # If ShinkaEvolve supports CLI args

# Or modify run_evo.py temporarily:
# evo_conf = EvolutionConfig(num_generations=2, ...)
```

**Expected:** Evolution loop runs, candidates evaluated via Docker.

## Known Limitations

1. **No public Docker image** – Users must build `nvidia/cvdp-sim:v1.0.0` locally.
2. **Windows Docker Desktop** – May have OCI runtime issues (WSL2 recommended).
3. **CVDP requires cocotb 2.0.1** – Older cocotb 1.4 images won't work.

## CI Status

- ✅ Unit tests pass (`requires_docker` markers exclude actual Docker)
- ✅ VerilogEval works locally (no Docker)
- ✅ CVDP harness extraction works (tested)
- ⚠️ CVDP Docker evaluation requires image build (manual step)

## Summary

The PR is **technically correct** and ready for merge. The Docker dependency is a deployment requirement, not a code issue. Users must build the CVDP image once, then evolution works.

For SakanaAI's CI: Add a `requires_docker` marker to skip actual Docker tests (already done). Run unit tests only.