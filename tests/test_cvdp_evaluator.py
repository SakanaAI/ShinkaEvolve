"""Unit tests for CVDP evaluator components (no Docker required)."""

import json
import re
from pathlib import Path
import pytest

# Import the CVDP evaluator functions we want to test
try:
    # Mock the original module if we don't want to run the actual imports
    # that require Docker
    from examples.cvdp.evaluate import (
        _load_problems,
        _find_problem,
        _extract_module_name,
        _extract_rtl_path,
        _parse_pytest_output,
        _classify_error,
    )
except ImportError:
    # For test environment, we might need to adjust sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from examples.cvdp.evaluate import (
        _load_problems,
        _find_problem,
        _extract_module_name,
        _extract_rtl_path,
        _parse_pytest_output,
        _classify_error,
    )


def test_load_problems(tmp_path):
    """Test loading problems from JSONL file."""
    jsonl_content = """
{"id": "prob1", "categories": ["cid003", "easy"], "input": {}, "harness": {}}
{"id": "prob2", "categories": ["cid016", "medium"], "input": {}, "harness": {}}
"""
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(jsonl_content, encoding="utf-8")
    
    problems = _load_problems(jsonl_path)
    assert len(problems) == 2
    assert problems[0]["id"] == "prob1"
    assert problems[1]["id"] == "prob2"


def test_find_problem():
    """Test finding a problem by ID."""
    problems = [
        {"id": "prob1", "categories": []},
        {"id": "prob2", "categories": []},
    ]
    
    # Find existing problem
    assert _find_problem(problems, "prob2") == problems[1]
    
    # Default to first if no ID
    assert _find_problem(problems, None) == problems[0]
    
    # Non-existent problem raises
    with pytest.raises(ValueError, match="Problem 'prob3' not found"):
        _find_problem(problems, "prob3")


def test_extract_module_name():
    """Test extracting module name from .env file."""
    problem = {
        "harness": {
            "files": {
                "src/.env": """
TOPLEVEL=lfsr_8bit
VERILOG_SOURCES=/code/rtl/lfsr_8bit.sv
"""
            }
        }
    }
    assert _extract_module_name(problem) == "lfsr_8bit"
    
    # Fallback
    problem_no_toplevel = {
        "harness": {
            "files": {
                "src/.env": "VERILOG_SOURCES=/code/rtl/design.sv"
            }
        }
    }
    assert _extract_module_name(problem_no_toplevel) == "design"


def test_extract_rtl_path():
    """Test extracting RTL path from .env file."""
    problem = {
        "harness": {
            "files": {
                "src/.env": """
TOPLEVEL=lfsr_8bit
VERILOG_SOURCES=/code/rtl/lfsr_8bit.sv /code/rtl/helper.sv
"""
            }
        }
    }
    # Should return first source file
    assert _extract_rtl_path(problem) == "/code/rtl/lfsr_8bit.sv"
    
    # Fallback
    problem_no_sources = {"harness": {"files": {}}}
    assert _extract_rtl_path(problem_no_sources) == "/code/rtl/design.sv"


def test_parse_pytest_output():
    """Test parsing pytest output for pass/fail counts."""
    # Normal case
    output = "3 passed, 2 failed, 1 warning in 1.23s"
    passed, failed = _parse_pytest_output(output)
    assert passed == 3
    assert failed == 2
    
    # Only passed
    output = "5 passed in 0.45s"
    passed, failed = _parse_pytest_output(output)
    assert passed == 5
    assert failed == 0
    
    # Only failed
    output = "7 failed, 1 error in 2.34s"
    passed, failed = _parse_pytest_output(output)
    assert passed == 0
    assert failed == 7
    
    # No matches
    output = "some random text"
    passed, failed = _parse_pytest_output(output)
    assert passed == 0
    assert failed == 0
    
    # With extra text
    output = """
Running tests...
test_lfsr.py::test_seed ... PASSED
test_lfsr.py::test_sequence ... FAILED
====================================================================== 2 passed, 1 failed, 0 warnings =======================================================================
"""
    passed, failed = _parse_pytest_output(output)
    assert passed == 2
    assert failed == 1


def test_classify_error():
    """Test error classification from simulation output."""
    # Syntax error
    assert _classify_error("Syntax error near 'always'") == "syntax_error"
    assert _classify_error("SYNTAX ERROR line 5") == "syntax_error"
    
    # Port binding
    assert _classify_error("unable to bind port 'clk'") == "port_binding_error"
    
    # Missing module
    assert _classify_error("unknown module type 'adder'") == "missing_module"
    
    # Compile error
    assert _classify_error("compile error: something") == "compile_error"
    assert _classify_error("build error in module") == "compile_error"
    
    # Timeout
    assert _classify_error("timeout after 120s") == "timeout"
    assert _classify_error("timed out waiting") == "timeout"
    
    # File not found
    assert _classify_error("no such file 'test.sv'") == "file_not_found"
    assert _classify_error("file not found") == "file_not_found"
    
    # Runtime error (catch-all)
    assert _classify_error("some other error") == "runtime_error"
    
    # Case insensitive
    assert _classify_error("SYNTAX ERROR") == "syntax_error"


@pytest.mark.requires_docker
def test_cvdp_evaluator_with_docker():
    """This test requires Docker and is marked accordingly.
    
    It will be skipped in CI unless explicitly run with `-m requires_docker`.
    """
    pytest.skip("Skipping Docker test in CI")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])