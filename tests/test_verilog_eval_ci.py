"""CI-friendly tests for VerilogEval evaluator (requires iverilog but no Docker)."""

import subprocess
from pathlib import Path
import pytest


def has_iverilog() -> bool:
    """Check if iverilog is installed."""
    try:
        result = subprocess.run(
            ["iverilog", "-V"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not has_iverilog(), reason="iverilog not installed")
class TestVerilogEvalInCI:
    """Tests that require iverilog but can run in CI."""
    
    def test_iverilog_version(self):
        """Test that iverilog is available and returns a sensible version."""
        result = subprocess.run(
            ["iverilog", "-V"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        # Should contain "Icarus Verilog"
        assert "Icarus Verilog" in result.stderr or "Icarus Verilog" in result.stdout
        
    def test_vvp_execution(self):
        """Test that vvp (the runtime) can execute a simple simulation."""
        # Create a minimal Verilog test
        simple_v = """
module test;
    initial begin
        $display("Hello from vvp");
        $finish;
    end
endmodule
"""
        test_dir = Path.cwd() / "tmp_test_vvp"
        test_dir.mkdir(exist_ok=True)
        
        try:
            v_file = test_dir / "test.v"
            v_file.write_text(simple_v, encoding="utf-8")
            
            # Compile
            compile_cmd = ["iverilog", "-o", str(test_dir / "test.vvp"), str(v_file)]
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert compile_result.returncode == 0, f"Compile failed: {compile_result.stderr}"
            
            # Run simulation
            sim_cmd = ["vvp", str(test_dir / "test.vvp")]
            sim_result = subprocess.run(
                sim_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert sim_result.returncode == 0
            assert "Hello from vvp" in sim_result.stdout
            
        finally:
            # Cleanup
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_verilog_syntax_check(self):
        """Test that we can compile a simple Verilog module (syntax check)."""
        from examples.verilog_eval.evaluate import _classify_compile_error
        
        # Simple valid Verilog
        valid_v = """
module test_module(
    input clk,
    input rst,
    output reg [7:0] out
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            out <= 8'h0;
        else
            out <= out + 1;
    end
endmodule
"""
        
        test_dir = Path.cwd() / "tmp_syntax_check"
        test_dir.mkdir(exist_ok=True)
        
        try:
            v_file = test_dir / "test_module.v"
            v_file.write_text(valid_v, encoding="utf-8")
            
            # Try to compile (should succeed)
            compile_cmd = ["iverilog", "-Wall", "-o", str(test_dir / "test.vvp"), str(v_file)]
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            # Compilation should succeed
            assert compile_result.returncode == 0, f"Valid Verilog failed to compile: {compile_result.stderr}"
            
            # Test error classification on a known error
            error_text = "syntax error near 'always'"
            error_class = _classify_compile_error(error_text)
            assert error_class == "syntax_error"
            
        finally:
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])