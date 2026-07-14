from shinka.edit import apply_diff_patch, apply_full_patch
from shinka.utils.languages import (
    get_code_fence_languages,
    get_evolve_comment_prefix,
    get_language_extension,
    normalize_language,
)


def test_apply_diff_patch_supports_verilog(tmp_path):
    original_content = """// EVOLVE-BLOCK-START
module TopModule (
  input clk,
  input reset,
  output reg [31:0] q
);
  always @(posedge clk) begin
    if (reset)
      q <= 32'h1;
    else
      q <= {q[0], q[31:1]};
  end
endmodule
// EVOLVE-BLOCK-END"""

    patch_content = """// EVOLVE-BLOCK-START
<<<<<<< SEARCH
      q <= {q[0], q[31:1]};
=======
      q <= {q[0] ^ q[31], q[31:22], q[21] ^ q[0], q[20:2], q[1] ^ q[0], q[0] ^ q[0]};
>>>>>>> REPLACE
// EVOLVE-BLOCK-END"""

    patch_dir = tmp_path / "verilog_diff_patch"
    result = apply_diff_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="verilog",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "q[21] ^ q[0]" in updated_content
    assert output_path == patch_dir / "main.sv"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "original.sv").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "q[21] ^ q[0]" in patch_txt

    search_replace_txt = (patch_dir / "search_replace.txt").read_text("utf-8")
    assert "EVOLVE-BLOCK-START" not in search_replace_txt
    assert "EVOLVE-BLOCK-END" not in search_replace_txt


def test_apply_full_patch_supports_verilog_and_sv_fence(tmp_path):
    original_content = """// EVOLVE-BLOCK-START
module TopModule (
  input clk,
  output reg q
);
  always @(posedge clk)
    q <= 1'b0;
endmodule
// EVOLVE-BLOCK-END
"""

    patch_content = """```sv
// EVOLVE-BLOCK-START
module TopModule (
  input clk,
  output reg q
);
  always @(posedge clk)
    q <= 1'b1;
endmodule
// EVOLVE-BLOCK-END
```"""

    patch_dir = tmp_path / "verilog_full_patch"
    result = apply_full_patch(
        patch_str=patch_content,
        original_str=original_content,
        patch_dir=patch_dir,
        language="verilog",
        verbose=False,
    )
    updated_content, num_applied, output_path, error, patch_txt, diff_path = result

    assert error is None
    assert num_applied == 1
    assert "q <= 1'b1" in updated_content
    assert output_path == patch_dir / "main.sv"
    assert output_path is not None and output_path.exists()
    assert (patch_dir / "rewrite.txt").exists()
    assert (patch_dir / "original.sv").exists()
    assert diff_path == patch_dir / "edit.diff"
    assert diff_path is not None and diff_path.exists()
    assert patch_txt is not None and "q <= 1'b1" in patch_txt


def test_language_helpers_support_verilog_aliases():
    assert normalize_language("verilog") == "verilog"
    assert normalize_language("sv") == "verilog"
    assert normalize_language("systemverilog") == "verilog"
    assert normalize_language("sverilog") == "verilog"
    assert normalize_language("Verilog") == "verilog"
    assert get_language_extension("verilog") == "sv"
    assert get_language_extension("sv") == "sv"
    assert get_evolve_comment_prefix("verilog") == "//"
    assert get_evolve_comment_prefix("systemverilog") == "//"

    fences = get_code_fence_languages("sv")
    assert fences[0] == "sv"
    assert "verilog" in fences
    assert "systemverilog" in fences
