// EVOLVE-BLOCK-START
module TopModule (
  input clk,
  input reset,
  output reg [31:0] q
);

  // 32-bit Galois LFSR with taps at positions 32, 22, 2, 1.
  // Active-high synchronous reset to 32'h1.
  always @(posedge clk) begin
    if (reset)
      q <= 32'h1;
    else begin
      q <= {q[0], q[31:1]};
    end
  end

endmodule
// EVOLVE-BLOCK-END
