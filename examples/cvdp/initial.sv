// CVDP Problem: cvdp_copilot_lfsr_0001 (cid003, easy)
// 8-bit Galois LFSR with polynomial x^8+x^6+x^5+x+1
// Active-LOW async reset loads lfsr_seed
// EVOLVE-BLOCK-START
module lfsr_8bit(
  input        clock,
  input        reset,
  input  [7:0] lfsr_seed,
  output reg [7:0] lfsr_out
);

  always @(posedge clock or negedge reset) begin
    if (!reset)
      lfsr_out <= lfsr_seed;
    else
      lfsr_out <= {lfsr_out[6:0], lfsr_out[7]};
  end

endmodule
// EVOLVE-BLOCK-END
