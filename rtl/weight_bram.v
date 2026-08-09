`timescale 1ns / 1ps
// ============================================================
// Module : weight_bram
// Project : FP4-SPARSA 4x4 Systolic Array Accelerator (v20 Final)
//
// Dual-port weight buffer - Xilinx BRAM36 inference.
//
// Memory layout (8 words × 128-bit):
//   Addr 0-3 : raw packed weight nibbles, rows 0-1 (lo half of 4x4 array)
//              [15:0]=col0  [31:16]=col1  [47:32]=col2  [63:48]=col3 (row0, low 64b)
//              [79:64]=col0 [95:80]=col1  [111:96]=col2 [127:112]=col3 (row1, high 64b)
//   Addr 4-7 : raw packed weight nibbles, rows 2-3 (hi half of 4x4 array), same layout
//   Decoding (FP4→INT8) happens combinationally in systolic_array.v via
//   build_dec_word() after read-out - nothing decoded is stored in this BRAM.
//
// Port A : AXI-Lite CPU write path (synchronous write)
// Port B : Systolic array read path (1-cycle registered output)
//
// FIX vs original:
//   - Removed `initial` block: unsupported by Xilinx BRAM primitives
//     in bitstream flow; use an explicit reset sequence instead.
//   - Attribute placement corrected: both ram_style and
//     rw_addr_collision on contiguous lines before the reg declaration.
//   - Port B read is purely registered (matches RAMB36E1 OREG=1 mode).
// ============================================================
module weight_bram (
    input  wire         clk,

    // Port A - CPU write
    input  wire         ena_a,
    input  wire [2:0]   addr_a,
    input  wire [127:0] din_a,

    // Port B - systolic array read (gated read enable)
    input  wire         enb_b,
    input  wire [2:0]   addr_b,
    output reg  [127:0] dout_b
);

    (* ram_style = "block" *)
    (* rw_addr_collision = "no" *)
    reg [127:0] mem [0:7];

    // Port A: synchronous write
    always @(posedge clk) begin
        if (ena_a)
            mem[addr_a] <= din_a;
    end

    // Port B: synchronous registered read (gated by enb_b)
    always @(posedge clk) begin
        if (enb_b)
            dout_b <= mem[addr_b];
    end

endmodule