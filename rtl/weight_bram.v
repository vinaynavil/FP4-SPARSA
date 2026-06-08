`timescale 1ns / 1ps
// ============================================================
// Module : weight_bram
// Project : FP4-SPARSA 4x4 Systolic Array Accelerator
//
// Dual-port weight buffer - Xilinx BRAM36 inference.
//
// Memory layout (8 words × 128-bit):
//   Addr 0-3 : weight_dec rows 0-3, one word per row
//              [31:0]=col0  [63:32]=col1  [95:64]=col2  [127:96]=col3
//   Addr 4-7 : weight_raw rows 0-3, lower 64 bits used
//              [15:0]=col0  [31:16]=col1  [47:32]=col2  [63:48]=col3
//              [127:64] unused
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

    // Port B - systolic array read (1-cycle latency)
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

    // Port B: synchronous registered read (BRAM OREG style, 1-cycle latency)
    always @(posedge clk) begin
        dout_b <= mem[addr_b];
    end

endmodule