`timescale 1ns / 1ps
// ============================================================================
// Module      : weight_bram
// Project     : FP4-SPARSA (FP4 Sparse Precision Accelerator using a Reconfigurable Systolic Array)
// Version     : 2.0
// ============================================================================
//
// Dual-port weight buffer - Xilinx BRAM36 inference.
//
// Memory layout (8 words × 128-bit capacity, 2 words actively used):
//   Only addr 0 and addr 4 are written by the top-level controller.
//   Remaining 6 addresses (1-3, 5-7) are unused — reserved headroom for
//   extension to larger weight matrices without changing the interface.
//
//   Addr 0 : raw packed weight nibbles, rows 0-1 (lo half of 4x4 array)
//              [15:0]=col0  [31:16]=col1  [47:32]=col2  [63:48]=col3 (row0, low 64b)
//              [79:64]=col0 [95:80]=col1  [111:96]=col2 [127:112]=col3 (row1, high 64b)
//   Addr 4 : raw packed weight nibbles, rows 2-3 (hi half of 4x4 array), same layout
//   Decoding (FP4→INT8) happens combinationally in systolic_array.v via
//   build_dec_word() after read-out - nothing decoded is stored in this BRAM.
//
// Usage pattern:
//   Weights are written once per inference load via the AXI-Lite CPU path.
//   Port B reads each word back exactly once (2 cycles after the write) and
//   the top-level pipeline latches the data into systolic_array weight registers.
//   This BRAM is idle during all MAC computation cycles.
//
// Port A : AXI-Lite CPU write path (synchronous write)
// Port B : Systolic array read path (1-cycle registered output)
//
// Implementation Notes:
//   - Simple dual-port (SDP): Port A is write-only (CPU path), Port B is read-only (array path).
//   - Memory contents are initialized via CPU AXI-Lite writes (Port A).
//   - Port B provides synchronous 1-cycle registered read latency.
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