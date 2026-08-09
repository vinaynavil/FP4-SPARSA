`timescale 1ns / 1ps
// ============================================================
// Module : act_fifo
// Project : FP4-SPARSA 4x4 Systolic Array Accelerator (v20 Final)
//
// Synchronous FIFO for activation inputs.
// Decouples the data producer (CPU/testbench) from the
// systolic array consumer.
//
// Parameters:
//   DEPTH = 64  : number of entries (must be power of 2)
//   WIDTH = 64  : 4 rows × 16 bits packed internally
//
// Write side (producer):
//   wr_en              : push one activation set
//   wr_row0/1/2/3      : four 16-bit activation rows
//   full               : asserted when FIFO cannot accept more data
//                        producer must NOT write when full=1
//
// Read side (consumer = systolic array):
//   rd_en              : pop one activation set (assert when array ready)
//   rd_row0/1/2/3      : four 16-bit activation rows (registered output)
//   rd_valid           : high for one cycle after rd_en, qualifies rd_row outputs
//   empty              : asserted when no data available
//                        rd_en while empty is a no-op (rd_valid stays low)
//
// Implementation:
//   64-entry × 64-bit distributed RAM (LUTRAM).
//   4-bit gray-code-free pointers (straight binary, single clock domain).
//   Full  : (wr_ptr[3:0] == rd_ptr[3:0]) && (wr_ptr[4] != rd_ptr[4])
//   Empty : wr_ptr == rd_ptr
//   This is a standard single-clock FIFO; no CDC needed.
//
// Timing:
//   All outputs registered. rd_valid lags rd_en by 1 cycle.
//   LUTRAM read is synchronous — no combinatorial path from
//   rd_ptr to rd_row outputs. Safe at 350 MHz.
// ============================================================
module act_fifo #(
    parameter DEPTH = 16   // must be power of 2
) (
    input  wire        clk,
    input  wire        rst,

    // Write port
    input  wire        wr_en,
    input  wire [15:0] wr_row0,
    input  wire [15:0] wr_row1,
    input  wire [15:0] wr_row2,
    input  wire [15:0] wr_row3,
    output wire        full,

    // Read port
    input  wire        rd_en,
    output reg  [15:0] rd_row0,
    output reg  [15:0] rd_row1,
    output reg  [15:0] rd_row2,
    output reg  [15:0] rd_row3,
    output reg         rd_valid,
    output wire        empty
);

    localparam PTR_W = $clog2(DEPTH) + 1;  // extra bit for full/empty disambiguation

    // ── Storage ──────────────────────────────────────────────
    (* ram_style = "distributed" *)
    reg [63:0] mem [0:DEPTH-1];

    // ── Pointers ─────────────────────────────────────────────
    reg [PTR_W-1:0] wr_ptr;
    reg [PTR_W-1:0] rd_ptr;

    // ── Status flags ─────────────────────────────────────────
    // Empty : all pointer bits equal
    // Full  : lower bits equal, MSBs differ (one full wrap difference)
    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[PTR_W-2:0] == rd_ptr[PTR_W-2:0]) &&
                   (wr_ptr[PTR_W-1]   != rd_ptr[PTR_W-1]);

    // ── Write logic ──────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            wr_ptr <= {PTR_W{1'b0}};
        end else if (wr_en && !full) begin
            mem[wr_ptr[PTR_W-2:0]] <= {wr_row3, wr_row2, wr_row1, wr_row0};
            wr_ptr <= wr_ptr + {{PTR_W-1{1'b0}}, 1'b1};
        end
    end

    // ── Read logic (registered output) ───────────────────────
    // rd_valid pulses one cycle after rd_en when data was available.
    // rd_row0-3 hold the popped values until the next valid read.
    always @(posedge clk) begin
        if (rst) begin
            rd_ptr   <= {PTR_W{1'b0}};
            rd_valid <= 1'b0;
            rd_row0  <= 16'd0;
            rd_row1  <= 16'd0;
            rd_row2  <= 16'd0;
            rd_row3  <= 16'd0;
        end else begin
            rd_valid <= 1'b0;  // default: no valid output
            if (rd_en && !empty) begin
                // Registered read: latch current rd_ptr entry, advance pointer
                rd_row0  <= mem[rd_ptr[PTR_W-2:0]][15: 0];
                rd_row1  <= mem[rd_ptr[PTR_W-2:0]][31:16];
                rd_row2  <= mem[rd_ptr[PTR_W-2:0]][47:32];
                rd_row3  <= mem[rd_ptr[PTR_W-2:0]][63:48];
                rd_ptr   <= rd_ptr + {{PTR_W-1{1'b0}}, 1'b1};
                rd_valid <= 1'b1;
            end
        end
    end

endmodule
