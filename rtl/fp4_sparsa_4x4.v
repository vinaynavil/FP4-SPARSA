`timescale 1ns / 1ps
// ============================================================
// Module : fp4_sparsa_4x4  (top-level wrapper - v15b)
// Project : FP4-SPARSA 4x4 Systolic Array Accelerator
//
//  IO pin fix: 256-bit weight_data split into two-phase
//  128-bit load (load_weight_lo / load_weight_hi).
//  Total IO: 287 pins  (was 414, limit 400)
// ============================================================
module fp4_sparsa_4x4 (
    input  wire          clk,
    input  wire          rst,
    input  wire          sparse_en,
    input  wire          mode,
    input  wire          load_weight_lo,
    input  wire          load_weight_hi,
    input  wire [127:0]  weight_data,
    input  wire [15:0]   act_row0,
    input  wire [15:0]   act_row1,
    input  wire [15:0]   act_row2,
    input  wire [15:0]   act_row3,
    input  wire          valid_in,
    output wire [17:0]   result_col0,
    output wire [17:0]   result_col1,
    output wire [17:0]   result_col2,
    output wire [17:0]   result_col3,
    output wire [6:0]    zero_skip_count,
    output wire [5:0]    sparse_act_count,
    output wire [5:0]    sparse_wgt_count,
    output wire          valid_out
);

    systolic_array u_array (
        .clk              (clk),
        .rst              (rst),
        .sparse_en        (sparse_en),
        .mode             (mode),
        .load_weight_lo   (load_weight_lo),
        .load_weight_hi   (load_weight_hi),
        .weight_data      (weight_data),
        .act_row0         (act_row0),
        .act_row1         (act_row1),
        .act_row2         (act_row2),
        .act_row3         (act_row3),
        .valid_in         (valid_in),
        .result_col0      (result_col0),
        .result_col1      (result_col1),
        .result_col2      (result_col2),
        .result_col3      (result_col3),
        .zero_skip_count  (zero_skip_count),
        .sparse_act_count (sparse_act_count),
        .sparse_wgt_count (sparse_wgt_count),
        .valid_out        (valid_out)
    );

endmodule