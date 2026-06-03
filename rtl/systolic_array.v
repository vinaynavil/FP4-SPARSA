`timescale 1ns / 1ps
// ============================================================
// Module : systolic_array
// Project : FP4-SPARSA v15d - Pre-decoded weights (TPU-style)
//
//  Fix (v15d final): Separate weight_dec[31:0] and weight_raw[15:0]
//  arrays replace the corrupted single-array approach where
//  build_weight_word overwrote d0/d1 with raw FP4 bits.
//
//  PE ports updated:
//    weight_dec_in [31:0] = { dec_lane3, dec_lane2, dec_lane1, dec_lane0 }
//    weight_raw_in [15:0] = { fp4_lane3, fp4_lane2, fp4_lane1, fp4_lane0 }
//
//  Sparsity counters, adder tree, pipeline: unchanged from v15c.
// ============================================================
module systolic_array (
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
    output reg  [6:0]    zero_skip_count,
    output reg  [5:0]    sparse_act_count,
    output reg  [5:0]    sparse_wgt_count,
    output wire          valid_out
);

// ── decode_fp4 function ───────────────────────────────────────
// Mirrors fp4_decoder module exactly (FTZ included).
// FP4 E2M1: [3]=sign [2:1]=exp [0]=man, bias=1.
// exp==00 → FTZ → magnitude 0 (subnormal flushed).
// Output scaled ×2 (Q1.1): 1.0→2, 1.5→3, 2.0→4, 3.0→6, 4.0→8, 6.0→12.
function automatic signed [7:0] decode_fp4;
    input [3:0] fp4;
    reg        sign;
    reg [1:0]  exp;
    reg        man;
    reg [6:0]  mag;
    begin
        sign = fp4[3];
        exp  = fp4[2:1];
        man  = fp4[0];
        if (exp == 2'b00) begin
            mag = 7'd0;          // FTZ: subnormal → 0
        end else begin
            case ({exp, man})
                3'b010: mag = 7'd2;
                3'b011: mag = 7'd3;
                3'b100: mag = 7'd4;
                3'b101: mag = 7'd6;
                3'b110: mag = 7'd8;
                3'b111: mag = 7'd12;
                default: mag = 7'd0;
            endcase
        end
        decode_fp4 = sign ? -$signed({1'b0, mag})
                          :  $signed({1'b0, mag});
    end
endfunction

// ── Weight registers ──────────────────────────────────────────
// Separate arrays: no aliasing, no overwrite corruption.
//
// weight_dec[r][c][31:0]:
//   [31:24] = decode_fp4(lane3)  signed 8-bit
//   [23:16] = decode_fp4(lane2)
//   [15: 8] = decode_fp4(lane1)
//   [ 7: 0] = decode_fp4(lane0)
//
// weight_raw[r][c][15:0]:
//   [15:12] = raw FP4 lane3 bits
//   [11: 8] = raw FP4 lane2 bits
//   [ 7: 4] = raw FP4 lane1 bits
//   [ 3: 0] = raw FP4 lane0 bits
//   Used by PE in INT4 mode (sign-extend each nibble to 8-bit).
reg [31:0] weight_dec [0:3][0:3];
reg [15:0] weight_raw [0:3][0:3];

// Build decoded 32-bit word from 16-bit packed FP4.
// packed[15:0] = { fp4_lane3[3:0], fp4_lane2[3:0],
//                  fp4_lane1[3:0], fp4_lane0[3:0] }
function automatic [31:0] build_dec_word;
    input [15:0] packed;
    reg signed [7:0] d0, d1, d2, d3;
    begin
        d0 = decode_fp4(packed[ 3: 0]);
        d1 = decode_fp4(packed[ 7: 4]);
        d2 = decode_fp4(packed[11: 8]);
        d3 = decode_fp4(packed[15:12]);
        build_dec_word = { d3[7:0], d2[7:0], d1[7:0], d0[7:0] };
    end
endfunction

integer ri, ci;
always @(posedge clk) begin
    if (rst) begin
        for (ri = 0; ri < 4; ri = ri + 1)
            for (ci = 0; ci < 4; ci = ci + 1) begin
                weight_dec[ri][ci] <= 32'd0;
                weight_raw[ri][ci] <= 16'd0;
            end
    end else begin
        if (load_weight_lo) begin
            weight_dec[0][0] <= build_dec_word(weight_data[ 15:  0]);
            weight_dec[0][1] <= build_dec_word(weight_data[ 31: 16]);
            weight_dec[0][2] <= build_dec_word(weight_data[ 47: 32]);
            weight_dec[0][3] <= build_dec_word(weight_data[ 63: 48]);
            weight_dec[1][0] <= build_dec_word(weight_data[ 79: 64]);
            weight_dec[1][1] <= build_dec_word(weight_data[ 95: 80]);
            weight_dec[1][2] <= build_dec_word(weight_data[111: 96]);
            weight_dec[1][3] <= build_dec_word(weight_data[127:112]);

            weight_raw[0][0] <= weight_data[ 15:  0];
            weight_raw[0][1] <= weight_data[ 31: 16];
            weight_raw[0][2] <= weight_data[ 47: 32];
            weight_raw[0][3] <= weight_data[ 63: 48];
            weight_raw[1][0] <= weight_data[ 79: 64];
            weight_raw[1][1] <= weight_data[ 95: 80];
            weight_raw[1][2] <= weight_data[111: 96];
            weight_raw[1][3] <= weight_data[127:112];
        end
        if (load_weight_hi) begin
            weight_dec[2][0] <= build_dec_word(weight_data[ 15:  0]);
            weight_dec[2][1] <= build_dec_word(weight_data[ 31: 16]);
            weight_dec[2][2] <= build_dec_word(weight_data[ 47: 32]);
            weight_dec[2][3] <= build_dec_word(weight_data[ 63: 48]);
            weight_dec[3][0] <= build_dec_word(weight_data[ 79: 64]);
            weight_dec[3][1] <= build_dec_word(weight_data[ 95: 80]);
            weight_dec[3][2] <= build_dec_word(weight_data[111: 96]);
            weight_dec[3][3] <= build_dec_word(weight_data[127:112]);

            weight_raw[2][0] <= weight_data[ 15:  0];
            weight_raw[2][1] <= weight_data[ 31: 16];
            weight_raw[2][2] <= weight_data[ 47: 32];
            weight_raw[2][3] <= weight_data[ 63: 48];
            weight_raw[3][0] <= weight_data[ 79: 64];
            weight_raw[3][1] <= weight_data[ 95: 80];
            weight_raw[3][2] <= weight_data[111: 96];
            weight_raw[3][3] <= weight_data[127:112];
        end
    end
end

// ── Interconnect wires ────────────────────────────────────────
wire [15:0] act_wire       [0:3][0:3];
wire [17:0] acc_wire       [0:4][0:3];
wire        valid_wire     [0:3][0:3];
wire        skip_wire      [0:3][0:3];
wire        act_zero_wire  [0:3][0:3];
wire        wgt_zero_wire  [0:3][0:3];
wire [2:0]  lane_skip_wire [0:3][0:3];

assign acc_wire[0][0] = 18'd0;
assign acc_wire[0][1] = 18'd0;
assign acc_wire[0][2] = 18'd0;
assign acc_wire[0][3] = 18'd0;

assign act_wire[0][0] = act_row0;
assign act_wire[1][0] = act_row1;
assign act_wire[2][0] = act_row2;
assign act_wire[3][0] = act_row3;

assign valid_wire[0][0] = valid_in;
assign valid_wire[1][0] = valid_in;
assign valid_wire[2][0] = valid_in;
assign valid_wire[3][0] = valid_in;

// ── PE array ──────────────────────────────────────────────────
wire valid_out_r3c3;

genvar r, c;
generate
    for (r = 0; r < 4; r = r + 1) begin : row
        for (c = 0; c < 3; c = c + 1) begin : col
            pe_pipelined u_pe (
                .clk             (clk),
                .rst             (rst),
                .sparse_en       (sparse_en),
                .mode            (mode),
                .weight_dec_in   (weight_dec[r][c]),
                .weight_raw_in   (weight_raw[r][c]),
                .act_in          (act_wire[r][c]),
                .act_out         (act_wire[r][c+1]),
                .acc_in          (acc_wire[r][c]),
                .acc_out         (acc_wire[r+1][c]),
                .valid_in        (valid_wire[r][c]),
                .valid_out       (valid_wire[r][c+1]),
                .skip_out        (skip_wire[r][c]),
                .act_zero_out    (act_zero_wire[r][c]),
                .wgt_zero_out    (wgt_zero_wire[r][c]),
                .lane_skip_count (lane_skip_wire[r][c])
            );
        end
    end

    for (r = 0; r < 3; r = r + 1) begin : row_c3
        pe_pipelined u_pe_c3 (
            .clk             (clk),
            .rst             (rst),
            .sparse_en       (sparse_en),
            .mode            (mode),
            .weight_dec_in   (weight_dec[r][3]),
            .weight_raw_in   (weight_raw[r][3]),
            .act_in          (act_wire[r][3]),
            .act_out         (),
            .acc_in          (acc_wire[r][3]),
            .acc_out         (acc_wire[r+1][3]),
            .valid_in        (valid_wire[r][3]),
            .valid_out       (),
            .skip_out        (skip_wire[r][3]),
            .act_zero_out    (act_zero_wire[r][3]),
            .wgt_zero_out    (wgt_zero_wire[r][3]),
            .lane_skip_count (lane_skip_wire[r][3])
        );
    end

    pe_pipelined u_pe_r3c3 (
        .clk             (clk),
        .rst             (rst),
        .sparse_en       (sparse_en),
        .mode            (mode),
        .weight_dec_in   (weight_dec[3][3]),
        .weight_raw_in   (weight_raw[3][3]),
        .act_in          (act_wire[3][3]),
        .act_out         (),
        .acc_in          (acc_wire[3][3]),
        .acc_out         (acc_wire[4][3]),
        .valid_in        (valid_wire[3][3]),
        .valid_out       (valid_out_r3c3),
        .skip_out        (skip_wire[3][3]),
        .act_zero_out    (act_zero_wire[3][3]),
        .wgt_zero_out    (wgt_zero_wire[3][3]),
        .lane_skip_count (lane_skip_wire[3][3])
    );
endgenerate

// ── Output assignments ────────────────────────────────────────
assign result_col0 = acc_wire[4][0];
assign result_col1 = acc_wire[4][1];
assign result_col2 = acc_wire[4][2];
assign result_col3 = acc_wire[4][3];
assign valid_out   = valid_out_r3c3;

// ── Sparsity counters - 2-stage pipelined adder tree ─────────
// (Unchanged from v15c)
reg [4:0] skip_row_sum [0:3];
reg [2:0] act_row_sum  [0:3];
reg [2:0] wgt_row_sum  [0:3];

always @(posedge clk) begin
    if (rst) begin
        skip_row_sum[0] <= 5'd0; skip_row_sum[1] <= 5'd0;
        skip_row_sum[2] <= 5'd0; skip_row_sum[3] <= 5'd0;
        act_row_sum[0]  <= 3'd0; act_row_sum[1]  <= 3'd0;
        act_row_sum[2]  <= 3'd0; act_row_sum[3]  <= 3'd0;
        wgt_row_sum[0]  <= 3'd0; wgt_row_sum[1]  <= 3'd0;
        wgt_row_sum[2]  <= 3'd0; wgt_row_sum[3]  <= 3'd0;
    end else begin
        skip_row_sum[0] <= {2'd0, lane_skip_wire[0][0]} + {2'd0, lane_skip_wire[0][1]} +
                           {2'd0, lane_skip_wire[0][2]} + {2'd0, lane_skip_wire[0][3]};
        skip_row_sum[1] <= {2'd0, lane_skip_wire[1][0]} + {2'd0, lane_skip_wire[1][1]} +
                           {2'd0, lane_skip_wire[1][2]} + {2'd0, lane_skip_wire[1][3]};
        skip_row_sum[2] <= {2'd0, lane_skip_wire[2][0]} + {2'd0, lane_skip_wire[2][1]} +
                           {2'd0, lane_skip_wire[2][2]} + {2'd0, lane_skip_wire[2][3]};
        skip_row_sum[3] <= {2'd0, lane_skip_wire[3][0]} + {2'd0, lane_skip_wire[3][1]} +
                           {2'd0, lane_skip_wire[3][2]} + {2'd0, lane_skip_wire[3][3]};

        act_row_sum[0] <= act_zero_wire[0][0] + act_zero_wire[0][1] +
                          act_zero_wire[0][2] + act_zero_wire[0][3];
        act_row_sum[1] <= act_zero_wire[1][0] + act_zero_wire[1][1] +
                          act_zero_wire[1][2] + act_zero_wire[1][3];
        act_row_sum[2] <= act_zero_wire[2][0] + act_zero_wire[2][1] +
                          act_zero_wire[2][2] + act_zero_wire[2][3];
        act_row_sum[3] <= act_zero_wire[3][0] + act_zero_wire[3][1] +
                          act_zero_wire[3][2] + act_zero_wire[3][3];

        wgt_row_sum[0] <= wgt_zero_wire[0][0] + wgt_zero_wire[0][1] +
                          wgt_zero_wire[0][2] + wgt_zero_wire[0][3];
        wgt_row_sum[1] <= wgt_zero_wire[1][0] + wgt_zero_wire[1][1] +
                          wgt_zero_wire[1][2] + wgt_zero_wire[1][3];
        wgt_row_sum[2] <= wgt_zero_wire[2][0] + wgt_zero_wire[2][1] +
                          wgt_zero_wire[2][2] + wgt_zero_wire[2][3];
        wgt_row_sum[3] <= wgt_zero_wire[3][0] + wgt_zero_wire[3][1] +
                          wgt_zero_wire[3][2] + wgt_zero_wire[3][3];
    end
end

always @(posedge clk) begin
    if (rst) begin
        zero_skip_count  <= 7'd0;
        sparse_act_count <= 6'd0;
        sparse_wgt_count <= 6'd0;
    end else begin
        zero_skip_count  <= {2'd0, skip_row_sum[0]} + {2'd0, skip_row_sum[1]} +
                            {2'd0, skip_row_sum[2]} + {2'd0, skip_row_sum[3]};
        sparse_act_count <= {3'd0, act_row_sum[0]}  + {3'd0, act_row_sum[1]}  +
                            {3'd0, act_row_sum[2]}  + {3'd0, act_row_sum[3]};
        sparse_wgt_count <= {3'd0, wgt_row_sum[0]}  + {3'd0, wgt_row_sum[1]}  +
                            {3'd0, wgt_row_sum[2]}  + {3'd0, wgt_row_sum[3]};
    end
end

endmodule