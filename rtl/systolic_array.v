`timescale 1ns / 1ps
// ============================================================
// Module : systolic_array
// Project : FP4-SPARSA v19
//
// // checked acc_wire routing for row/col edge cases, looks right
// act_dec_wire duplicate decode - looks redundant but is needed,
// leave it (col>0 gets pre-decoded act from prev PE's act_out)
// skip_wire is dead above PE level, vivado should trim it, not
// removing the port for now
// ============================================================
module systolic_array (
    input  wire          clk,
    input  wire          rst,
    input  wire          sparse_en,
    input  wire          mode,
    input  wire          load_weight_lo,
    input  wire          load_weight_hi,
    input  wire          bank_switch,
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
    output wire [3:0]    sat_flags,
    output wire          valid_out
);

// ── decode_fp4 function ───────────────────────────────────────
// FP4 E2M1: sign[3] exp[2:1] man[0]
// Subnormals (exp=00) → FTZ (flush to zero), matches TPU/ANE behaviour.
// Decoded value is scaled ×2 to map to Q1.1 integers for DSP48E1.
function signed [7:0] decode_fp4;
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
            mag = 7'd0;  // FTZ: subnormals → 0
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

// ── build_dec_word ────────────────────────────────────────────
// Decode 4 packed FP4 nibbles into four signed 8-bit integers.
// Output [7:0]=lane0, [15:8]=lane1, [23:16]=lane2, [31:24]=lane3.
function [31:0] build_dec_word;
    input [15:0] packed;
    begin
        build_dec_word = { decode_fp4(packed[15:12]),
                           decode_fp4(packed[11: 8]),
                           decode_fp4(packed[ 7: 4]),
                           decode_fp4(packed[ 3: 0]) };
    end
endfunction

// ── [U4] Ping-pong weight buffers ────────────────────────────
// Two banks: active_bank is live (read by PEs), idle_bank is written.
// bank_switch (1-cycle pulse from AXI ctrl) swaps after new weights loaded.
//
// TIMING FIX: BRAM output (weight_data = bram_rdata_w) has 1.800ns
// propagation delay on Kintex-7. Feeding build_dec_word() directly from
// BRAM output leaves only ~1.0ns for LUT+route+setup - violates 350MHz.
//
// Solution: two-cycle weight load pipeline:
//   Cycle N   : load_weight_lo/hi pulse arrives, weight_data valid on BRAM dout_b
//   Cycle N   : Stage A - register raw weight_data into wdata_stage[] (FF, not LUT)
//   Cycle N+1 : Stage B - decode from wdata_stage (LUT input is fabric FF, ~0.1ns)
//                          write decoded result into weight_dec / weight_raw
//
// load_lo/hi pulses are also registered (load_lo_q/hi_q) to trigger Stage B.
// This adds 1 cycle to weight configuration latency only (not inference path).
reg [127:0] wdata_stage;       // Stage A: raw BRAM data registered
reg         load_lo_q;         // Stage B trigger for rows 0-1
reg         load_hi_q;         // Stage B trigger for rows 2-3

reg [31:0] weight_dec [0:1][0:3][0:3];
reg [15:0] weight_raw [0:1][0:3][0:3];
reg        active_bank;

wire idle_bank = ~active_bank;

integer ri, ci, bk;
always @(posedge clk) begin
    if (rst) begin
        active_bank <= 1'b0;
        wdata_stage <= 128'd0;
        load_lo_q   <= 1'b0;
        load_hi_q   <= 1'b0;
        for (bk = 0; bk < 2; bk = bk + 1)
            for (ri = 0; ri < 4; ri = ri + 1)
                for (ci = 0; ci < 4; ci = ci + 1) begin
                    weight_dec[bk][ri][ci] <= 32'd0;
                    weight_raw[bk][ri][ci] <= 16'd0;
                end
    end else begin
        if (bank_switch) active_bank <= ~active_bank;

        // ── Stage A: register raw BRAM output (breaks BRAM→LUT path) ──
        // weight_data is bram_rdata_w (RAMB36E1 DOBDO/DOADO, 1.800ns delay).
        // Registering it here means Stage B decode LUTs see a fabric FF
        // input (~0.1ns), giving the full 2.857ns budget for decode+route.
        load_lo_q   <= load_weight_lo;
        load_hi_q   <= load_weight_hi;
        if (load_weight_lo || load_weight_hi)
            wdata_stage <= weight_data;

        // ── Stage B: decode from registered data, write weight buffers ──
        // Weight word layout per row:
        //   [15:0]=col0  [31:16]=col1  [47:32]=col2  [63:48]=col3
        //   [79:64]=next_row_col0 ... [127:112]=next_row_col3
        if (load_lo_q) begin
            weight_dec[idle_bank][0][0] <= build_dec_word(wdata_stage[ 15:  0]);
            weight_dec[idle_bank][0][1] <= build_dec_word(wdata_stage[ 31: 16]);
            weight_dec[idle_bank][0][2] <= build_dec_word(wdata_stage[ 47: 32]);
            weight_dec[idle_bank][0][3] <= build_dec_word(wdata_stage[ 63: 48]);
            weight_dec[idle_bank][1][0] <= build_dec_word(wdata_stage[ 79: 64]);
            weight_dec[idle_bank][1][1] <= build_dec_word(wdata_stage[ 95: 80]);
            weight_dec[idle_bank][1][2] <= build_dec_word(wdata_stage[111: 96]);
            weight_dec[idle_bank][1][3] <= build_dec_word(wdata_stage[127:112]);
            weight_raw[idle_bank][0][0] <= wdata_stage[ 15:  0];
            weight_raw[idle_bank][0][1] <= wdata_stage[ 31: 16];
            weight_raw[idle_bank][0][2] <= wdata_stage[ 47: 32];
            weight_raw[idle_bank][0][3] <= wdata_stage[ 63: 48];
            weight_raw[idle_bank][1][0] <= wdata_stage[ 79: 64];
            weight_raw[idle_bank][1][1] <= wdata_stage[ 95: 80];
            weight_raw[idle_bank][1][2] <= wdata_stage[111: 96];
            weight_raw[idle_bank][1][3] <= wdata_stage[127:112];
        end

        if (load_hi_q) begin
            weight_dec[idle_bank][2][0] <= build_dec_word(wdata_stage[ 15:  0]);
            weight_dec[idle_bank][2][1] <= build_dec_word(wdata_stage[ 31: 16]);
            weight_dec[idle_bank][2][2] <= build_dec_word(wdata_stage[ 47: 32]);
            weight_dec[idle_bank][2][3] <= build_dec_word(wdata_stage[ 63: 48]);
            weight_dec[idle_bank][3][0] <= build_dec_word(wdata_stage[ 79: 64]);
            weight_dec[idle_bank][3][1] <= build_dec_word(wdata_stage[ 95: 80]);
            weight_dec[idle_bank][3][2] <= build_dec_word(wdata_stage[111: 96]);
            weight_dec[idle_bank][3][3] <= build_dec_word(wdata_stage[127:112]);
            weight_raw[idle_bank][2][0] <= wdata_stage[ 15:  0];
            weight_raw[idle_bank][2][1] <= wdata_stage[ 31: 16];
            weight_raw[idle_bank][2][2] <= wdata_stage[ 47: 32];
            weight_raw[idle_bank][2][3] <= wdata_stage[ 63: 48];
            weight_raw[idle_bank][3][0] <= wdata_stage[ 79: 64];
            weight_raw[idle_bank][3][1] <= wdata_stage[ 95: 80];
            weight_raw[idle_bank][3][2] <= wdata_stage[111: 96];
            weight_raw[idle_bank][3][3] <= wdata_stage[127:112];
        end
    end
end

// ── [U3] Activation pre-decode (column 0 only) ───────────────
// PEs in column 0 receive pre-decoded activations directly.
// PEs in columns 1-3 receive act_out from the previous column's PE
// (registered 1-cycle delay) and re-decode from that registered raw value.
wire [31:0] act_dec_row [0:3];
assign act_dec_row[0] = build_dec_word(act_row0);
assign act_dec_row[1] = build_dec_word(act_row1);
assign act_dec_row[2] = build_dec_word(act_row2);
assign act_dec_row[3] = build_dec_word(act_row3);

// ── Interconnect wires ────────────────────────────────────────
wire [15:0] act_wire       [0:3][0:3];   // [row][col]: raw activation into PE[r][c]
wire [31:0] act_dec_wire   [0:3][0:3];   // [row][col]: decoded activation into PE[r][c]
wire [17:0] acc_wire       [0:4][0:3];   // [row_out][col]: partial accumulator (row 4 = output)
wire        valid_wire     [0:3][0:3];
wire        skip_wire      [0:3][0:3];   // driven, unused above PE level (Vivado trims)
wire        act_zero_wire  [0:3][0:3];
wire        wgt_zero_wire  [0:3][0:3];
wire [2:0]  lane_skip_wire [0:3][0:3];
wire        sat_flag_wire  [0:3][0:3];

// Accumulator chain initialised to zero at top of each column
assign acc_wire[0][0] = 18'd0;
assign acc_wire[0][1] = 18'd0;
assign acc_wire[0][2] = 18'd0;
assign acc_wire[0][3] = 18'd0;

// Column 0 activations: sourced directly from top-level inputs
assign act_wire[0][0]     = act_row0;
assign act_wire[1][0]     = act_row1;
assign act_wire[2][0]     = act_row2;
assign act_wire[3][0]     = act_row3;

// Column 0 pre-decoded activations
assign act_dec_wire[0][0] = act_dec_row[0];
assign act_dec_wire[1][0] = act_dec_row[1];
assign act_dec_wire[2][0] = act_dec_row[2];
assign act_dec_wire[3][0] = act_dec_row[3];

// valid_in broadcast to all rows, column 0
// Weight-stationary: all rows process the same activation word each cycle.
assign valid_wire[0][0] = valid_in;
assign valid_wire[1][0] = valid_in;
assign valid_wire[2][0] = valid_in;
assign valid_wire[3][0] = valid_in;

// ── PE array ──────────────────────────────────────────────────
genvar r, c;
generate
    // Columns 1-3 decoded from the registered act_wire (PE act_out = act_in delayed 1 cycle)
    for (r = 0; r < 4; r = r + 1) begin : act_dec_cols
        for (c = 1; c < 4; c = c + 1) begin : col_dec
            assign act_dec_wire[r][c] = build_dec_word(act_wire[r][c]);
        end
    end

    // PE instances: rows 0-3, columns 0-2
    for (r = 0; r < 4; r = r + 1) begin : row
        for (c = 0; c < 3; c = c + 1) begin : col
            pe_pipelined u_pe (
                .clk             (clk),
                .rst             (rst),
                .sparse_en       (sparse_en),
                .mode            (mode),
                .weight_dec_in   (weight_dec[active_bank][r][c]),
                .weight_raw_in   (weight_raw[active_bank][r][c]),
                .act_in          (act_wire[r][c]),
                .act_dec_in      (act_dec_wire[r][c]),
                .act_out         (act_wire[r][c+1]),     // feeds col+1 with 1-cycle delay
                .acc_in          (acc_wire[r][c]),
                .acc_out         (acc_wire[r+1][c]),     // feeds row+1
                .valid_in        (valid_wire[r][c]),
                .valid_out       (valid_wire[r][c+1]),
                .skip_out        (skip_wire[r][c]),
                .act_zero_out    (act_zero_wire[r][c]),
                .wgt_zero_out    (wgt_zero_wire[r][c]),
                .lane_skip_count (lane_skip_wire[r][c]),
                .sat_flag        (sat_flag_wire[r][c])
            );
        end
    end

    // PE instances: rows 0-2, column 3
    for (r = 0; r < 3; r = r + 1) begin : row_c3
        pe_pipelined u_pe_c3 (
            .clk             (clk),
            .rst             (rst),
            .sparse_en       (sparse_en),
            .mode            (mode),
            .weight_dec_in   (weight_dec[active_bank][r][3]),
            .weight_raw_in   (weight_raw[active_bank][r][3]),
            .act_in          (act_wire[r][3]),
            .act_dec_in      (act_dec_wire[r][3]),
            .act_out         (),                         // last column: no right neighbour
            .acc_in          (acc_wire[r][3]),
            .acc_out         (acc_wire[r+1][3]),
            .valid_in        (valid_wire[r][3]),
            .valid_out       (),
            .skip_out        (skip_wire[r][3]),
            .act_zero_out    (act_zero_wire[r][3]),
            .wgt_zero_out    (wgt_zero_wire[r][3]),
            .lane_skip_count (lane_skip_wire[r][3]),
            .sat_flag        (sat_flag_wire[r][3])
        );
    end
endgenerate

// PE instance: row 3, column 3 (corner - no right neighbour, no row output)
pe_pipelined u_pe_r3c3 (
    .clk             (clk),
    .rst             (rst),
    .sparse_en       (sparse_en),
    .mode            (mode),
    .weight_dec_in   (weight_dec[active_bank][3][3]),
    .weight_raw_in   (weight_raw[active_bank][3][3]),
    .act_in          (act_wire[3][3]),
    .act_dec_in      (act_dec_wire[3][3]),
    .act_out         (),
    .acc_in          (acc_wire[3][3]),
    .acc_out         (acc_wire[4][3]),
    .valid_in        (valid_wire[3][3]),
    .valid_out       (),
    .skip_out        (skip_wire[3][3]),
    .act_zero_out    (act_zero_wire[3][3]),
    .wgt_zero_out    (wgt_zero_wire[3][3]),
    .lane_skip_count (lane_skip_wire[3][3]),
    .sat_flag        (sat_flag_wire[3][3])
);

// ── Output assignments ────────────────────────────────────────
assign result_col0 = acc_wire[4][0];
assign result_col1 = acc_wire[4][1];
assign result_col2 = acc_wire[4][2];
assign result_col3 = acc_wire[4][3];

// ── Saturation flags (column OR-reduction) ────────────────────
assign sat_flags[0] = sat_flag_wire[0][0] | sat_flag_wire[1][0] |
                      sat_flag_wire[2][0] | sat_flag_wire[3][0];
assign sat_flags[1] = sat_flag_wire[0][1] | sat_flag_wire[1][1] |
                      sat_flag_wire[2][1] | sat_flag_wire[3][1];
assign sat_flags[2] = sat_flag_wire[0][2] | sat_flag_wire[1][2] |
                      sat_flag_wire[2][2] | sat_flag_wire[3][2];
assign sat_flags[3] = sat_flag_wire[0][3] | sat_flag_wire[1][3] |
                      sat_flag_wire[2][3] | sat_flag_wire[3][3];

// ── valid_out: 5-stage shift register ────────────────────────
// Matches 5-cycle PE pipeline latency (Stages 1+2+3a+3b+outreg).
reg [4:0] valid_pipe;
always @(posedge clk) begin
    if (rst) valid_pipe <= 5'd0;
    else     valid_pipe <= {valid_pipe[3:0], valid_in};
end
assign valid_out = valid_pipe[4];

// ── Sparsity counters: 2-stage pipelined adder tree ──────────
// Stage 0 (pre-register): break active_bank → lane_skip → adder
//   combinatorial path that violated timing at 350 MHz.
reg [2:0] lane_skip_r [0:3][0:3];
reg       act_zero_r  [0:3][0:3];
reg       wgt_zero_r  [0:3][0:3];

integer ri2, ci2;
always @(posedge clk) begin
    if (rst) begin
        for (ri2 = 0; ri2 < 4; ri2 = ri2 + 1)
            for (ci2 = 0; ci2 < 4; ci2 = ci2 + 1) begin
                lane_skip_r[ri2][ci2] <= 3'd0;
                act_zero_r [ri2][ci2] <= 1'b0;
                wgt_zero_r [ri2][ci2] <= 1'b0;
            end
    end else begin
        for (ri2 = 0; ri2 < 4; ri2 = ri2 + 1)
            for (ci2 = 0; ci2 < 4; ci2 = ci2 + 1) begin
                lane_skip_r[ri2][ci2] <= lane_skip_wire[ri2][ci2];
                act_zero_r [ri2][ci2] <= act_zero_wire [ri2][ci2];
                wgt_zero_r [ri2][ci2] <= wgt_zero_wire [ri2][ci2];
            end
    end
end

// Stage 1: sum across columns per row (registered)
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
        skip_row_sum[0] <= {2'd0,lane_skip_r[0][0]} + {2'd0,lane_skip_r[0][1]} +
                           {2'd0,lane_skip_r[0][2]} + {2'd0,lane_skip_r[0][3]};
        skip_row_sum[1] <= {2'd0,lane_skip_r[1][0]} + {2'd0,lane_skip_r[1][1]} +
                           {2'd0,lane_skip_r[1][2]} + {2'd0,lane_skip_r[1][3]};
        skip_row_sum[2] <= {2'd0,lane_skip_r[2][0]} + {2'd0,lane_skip_r[2][1]} +
                           {2'd0,lane_skip_r[2][2]} + {2'd0,lane_skip_r[2][3]};
        skip_row_sum[3] <= {2'd0,lane_skip_r[3][0]} + {2'd0,lane_skip_r[3][1]} +
                           {2'd0,lane_skip_r[3][2]} + {2'd0,lane_skip_r[3][3]};

        act_row_sum[0] <= act_zero_r[0][0] + act_zero_r[0][1] +
                          act_zero_r[0][2] + act_zero_r[0][3];
        act_row_sum[1] <= act_zero_r[1][0] + act_zero_r[1][1] +
                          act_zero_r[1][2] + act_zero_r[1][3];
        act_row_sum[2] <= act_zero_r[2][0] + act_zero_r[2][1] +
                          act_zero_r[2][2] + act_zero_r[2][3];
        act_row_sum[3] <= act_zero_r[3][0] + act_zero_r[3][1] +
                          act_zero_r[3][2] + act_zero_r[3][3];

        wgt_row_sum[0] <= wgt_zero_r[0][0] + wgt_zero_r[0][1] +
                          wgt_zero_r[0][2] + wgt_zero_r[0][3];
        wgt_row_sum[1] <= wgt_zero_r[1][0] + wgt_zero_r[1][1] +
                          wgt_zero_r[1][2] + wgt_zero_r[1][3];
        wgt_row_sum[2] <= wgt_zero_r[2][0] + wgt_zero_r[2][1] +
                          wgt_zero_r[2][2] + wgt_zero_r[2][3];
        wgt_row_sum[3] <= wgt_zero_r[3][0] + wgt_zero_r[3][1] +
                          wgt_zero_r[3][2] + wgt_zero_r[3][3];
    end
end

// Stage 2: sum across rows (registered output to top-level)
always @(posedge clk) begin
    if (rst) begin
        zero_skip_count  <= 7'd0;
        sparse_act_count <= 6'd0;
        sparse_wgt_count <= 6'd0;
    end else begin
        zero_skip_count  <= {2'd0,skip_row_sum[0]} + {2'd0,skip_row_sum[1]} +
                            {2'd0,skip_row_sum[2]} + {2'd0,skip_row_sum[3]};
        sparse_act_count <= {3'd0,act_row_sum[0]}  + {3'd0,act_row_sum[1]}  +
                            {3'd0,act_row_sum[2]}  + {3'd0,act_row_sum[3]};
        sparse_wgt_count <= {3'd0,wgt_row_sum[0]}  + {3'd0,wgt_row_sum[1]}  +
                            {3'd0,wgt_row_sum[2]}  + {3'd0,wgt_row_sum[3]};
    end
end

endmodule