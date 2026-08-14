`timescale 1ns / 1ps
// ============================================================
// Module : systolic_array
// Project: FP4-SPARSA (v20 Final)
// ============================================================
module systolic_array (
    input  wire          clk,
    input  wire          rst,
    input  wire          sparse_en,
    input  wire          mode,             // 0 = FP4 E2M1, 1 = INT4
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
    output wire [3:0]    sat_flags,
    output wire          valid_out
);

// ── decode_fp4 function ───────────────────────────────────────
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
function [31:0] build_dec_word;
    input [15:0] packed;
    begin
        build_dec_word = { decode_fp4(packed[15:12]),
                           decode_fp4(packed[11: 8]),
                           decode_fp4(packed[ 7: 4]),
                           decode_fp4(packed[ 3: 0]) };
    end
endfunction

// ── calc_zero_mask (v20 mode-aware fix) ───────────────────────
// Returns 4-bit zero mask for 4 packed FP4/INT4 nibbles.
// Prevents INT4 -8 (4'b1000) from being misidentified as zero.
function [3:0] calc_zero_mask;
    input [15:0] raw16;
    input        mode_in;
    begin
        if (mode_in) begin
            // INT4 mode: exact 4-bit zero check
            calc_zero_mask[0] = (raw16[ 3: 0] == 4'b0000);
            calc_zero_mask[1] = (raw16[ 7: 4] == 4'b0000);
            calc_zero_mask[2] = (raw16[11: 8] == 4'b0000);
            calc_zero_mask[3] = (raw16[15:12] == 4'b0000);
        end else begin
            // FP4 E2M1 mode: 3-bit check ignoring sign bit (+0 [0000] & -0 [1000])
            calc_zero_mask[0] = (raw16[ 2: 1] == 2'b00);
            calc_zero_mask[1] = (raw16[ 6: 5] == 2'b00);
            calc_zero_mask[2] = (raw16[10: 9] == 2'b00);
            calc_zero_mask[3] = (raw16[14:13] == 2'b00);
        end
    end
endfunction

// ── [U4] Ping-pong weight buffers ────────────────────────────
reg [127:0] wdata_stage;       // Stage A: raw BRAM data registered
reg         load_lo_q;         // Stage B trigger for rows 0-1
reg         load_hi_q;         // Stage B trigger for rows 2-3

reg [31:0] weight_dec  [0:1][0:3][0:3];
reg [15:0] weight_raw  [0:1][0:3][0:3];
reg [3:0]  weight_zero [0:1][0:3][0:3];  // Pre-computed weight zero masks
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
                    weight_dec[bk][ri][ci]  <= 32'd0;
                    weight_raw[bk][ri][ci]  <= 16'd0;
                    weight_zero[bk][ri][ci] <= 4'd0;
                end
    end else begin
        if (bank_switch) active_bank <= ~active_bank;

        load_lo_q   <= load_weight_lo;
        load_hi_q   <= load_weight_hi;
        if (load_weight_lo || load_weight_hi)
            wdata_stage <= weight_data;

        if (load_lo_q) begin
            weight_dec[idle_bank][0][0]  <= build_dec_word(wdata_stage[ 15:  0]);
            weight_dec[idle_bank][0][1]  <= build_dec_word(wdata_stage[ 31: 16]);
            weight_dec[idle_bank][0][2]  <= build_dec_word(wdata_stage[ 47: 32]);
            weight_dec[idle_bank][0][3]  <= build_dec_word(wdata_stage[ 63: 48]);
            weight_dec[idle_bank][1][0]  <= build_dec_word(wdata_stage[ 79: 64]);
            weight_dec[idle_bank][1][1]  <= build_dec_word(wdata_stage[ 95: 80]);
            weight_dec[idle_bank][1][2]  <= build_dec_word(wdata_stage[111: 96]);
            weight_dec[idle_bank][1][3]  <= build_dec_word(wdata_stage[127:112]);
            weight_raw[idle_bank][0][0]  <= wdata_stage[ 15:  0];
            weight_raw[idle_bank][0][1]  <= wdata_stage[ 31: 16];
            weight_raw[idle_bank][0][2]  <= wdata_stage[ 47: 32];
            weight_raw[idle_bank][0][3]  <= wdata_stage[ 63: 48];
            weight_raw[idle_bank][1][0]  <= wdata_stage[ 79: 64];
            weight_raw[idle_bank][1][1]  <= wdata_stage[ 95: 80];
            weight_raw[idle_bank][1][2]  <= wdata_stage[111: 96];
            weight_raw[idle_bank][1][3]  <= wdata_stage[127:112];
            weight_zero[idle_bank][0][0] <= calc_zero_mask(wdata_stage[ 15:  0], mode);
            weight_zero[idle_bank][0][1] <= calc_zero_mask(wdata_stage[ 31: 16], mode);
            weight_zero[idle_bank][0][2] <= calc_zero_mask(wdata_stage[ 47: 32], mode);
            weight_zero[idle_bank][0][3] <= calc_zero_mask(wdata_stage[ 63: 48], mode);
            weight_zero[idle_bank][1][0] <= calc_zero_mask(wdata_stage[ 79: 64], mode);
            weight_zero[idle_bank][1][1] <= calc_zero_mask(wdata_stage[ 95: 80], mode);
            weight_zero[idle_bank][1][2] <= calc_zero_mask(wdata_stage[111: 96], mode);
            weight_zero[idle_bank][1][3] <= calc_zero_mask(wdata_stage[127:112], mode);
        end

        if (load_hi_q) begin
            weight_dec[idle_bank][2][0]  <= build_dec_word(wdata_stage[ 15:  0]);
            weight_dec[idle_bank][2][1]  <= build_dec_word(wdata_stage[ 31: 16]);
            weight_dec[idle_bank][2][2]  <= build_dec_word(wdata_stage[ 47: 32]);
            weight_dec[idle_bank][2][3]  <= build_dec_word(wdata_stage[ 63: 48]);
            weight_dec[idle_bank][3][0]  <= build_dec_word(wdata_stage[ 79: 64]);
            weight_dec[idle_bank][3][1]  <= build_dec_word(wdata_stage[ 95: 80]);
            weight_dec[idle_bank][3][2]  <= build_dec_word(wdata_stage[111: 96]);
            weight_dec[idle_bank][3][3]  <= build_dec_word(wdata_stage[127:112]);
            weight_raw[idle_bank][2][0]  <= wdata_stage[ 15:  0];
            weight_raw[idle_bank][2][1]  <= wdata_stage[ 31: 16];
            weight_raw[idle_bank][2][2]  <= wdata_stage[ 47: 32];
            weight_raw[idle_bank][2][3]  <= wdata_stage[ 63: 48];
            weight_raw[idle_bank][3][0]  <= wdata_stage[ 79: 64];
            weight_raw[idle_bank][3][1]  <= wdata_stage[ 95: 80];
            weight_raw[idle_bank][3][2]  <= wdata_stage[111: 96];
            weight_raw[idle_bank][3][3]  <= wdata_stage[127:112];
            weight_zero[idle_bank][2][0] <= calc_zero_mask(wdata_stage[ 15:  0], mode);
            weight_zero[idle_bank][2][1] <= calc_zero_mask(wdata_stage[ 31: 16], mode);
            weight_zero[idle_bank][2][2] <= calc_zero_mask(wdata_stage[ 47: 32], mode);
            weight_zero[idle_bank][2][3] <= calc_zero_mask(wdata_stage[ 63: 48], mode);
            weight_zero[idle_bank][3][0] <= calc_zero_mask(wdata_stage[ 79: 64], mode);
            weight_zero[idle_bank][3][1] <= calc_zero_mask(wdata_stage[ 95: 80], mode);
            weight_zero[idle_bank][3][2] <= calc_zero_mask(wdata_stage[111: 96], mode);
            weight_zero[idle_bank][3][3] <= calc_zero_mask(wdata_stage[127:112], mode);
        end
    end
end

// ── [U3] Activation pre-decode (column 0 only) ───────────────
wire [31:0] act_dec_row [0:3];
assign act_dec_row[0] = build_dec_word(act_row0);
assign act_dec_row[1] = build_dec_word(act_row1);
assign act_dec_row[2] = build_dec_word(act_row2);
assign act_dec_row[3] = build_dec_word(act_row3);

// ── Interconnect wires ────────────────────────────────────────
wire [15:0] act_wire       [0:3][0:3];   // [row][col]: raw activation into PE[r][c]
wire [31:0] act_dec_wire   [0:3][0:3];   // [row][col]: decoded activation into PE[r][c]
wire [3:0]  act_zero_wire  [0:3][0:3];   // [row][col]: activation zero mask into PE[r][c]
wire [17:0] acc_wire       [0:4][0:3];   // [row_out][col]: partial accumulator (row 4 = output)
wire        valid_wire     [0:3][0:3];
wire        sat_flag_wire  [0:3][0:3];

// Accumulator chain initialised to zero at top of each column
assign acc_wire[0][0] = 18'd0;
assign acc_wire[0][1] = 18'd0;
assign acc_wire[0][2] = 18'd0;
assign acc_wire[0][3] = 18'd0;

// Column 0 activations: sourced directly from top-level inputs
assign act_wire[0][0]      = act_row0;
assign act_wire[1][0]      = act_row1;
assign act_wire[2][0]      = act_row2;
assign act_wire[3][0]      = act_row3;

// Column 0 pre-computed activation zero masks (v20 mode-aware fix)
assign act_zero_wire[0][0] = calc_zero_mask(act_row0, mode);
assign act_zero_wire[1][0] = calc_zero_mask(act_row1, mode);
assign act_zero_wire[2][0] = calc_zero_mask(act_row2, mode);
assign act_zero_wire[3][0] = calc_zero_mask(act_row3, mode);

// Column 0 pre-decoded activations
assign act_dec_wire[0][0]  = act_dec_row[0];
assign act_dec_wire[1][0]  = act_dec_row[1];
assign act_dec_wire[2][0]  = act_dec_row[2];
assign act_dec_wire[3][0]  = act_dec_row[3];

// valid_in broadcast to all rows, column 0
assign valid_wire[0][0] = valid_in;
assign valid_wire[1][0] = valid_in;
assign valid_wire[2][0] = valid_in;
assign valid_wire[3][0] = valid_in;

// ── PE array ──────────────────────────────────────────────────
genvar r, c;
generate
    // Columns 1-3 decoded from the registered act_wire
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
                .wgt_zero_in     (weight_zero[active_bank][r][c]),
                .act_in          (act_wire[r][c]),
                .act_dec_in      (act_dec_wire[r][c]),
                .act_zero_in     (act_zero_wire[r][c]),
                .act_out         (act_wire[r][c+1]),     // feeds col+1 with 1-cycle delay
                .act_zero_out    (act_zero_wire[r][c+1]),// feeds col+1 with 1-cycle delay
                .acc_in          (acc_wire[r][c]),
                .acc_out         (acc_wire[r+1][c]),     // feeds row+1
                .valid_in        (valid_wire[r][c]),
                .valid_out       (valid_wire[r][c+1]),
                .skip_out        (),
                .wgt_zero_out    (),
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
            .wgt_zero_in     (weight_zero[active_bank][r][3]),
            .act_in          (act_wire[r][3]),
            .act_dec_in      (act_dec_wire[r][3]),
            .act_zero_in     (act_zero_wire[r][3]),
            .act_out         (),
            .act_zero_out    (),
            .acc_in          (acc_wire[r][3]),
            .acc_out         (acc_wire[r+1][3]),
            .valid_in        (valid_wire[r][3]),
            .valid_out       (),
            .skip_out        (),
            .wgt_zero_out    (),
            .sat_flag        (sat_flag_wire[r][3])
        );
    end
endgenerate

// PE instance: row 3, column 3 (corner)
pe_pipelined u_pe_r3c3 (
    .clk             (clk),
    .rst             (rst),
    .sparse_en       (sparse_en),
    .mode            (mode),
    .weight_dec_in   (weight_dec[active_bank][3][3]),
    .weight_raw_in   (weight_raw[active_bank][3][3]),
    .wgt_zero_in     (weight_zero[active_bank][3][3]),
    .act_in          (act_wire[3][3]),
    .act_dec_in      (act_dec_wire[3][3]),
    .act_zero_in     (act_zero_wire[3][3]),
    .act_out         (),
    .act_zero_out    (),
    .acc_in          (acc_wire[3][3]),
    .acc_out         (acc_wire[4][3]),
    .valid_in        (valid_wire[3][3]),
    .valid_out       (),
    .skip_out        (),
    .wgt_zero_out    (),
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

// ── valid_out: full convergence-latency shift register ───────
// Total latency = vertical (4 rows × 4 stages = 16 cycles)
//               + horizontal (3 cols × 1-cycle PE valid delay = 3 cycles)
//               = 19 cycles.
// Column 0 converges at posedge 15, column 3 at posedge 18.
// valid_out must wait for the slowest column (col 3).
reg [18:0] valid_pipe;
always @(posedge clk) begin
    if (rst) valid_pipe <= 19'd0;
    else     valid_pipe <= {valid_pipe[17:0], valid_in};
end
assign valid_out = valid_pipe[18];

endmodule
