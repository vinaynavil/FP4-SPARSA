`timescale 1ns/1ps
// ============================================================
// Module : pe_pipelined
// Project : FP4-SPARSA v19
//
// Pipeline: 5 cycles end-to-end
//   Stage 1  : operand register (w_op, a_op, acc, lane_zero)
//   Stage 2  : 4 × 8×8 signed multiply → prod_s2[0:3]
//   Stage 3a : Wallace tree level-1 partial sums (registered)
//   Stage 3b : accumulate + saturate → acc_s3 (registered)
//   Output   : acc_out = acc_s3 (wire, 0 extra cycles)
//
// FIX vs v16:
//   - act_out: was latching act_pass_s3a (Stage 3a output),
//     giving 4 cycles of delay instead of the 1-cycle skew
//     required for systolic left-to-right propagation.
//     FIXED: act_out now latches act_in directly in Stage 1
//     (act_pass_s1), aligned with the operand pipeline.
//     This is the correct behaviour for a weight-stationary
//     systolic array where activations pass through in 1 cycle.
//   - valid_out: was delayed one extra cycle beyond acc_out
//     (valid_out ← valid_s3 ← valid_s3a; acc_out ← acc_s3).
//     FIXED: valid_out now registered alongside acc_s3 in the
//     same always block so they are cycle-aligned.
//   - sat_flag: now captures sat_overflow from Stage 3b in the
//     same register stage as acc_s3 (was already correct but
//     comment clarified).
//   - Removed unused lane_zero_s2 pipeline (was carried through
//     Stage 2 but never read in Stage 3a/3b - dead registers,
//     saves 4 FFs).
//   - sparse_en_s2 pipeline removed: sparse_en gating already
//     applied in Stage 2 multiply using sparse_en_s1 (correct).
//     The additional s2 copy was never used downstream.
// ============================================================
module pe_pipelined (
    input  wire        clk,
    input  wire        rst,
    input  wire        sparse_en,
    input  wire        mode,
    input  wire [31:0] weight_dec_in,
    input  wire [15:0] weight_raw_in,
    input  wire [15:0] act_in,
    input  wire [31:0] act_dec_in,
    output reg  [15:0] act_out,         // 1-cycle delayed act_in (systolic pass-through)
    input  wire [17:0] acc_in,
    output wire [17:0] acc_out,
    input  wire        valid_in,
    output reg         valid_out,       // aligned with acc_out (5 cycles after valid_in)
    output wire        skip_out,
    output wire        act_zero_out,
    output wire        wgt_zero_out,
    output wire [2:0]  lane_skip_count,
    output reg         sat_flag
);

    // ── Operand decode wires ─────────────────────────────────
    wire signed [7:0] w_dec [0:3];
    assign w_dec[0] = $signed(weight_dec_in[ 7: 0]);
    assign w_dec[1] = $signed(weight_dec_in[15: 8]);
    assign w_dec[2] = $signed(weight_dec_in[23:16]);
    assign w_dec[3] = $signed(weight_dec_in[31:24]);

    wire [3:0] a_lane [0:3];
    assign a_lane[0] = act_in[ 3: 0];
    assign a_lane[1] = act_in[ 7: 4];
    assign a_lane[2] = act_in[11: 8];
    assign a_lane[3] = act_in[15:12];

    wire signed [7:0] a_dec [0:3];
    assign a_dec[0] = $signed(act_dec_in[ 7: 0]);
    assign a_dec[1] = $signed(act_dec_in[15: 8]);
    assign a_dec[2] = $signed(act_dec_in[23:16]);
    assign a_dec[3] = $signed(act_dec_in[31:24]);

    wire [3:0] w_lane_raw [0:3];
    assign w_lane_raw[0] = weight_raw_in[ 3: 0];
    assign w_lane_raw[1] = weight_raw_in[ 7: 4];
    assign w_lane_raw[2] = weight_raw_in[11: 8];
    assign w_lane_raw[3] = weight_raw_in[15:12];

    // INT4 sign-extension
    genvar ln;
    wire signed [7:0] w_int4 [0:3];
    wire signed [7:0] a_int4 [0:3];
    generate
        for (ln = 0; ln < 4; ln = ln + 1) begin : int4_lanes
            assign w_int4[ln] = {{4{w_lane_raw[ln][3]}}, w_lane_raw[ln]};
            assign a_int4[ln] = {{4{a_lane[ln][3]}},     a_lane[ln]};
        end
    endgenerate

    // Mode mux: FP4-decoded (mode=0) vs INT4 raw (mode=1)
    wire signed [7:0] w_op [0:3];
    wire signed [7:0] a_op [0:3];
    generate
        for (ln = 0; ln < 4; ln = ln + 1) begin : op_mux
            assign w_op[ln] = mode ? w_int4[ln] : w_dec[ln];
            assign a_op[ln] = mode ? a_int4[ln] : a_dec[ln];
        end
    endgenerate

    // ── Sparsity detection (combinatorial) ──────────────────
    // Zero detection uses [2:0] of raw FP4/INT4 - exponent+mantissa
    // (sign bit excluded so -0 treated as sparse, matching FTZ).
    wire act_zero_lane [0:3];
    wire wgt_zero_lane [0:3];
    generate
        for (ln = 0; ln < 4; ln = ln + 1) begin : g_sparse
            assign act_zero_lane[ln] = (a_lane[ln][2:0]     == 3'b000);
            assign wgt_zero_lane[ln] = (w_lane_raw[ln][2:0] == 3'b000);
        end
    endgenerate

    wire lane_zero [0:3];
    generate
        for (ln = 0; ln < 4; ln = ln + 1) begin : g_lane_zero
            assign lane_zero[ln] = act_zero_lane[ln] | wgt_zero_lane[ln];
        end
    endgenerate

    wire act_zero_all = act_zero_lane[0] & act_zero_lane[1] &
                        act_zero_lane[2] & act_zero_lane[3];
    wire wgt_zero_all = wgt_zero_lane[0] & wgt_zero_lane[1] &
                        wgt_zero_lane[2] & wgt_zero_lane[3];

    assign act_zero_out    = act_zero_all;
    assign wgt_zero_out    = wgt_zero_all;
    assign skip_out        = sparse_en & (lane_zero[0] | lane_zero[1] |
                                          lane_zero[2] | lane_zero[3]);
    assign lane_skip_count = sparse_en ?
        ({2'b00, lane_zero[0]} + {2'b00, lane_zero[1]} +
         {2'b00, lane_zero[2]} + {2'b00, lane_zero[3]}) : 3'd0;

    // ── Stage 1: operand register ────────────────────────────
    reg signed [7:0]  w_s1 [0:3];
    reg signed [7:0]  a_s1 [0:3];
    (* shreg_extract = "no" *) reg signed [17:0] acc_s1;
    (* shreg_extract = "no" *) reg               valid_s1;
    reg               lane_zero_s1 [0:3];
    reg               sparse_en_s1;

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 4; i = i + 1) begin
                w_s1[i]         <= 8'sd0;
                a_s1[i]         <= 8'sd0;
                lane_zero_s1[i] <= 1'b0;
            end
            acc_s1       <= 18'sd0;
            valid_s1     <= 1'b0;
            act_out      <= 16'd0;   // ← systolic pass-through: 1-cycle delay on act_in
            sparse_en_s1 <= 1'b0;
        end else begin
            for (i = 0; i < 4; i = i + 1) begin
                w_s1[i]         <= w_op[i];
                a_s1[i]         <= a_op[i];
                lane_zero_s1[i] <= lane_zero[i];
            end
            acc_s1       <= acc_in;
            valid_s1     <= valid_in;
            act_out      <= act_in;
            sparse_en_s1 <= sparse_en;
        end
    end

    // ── Stage 2: 4 × signed multiply ────────────────────────
    (* use_dsp = "yes" *) reg signed [16:0] prod_s2 [0:3];
    reg signed [17:0] acc_s2;
    reg               valid_s2;

    always @(posedge clk) begin
        if (rst) begin
            prod_s2[0] <= 17'sd0; prod_s2[1] <= 17'sd0;
            prod_s2[2] <= 17'sd0; prod_s2[3] <= 17'sd0;
            acc_s2  <= 18'sd0;
            valid_s2 <= 1'b0;
        end else begin
            prod_s2[0] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[0]))
                          ? 17'sd0 : w_s1[0] * a_s1[0];
            prod_s2[1] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[1]))
                          ? 17'sd0 : w_s1[1] * a_s1[1];
            prod_s2[2] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[2]))
                          ? 17'sd0 : w_s1[2] * a_s1[2];
            prod_s2[3] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[3]))
                          ? 17'sd0 : w_s1[3] * a_s1[3];
            acc_s2  <= acc_s1;
            valid_s2 <= valid_s1;
        end
    end

    // ── Stage 3a: Wallace tree level-1 (registered) ─────────
    // Two adder pairs → two 18-bit partial sums, registered to
    // break the DSP MREG=1 critical path.
    wire signed [17:0] wt_l1_hi_comb =
        $signed({prod_s2[0][16], prod_s2[0]}) +
        $signed({prod_s2[1][16], prod_s2[1]});

    wire signed [17:0] wt_l1_lo_comb =
        $signed({prod_s2[2][16], prod_s2[2]}) +
        $signed({prod_s2[3][16], prod_s2[3]});

    reg signed [17:0] wt_l1_hi_s3a;
    reg signed [17:0] wt_l1_lo_s3a;
    reg signed [17:0] acc_s3a;
    reg               valid_s3a;

    always @(posedge clk) begin
        if (rst) begin
            wt_l1_hi_s3a <= 18'sd0;
            wt_l1_lo_s3a <= 18'sd0;
            acc_s3a      <= 18'sd0;
            valid_s3a    <= 1'b0;
        end else begin
            wt_l1_hi_s3a <= wt_l1_hi_comb;
            wt_l1_lo_s3a <= wt_l1_lo_comb;
            acc_s3a      <= acc_s2;
            valid_s3a    <= valid_s2;
        end
    end

    // ── Stage 3b: accumulate + saturate ─────────────────────
    // 19-bit intermediate to detect overflow before truncation.
    wire signed [18:0] mac_result =
        (~valid_s3a)
            ? $signed({acc_s3a[17], acc_s3a})
            : $signed({acc_s3a[17],    acc_s3a})
            + $signed({wt_l1_hi_s3a[17], wt_l1_hi_s3a})
            + $signed({wt_l1_lo_s3a[17], wt_l1_lo_s3a});

    wire sat_overflow = (mac_result[18] != mac_result[17]);
    wire [17:0] sat_result = sat_overflow
        ? (mac_result[18] ? 18'h20000 : 18'h1FFFF)
        : mac_result[17:0];

    reg signed [17:0] acc_s3;

    always @(posedge clk) begin
        if (rst) begin
            acc_s3    <= 18'sd0;
            valid_out <= 1'b0;
            sat_flag  <= 1'b0;
        end else begin
            // Only update acc_s3 when valid data exits the pipeline.
            // This preserves the last computed result after valid_in goes low.
            if (valid_s3a) begin
                acc_s3   <= sat_result;
                sat_flag <= sat_overflow;
            end
            valid_out <= valid_s3a;
        end
    end

    assign acc_out = acc_s3;

endmodule