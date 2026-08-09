`timescale 1ns/1ps
// ============================================================
// Module : pe_pipelined (v20 Final)
// Project : FP4-SPARSA 
//
// Pipeline: 4 cycles end-to-end
//   Stage 1  : operand register + skip pre-computation
//   Stage 2  : 4 × 8×8 signed multiply → prod_s2[0:3]
//   Stage 3a : pairwise adder tree, level 1, partial sums (registered)
//   Stage 3b : accumulate + saturate → acc_s3 (registered)
//   Output   : acc_out = acc_s3 (wire, 0 extra cycles)
//
// fixed act_out - was latching from stage 3a, gave 4 cycle delay
// instead of the 1 cycle skew systolic needs. now latches act_in
// straight in stage 1. (spent way too long on this one)
// also fixed valid_out being 1 cycle behind acc_out - same root cause
// removed lane_zero_s2, wasn't even used, just extra FFs
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
//
// 4-stage pipelined Processing Element (PE).
// Stage 1: input latch + pre-computed zero mask evaluation
// Stage 2: 4 signed 8x8 multipliers (with operand isolation)
// Stage 3a: 2-stage pairwise adder tree (pipelined)
// Stage 3b: 18-bit signed accumulation & 18-bit saturation
// ============================================================
module pe_pipelined (
    input  wire        clk,
    input  wire        rst,
    input  wire        sparse_en,
    input  wire        mode,
    input  wire [31:0] weight_dec_in,
    input  wire [15:0] weight_raw_in,
    input  wire [3:0]  wgt_zero_in,     // Pre-computed weight zero mask (v20)
    input  wire [15:0] act_in,
    input  wire [31:0] act_dec_in,
    input  wire [3:0]  act_zero_in,     // Pre-computed activation zero mask (v20)
    output reg  [15:0] act_out,         // 1-cycle delayed act_in
    output reg  [3:0]  act_zero_out,    // 1-cycle delayed act_zero_in (v20)
    input  wire [17:0] acc_in,
    output wire [17:0] acc_out,
    input  wire        valid_in,
    output reg         valid_out,       // aligned with acc_out (4 cycles after valid_in)
    output wire        skip_out,
    output wire        wgt_zero_out,
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

    // ── Sparsity evaluation (v20 pre-computed masks) ─────────
    wire lane_zero [0:3];
    assign lane_zero[0] = act_zero_in[0] | wgt_zero_in[0];
    assign lane_zero[1] = act_zero_in[1] | wgt_zero_in[1];
    assign lane_zero[2] = act_zero_in[2] | wgt_zero_in[2];
    assign lane_zero[3] = act_zero_in[3] | wgt_zero_in[3];

    wire act_zero_all = act_zero_in[0] & act_zero_in[1] &
                        act_zero_in[2] & act_zero_in[3];
    wire wgt_zero_all = wgt_zero_in[0] & wgt_zero_in[1] &
                        wgt_zero_in[2] & wgt_zero_in[3];

    assign wgt_zero_out    = wgt_zero_all;
    assign skip_out        = sparse_en & (lane_zero[0] | lane_zero[1] |
                                          lane_zero[2] | lane_zero[3]);

    // ── Stage 1: operand register + skip pre-computation ─────
    reg signed [7:0]  w_s1 [0:3];
    reg signed [7:0]  a_s1 [0:3];
    (* shreg_extract = "no" *) reg signed [17:0] acc_s1;
    (* shreg_extract = "no" *) reg               valid_s1;
    reg               skip_s1 [0:3];   // pre-computed gate for Stage 2 MUX

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 4; i = i + 1) begin
                w_s1[i]    <= 8'sd0;
                a_s1[i]    <= 8'sd0;
                skip_s1[i] <= 1'b1;
            end
            acc_s1   <= 18'sd0;
            valid_s1     <= 1'b0;
            act_out      <= 16'd0;   // ← systolic pass-through: 1-cycle delay on act_in
            act_zero_out <= 4'd0;
        end else begin
            for (i = 0; i < 4; i = i + 1) begin
                // Gate operand register updates with valid_in (reduces clock switching)
                if (valid_in) begin
                    w_s1[i] <= w_op[i];
                    a_s1[i] <= a_op[i];
                end
                // Pre-compute Stage 2 gate: folds ~valid + sparse & lane_zero
                // into single FF. Stage 2 MUX select is 0 comb. LUT levels.
                skip_s1[i] <= ~valid_in | (sparse_en & lane_zero[i]);
            end
            acc_s1       <= acc_in;
            valid_s1     <= valid_in;
            act_out      <= act_in;
            act_zero_out <= act_zero_in;
        end
    end

    // ── Dual-Operand Isolation (Weight & Activation) ─────────────
    wire signed [7:0] w_iso [0:3];
    wire signed [7:0] a_iso [0:3];

    generate
        for (ln = 0; ln < 4; ln = ln + 1) begin : g_operand_isolation
            assign w_iso[ln] = (skip_s1[ln]) ? 8'sd0 : w_s1[ln];
            assign a_iso[ln] = (skip_s1[ln]) ? 8'sd0 : a_s1[ln];
        end
    endgenerate

    // ── Stage 2: 4 × signed multiply ────────────────────────
    (* use_dsp = "yes" *) reg signed [16:0] prod_s2 [0:3];
    reg signed [17:0] acc_s2;
    reg               valid_s2;

    always @(posedge clk) begin
        if (rst) begin
            prod_s2[0] <= 17'sd0; prod_s2[1] <= 17'sd0;
            prod_s2[2] <= 17'sd0; prod_s2[3] <= 17'sd0;
            acc_s2   <= 18'sd0;
            valid_s2 <= 1'b0;
        end else begin
            prod_s2[0] <= valid_s1 ? (w_iso[0] * a_iso[0]) : 17'sd0;
            prod_s2[1] <= valid_s1 ? (w_iso[1] * a_iso[1]) : 17'sd0;
            prod_s2[2] <= valid_s1 ? (w_iso[2] * a_iso[2]) : 17'sd0;
            prod_s2[3] <= valid_s1 ? (w_iso[3] * a_iso[3]) : 17'sd0;
            acc_s2   <= acc_s1;
            valid_s2 <= valid_s1;
        end
    end

    // ── Stage 3a: pairwise adder tree, level 1 (registered) ─────
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
            if (valid_s2) begin
                wt_l1_hi_s3a <= wt_l1_hi_comb;
                wt_l1_lo_s3a <= wt_l1_lo_comb;
                acc_s3a      <= acc_s2;
            end
            valid_s3a <= valid_s2;
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