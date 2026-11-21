`timescale 1ns/1ps
// ============================================================================
// Module      : pe_pipelined
// Project     : FP4-SPARSA (FP4 Sparse Precision Accelerator using a Reconfigurable Systolic Array)
// Version     : 20.0
// ============================================================================
//
// Pipeline Architecture:
//   Stage 1  : Input latching + pre-computed zero mask evaluation.
//              act_in, act_zero_in, valid_in registered with 1-cycle delay
//              to provide horizontal systolic skewing across columns.
//   Stage 2  : 4 × signed 8×8 multipliers with dual-operand isolation gating.
//   Stage 3a : Pairwise adder tree (Level 1 partial sums, 13-bit signed, registered).
//   Stage 3b : 13-bit signed accumulation (acc_in used directly at stage 3b, acc_s3).
//   Output   : acc_out = acc_s3 (wire, 4 cycles end-to-end latency).
//
// Signal Flow & Timing:
//   - Horizontal: act_out, act_zero_out, valid_out output at Stage 1 (1 cycle).
//   - Vertical  : acc_in used directly at Stage 3b; acc_out passes to next row PE at Stage 3b.
//   - Internal  : valid_s1/s2/s3a pipeline gates datapath and preserves results.
// ============================================================
module pe_pipelined (
    input  wire        clk,
    input  wire        rst,
    input  wire        sparse_en,
    input  wire        mode,
    input  wire [31:0] weight_dec_in,
    input  wire [15:0] weight_raw_in,
    input  wire [3:0]  wgt_zero_in,     // Pre-computed weight zero mask
    input  wire [15:0] act_in,
    input  wire [31:0] act_dec_in,
    input  wire [3:0]  act_zero_in,     // Pre-computed activation zero mask
    output reg  [15:0] act_out,         // 1-cycle delayed act_in
    output reg  [3:0]  act_zero_out,    // 1-cycle delayed activation zero mask
    input  wire [12:0] acc_in,
    output wire [12:0] acc_out,
    input  wire        valid_in,
    output reg         valid_out        // 1-cycle delayed valid_in, aligned with act_out
);

    // ── Operand decode wires ─────────────────────────────────
    wire signed [7:0] w_dec [0:3];
    assign w_dec[0] = $signed(weight_dec_in[ 7: 0]);
    assign w_dec[1] = $signed(weight_dec_in[15: 8]);
    assign w_dec[2] = $signed(weight_dec_in[23:16]);
    assign w_dec[3] = $signed(weight_dec_in[31:24]);
    
    wire [3:0] w_lane_raw [0:3];
    assign w_lane_raw[0] = weight_raw_in[ 3: 0];
    assign w_lane_raw[1] = weight_raw_in[ 7: 4];
    assign w_lane_raw[2] = weight_raw_in[11: 8];
    assign w_lane_raw[3] = weight_raw_in[15:12];

    wire signed [7:0] a_dec [0:3];
    assign a_dec[0] = $signed(act_dec_in[ 7: 0]);
    assign a_dec[1] = $signed(act_dec_in[15: 8]);
    assign a_dec[2] = $signed(act_dec_in[23:16]);
    assign a_dec[3] = $signed(act_dec_in[31:24]); 
    
    wire [3:0] a_lane [0:3];
    assign a_lane[0] = act_in[ 3: 0];
    assign a_lane[1] = act_in[ 7: 4];
    assign a_lane[2] = act_in[11: 8];
    assign a_lane[3] = act_in[15:12];

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

    // ── Sparsity evaluation (pre-computed zero masks) ────────
    wire lane_zero [0:3];
    assign lane_zero[0] = act_zero_in[0] | wgt_zero_in[0];
    assign lane_zero[1] = act_zero_in[1] | wgt_zero_in[1];
    assign lane_zero[2] = act_zero_in[2] | wgt_zero_in[2];
    assign lane_zero[3] = act_zero_in[3] | wgt_zero_in[3];

    // ── Stage 1: operand register + skip pre-computation ─────
    reg signed [7:0]  w_s1 [0:3];
    reg signed [7:0]  a_s1 [0:3];
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
            valid_s1     <= 1'b0;
            act_out      <= 16'd0;   // ← systolic pass-through: 1-cycle delay on act_in
            act_zero_out <= 4'd0;
            valid_out    <= 1'b0;    // ← systolic pass-through: 1-cycle delay on valid_in
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
            valid_s1     <= valid_in;
            act_out      <= act_in;
            act_zero_out <= act_zero_in;
            valid_out    <= valid_in;  // 1-cycle delay, matches act_out
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
    reg               valid_s2;

    always @(posedge clk) begin
        if (rst) begin
            prod_s2[0] <= 17'sd0; prod_s2[1] <= 17'sd0;
            prod_s2[2] <= 17'sd0; prod_s2[3] <= 17'sd0;
            valid_s2 <= 1'b0;
        end else begin
            prod_s2[0] <= valid_s1 ? (w_iso[0] * a_iso[0]) : 17'sd0;
            prod_s2[1] <= valid_s1 ? (w_iso[1] * a_iso[1]) : 17'sd0;
            prod_s2[2] <= valid_s1 ? (w_iso[2] * a_iso[2]) : 17'sd0;
            prod_s2[3] <= valid_s1 ? (w_iso[3] * a_iso[3]) : 17'sd0;
            valid_s2 <= valid_s1;
        end
    end

    // ── Stage 3a: pairwise adder tree, level 1 (registered) ─────
    // Two adder pairs → two 13-bit partial sums, registered to
    // break the DSP MREG=1 critical path.
    wire signed [12:0] wt_l1_hi_comb =
        $signed(prod_s2[0]) +
        $signed(prod_s2[1]);

    wire signed [12:0] wt_l1_lo_comb =
        $signed(prod_s2[2]) +
        $signed(prod_s2[3]);

    reg signed [12:0] wt_l1_hi_s3a;
    reg signed [12:0] wt_l1_lo_s3a;
    reg               valid_s3a;

    always @(posedge clk) begin
        if (rst) begin
            wt_l1_hi_s3a <= 13'sd0;
            wt_l1_lo_s3a <= 13'sd0;
            valid_s3a    <= 1'b0;
        end else begin
            if (valid_s2) begin
                wt_l1_hi_s3a <= wt_l1_hi_comb;
                wt_l1_lo_s3a <= wt_l1_lo_comb;
            end
            valid_s3a <= valid_s2;
        end
    end

    // ── Stage 3b: accumulate (13-bit signed) ────────────────
    wire signed [12:0] acc_sum =
        $signed(acc_in)
        + wt_l1_hi_s3a
        + wt_l1_lo_s3a;

    reg signed [12:0] acc_s3;

    always @(posedge clk) begin
        if (rst) begin
            acc_s3 <= 13'sd0;
        end else begin
            // Only update acc_s3 when valid data exits the pipeline.
            // This preserves the last computed result after valid_in goes low.
            if (valid_s3a) begin
                acc_s3 <= acc_sum;
            end
        end
    end

    assign acc_out = acc_s3;

endmodule
