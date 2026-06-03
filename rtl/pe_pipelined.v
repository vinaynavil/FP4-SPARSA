`timescale 1ns/1ps
// ============================================================
// Module : pe_pipelined
// Project : FP4-SPARSA v15d - Pre-decoded weights (TPU-style)
//
//  Changes from v15c:
//    - weight_in port: [15:0] packed FP4 → [31:0] pre-decoded
//      (4 lanes × signed 8-bit, decoded once at load time
//       in systolic_array; removes fp4_decoder from weight
//       critical path. Activation decode unchanged.)
//    - u_dec_w instances removed from decode_lanes generate.
//    - w_dec[ln] wired directly from weight_in slices.
//    - All other logic, ports, pipeline stages unchanged.
// ============================================================
module pe_pipelined (
    input  wire        clk,
    input  wire        rst,
    input  wire        sparse_en,
    input  wire        mode,
    input  wire [31:0] weight_dec_in,
    input  wire [15:0] weight_raw_in,      // PRE-DECODED: 4 × signed [7:0]
    input  wire [15:0] act_in,
    output reg  [15:0] act_out,
    input  wire [17:0] acc_in,
    output wire [17:0] acc_out,
    input  wire        valid_in,
    output reg         valid_out,
    output wire        skip_out,
    output wire        act_zero_out,
    output wire        wgt_zero_out,
    output wire [2:0]  lane_skip_count
);

// ── Unpack pre-decoded weights (from systolic_array registers) ─
// weight_dec_in[31:0] =
//   { lane3[7:0], lane2[7:0], lane1[7:0], lane0[7:0] }
//
// Each lane already decoded from FP4 → signed 8-bit fixed-point
// during load_weight in systolic_array (TPU-style predecode).
wire signed [7:0] w_dec [0:3];

assign w_dec[0] = $signed(weight_dec_in[ 7: 0]);
assign w_dec[1] = $signed(weight_dec_in[15: 8]);
assign w_dec[2] = $signed(weight_dec_in[23:16]);
assign w_dec[3] = $signed(weight_dec_in[31:24]);
// ── Unpack raw FP4 lanes (activations, for sparsity detection) ─
wire [3:0] a_lane [0:3];
assign a_lane[0] = act_in[ 3: 0];
assign a_lane[1] = act_in[ 7: 4];
assign a_lane[2] = act_in[11: 8];
assign a_lane[3] = act_in[15:12];

// ── Activation decode (still per-cycle, not pre-decoded) ───────
wire signed [7:0] a_dec [0:3];
genvar ln;
generate
    for (ln = 0; ln < 4; ln = ln + 1) begin : decode_lanes
        fp4_decoder u_dec_a (.fp4_in(a_lane[ln]), .decoded(a_dec[ln]));
    end
endgenerate

// ── INT4 paths (mode=1) ────────────────────────────────────────
// weight_raw_in[15:0] stores the original packed FP4 nibbles.
//
// In INT4 mode, each nibble is reinterpreted directly as signed
// INT4 and sign-extended to signed 8-bit.
//
// Layout:
//   [15:12] = lane3
//   [11: 8] = lane2
//   [ 7: 4] = lane1
//   [ 3: 0] = lane0
wire [3:0] w_lane_raw [0:3];

assign w_lane_raw[0] = weight_raw_in[ 3: 0];
assign w_lane_raw[1] = weight_raw_in[ 7: 4];
assign w_lane_raw[2] = weight_raw_in[11: 8];
assign w_lane_raw[3] = weight_raw_in[15:12];

wire signed [7:0] w_int4 [0:3];
wire signed [7:0] a_int4 [0:3];
generate
    for (ln = 0; ln < 4; ln = ln + 1) begin : int4_lanes
        assign w_int4[ln] = {{4{w_lane_raw[ln][3]}}, w_lane_raw[ln]};
        assign a_int4[ln] = {{4{a_lane[ln][3]}},     a_lane[ln]};
    end
endgenerate

// ── Operand mux: FP4 decoded vs INT4 ──────────────────────────
wire signed [7:0] w_op [0:3];
wire signed [7:0] a_op [0:3];
generate
    for (ln = 0; ln < 4; ln = ln + 1) begin : op_mux
        assign w_op[ln] = mode ? w_int4[ln] : w_dec[ln];
        assign a_op[ln] = mode ? a_int4[ln] : a_dec[ln];
    end
endgenerate

// ── Sparsity detection ─────────────────────────────────────────
// Weight zero: check pre-decoded value == 0
// (equivalent to checking original FP4 bits[2:0]==0, kept consistent)
wire act_zero_lane [0:3];
wire wgt_zero_lane [0:3];
generate
    for (ln = 0; ln < 4; ln = ln + 1) begin : g_sparse
        assign act_zero_lane[ln] = (a_lane[ln][2:0] == 3'b000);
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

assign act_zero_out = act_zero_all;
assign wgt_zero_out = wgt_zero_all;
assign skip_out     = sparse_en & (lane_zero[0] | lane_zero[1] |
                                   lane_zero[2] | lane_zero[3]);
assign lane_skip_count = sparse_en ?
    ({2'b00, lane_zero[0]} + {2'b00, lane_zero[1]} +
     {2'b00, lane_zero[2]} + {2'b00, lane_zero[3]}) : 3'd0;

// ── Stage 1 registers ──────────────────────────────────────────
reg signed [7:0]  w_s1 [0:3];
reg signed [7:0]  a_s1 [0:3];
reg signed [17:0] acc_s1;
reg               valid_s1;
reg [15:0]        act_pass_s1;
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
        act_pass_s1  <= 16'd0;
        sparse_en_s1 <= 1'b0;
    end else begin
        for (i = 0; i < 4; i = i + 1) begin
            w_s1[i]         <= w_op[i];
            a_s1[i]         <= a_op[i];
            lane_zero_s1[i] <= lane_zero[i];
        end
        acc_s1       <= acc_in;
        valid_s1     <= valid_in;
        act_pass_s1  <= act_in;
        sparse_en_s1 <= sparse_en;
    end
end

// ── Stage 2: multiply + Wallace tree adder ─────────────────────
(* use_dsp = "yes" *) reg signed [16:0] prod_s2 [0:3];
reg signed [17:0] wt_l1_hi_s2;
reg signed [17:0] wt_l1_lo_s2;
reg signed [17:0] acc_s2;
reg               valid_s2;
reg [15:0]        act_pass_s2;

always @(posedge clk) begin
    if (rst) begin
        prod_s2[0]  <= 17'sd0; prod_s2[1] <= 17'sd0;
        prod_s2[2]  <= 17'sd0; prod_s2[3] <= 17'sd0;
        wt_l1_hi_s2 <= 18'sd0;
        wt_l1_lo_s2 <= 18'sd0;
        acc_s2      <= 18'sd0;
        valid_s2    <= 1'b0;
        act_pass_s2 <= 16'd0;
    end else begin
        prod_s2[0] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[0]))
                      ? 17'sd0 : w_s1[0] * a_s1[0];
        prod_s2[1] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[1]))
                      ? 17'sd0 : w_s1[1] * a_s1[1];
        prod_s2[2] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[2]))
                      ? 17'sd0 : w_s1[2] * a_s1[2];
        prod_s2[3] <= (~valid_s1 || (sparse_en_s1 & lane_zero_s1[3]))
                      ? 17'sd0 : w_s1[3] * a_s1[3];

        wt_l1_hi_s2 <= $signed({prod_s2[0][16], prod_s2[0]}) +
                        $signed({prod_s2[1][16], prod_s2[1]});
        wt_l1_lo_s2 <= $signed({prod_s2[2][16], prod_s2[2]}) +
                        $signed({prod_s2[3][16], prod_s2[3]});
        acc_s2      <= acc_s1;
        valid_s2    <= valid_s1;
        act_pass_s2 <= act_pass_s1;
    end
end

// ── Stage 3: accumulate + saturate ────────────────────────────
wire signed [18:0] mac_result =
    (~valid_s2) ? $signed({acc_s2[17], acc_s2})
                : $signed({acc_s2[17], acc_s2}) +
                  $signed({wt_l1_hi_s2[17], wt_l1_hi_s2}) +
                  $signed({wt_l1_lo_s2[17], wt_l1_lo_s2});

wire sat_overflow = (mac_result[18] != mac_result[17]);
wire [17:0] sat_result = sat_overflow
    ? (mac_result[18] ? 18'h20000 : 18'h1FFFF)
    : mac_result[17:0];

reg signed [17:0] acc_s3;
reg               valid_s3;

always @(posedge clk) begin
    if (rst) begin
        acc_s3    <= 18'sd0;
        valid_s3  <= 1'b0;
        valid_out <= 1'b0;
        act_out   <= 16'd0;
    end else begin
        acc_s3    <= sat_result;
        valid_s3  <= valid_s2;
        valid_out <= valid_s3;
        act_out   <= act_pass_s2;
    end
end

assign acc_out = acc_s3;

endmodule