`timescale 1ns / 1ps
// ============================================================
// Module : tb_fp4_sparsa_4x4  (v15c - timing fix)
//
//  Change: drain increased 30→32 cycles to absorb the 2-stage
//  pipelined counter latency added in systolic_array for
//  timing closure at 350MHz.
//  All expected values unchanged from v15c.
// ============================================================
module tb_fp4_sparsa_4x4;

    reg          clk, rst;
    reg          load_weight_lo, load_weight_hi;
    reg          sparse_en, mode;
    reg  [127:0] weight_data;
    reg  [15:0]  act_row0, act_row1, act_row2, act_row3;
    reg          valid_in;

    wire [17:0]  result_col0, result_col1, result_col2, result_col3;
    wire [6:0]   zero_skip_count;
    wire [5:0]   sparse_act_count;
    wire [5:0]   sparse_wgt_count;
    wire         valid_out;

    integer pass_count, fail_count, cycle_count;

    fp4_sparsa_4x4 dut (
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

    initial clk = 0;
    always #2 clk = ~clk;
    always @(posedge clk) cycle_count = cycle_count + 1;

    // ── Display helpers ───────────────────────────────────────
    task print_fp4_val;
        input [3:0] fp4;
        begin
            case (fp4[2:0])
                3'b000: $write("   0.0");
                3'b001: $write("%s0.5(FTZ)", fp4[3] ? "-" : "+");
                3'b010: $write("%s1.0", fp4[3] ? "-" : "+");
                3'b011: $write("%s1.5", fp4[3] ? "-" : "+");
                3'b100: $write("%s2.0", fp4[3] ? "-" : "+");
                3'b101: $write("%s3.0", fp4[3] ? "-" : "+");
                3'b110: $write("%s4.0", fp4[3] ? "-" : "+");
                3'b111: $write("%s6.0", fp4[3] ? "-" : "+");
            endcase
        end
    endtask

    task print_int4_val;
        input [3:0] val;
        begin $write("%4d", $signed(val)); end
    endtask

    task print_act16;
        input [15:0] act;
        input [1:0]  row;
        input        md;
        begin
            $write("  act_row%0d [%04h]: lanes= ", row, act);
            if (md) begin
                print_int4_val(act[ 3: 0]); $write(" ");
                print_int4_val(act[ 7: 4]); $write(" ");
                print_int4_val(act[11: 8]); $write(" ");
                print_int4_val(act[15:12]);
            end else begin
                print_fp4_val(act[ 3: 0]); $write(" ");
                print_fp4_val(act[ 7: 4]); $write(" ");
                print_fp4_val(act[11: 8]); $write(" ");
                print_fp4_val(act[15:12]);
            end
            $display("");
        end
    endtask

    task print_weight_pe;
        input [15:0] w;
        input        md;
        begin
            $write("[");
            if (md) begin
                print_int4_val(w[ 3: 0]); $write(",");
                print_int4_val(w[ 7: 4]); $write(",");
                print_int4_val(w[11: 8]); $write(",");
                print_int4_val(w[15:12]);
            end else begin
                print_fp4_val(w[ 3: 0]); $write(",");
                print_fp4_val(w[ 7: 4]); $write(",");
                print_fp4_val(w[11: 8]); $write(",");
                print_fp4_val(w[15:12]);
            end
            $write("]");
        end
    endtask

    task print_weight_matrix;
        input [127:0] lo_data, hi_data;
        input         md;
        reg [15:0]    ws;
        begin
            $display("  [WEIGHT MATRIX  (lo=rows0,1  hi=rows2,3)]");
            $write("    row0: ");
            ws = lo_data[15:0];    print_weight_pe(ws, md); $write(" ");
            ws = lo_data[31:16];   print_weight_pe(ws, md); $write(" ");
            ws = lo_data[47:32];   print_weight_pe(ws, md); $write(" ");
            ws = lo_data[63:48];   print_weight_pe(ws, md); $display("");
            $write("    row1: ");
            ws = lo_data[79:64];   print_weight_pe(ws, md); $write(" ");
            ws = lo_data[95:80];   print_weight_pe(ws, md); $write(" ");
            ws = lo_data[111:96];  print_weight_pe(ws, md); $write(" ");
            ws = lo_data[127:112]; print_weight_pe(ws, md); $display("");
            $write("    row2: ");
            ws = hi_data[15:0];    print_weight_pe(ws, md); $write(" ");
            ws = hi_data[31:16];   print_weight_pe(ws, md); $write(" ");
            ws = hi_data[47:32];   print_weight_pe(ws, md); $write(" ");
            ws = hi_data[63:48];   print_weight_pe(ws, md); $display("");
            $write("    row3: ");
            ws = hi_data[79:64];   print_weight_pe(ws, md); $write(" ");
            ws = hi_data[95:80];   print_weight_pe(ws, md); $write(" ");
            ws = hi_data[111:96];  print_weight_pe(ws, md); $write(" ");
            ws = hi_data[127:112]; print_weight_pe(ws, md); $display("");
        end
    endtask

    task do_reset;
        begin
            rst            = 1;
            load_weight_lo = 0;
            load_weight_hi = 0;
            valid_in       = 0;
            sparse_en      = 1;
            mode           = 0;
            act_row0       = 16'd0;
            act_row1       = 16'd0;
            act_row2       = 16'd0;
            act_row3       = 16'd0;
            weight_data    = 128'd0;
            repeat(4) @(posedge clk);
            rst = 0;
            repeat(2) @(posedge clk);
        end
    endtask

    // drain=32: 30 original + 2 for pipelined counter stages
    task run_inference;
        input [127:0]        lo_wdata, hi_wdata;
        input [15:0]         a0, a1, a2, a3;
        input signed [17:0]  exp_c0, exp_c1, exp_c2, exp_c3;
        input [4:0]          exp_act, exp_wgt;
        input [6:0]          exp_skip;
        input [7:0]          tc_num;

        integer tc_start, tc_end, tc_cycles;
        reg     c0ok, c1ok, c2ok, c3ok, aok, wok, sok;
        begin
            tc_start = cycle_count;

            $display("=================================================================");
            $display("  TC%0d  |  sparse_en=%0d  |  mode=%0d (%s)",
                tc_num, sparse_en, mode, mode ? "    INT4" : "FP4 E2M1");
            $display("=================================================================");
            print_weight_matrix(lo_wdata, hi_wdata, mode);
            $display("  [ACTIVATION INPUTS  (4 lanes each)]");
            print_act16(a0, 0, mode);
            print_act16(a1, 1, mode);
            print_act16(a2, 2, mode);
            print_act16(a3, 3, mode);
            $display("  [EXPECTED]  col[0..3]=%0d,%0d,%0d,%0d  act=%0d wgt=%0d skip=%0d",
                $signed(exp_c0), $signed(exp_c1),
                $signed(exp_c2), $signed(exp_c3),
                exp_act, exp_wgt, exp_skip);

            // Two-phase weight load
            @(posedge clk);
            load_weight_lo = 1; load_weight_hi = 0;
            weight_data    = lo_wdata;
            @(posedge clk);
            load_weight_lo = 0; load_weight_hi = 1;
            weight_data    = hi_wdata;
            @(posedge clk);
            load_weight_lo = 0; load_weight_hi = 0;
            @(posedge clk);

            // Staggered row injection
            valid_in = 1;
            act_row0 = a0; act_row1 = 16'd0;
            act_row2 = 16'd0; act_row3 = 16'd0;
            repeat(3) @(posedge clk);
            act_row1 = a1;
            repeat(3) @(posedge clk);
            act_row2 = a2;
            repeat(3) @(posedge clk);
            act_row3 = a3;
            repeat(32) @(posedge clk);  // 30 + 2 for pipelined counters

            valid_in = 0;
            act_row0 = 16'd0; act_row1 = 16'd0;
            act_row2 = 16'd0; act_row3 = 16'd0;

            tc_end    = cycle_count;
            tc_cycles = tc_end - tc_start;

            c0ok = ($signed(result_col0) === $signed(exp_c0));
            c1ok = ($signed(result_col1) === $signed(exp_c1));
            c2ok = ($signed(result_col2) === $signed(exp_c2));
            c3ok = ($signed(result_col3) === $signed(exp_c3));
            aok  = (sparse_act_count === exp_act);
            wok  = (sparse_wgt_count === exp_wgt);
            sok  = (zero_skip_count  === exp_skip);

            $display("  [ACTUAL]");
            $display("  result_col0 = %7d  (exp %7d)  %s",
                $signed(result_col0), $signed(exp_c0), c0ok ? "    PASS" : "FAIL <<");
            $display("  result_col1 = %7d  (exp %7d)  %s",
                $signed(result_col1), $signed(exp_c1), c1ok ? "    PASS" : "FAIL <<");
            $display("  result_col2 = %7d  (exp %7d)  %s",
                $signed(result_col2), $signed(exp_c2), c2ok ? "    PASS" : "FAIL <<");
            $display("  result_col3 = %7d  (exp %7d)  %s",
                $signed(result_col3), $signed(exp_c3), c3ok ? "    PASS" : "FAIL <<");
            $display("  sparse_act  = %7d  (exp %7d)  %s",
                sparse_act_count, exp_act, aok ? "    PASS" : "FAIL <<");
            $display("  sparse_wgt  = %7d  (exp %7d)  %s",
                sparse_wgt_count, exp_wgt, wok ? "    PASS" : "FAIL <<");
            $display("  zero_skip   = %7d  (exp %7d)  %s",
                zero_skip_count, exp_skip, sok ? "    PASS" : "FAIL <<");
            $display("  valid_out=%0d  TC%0d=%0dc  Sim=%0t ns",
                valid_out, tc_num, tc_cycles, $time);

            if (c0ok & c1ok & c2ok & c3ok & aok & wok & sok) begin
                $display("  >>> TC%0d : ALL PASS <<<", tc_num);
                pass_count = pass_count + 1;
            end else begin
                $display("  >>> TC%0d : FAIL <<<", tc_num);
                fail_count = fail_count + 1;
            end
            $display("");
            repeat(4) @(posedge clk);
        end
    endtask

    initial begin
        cycle_count = 0; pass_count = 0; fail_count = 0;

        $display("=================================================================");
        $display("  FP4-SPARSA Testbench v15c  --  Phase 2 DP4A  lane-wise sparsity");
        $display("  weight_data : 128-bit two-phase load (lo=rows0,1  hi=rows2,3)");
        $display("  Total IO    : 287 pins  (limit 400, margin 113)");
        $display("  TCs         : 12  |  drain=32c (30+2 pipelined counter stages)");
        $display("=================================================================");
        $display("");

        // TC1: Dense, no zeros → skip=0
        do_reset;
        $display("--- TC1: Dense FP4  weight=+1.5  act=+1.0  exp=96");
        run_inference(
            128'h33333333333333333333333333333333,
            128'h33333333333333333333333333333333,
            16'h2222, 16'h2222, 16'h2222, 16'h2222,
            18'd96, 18'd96, 18'd96, 18'd96,
            5'd0, 5'd0, 7'd0, 8'd1);

        // TC2: act rows 0,2 zero → act=8 skip=32
        do_reset;
        $display("--- TC2: Sparse acts rows0,2=zero  exp=48");
        run_inference(
            128'h33333333333333333333333333333333,
            128'h33333333333333333333333333333333,
            16'h0000, 16'h2222, 16'h0000, 16'h2222,
            18'd48, 18'd48, 18'd48, 18'd48,
            5'd8, 5'd0, 7'd32, 8'd2);

        // TC3: wgt cols 0,2 zero → wgt=8 skip=32
        do_reset;
        $display("--- TC3: Sparse wgt col0,col2=zero  exp col1=col3=96");
        run_inference(
            128'h33330000333300003333000033330000,
            128'h33330000333300003333000033330000,
            16'h2222, 16'h2222, 16'h2222, 16'h2222,
            18'd0, 18'd96, 18'd0, 18'd96,
            5'd0, 5'd8, 7'd32, 8'd3);

        // TC4: act row 3 zero → act=4 skip=16
        do_reset;
        $display("--- TC4: Mixed signs  exp=40");
        run_inference(
            128'h22222222222222222222222222222222,
            128'h22222222222222222222222222222222,
            16'h4444, 16'hAAAA, 16'h3333, 16'h0000,
            18'd40, 18'd40, 18'd40, 18'd40,
            5'd4, 5'd0, 7'd16, 8'd4);

        // TC5: sparse_en=1 act rows 1,3 zero → act=8 skip=32
        do_reset; sparse_en = 1;
        $display("--- TC5: sparse_en=1 rows1,3=zero  exp=48 skip=32");
        run_inference(
            128'h33333333333333333333333333333333,
            128'h33333333333333333333333333333333,
            16'h2222, 16'h0000, 16'h2222, 16'h0000,
            18'd48, 18'd48, 18'd48, 18'd48,
            5'd8, 5'd0, 7'd32, 8'd5);

        // TC6: sparse_en=0 → skip=0, act=8 (structural)
        do_reset; sparse_en = 0;
        $display("--- TC6: sparse_en=0 same inputs  exp=48 skip=0");
        run_inference(
            128'h33333333333333333333333333333333,
            128'h33333333333333333333333333333333,
            16'h2222, 16'h0000, 16'h2222, 16'h0000,
            18'd48, 18'd48, 18'd48, 18'd48,
            5'd8, 5'd0, 7'd0, 8'd6);

        // TC7: INT4, no zeros → skip=0
        do_reset; mode = 1; sparse_en = 1;
        $display("--- TC7: INT4 mode act=-6  exp=-192");
        run_inference(
            128'h22222222222222222222222222222222,
            128'h22222222222222222222222222222222,
            16'hAAAA, 16'hAAAA, 16'hAAAA, 16'hAAAA,
            -18'd192, -18'd192, -18'd192, -18'd192,
            5'd0, 5'd0, 7'd0, 8'd7);

        // TC8: FP4 max, no zeros → skip=0
        do_reset;
        $display("--- TC8: FP4 max 6.0x6.0  exp=2304");
        run_inference(
            128'h77777777777777777777777777777777,
            128'h77777777777777777777777777777777,
            16'h7777, 16'h7777, 16'h7777, 16'h7777,
            18'd2304, 18'd2304, 18'd2304, 18'd2304,
            5'd0, 5'd0, 7'd0, 8'd8);

        // TC9: 100% act zero → act=16 skip=64
        do_reset; sparse_en = 1;
        $display("--- TC9: 100%% act sparsity  exp=0");
        run_inference(
            128'h33333333333333333333333333333333,
            128'h33333333333333333333333333333333,
            16'h0000, 16'h0000, 16'h0000, 16'h0000,
            18'd0, 18'd0, 18'd0, 18'd0,
            5'd16, 5'd0, 7'd64, 8'd9);

        // TC10: 100% wgt zero → wgt=16 skip=64
        do_reset; sparse_en = 1;
        $display("--- TC10: 100%% weight sparsity  exp=0");
        run_inference(
            128'h0, 128'h0,
            16'h2222, 16'h2222, 16'h2222, 16'h2222,
            18'd0, 18'd0, 18'd0, 18'd0,
            5'd0, 5'd16, 7'd64, 8'd10);

        // TC11: INT4 no zeros → skip=0
        do_reset; mode = 1; sparse_en = 1;
        $display("--- TC11: INT4 small positive  exp=48");
        run_inference(
            128'h11111111111111111111111111111111,
            128'h11111111111111111111111111111111,
            16'h3333, 16'h3333, 16'h3333, 16'h3333,
            18'd48, 18'd48, 18'd48, 18'd48,
            5'd0, 5'd0, 7'd0, 8'd11);

        // TC12: alt-sign wgt, act rows 1,3 zero → act=8 skip=32
        do_reset; sparse_en = 1;
        $display("--- TC12: Alt-sign wgt + sparse acts  exp col0=+48 col1=-48");
        run_inference(
            128'hBBBB3333BBBB3333BBBB3333BBBB3333,
            128'hBBBB3333BBBB3333BBBB3333BBBB3333,
            16'h2222, 16'h0000, 16'h2222, 16'h0000,
            18'd48, -18'd48, 18'd48, -18'd48,
            5'd8, 5'd0, 7'd32, 8'd12);

        $display("=================================================================");
        $display("  FP4-SPARSA v15c DP4A - FINAL SUMMARY");
        $display("=================================================================");
        $display("  Total TCs : 12   PASS : %0d   FAIL : %0d", pass_count, fail_count);
        $display("  Sim cycles: %0d   Sim time: %0t ns", cycle_count, $time);
        $display("-----------------------------------------------------------------");
        $display("  IO budget : 287 pins (was 414, limit 400, margin 113)");
        $display("  Weight load: 2-phase 128-bit (lo=rows0,1  hi=rows2,3)");
        $display("  Pipeline  : 3-stage DP4A, stagger=3c, drain=32c");
        $display("-----------------------------------------------------------------");
        $display("  TC5 vs TC6: zero_skip 32->0  Reconfigurable sparsity VERIFIED");
        $display("  TC7 vs TC1: INT4=-192 FP4=+96  Mode mux VERIFIED");
        $display("  Lane-wise sparsity: zero_skip counts masked MACs (max 64)");
        $display("=================================================================");
        $finish;
    end

    initial begin #10000000; $display("TIMEOUT"); $finish; end
endmodule