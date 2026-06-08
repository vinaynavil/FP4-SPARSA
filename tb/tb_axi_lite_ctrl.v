`timescale 1ns / 1ps
// ============================================================
// Testbench : tb_axi_lite_ctrl
// Tests AXI-Lite register write and read-back for
// fp4_sparsa_4x4 v17
// ============================================================
module tb_axi_lite_ctrl;

    // --------------------------------------------------------
    // Clock & reset
    // --------------------------------------------------------
    reg clk = 0;
    reg aresetn = 0;
    always #1.43 clk = ~clk;   // ~350 MHz

    // --------------------------------------------------------
    // AXI-Lite signals
    // --------------------------------------------------------
    reg  [3:0]  awaddr;
    reg         awvalid;
    wire        awready;
    reg  [31:0] wdata;
    reg  [3:0]  wstrb;
    reg         wvalid;
    wire        wready;
    wire [1:0]  bresp;
    wire        bvalid;
    reg         bready;
    reg  [3:0]  araddr;
    reg         arvalid;
    wire        arready;
    wire [31:0] rdata;
    wire [1:0]  rresp;
    wire        rvalid;
    reg         rready;

    // Dummy status inputs
    reg        valid_out      = 1;
    reg [6:0]  zero_skip_count = 7'd42;
    reg [5:0]  sparse_act_count = 6'd15;
    reg [5:0]  sparse_wgt_count = 6'd7;
    reg [3:0]  sat_flags      = 4'b1010;

    // Control outputs to check
    wire sparse_en, mode, bank_switch;
    wire load_weight_lo, load_weight_hi;

    // --------------------------------------------------------
    // DUT
    // --------------------------------------------------------
    axi_lite_ctrl dut (
        .s_axi_aclk       (clk),
        .s_axi_aresetn    (aresetn),
        .s_axi_awaddr     (awaddr),
        .s_axi_awvalid    (awvalid),
        .s_axi_awready    (awready),
        .s_axi_wdata      (wdata),
        .s_axi_wstrb      (wstrb),
        .s_axi_wvalid     (wvalid),
        .s_axi_wready     (wready),
        .s_axi_bresp      (bresp),
        .s_axi_bvalid     (bvalid),
        .s_axi_bready     (bready),
        .s_axi_araddr     (araddr),
        .s_axi_arvalid    (arvalid),
        .s_axi_arready    (arready),
        .s_axi_rdata      (rdata),
        .s_axi_rresp      (rresp),
        .s_axi_rvalid     (rvalid),
        .s_axi_rready     (rready),
        .valid_out        (valid_out),
        .zero_skip_count  (zero_skip_count),
        .sparse_act_count (sparse_act_count),
        .sparse_wgt_count (sparse_wgt_count),
        .sat_flags        (sat_flags),
        .sparse_en        (sparse_en),
        .mode             (mode),
        .bank_switch      (bank_switch),
        .load_weight_lo   (load_weight_lo),
        .load_weight_hi   (load_weight_hi)
    );

    // --------------------------------------------------------
    // AXI write task
    // --------------------------------------------------------
    task axi_write;
        input [3:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk);
            awaddr  = addr;   awvalid = 1;
            wdata   = data;   wstrb   = 4'hF;   wvalid = 1;
            bready  = 1;
            @(posedge clk);
            while (!awready) @(posedge clk);
            awvalid = 0;
            while (!wready)  @(posedge clk);
            wvalid  = 0;
            while (!bvalid)  @(posedge clk);
            @(posedge clk);
            bready  = 0;
        end
    endtask

    // AXI read task
    task axi_read;
        input  [3:0]  addr;
        output [31:0] data;
        begin
            @(posedge clk);
            araddr  = addr;  arvalid = 1;
            rready  = 1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            arvalid = 0;
            while (!rvalid)  @(posedge clk);
            data = rdata;
            @(posedge clk);
            rready  = 0;
        end
    endtask

    // --------------------------------------------------------
    // Stimulus
    // --------------------------------------------------------
    integer errors = 0;
    reg [31:0] rd;

    initial begin
        // Init
        awaddr=0; awvalid=0; wdata=0; wstrb=0; wvalid=0; bready=0;
        araddr=0; arvalid=0; rready=0;

        // Release reset
        repeat(4) @(posedge clk);
        aresetn = 1;
        repeat(2) @(posedge clk);

        // ---- TC1: Write CTRL register (sparse_en=1, mode=1, bank_switch=1) ----
        $display("TC1: Write CTRL = 0x07");
        axi_write(4'h0, 32'h00000007);
        if (sparse_en !== 1 || mode !== 1 || bank_switch !== 1) begin
            $display("  FAIL: sparse_en=%b mode=%b bank_switch=%b", sparse_en, mode, bank_switch);
            errors = errors + 1;
        end else
            $display("  PASS");

        // ---- TC2: Read back CTRL register ----
        $display("TC2: Read CTRL");
        axi_read(4'h0, rd);
        if (rd[2:0] !== 3'b111) begin
            $display("  FAIL: rdata[2:0]=%b", rd[2:0]);
            errors = errors + 1;
        end else
            $display("  PASS: rdata=0x%08X", rd);

        // ---- TC3: Write WEIGHT_CTRL — load_weight_lo pulse ----
        $display("TC3: Pulse load_weight_lo");
        axi_write(4'h4, 32'h00000001);
        // load_weight_lo should have been 1 for one cycle during write
        // After the transaction it auto-clears — check it's 0 now
        @(posedge clk);
        if (load_weight_lo !== 0) begin
            $display("  FAIL: load_weight_lo did not auto-clear");
            errors = errors + 1;
        end else
            $display("  PASS: pulse auto-cleared");

        // ---- TC4: Read STATUS register ----
        $display("TC4: Read STATUS");
        axi_read(4'h8, rd);
        // Expected: valid_out=1, zero_skip=42, sparse_act=15, sparse_wgt=7, sat_flags=1010
        // [0]=1, [7:1]=42, [13:8]=15, [19:14]=7, [23:20]=1010
        $display("  STATUS = 0x%08X", rd);
        if (rd[0] !== 1'b1) begin
            $display("  FAIL: valid_out bit"); errors=errors+1;
        end
        if (rd[7:1] !== 7'd42) begin
            $display("  FAIL: zero_skip_count=%0d", rd[7:1]); errors=errors+1;
        end
        if (rd[13:8] !== 6'd15) begin
            $display("  FAIL: sparse_act_count=%0d", rd[13:8]); errors=errors+1;
        end
        if (rd[19:14] !== 6'd7) begin
            $display("  FAIL: sparse_wgt_count=%0d", rd[19:14]); errors=errors+1;
        end
        if (rd[23:20] !== 4'b1010) begin
            $display("  FAIL: sat_flags=%b", rd[23:20]); errors=errors+1;
        end
        if (errors == 0) $display("  PASS");

        // ---- TC5: Clear sparse_en ----
        $display("TC5: Clear sparse_en via CTRL write");
        axi_write(4'h0, 32'h00000006);  // mode=1, bank_switch=1, sparse_en=0
        if (sparse_en !== 0) begin
            $display("  FAIL: sparse_en still asserted"); errors=errors+1;
        end else
            $display("  PASS");

        // ---- Summary ----
        repeat(4) @(posedge clk);
        if (errors == 0)
            $display("\n*** ALL TESTS PASSED ***");
        else
            $display("\n*** %0d TEST(S) FAILED ***", errors);

        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
